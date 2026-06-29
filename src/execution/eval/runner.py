"""Evaluation runner: drive a policy over dataset episodes and collect metrics.

Step 1 uses this to validate the environment end-to-end: run TWAP across the
held-out test episodes and report the implementation-shortfall distribution, plus
the sanity diagnostics (instant liquidation = worst cost; do-nothing =
force-execution penalty). The loop is generic and carries forward to the
Layer-1/2 evaluation in Step 5.

CLI (smoke test):
    PYTHONPATH=src .venv/bin/python -m execution.eval.runner \
        --scratch-root "<path-to-scratch_hyperliquid>" --split test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from execution.env.benchmarks import (
    AlmgrenChriss,
    FixedLiquidityParticipation,
    TWAP,
)
from execution.env.calibration import (
    calibrate_temporary_impact,
    estimate_relative_daily_vol,
    estimate_volatility,
)
from execution.env.episode_store import EpisodeStore
from execution.env.normalize import FeatureNormalizer
from execution.env.real_data_env import RealDataExecutionEnv


# ----------------------------------------------------------------- diagnostics
class InstantLiquidation:
    """Buy the entire meta-order at the first step (maximum impact baseline)."""

    absolute = True

    def __init__(self, initial_inventory: float) -> None:
        self.q = float(initial_inventory)

    def reset(self, env) -> None:  # noqa: ARG002
        self._spent = False

    def action(self, obs=None, info=None) -> float:
        if self._spent:
            return 0.0
        self._spent = True
        return self.q


class DoNothing:
    """Never trade; the whole order is force-completed at the deadline."""

    absolute = True

    def reset(self, env) -> None:  # noqa: ARG002
        pass

    def action(self, obs=None, info=None) -> float:
        return 0.0


# -------------------------------------------------------------------- the loop
def run_episodes(env: RealDataExecutionEnv, policy, episode_indices) -> pd.DataFrame:
    """Run ``policy`` over the given episode indices; return a per-episode frame."""
    env.set_absolute_quantity(getattr(policy, "absolute", False))
    rows = []
    for idx in episode_indices:
        obs, info = env.reset(options={"episode_index": idx})
        policy.reset(env)
        done = False
        while not done:
            obs, reward, done, _, info = env.step(policy.action(obs, info))
        rows.append({
            "episode_index": idx,
            "episode_id": info["episode_id"],
            "regime": info["regime"],
            "arrival_price": info["arrival_price"],
            "realized_cost": info["realized_cost"],
            "is_bps": info["is_bps"],
            "residual_btc": info["residual_btc"],
            "residual_bps": info["residual_bps"],
            "filled": env.n_filled,
            "inventory_left": info["inventory"],
        })
    return pd.DataFrame(rows)


def _stats(df: pd.DataFrame) -> dict:
    s = df["is_bps"]
    return {
        "n": int(len(s)), "mean": float(s.mean()), "median": float(s.median()),
        "std": float(s.std()), "p05": float(s.quantile(0.05)), "p95": float(s.quantile(0.95)),
    }


def summarize(df: pd.DataFrame) -> dict:
    out = {"all": _stats(df)}
    for reg, g in df.groupby("regime"):
        out[str(reg)] = _stats(g)
    return out


def _fmt(name: str, st: dict) -> str:
    return (f"  {name:9s} n={st['n']:6d}  mean={st['mean']:8.3f}  median={st['median']:8.3f}"
            f"  std={st['std']:8.3f}  [p05 {st['p05']:7.3f}, p95 {st['p95']:7.3f}] bps")


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase-2 Step-1 env smoke test (TWAP on real data)")
    ap.add_argument("--scratch-root", required=True, help="path to scratch_hyperliquid/")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--max-episodes", type=int, default=None, help="cap episodes (default: all)")
    args = ap.parse_args()

    cfg_all = yaml.safe_load(Path(args.config).read_text())
    cfg = cfg_all["env"]
    obs_features = tuple(cfg_all.get("obs", {}).get("features", []))
    ds_dir = Path(args.scratch_root) / "dataset"

    # The feature normalizer is ALWAYS fit on TRAIN (leakage-safe), then frozen and
    # applied to whichever split we evaluate. Persisted for training/eval reuse.
    train_store = EpisodeStore.from_parquet(ds_dir / "train.parquet", n_steps=cfg["n_steps"])
    normalizer = FeatureNormalizer.fit(train_store.features, train_store.feature_names)
    normalizer.to_json(ds_dir / "normalization.json")

    store = (train_store if args.split == "train"
             else EpisodeStore.from_parquet(ds_dir / "test.parquet", n_steps=cfg["n_steps"]))
    n = store.n_episodes if args.max_episodes is None else min(args.max_episodes, store.n_episodes)
    idxs = list(range(n))

    inv = cfg["initial_inventory"]
    N = cfg["n_steps"]
    adv = cfg.get("adv_btc")

    # Calibrate impact from the TRAIN book (real, leakage-safe): linear eta (the AC
    # reference) AND the per-regime square-root coefficient (the deadline residual).
    q_grid = [1, 2, 5, 10, 20]
    sigma = estimate_volatility(train_store)
    cal = calibrate_temporary_impact(train_store, q_grid, adv=adv)

    residual_coef_by_regime: dict[str, float] = {}
    print("\n=== Phase 3 benchmarks (real Hyperliquid L2) ===")
    print(f"split: {args.split}   episodes: {n}   meta-order: {inv} BTC over {N} steps   "
          f"ADV: {adv} BTC\n")
    print("Calibrated impact (TRAIN book):")
    print(f"  sigma (USD per-min mid vol): {sigma:.4f}")
    print(f"  eta (linear temp-impact, USD/BTC^2): {cal.eta:.6f}   R^2={cal.r2_linear:.3f}  "
          f"(R^2<1 => real book is convex, not linear -> AC is a reference)")
    print(f"  sqrt_coef (Y*sigma, deadline residual): {cal.sqrt_coef:.4f}   R^2={cal.r2_sqrt:.3f}")
    for reg in ("calm", "volatile"):
        sig_rel = estimate_relative_daily_vol(train_store, regime=reg)
        c_r = calibrate_temporary_impact(train_store, q_grid, adv=adv, regime=reg)
        residual_coef_by_regime[reg] = c_r.sqrt_coef
        y_impl = (c_r.sqrt_coef / sig_rel) if (sig_rel and np.isfinite(sig_rel)) else float("nan")
        print(f"  [{reg:8}] eta={c_r.eta:.6f} (R^2={c_r.r2_linear:.3f})  "
              f"sqrt_coef={c_r.sqrt_coef:.4f} (R^2={c_r.r2_sqrt:.3f})  "
              f"sigma_daily={sig_rel:.4f}  implied Y={y_impl:.3f}")
    print("\nImplied AC risk-aversion (lambda = kappa^2 * eta / sigma^2):")
    for kT in (1.0, 2.0):
        lam = (kT / N) ** 2 * cal.eta / sigma ** 2 if sigma > 0 else float("nan")
        print(f"  urgency kappa*T={kT:.0f}  ->  lambda={lam:.3e}")

    env = RealDataExecutionEnv(
        store, initial_inventory=inv, final_penalty=cfg["final_penalty"],
        actions=tuple(cfg["actions"]), obs_features=obs_features, normalizer=normalizer,
        test_mode=True, residual_coef_by_regime=residual_coef_by_regime, adv_btc=adv,
    )

    # Core benchmarks, all priced through the same real-L2 fill engine.
    benches = {
        "TWAP": TWAP(inv, N),
        "AlmgrenChriss(kT=1)": AlmgrenChriss.from_urgency(inv, N, 1.0),
        "AlmgrenChriss(kT=2)": AlmgrenChriss.from_urgency(inv, N, 2.0),
        "FixedLiqParticip": FixedLiquidityParticipation(fraction=0.2, levels=5),
    }
    print("\nImplementation shortfall (bps vs arrival), per regime:")
    for name, pol in benches.items():
        df = run_episodes(env, pol, idxs)
        s = summarize(df)
        full = (df["inventory_left"].abs() < 1e-6).mean() * 100
        resid_freq = (df["residual_btc"] > 1e-9).mean() * 100
        resid_mean = df.loc[df["residual_btc"] > 1e-9, "residual_btc"].mean()
        print(f"\n{name}  (fully executed {full:.1f}%; deadline residual fired "
              f"{resid_freq:.2f}% of episodes, mean {resid_mean:.3f} BTC):")
        for k in ("all", "calm", "volatile"):
            if k in s:
                print(_fmt(k, s[k]))

    # Sanity floor: every scheduled benchmark should beat dumping the order at once.
    instant = run_episodes(env, InstantLiquidation(inv), idxs)
    print(f"\n  (sanity) instant liquidation mean IS: {instant['is_bps'].mean():.3f} bps "
          f"-- scheduled benchmarks should sit below this")


if __name__ == "__main__":
    main()
