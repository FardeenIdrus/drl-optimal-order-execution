"""Train DQN (and, in Step 7, PPO) on the real-data execution env.

Pipeline per run: load ``train.parquet`` -> chronological ``(train_sub, val)``
split -> fit the normalizer on ``train_sub`` ONLY (leakage-free) -> calibrate the
per-regime square-root residual on ``train_sub`` -> train on ``train_sub`` with the
bps-scaled reward, logging the paired validation curve -> save a self-contained run
bundle (model + normalizer + resolved config + curve + timing). The test set is
never touched here.

Machine-agnostic: all paths come from ``--scratch-root`` / ``--out-root`` and the
device defaults to CPU (the probe showed MPS is ~9x slower for this small net), so
the identical command runs locally or on a server.

CLI:
    PYTHONPATH=src .venv/bin/python -m execution.agents.train \
        --scratch-root "<scratch_hyperliquid>" --algo dqn
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3.common.monitor import Monitor

from execution.agents.build import make_dqn, make_ppo
from execution.agents.callbacks import ValidationCurveCallback
from execution.agents.policy import SB3Policy
from execution.env.benchmarks import TWAP
from execution.env.calibration import calibrate_temporary_impact
from execution.env.episode_store import EpisodeStore
from execution.env.normalize import FeatureNormalizer
from execution.env.real_data_env import RealDataExecutionEnv
from execution.eval.runner import run_episodes

Q_GRID = [1, 2, 5, 10, 20]   # BTC probe sizes for the impact calibration (as in runner)


@dataclass(frozen=True)
class TrainSetup:
    """Leakage-free training inputs derived once and shared across seeds."""

    train_sub: EpisodeStore
    val: EpisodeStore
    normalizer: FeatureNormalizer
    residual_coef: dict[str, float]
    adv: float | None


def setup_data(scratch_root: str | Path, cfg: dict) -> TrainSetup:
    """Load train.parquet, carve the chronological validation split, fit the
    train-sub normalizer, and calibrate the per-regime residual coefficient."""
    ds = Path(scratch_root) / cfg.get("dataset_dir", "dataset")   # dataset_10s at 10s
    n_steps = cfg["env"]["n_steps"]
    full = EpisodeStore.from_parquet(ds / "train.parquet", n_steps=n_steps)
    train_sub, val = full.chrono_split(cfg["split"]["val_frac"])
    normalizer = FeatureNormalizer.fit(train_sub.features, train_sub.feature_names)
    adv = cfg["env"].get("adv_btc")
    residual_coef = {
        reg: calibrate_temporary_impact(train_sub, Q_GRID, adv=adv, regime=reg).sqrt_coef
        for reg in ("calm", "volatile")
    }
    return TrainSetup(train_sub, val, normalizer, residual_coef, adv)


def _val_env(setup: TrainSetup, *, size: float, cfg: dict, obs_features: tuple[str, ...]):
    """Validation env in force-completion (test_mode) accounting with the residual."""
    return RealDataExecutionEnv(
        setup.val, initial_inventory=size, actions=tuple(cfg["env"]["actions"]),
        action_mode=cfg["env"].get("action_mode", "twap_pace_multiple"),
        obs_features=obs_features, normalizer=setup.normalizer, test_mode=True,
        residual_coef_by_regime=setup.residual_coef, adv_btc=setup.adv,
    )


def train_one_seed(setup: TrainSetup, *, size: float, cfg: dict, seed: int,
                   total_timesteps: int, out_dir: Path, algo: str = "dqn",
                   device: str = "cpu", eval_freq: int = 100_000, n_val_eval: int = 400,
                   verbose: int = 0) -> dict:
    """Train a single seed of ``algo`` ("dqn" or "ppo"); save the run bundle and
    return its metadata. Both algorithms share the env, reward, action grid, split,
    and order size, so a run differs only by the algorithm and its hyperparameters."""
    obs_features = tuple(cfg["obs"]["features"])
    acfg = cfg["agent"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fixed validation subset (same across seeds -> comparable curves).
    rng = np.random.default_rng(12345)
    n_val = min(n_val_eval, setup.val.n_episodes)
    val_indices = sorted(rng.choice(setup.val.n_episodes, size=n_val, replace=False).tolist())

    val_env = _val_env(setup, size=size, cfg=cfg, obs_features=obs_features)
    twap_is = run_episodes(val_env, TWAP(size, cfg["env"]["n_steps"]), val_indices)["is_bps"].to_numpy()

    train_env = Monitor(RealDataExecutionEnv(
        setup.train_sub, initial_inventory=size, actions=tuple(cfg["env"]["actions"]),
        action_mode=cfg["env"].get("action_mode", "twap_pace_multiple"),
        obs_features=obs_features, normalizer=setup.normalizer, test_mode=False,
        reward_bps=True, leftover_penalty_bps=acfg["leftover_penalty_bps"], seed=seed,
    ))
    if algo == "dqn":
        model = make_dqn(train_env, acfg["dqn"], seed=seed, device=device,
                         total_timesteps=total_timesteps)
    elif algo == "ppo":
        model = make_ppo(train_env, acfg["ppo"], seed=seed, device=device)
    else:
        raise ValueError(f"unknown algo: {algo!r}")
    cb = ValidationCurveCallback(val_env, val_indices, twap_is, eval_freq=eval_freq,
                                 csv_path=out_dir / "curve.csv", verbose=verbose)

    t0 = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=False)
    train_time = time.perf_counter() - t0

    model.save(out_dir / "model.zip")
    setup.normalizer.to_json(out_dir / "normalizer.json")
    final = cb.rows[-1] if cb.rows else {}
    meta = {
        "algo": algo, "size_btc": size, "seed": seed,
        "total_timesteps": total_timesteps, "train_time_s": round(train_time, 1),
        "device": device, "reward_bps": True,
        "leftover_penalty_bps": acfg["leftover_penalty_bps"],
        "n_val_eval": n_val, "eval_freq": eval_freq,
        "adv_btc": setup.adv, "residual_coef": setup.residual_coef,
        f"{algo}_hp": acfg[algo], "obs_features": list(obs_features),
        "val_vs_twap_final": final.get("val_vs_twap_mean"),
        "val_full_exec_final": final.get("val_full_exec_frac"),
        "val_residual_freq_final": final.get("val_residual_freq"),
    }
    if algo == "dqn":
        meta["exploration_fraction"] = float(model.exploration_fraction)  # budget-derived
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def _plot_curves(run_dirs: list[Path], out_path: Path, title: str) -> None:
    """Overlay each seed's paired validation curve (val - TWAP, bps) vs timesteps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for d in run_dirs:
        csv = d / "curve.csv"
        if not csv.exists():
            continue
        import pandas as pd
        c = pd.read_csv(csv)
        ax.plot(c["timesteps"], c["val_vs_twap_mean"], alpha=0.8, label=d.name)
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("timesteps")
    ax.set_ylabel("validation IS - TWAP (bps, paired mean)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train DQN or PPO on the real-data execution env")
    ap.add_argument("--scratch-root", required=True)
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--algo", default="dqn", choices=["dqn", "ppo"])
    ap.add_argument("--size", type=float, default=None, help="order size BTC (default: primary)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="override config seeds")
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--eval-freq", type=int, default=100_000)
    ap.add_argument("--n-val-eval", type=int, default=400)
    ap.add_argument("--out-root", default=None, help="default: <scratch>/runs")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    acfg = cfg["agent"]
    size = args.size if args.size is not None else cfg["env"]["initial_inventory"]
    seeds = args.seeds if args.seeds is not None else acfg["seeds"]
    total = args.total_timesteps if args.total_timesteps is not None else acfg["total_timesteps"]
    out_root = Path(args.out_root) if args.out_root else Path(args.scratch_root) / "runs"

    print(f"setup: loading data, splitting, calibrating residual ...")
    setup = setup_data(args.scratch_root, cfg)
    print(f"  train_sub {setup.train_sub.n_episodes} | val {setup.val.n_episodes} eps | "
          f"size {size} BTC | seeds {seeds} | budget {total:,} | device {args.device}")
    print(f"  residual_coef {setup.residual_coef}")

    run_dirs = []
    for seed in seeds:
        out_dir = out_root / f"{args.algo}_size{size:g}_seed{seed}"
        print(f"\n[{args.algo} seed {seed}] training -> {out_dir}")
        meta = train_one_seed(setup, size=size, cfg=cfg, seed=seed, total_timesteps=total,
                              out_dir=out_dir, algo=args.algo, device=args.device,
                              eval_freq=args.eval_freq, n_val_eval=args.n_val_eval, verbose=1)
        run_dirs.append(out_dir)
        print(f"[seed {seed}] done in {meta['train_time_s']}s  "
              f"val_vs_twap_final={meta['val_vs_twap_final']:+.3f} bps  "
              f"full_exec={meta['val_full_exec_final']*100:.1f}%")

    plot_path = out_root / f"{args.algo}_size{size:g}_curves.png"
    _plot_curves(run_dirs, plot_path, f"{args.algo.upper()} val-vs-TWAP, size {size:g} BTC")
    print(f"\nall seeds done. curves -> {plot_path}")


if __name__ == "__main__":
    main()
