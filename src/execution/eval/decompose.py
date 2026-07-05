"""Authoritative full-val three-baseline decomposition for trained agents.

The training loop logs a *400-episode* validation-curve value (`n_val_eval`), a
cheap proxy that can drift from the truth by a few hundredths of a bp. This module
re-scores each saved run bundle on the ENTIRE validation split through the same
`run_episodes` loop and fill engine as the benchmarks, against THREE baselines, so
the reported number is the paired, full-population one used for the gate:

    * fixed-TWAP   -- equal absolute slices (the headline benchmark).
    * always-1x    -- the action-grid `mult = 1` policy = adaptive TWAP
                      (`remaining / steps_left`); the mechanical pacing reference.
    * agent        -- the trained model (deterministic).

Reported decomposition (per size, per seed, overall + per regime):

    adaptive-pacing value = always-1x  - fixed-TWAP   (mechanical, seed-independent)
    learned-timing value  = agent      - always-1x    (what attribution would explain)
    total                 = agent      - fixed-TWAP

with the paired Wilcoxon signed-rank p-value and a paired-bootstrap 95% CI on each
mean difference (+ => the strategy is WORSE, i.e. higher implementation shortfall).
Each seed's on-policy action distribution is also recorded, so a degenerate policy
(e.g. DQN's deadline-dump collapse: ~all mass on the 0.0 multiple) is visible rather
than hidden inside an averaged bp.

The env is reconstructed via `agents.train.setup_data` / `_val_env`, so the
normalizer, per-regime residual, split, and order size are byte-identical to the
run that produced each bundle. TEST is never loaded.

CLI:
    PYTHONPATH=src .venv/bin/python -m execution.eval.decompose \
        --scratch-root "<scratch_hyperliquid>" --config configs/experiment_10s_10min.yaml \
        --runs-dir "<scratch>/runs_10s_10min" --sizes 96.57 193.13 --algos dqn ppo \
        --out "<scratch>/runs_10s_10min/full_val_decomposition.json"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon
from stable_baselines3 import DQN, PPO

from execution.agents.train import _val_env, setup_data
from execution.env.benchmarks import TWAP
from execution.eval.runner import run_episodes

_ALGOS = {"dqn": DQN, "ppo": PPO}


# --------------------------------------------------------------------- policies
class AlwaysMultiple:
    """Always select the action-grid index for a fixed pace multiple.

    With ``mult = 1`` in ``twap_pace_multiple`` mode this is *adaptive* TWAP:
    ``qty = 1 * remaining / steps_left`` at every step (self-correcting, so it
    fully executes). It is the mechanical pacing baseline the learned policy is
    measured against.
    """

    absolute = False

    def __init__(self, action_index: int) -> None:
        self._a = int(action_index)

    def reset(self, env: Any = None) -> None:  # noqa: ARG002 -- protocol parity
        pass

    def action(self, obs: Any = None, info: Any = None) -> int:  # noqa: ARG002
        return self._a


class RecordingSB3Policy:
    """Trained SB3 model as a deterministic policy that tallies its own actions."""

    absolute = False

    def __init__(self, model: Any, n_actions: int) -> None:
        self.model = model
        self.counts = np.zeros(n_actions, dtype=np.int64)

    def reset(self, env: Any = None) -> None:  # noqa: ARG002
        pass

    def action(self, obs: Any, info: Any = None) -> int:  # noqa: ARG002
        a, _ = self.model.predict(obs, deterministic=True)
        a = int(a)
        self.counts[a] += 1
        return a


# ----------------------------------------------------------------------- paired
def _bootstrap_ci(diff: np.ndarray, *, n_boot: int = 10_000, seed: int = 0) -> list[float]:
    """Paired-bootstrap 95% CI on the mean of ``diff`` (resample episodes w/ replacement)."""
    if diff.size == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    means = diff[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _paired(strategy: np.ndarray, baseline: np.ndarray) -> dict:
    """Paired stats for (strategy - baseline); + => strategy has HIGHER IS (worse)."""
    diff = np.asarray(strategy, float) - np.asarray(baseline, float)
    out: dict[str, Any] = {
        "n": int(diff.size),
        "mean_diff_bps": float(diff.mean()) if diff.size else float("nan"),
        "median_diff_bps": float(np.median(diff)) if diff.size else float("nan"),
        "n_nonzero": int(np.count_nonzero(diff)),
        "ci95_bps": _bootstrap_ci(diff),
    }
    nz = diff[diff != 0.0]
    if nz.size:
        try:
            out["wilcoxon_p"] = float(wilcoxon(nz)[1])
        except ValueError:
            out["wilcoxon_p"] = float("nan")
    else:
        out["wilcoxon_p"] = float("nan")   # every episode identical (all full-fill)
    return out


def _by_regime(df_s: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    """`_paired` overall and split by regime, aligned on episode_index."""
    m = df_s[["episode_index", "regime", "is_bps"]].merge(
        df_b[["episode_index", "is_bps"]], on="episode_index", suffixes=("_s", "_b"))
    res = {"all": _paired(m["is_bps_s"].to_numpy(), m["is_bps_b"].to_numpy())}
    for reg, g in m.groupby("regime"):
        res[str(reg)] = _paired(g["is_bps_s"].to_numpy(), g["is_bps_b"].to_numpy())
    return res


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description="Full-val three-baseline agent decomposition")
    ap.add_argument("--scratch-root", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--runs-dir", required=True, help="dir holding <algo>_size<size>_seed<seed>/")
    ap.add_argument("--sizes", type=float, nargs="+", required=True)
    ap.add_argument("--algos", nargs="+", default=["dqn", "ppo"], choices=list(_ALGOS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    obs_features = tuple(cfg["obs"]["features"])
    N = cfg["env"]["n_steps"]
    actions = [float(a) for a in cfg["env"]["actions"]]
    onex_idx = actions.index(1.0)   # the mult=1 (adaptive-TWAP) action
    runs_dir = Path(args.runs_dir)

    print("setup: loading data, splitting, calibrating residual (train_sub) ...")
    setup = setup_data(args.scratch_root, cfg)
    full_idx = list(range(setup.val.n_episodes))
    print(f"  val episodes (FULL): {len(full_idx)}  (train logged only "
          f"{cfg.get('agent', {}).get('n_val_eval', '?')})")

    report: dict[str, Any] = {
        "config": args.config, "n_steps": N, "actions": actions,
        "n_val_episodes": len(full_idx), "sizes": {},
    }
    for size in args.sizes:
        val_env = _val_env(setup, size=size, cfg=cfg, obs_features=obs_features)
        twap = run_episodes(val_env, TWAP(size, N), full_idx)
        onex = run_episodes(val_env, AlwaysMultiple(onex_idx), full_idx)
        pacing = _by_regime(onex, twap)   # always-1x - fixed-TWAP (mechanical)
        print(f"\n=== size {size:g} BTC ===")
        print(f"  adaptive-pacing (always-1x - fixed-TWAP), all: "
              f"{pacing['all']['mean_diff_bps']:+.4f} bps  p={pacing['all']['wilcoxon_p']:.3g}")

        seeds_out: dict[str, Any] = {}
        for algo in args.algos:
            for seed in args.seeds:
                d = runs_dir / f"{algo}_size{size:g}_seed{seed}"
                mdl = d / "model.zip"
                if not mdl.exists():
                    print(f"  [skip] {d.name}: no model.zip")
                    continue
                model = _ALGOS[algo].load(mdl, device="cpu")
                pol = RecordingSB3Policy(model, len(actions))
                agent = run_episodes(val_env, pol, full_idx)
                full_exec = float((agent["inventory_left"].abs() < 1e-6).mean())
                resid_freq = float((agent["residual_btc"] > 1e-9).mean())
                rec = {
                    "vs_fixed_twap": _by_regime(agent, twap),   # total
                    "vs_always_1x": _by_regime(agent, onex),    # learned-timing
                    "full_exec_frac": full_exec,
                    "residual_freq": resid_freq,
                    "action_dist": {str(a): int(c) for a, c in zip(actions, pol.counts)},
                    "action_frac": {str(a): float(c / pol.counts.sum())
                                    for a, c in zip(actions, pol.counts)},
                }
                seeds_out[f"{algo}_seed{seed}"] = rec
                tot = rec["vs_fixed_twap"]["all"]
                lt = rec["vs_always_1x"]["all"]
                top = max(rec["action_frac"], key=rec["action_frac"].get)
                print(f"  {algo} s{seed}: total {tot['mean_diff_bps']:+.4f} "
                      f"(p={tot['wilcoxon_p']:.3g})  learned-timing {lt['mean_diff_bps']:+.4f} "
                      f"(p={lt['wilcoxon_p']:.3g})  exec {full_exec*100:.0f}%  "
                      f"top-action {top}x@{rec['action_frac'][top]*100:.0f}%")

        report["sizes"][f"{size:g}"] = {"adaptive_pacing": pacing, "seeds": seeds_out}

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nwritten -> {args.out}")


if __name__ == "__main__":
    main()
