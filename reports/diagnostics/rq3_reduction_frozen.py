"""Policy reduction on the frozen-replay track, extended to ALL THREE datasets.

WHY THIS EXISTS. The reduction -- how much of an agent's behaviour is one constant front-loading
dose -- was already run on the frozen track by the inversion study (l2_inversion_stage3), but
that script is hard-wired to a single dataset, `runs_10s`. Its 30 agents are the whole of that
dataset and nothing was lost, but the RQ3 attribution covers all THREE frozen datasets (35
agents per algorithm). Reporting attribution on three datasets and reduction on one would mean
the two halves of the RQ3 argument rest on different samples, which is exactly the kind of
quiet scope mismatch a reader is entitled to object to.

This script closes that gap: same decomposition, same probe, all three datasets.

    (agent - TWAP)_e  =  alpha + beta * (frontload - TWAP)_e

  beta  = the agent's effective front-loading dose
  alpha = what it achieves that a constant dose does not explain
  r     = how completely one constant reproduces its per-episode pattern

VALIDATION SPLIT ONLY. The sealed test block is spent and its verdict is published and fixed.
This is a descriptive diagnostic: it applies no pass/fail rule, selects no agent and claims no
edge, so it has no business touching sealed data. `runs_10s` is re-measured here as well, on the
validation split, which gives a free consistency check against stage 3's independent
implementation -- if the two disagree, one of them is wrong and I want to know.

MEMORY. Frozen-replay stores are the largest objects in this project and have frozen this
machine before. Exactly ONE store is resident at a time; it is deleted and collected before the
next dataset loads, and peak RSS is reported.
"""
from __future__ import annotations

import argparse
import gc
import json
import resource
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/code/drl-optimal-order-execution")
S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
OUT = S / "oxford_l4" / "rq3_attribution"
VAL_SUBSET_SEED, N_EVAL = 12345, 400

DATASETS = {
    "runs": "configs/experiment.yaml",
    "runs_10s": "configs/experiment_10s.yaml",
    "runs_10s_10min": "configs/experiment_10s_10min.yaml",
}


class FixedPace:
    """The 2.0x pace multiple -- a member of the agents' own action grid, not an outside rule."""
    absolute = False

    def __init__(self, actions, mult):
        self.idx = int(np.argmin([abs(a - mult) for a in actions]))

    def reset(self, env=None):
        pass

    def action(self, obs=None, info=None):
        return self.idx


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def eval_indices(n: int) -> np.ndarray:
    rng = np.random.default_rng(VAL_SUBSET_SEED)
    return np.sort(rng.choice(n, size=min(N_EVAL, n), replace=False))


def guarded_out(path: Path, force: bool) -> Path:
    """Refuse to silently overwrite an existing result file. See rq3_attribution.py."""
    if path.exists() and not force:
        raise SystemExit(
            f"REFUSING TO OVERWRITE an existing result file:\n    {path}\n"
            f"Delete it, change --tag, or pass --force if overwriting is intended.")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=N_EVAL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from execution.env.benchmarks import TWAP
    from execution.env.episode_store import EpisodeStore
    from execution.env.normalize import FeatureNormalizer
    from execution.env.real_data_env import RealDataExecutionEnv
    from execution.eval.runner import run_episodes
    from execution.agents.policy import SB3Policy
    from stable_baselines3 import PPO, DQN

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = guarded_out(OUT / f"rq3_reduction_frozen_all{args.tag}.json", args.force)

    results, t0 = [], time.time()
    for ds_name, cfg_path in DATASETS.items():
        cfg = yaml.safe_load((REPO / cfg_path).read_text())
        n_steps = cfg["env"]["n_steps"]
        actions = cfg["env"]["actions"]
        val_frac = cfg["split"]["val_frac"]
        dd = S / cfg["dataset_dir"]

        runs = sorted(d for d in (S / ds_name).iterdir()
                      if d.is_dir() and (d / "model.zip").exists())
        if args.limit:
            runs = runs[:args.limit]

        # ---- ONE store resident at a time ------------------------------------------------
        full = EpisodeStore.from_parquet(dd / "train.parquet", n_steps=n_steps)
        _train_sub, store = full.chrono_split(val_frac)
        del full, _train_sub
        gc.collect()
        idx = eval_indices(store.n_episodes)
        print(f"[{ds_name}] {len(runs)} agents, validation store {store.n_episodes} episodes, "
              f"{len(idx)} evaluated, peak RSS {peak_rss_mb():.0f} MB", flush=True)

        # TWAP and the probe depend only on (size, residual, adv), not on the agent.
        twap_cache: dict = {}
        for i, rd in enumerate(runs, 1):
            meta = json.loads((rd / "meta.json").read_text())
            algo = meta["algo"]
            size = float(meta["size_btc"])
            resid = {k: float(v) for k, v in meta["residual_coef"].items()}
            adv = meta.get("adv_btc")
            norm = FeatureNormalizer.from_json(rd / "normalizer.json")

            def make_env():
                return RealDataExecutionEnv(
                    store, initial_inventory=size, actions=actions,
                    action_mode=cfg["env"]["action_mode"],
                    obs_features=tuple(cfg["obs"]["features"]),
                    normalizer=norm, test_mode=True,
                    residual_coef_by_regime=resid, adv_btc=adv)

            key = (size, tuple(sorted(resid.items())), adv)
            if key not in twap_cache:
                tw = run_episodes(make_env(), TWAP(size, n_steps), idx)["is_bps"].to_numpy()
                pr = run_episodes(make_env(), FixedPace(actions, 2.0), idx)["is_bps"].to_numpy()
                twap_cache[key] = (tw, pr - tw)
            twap, probe = twap_cache[key]

            model = (PPO if algo == "ppo" else DQN).load(str(rd / "model.zip"), device="cpu")
            agent = run_episodes(make_env(), SB3Policy(model), idx)["is_bps"].to_numpy()
            d = agent - twap

            if np.std(probe) > 1e-12:
                beta, alpha = np.polyfit(probe, d, 1)
                r = float(np.corrcoef(probe, d)[0, 1])
            else:
                beta = alpha = r = float("nan")

            results.append({
                "run": rd.name, "dataset": ds_name, "algo": algo, "size_btc": size,
                "split": "val", "n_episodes": int(len(idx)),
                "mean_diff_bps": float(d.mean()),
                "se_bps": float(d.std(ddof=1) / np.sqrt(len(d))),
                "beta_frontload_dose": float(beta),
                "alpha_bps": float(alpha),
                "corr_with_probe": r,
                "probe_mean_bps": float(probe.mean()),
            })
            print(f"  [{i}/{len(runs)}] {rd.name:<26} mean {d.mean():+.4f}  beta {beta:+.3f}  "
                  f"alpha {alpha:+.4f}  r {r:+.3f}", flush=True)
            out_path.write_text(json.dumps(results, indent=1))
            del model
            gc.collect()

        del store, twap_cache
        gc.collect()

    print(f"\nwrote {out_path}  ({len(results)} agents, {(time.time()-t0)/60:.1f} min, "
          f"peak RSS {peak_rss_mb():.0f} MB)", flush=True)

    # ---- consistency check against stage 3's independent implementation -----------------
    st3 = json.loads((S / "l2_test_results" / "l2_inversion_stage3.json").read_text())
    mine = {r["run"]: r for r in results if r["dataset"] == "runs_10s"}
    diffs = []
    for name, ref in st3["validation"].items():
        if name in mine:
            diffs.append(abs(mine[name]["corr_with_probe"] - ref["corr_with_probe"]))
    if diffs:
        print(f"consistency vs l2_inversion_stage3 on runs_10s validation: "
              f"{len(diffs)} agents, max |delta r| = {max(diffs):.2e}")
        if max(diffs) > 1e-6:
            print("  !! the two implementations DISAGREE -- do not quote either until resolved")


if __name__ == "__main__":
    main()
