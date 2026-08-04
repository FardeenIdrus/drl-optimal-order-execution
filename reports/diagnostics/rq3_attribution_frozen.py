"""RQ3 attribution on the FROZEN-REPLAY track, split by market regime.

WHY THIS EXISTS SEPARATELY. The reacting-simulator attribution (rq3_attribution.py) works on a
28-dimensional observation of which 2K entries are raw queue-slot values; those had to be
grouped into blocks before they could be interpreted, and the grouping is a construct. The
frozen-replay observation is SEVEN individually named dimensions:

    inventory remaining, time remaining, spread, queue imbalance, recent return,
    rolling volatility, ask depth

so attribution here is directly readable and needs no grouping decision at all.

WHY IT MATTERS MORE THAN THE REACTIVE ATTRIBUTION. Queue imbalance is the SAME signal this
study measured on the venue, injected into the simulator at its real strength, and showed a
one-line rule could monetise for 0.313/0.625 bps. Attribution on this track therefore answers a
question nothing else can: do the agents place any weight on the specific feature that was
demonstrably worth money? A near-zero imbalance share would give the null a named mechanism --
they ignored the thing that paid -- rather than leaving it as an absence of skill.

SPLIT BY REGIME. Episodes carry calm/volatile labels, so each agent is attributed twice, once
per regime, on episodes of that regime only. (An earlier draft of the RQ3 scope excluded this
track on the stated ground that it had no regime split. That was simply wrong -- the labels are
in the data -- and the exclusion is withdrawn.)

SPLIT USED: VALIDATION only. The test split is spent; attribution is a diagnostic and does not
need it, so there is no question of re-touching a sealed block.

MEMORY. Frozen-replay stores are the largest objects in this project and have frozen this
machine before. Exactly ONE dataset is resident at a time; the store is deleted and collected
before the next is loaded, and peak RSS is reported.
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

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
OUT = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4"
           "/rq3_attribution")
VAL_SUBSET_SEED = 12345

DATASETS = {                      # runs dir -> config
    "runs": "configs/experiment.yaml",
    "runs_10s": "configs/experiment_10s.yaml",
    "runs_10s_10min": "configs/experiment_10s_10min.yaml",
}


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def eval_indices(n: int, k: int) -> np.ndarray:
    rng = np.random.default_rng(VAL_SUBSET_SEED)
    return np.sort(rng.choice(n, size=min(k, n), replace=False))



def guarded_out(path, force: bool):
    """Refuse to silently overwrite an existing result file.

    An earlier version of this script built its output name without the algorithm, so a
    second pass over the same runs directory OVERWROTE the first and 28 rows were lost
    without any warning. The name is fixed, but a name can be got wrong again; this makes
    the failure loud instead of silent. Pass --force to overwrite deliberately.
    """
    if path.exists() and not force:
        raise SystemExit(
            f"REFUSING TO OVERWRITE an existing result file:\n    {path}\n"
            f"Delete it, change --tag, or pass --force if overwriting is intended.")
    return path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, choices=list(DATASETS))
    ap.add_argument("--algo", default="ppo")
    ap.add_argument("--n-episodes", type=int, default=60)
    ap.add_argument("--max-states", type=int, default=200)
    ap.add_argument("--background", type=int, default=40)
    ap.add_argument("--nsamples", type=int, default=200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing result file (off by default)")
    args = ap.parse_args()

    import shap
    from execution.env.episode_store import EpisodeStore
    from execution.env.normalize import FeatureNormalizer
    from execution.env.real_data_env import RealDataExecutionEnv
    from execution.eval.runner import run_episodes
    from execution.agents.policy import SB3Policy
    from stable_baselines3 import PPO, DQN

    REPO = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((REPO / DATASETS[args.runs]).read_text())
    n_steps = cfg["env"]["n_steps"]
    actions = cfg["env"]["actions"]
    obs_features = tuple(cfg["obs"]["features"])
    val_frac = cfg["split"]["val_frac"]
    ds = S / cfg["dataset_dir"]

    FEATURE_NAMES = ["inventory remaining", "time remaining"] + list(obs_features)
    PRETTY = {"spread_bps": "spread", "imbalance": "queue imbalance",
              "recent_return": "recent return", "rolling_vol": "rolling volatility",
              "ask_depth": "ask depth"}
    FEATURE_NAMES = [PRETTY.get(f, f) for f in FEATURE_NAMES]

    runs = sorted(d for d in (S / args.runs).iterdir()
                  if d.is_dir() and (d / "model.zip").exists()
                  and d.name.startswith(args.algo))
    if args.limit:
        runs = runs[:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = guarded_out(OUT / f"rq3_attribution_frozen_{args.runs}_{args.algo}{args.tag}.json", args.force)
    print(f"{len(runs)} {args.algo.upper()} agents on {args.runs} "
          f"({len(FEATURE_NAMES)} named features)", flush=True)

    # ---- ONE store resident at a time ------------------------------------------------
    full = EpisodeStore.from_parquet(ds / "train.parquet", n_steps=n_steps)
    train_sub, store = full.chrono_split(val_frac)
    norm_refit = FeatureNormalizer.fit(train_sub.features, train_sub.feature_names)
    del full, train_sub
    gc.collect()
    print(f"  validation store: {store.n_episodes} episodes, peak RSS {peak_rss_mb():.0f} MB",
          flush=True)

    idx_all = eval_indices(store.n_episodes, 400)
    regimes = np.asarray(store.regime)[idx_all]
    by_regime = {r: idx_all[regimes == r] for r in np.unique(regimes)}
    print(f"  regime split: " + ", ".join(f"{r} {len(v)}" for r, v in by_regime.items()),
          flush=True)

    results, t0 = [], time.time()
    for i, rd in enumerate(runs, 1):
        meta = json.loads((rd / "meta.json").read_text())
        size = float(meta["size_btc"])
        resid = {k: float(v) for k, v in meta["residual_coef"].items()}
        saved_norm = FeatureNormalizer.from_json(rd / "normalizer.json")
        model = (PPO if args.algo == "ppo" else DQN).load(str(rd / "model.zip"), device="cpu")

        row = {"run": rd.name, "algo": args.algo, "dataset": args.runs,
               "size_btc": size, "features": FEATURE_NAMES, "by_regime": {}}
        for regime, idxs in by_regime.items():
            if len(idxs) < 5:
                continue
            env = RealDataExecutionEnv(
                store, initial_inventory=size, actions=actions,
                action_mode=cfg["env"]["action_mode"], obs_features=obs_features,
                normalizer=saved_norm, test_mode=True,
                residual_coef_by_regime=resid, adv_btc=meta.get("adv_btc"))
            seen: list = []

            class Capture(SB3Policy):
                def action(self, obs, info=None):
                    if len(seen) < args.max_states:
                        seen.append(np.asarray(obs, dtype=np.float32))
                    return super().action(obs, info)

            run_episodes(env, Capture(model), idxs[:args.n_episodes])
            states = np.array(seen[:args.max_states], dtype=np.float32)
            if len(states) < args.background + 10:
                continue
            rng = np.random.default_rng(0)
            bg = states[rng.choice(len(states), args.background, replace=False)]

            def f(x):
                a, _ = model.predict(np.asarray(x, dtype=np.float32), deterministic=True)
                return np.asarray(a, dtype=float).reshape(-1)

            sv = np.asarray(shap.KernelExplainer(f, bg)
                            .shap_values(states, nsamples=args.nsamples, silent=True))
            if sv.ndim == 3:
                sv = sv[..., 0]
            mean_abs = np.abs(sv).mean(axis=0)
            tot = float(mean_abs.sum())
            share = {n: float(v / tot) if tot > 0 else 0.0
                     for n, v in zip(FEATURE_NAMES, mean_abs)}
            acts, _ = model.predict(states, deterministic=True)
            acts = np.asarray(acts).reshape(-1)
            p = np.bincount(acts, minlength=len(actions)).astype(float)
            p /= p.sum()
            row["by_regime"][str(regime)] = {
                "n_states": int(len(states)),
                "mean_abs_shap": {n: float(v) for n, v in zip(FEATURE_NAMES, mean_abs)},
                "share": share,
                "imbalance_share": share.get("queue imbalance", 0.0),
                "top_action": float(actions[int(np.argmax(p))]),
                "top_action_share": float(p.max()),
            }
            del env
        results.append(row)
        msg = "  ".join(f"{r}: imbalance {v['imbalance_share']:.1%}"
                        for r, v in row["by_regime"].items())
        print(f"  [{i}/{len(runs)}] {rd.name:<26} {msg}", flush=True)
        out_path.write_text(json.dumps(results, indent=1))
        del model
        gc.collect()

    del store
    gc.collect()
    print(f"wrote {out_path}  ({len(results)} agents, {(time.time()-t0)/60:.1f} min, "
          f"peak RSS {peak_rss_mb():.0f} MB)", flush=True)


if __name__ == "__main__":
    main()
