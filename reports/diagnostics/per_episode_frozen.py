"""Per-episode paired cost differences for the frozen-replay track, both splits.

WHY A SEPARATE SCRIPT AND NOT AN EDIT TO `test_evaluator.py`. That module produced the sealed
frozen-replay record. Editing it to add a diagnostic output would put the file that generated
the verdicts of record back under the keyboard, and any accidental change to its evaluation
path would silently invalidate a result nobody would think to re-check. This script imports
that module's own helpers and replicates its setup exactly, and every run it scores is checked
against the recorded mean before its array is kept.

WHAT IT DOES. For each of the 30 `runs_10s` agents, on the validation split and on the sealed
test split, it recomputes the per-episode paired difference (agent - TWAP, bps) on the SAME
fixed evaluation subset (VAL_SUBSET_SEED = 12345, n = 400) and saves the full array. The
project has never stored these; only per-run means and p-values survived.

WHY THIS TRACK MATTERS MOST FOR THE QUESTION BEING ASKED. In the simulator a "block" is a
range of common-random-number seeds drawn from one calibrated process, so stable dispersion
across blocks is close to expected. Here the two splits are genuinely different calendar
periods of recorded market data -- and Part C's central finding is that those two periods
behave differently. Whether episode-level DISPERSION is also unstable between them is an open
empirical question with a real chance of coming out against the convenient answer. It is
reported either way.

NO VERDICT IS ISSUED. The test split is already spent; re-scoring it computes a variance, not
a hypothesis test, and cannot create or destroy an edge claim.

INTEGRITY GATE, ABORTING. Every run's recomputed mean paired difference must EXACTLY equal the
recorded value in `l2_test_results/{val_recheck,test_runs_10s}.json`. The l2 inversion
investigation already reproduced every recorded validation number by this route at
max |difference| = 0, so exact equality is the standard, not a tolerance.

Run:  PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python \
        reports/diagnostics/per_episode_frozen.py
"""
from __future__ import annotations

import gc
import json
import resource
import time
from pathlib import Path

import numpy as np
import yaml

from execution.agents.policy import SB3Policy
from execution.env.benchmarks import TWAP
from execution.env.episode_store import EpisodeStore
from execution.env.normalize import FeatureNormalizer
from execution.env.real_data_env import RealDataExecutionEnv
from execution.eval.runner import run_episodes
from execution.eval.test_evaluator import (VAL_SUBSET_SEED, _eval_indices, _load_model,
                                           discover_runs)

SCRATCH = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
CODE = Path(__file__).resolve().parents[2]
OUT = SCRATCH / "oxford_l4" / "per_episode_frozen"
RUNS_DIR = "runs_10s"
CONFIG = CODE / "configs" / "experiment_10s.yaml"
N_EVAL = 400
RECORD = {"val": "val_recheck.json", "test": "test_runs_10s.json"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text())
    env_cfg = cfg["env"]
    n_steps = env_cfg["n_steps"]
    actions = tuple(env_cfg["actions"])
    action_mode = env_cfg.get("action_mode", "twap_pace_multiple")
    obs_features = tuple(cfg["obs"]["features"])
    val_frac = cfg["split"]["val_frac"]
    ds_dir = SCRATCH / cfg.get("dataset_dir", "dataset")
    run_dirs = discover_runs(SCRATCH / RUNS_DIR)

    manifest = {"runs_dir": RUNS_DIR, "config": str(CONFIG), "n_eval": N_EVAL,
                "subset_seed": VAL_SUBSET_SEED, "n_runs_discovered": len(run_dirs),
                "integrity_gate": "recomputed mean == recorded mean, EXACTLY, per run",
                "splits": {}}
    t_all = time.time()

    for split in ("val", "test"):
        rec_doc = json.loads((SCRATCH / "l2_test_results" / RECORD[split]).read_text())
        recorded = {r["run"]: r["mean_paired_diff_bps"]
                    for ds in rec_doc["datasets"] for r in ds["runs"]}

        full_train = EpisodeStore.from_parquet(ds_dir / "train.parquet", n_steps=n_steps)
        if split == "val":
            _train_sub, store = full_train.chrono_split(val_frac)
        else:
            store = EpisodeStore.from_parquet(ds_dir / "test.parquet", n_steps=n_steps)
        idx = _eval_indices(store.n_episodes, N_EVAL)

        arrays, rows, twap_cache = {}, {}, {}
        for rd in run_dirs:
            meta = json.loads((rd / "meta.json").read_text())
            size = float(meta["size_btc"])
            residual_coef = {k: float(v) for k, v in meta["residual_coef"].items()}
            adv = meta.get("adv_btc")
            env = RealDataExecutionEnv(
                store, initial_inventory=size, actions=actions, action_mode=action_mode,
                obs_features=obs_features,
                normalizer=FeatureNormalizer.from_json(rd / "normalizer.json"),
                test_mode=True, residual_coef_by_regime=residual_coef, adv_btc=adv)

            tkey = (size, tuple(sorted(residual_coef.items())), adv)
            if tkey not in twap_cache:
                twap_cache[tkey] = run_episodes(
                    env, TWAP(size, n_steps), idx)["is_bps"].to_numpy()

            model = _load_model(meta["algo"], rd / "model.zip")
            diff = run_episodes(env, SB3Policy(model), idx)["is_bps"].to_numpy() \
                - twap_cache[tkey]

            got, want = float(diff.mean()), recorded.get(rd.name)
            ok = want is not None and got == want
            rows[rd.name] = {"recomputed_mean": got, "recorded_mean": want,
                             "exact_match": bool(ok), "sd_paired": float(diff.std(ddof=1)),
                             "size_btc": size, "algo": meta["algo"]}
            print(f"[{split}] {rd.name}: mean {got:+.4f} (record "
                  f"{'None' if want is None else f'{want:+.4f}'}) exact={ok} "
                  f"sd {diff.std(ddof=1):.4f}", flush=True)
            if not ok:
                (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
                raise SystemExit(f"INTEGRITY MISMATCH on {rd.name} ({split}): "
                                 f"{got!r} != {want!r} -- aborting")
            arrays[rd.name] = diff
            del model
            gc.collect()

        np.savez_compressed(OUT / f"{split}.npz", **arrays)
        manifest["splits"][split] = {"n_runs": len(rows), "runs": rows}
        del arrays, store, full_train, twap_cache
        gc.collect()
        print(f"[{split}] saved -> {OUT}/{split}.npz", flush=True)

    manifest["wall_clock_s"] = round(time.time() - t_all, 1)
    manifest["peak_rss_gb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1073741824, 2)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE frozen: {manifest['wall_clock_s']}s, peak {manifest['peak_rss_gb']} GB")


if __name__ == "__main__":
    main()
