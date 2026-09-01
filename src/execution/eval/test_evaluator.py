"""Sealed-exam evaluator for the L2 frozen-replay track (val OR test split).

This module is the SINGLE piece of machinery that scores every trained agent on a
chosen split of ITS OWN dataset. It is written so that the split is nothing but a
parameter: the *identical* code path constructs episodes, benchmarks, and the
paired-vs-TWAP metric whether ``--split val`` or ``--split test``. Proving it
reproduces the recorded VALIDATION numbers (which were logged at train time by
``execution.agents.callbacks.ValidationCurveCallback``) proves it is the same
machinery -- so it can later be pointed, unchanged, at the sealed ``test`` split.

Assembly, not design: every ingredient is reused from the training/eval code --
``setup_data``-equivalent episode splitting (``EpisodeStore.chrono_split``), the
run's SAVED ``normalizer.json``, the run's recorded ``residual_coef`` / ``adv``,
the shared ``run_episodes`` loop + real-L2 fill engine, ``TWAP`` and ``SB3Policy``.
Nothing about the metric is re-derived here.

Metric (per §4 of ``reports/l2_test_protocol.md``, matching the recorded
validation): the agent runs the deterministic (no-exploration) policy over a fixed
episode subset -- the SAME fixed subset construction as training: the first
``min(n_eval, n_episodes)`` draws of ``numpy.random.default_rng(12345).choice(
n_episodes, size=..., replace=False)`` sorted ascending -- through the
force-completion (``test_mode``) env with the square-root deadline residual. The
per-episode implementation shortfall in bps is paired against TWAP on the identical
episodes; ``mean(agent_is_bps - twap_is_bps)`` (negative == agent cheaper) is the
run's headline number, byte-for-byte the ``val_vs_twap_mean`` the callback logged.

RAM safety (protocol §5): exactly ONE dataset parquet is resident at a time. All
runs sharing a dataset are scored against a single in-memory store, which is then
released (``del`` + ``gc.collect()``) before the next dataset is loaded. TWAP costs
are cached per order size within a dataset (they do not depend on the agent). Peak
process RSS is reported from ``resource.getrusage``.

CLI (only ever executed with ``--split val`` by the build/proof session):
    PYTHONPATH=src .venv/bin/python -m execution.eval.test_evaluator \
        --scratch-root "<scratch_hyperliquid>" --split val \
        --runs-dir runs --config configs/experiment.yaml --out out.json
    # or, all three datasets sequentially (one parquet resident at a time):
    PYTHONPATH=src .venv/bin/python -m execution.eval.test_evaluator \
        --scratch-root "<scratch_hyperliquid>" --split val --all --out out.json
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

from execution.agents.policy import SB3Policy
from execution.env.benchmarks import TWAP
from execution.env.episode_store import EpisodeStore
from execution.env.normalize import FeatureNormalizer
from execution.env.real_data_env import RealDataExecutionEnv
from execution.eval.runner import run_episodes

# Fixed subset construction, IDENTICAL to execution.agents.train.train_one_seed:
#   rng = np.random.default_rng(12345); indices = sorted(rng.choice(n, size=k, ...))
# with k = min(N_EVAL_DEFAULT, n_episodes) and N_EVAL_DEFAULT = the train CLI default.
VAL_SUBSET_SEED = 12345
N_EVAL_DEFAULT = 400

# One canonical config per runs directory (dataset resolution x horizon).
CONFIG_FOR_RUNS = {
    "runs": "configs/experiment.yaml",
    "runs_10s": "configs/experiment_10s.yaml",
    "runs_10s_10min": "configs/experiment_10s_10min.yaml",
}

_RUN_RE = re.compile(r"^(?P<algo>ppo|dqn)_size(?P<size>[0-9.]+)_seed(?P<seed>\d+)$")


def _peak_rss_mb() -> float:
    """Process peak resident set size in MB (macOS ru_maxrss is bytes)."""
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS reports bytes. Detect by magnitude.
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def discover_runs(runs_dir: Path) -> list[Path]:
    """Complete run bundles (model + normalizer + meta) under ``runs_dir``, sorted."""
    out = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        if not _RUN_RE.match(child.name):
            continue
        if all((child / f).exists() for f in ("model.zip", "normalizer.json", "meta.json")):
            out.append(child)
    return out


def _load_model(algo: str, model_path: Path):
    """Load a saved SB3 model on CPU (weights are reconstructed bit-exact)."""
    from stable_baselines3 import DQN, PPO
    if algo == "ppo":
        return PPO.load(str(model_path), device="cpu")
    if algo == "dqn":
        return DQN.load(str(model_path), device="cpu")
    raise ValueError(f"unknown algo: {algo!r}")


def _eval_indices(n_episodes: int, n_eval: int) -> list[int]:
    """The fixed evaluation subset -- identical construction to training."""
    rng = np.random.default_rng(VAL_SUBSET_SEED)
    k = min(n_eval, n_episodes)
    return sorted(rng.choice(n_episodes, size=k, replace=False).tolist())


def _paired_stats(diff: np.ndarray) -> dict:
    """Per-run paired-difference statistics (agent - TWAP, bps; negative cheaper)."""
    diff = np.asarray(diff, dtype=float)
    n = int(diff.size)
    out = {
        "n_episodes": n,
        "mean_paired_diff_bps": float(np.mean(diff)),
        "std_paired_diff_bps": float(np.std(diff, ddof=1)) if n > 1 else float("nan"),
        "wilcoxon_p_two_sided": float("nan"),
        "wilcoxon_p_less": float("nan"),   # H1: agent cheaper than TWAP
    }
    try:                                   # wilcoxon raises if all diffs are zero
        out["wilcoxon_p_two_sided"] = float(stats.wilcoxon(diff)[1])
        out["wilcoxon_p_less"] = float(stats.wilcoxon(diff, alternative="less")[1])
    except ValueError:
        pass
    return out


def evaluate_runs_dir(
    scratch_root: str | Path,
    runs_dir_name: str,
    split: str,
    config_path: str | Path,
    *,
    n_eval: int = N_EVAL_DEFAULT,
) -> dict:
    """Evaluate every complete run under one runs directory on ``split``.

    ONE dataset parquet is resident for the whole call; released on return. For
    ``split == 'val'`` the store is the chronological validation carve of
    ``train.parquet`` (never test); for ``split == 'test'`` it is ``test.parquet``
    of the same dataset. The test branch is present so the split is a pure
    parameter; the build/proof session executes this function ONLY with 'val'.
    """
    scratch_root = Path(scratch_root)
    cfg = yaml.safe_load(Path(config_path).read_text())
    env_cfg = cfg["env"]
    n_steps = env_cfg["n_steps"]
    actions = tuple(env_cfg["actions"])
    action_mode = env_cfg.get("action_mode", "twap_pace_multiple")
    obs_features = tuple(cfg["obs"]["features"])
    val_frac = cfg["split"]["val_frac"]
    ds_dir = scratch_root / cfg.get("dataset_dir", "dataset")

    runs_dir = scratch_root / runs_dir_name
    run_dirs = discover_runs(runs_dir)

    # ---- load exactly one dataset store for this split -----------------------
    full_train = EpisodeStore.from_parquet(ds_dir / "train.parquet", n_steps=n_steps)
    if split == "val":
        _train_sub, eval_store = full_train.chrono_split(val_frac)
        # Cross-check only: the normalizer refit on train_sub must equal each run's
        # saved normalizer.json (evidence the dataset is unchanged since training).
        refit_norm = FeatureNormalizer.fit(_train_sub.features, _train_sub.feature_names)
    elif split == "test":
        # SEALED SPLIT. Reached only under --split test, which the proof never runs.
        eval_store = EpisodeStore.from_parquet(ds_dir / "test.parquet", n_steps=n_steps)
        refit_norm = None
    else:
        raise ValueError(f"unknown split: {split!r}")

    eval_indices = _eval_indices(eval_store.n_episodes, n_eval)

    results = []
    twap_cache: dict[tuple, np.ndarray] = {}
    for rd in run_dirs:
        meta = json.loads((rd / "meta.json").read_text())
        algo = meta["algo"]
        size = float(meta["size_btc"])
        residual_coef = {k: float(v) for k, v in meta["residual_coef"].items()}
        adv = meta.get("adv_btc")
        saved_norm = FeatureNormalizer.from_json(rd / "normalizer.json")

        # Normalizer sanity (val only): saved must match a fresh train_sub fit.
        norm_max_abs_diff = None
        if refit_norm is not None:
            norm_max_abs_diff = float(max(
                np.max(np.abs(saved_norm.center - refit_norm.center)),
                np.max(np.abs(saved_norm.scale - refit_norm.scale)),
            ))

        env = RealDataExecutionEnv(
            eval_store, initial_inventory=size, actions=actions, action_mode=action_mode,
            obs_features=obs_features, normalizer=saved_norm, test_mode=True,
            residual_coef_by_regime=residual_coef, adv_btc=adv,
        )

        # TWAP is agent-independent: cache per (size, residual, adv) within dataset.
        tkey = (size, tuple(sorted(residual_coef.items())), adv)
        if tkey not in twap_cache:
            twap_df = run_episodes(env, TWAP(size, n_steps), eval_indices)
            twap_cache[tkey] = twap_df["is_bps"].to_numpy()
        twap_is = twap_cache[tkey]

        model = _load_model(algo, rd / "model.zip")
        agent_df = run_episodes(env, SB3Policy(model), eval_indices)
        agent_is = agent_df["is_bps"].to_numpy()
        diff = agent_is - twap_is                       # paired: agent - TWAP
        # Deadline-leaning diagnostics (same definitions as the training callback /
        # v1 record): residual_freq = fraction of episodes whose forced deadline buy
        # went beyond visible depth; DL flag (v1) = residual_freq > 0.10.
        agent_residual_freq = float((agent_df["residual_btc"] > 1e-9).mean())
        agent_full_exec_frac = float((agent_df["inventory_left"].abs() < 1e-6).mean())

        stats_row = _paired_stats(diff)
        recorded = meta.get("val_vs_twap_final")
        row = {
            "run": rd.name,
            "runs_dir": runs_dir_name,
            "algo": algo,
            "size_btc": size,
            "seed": int(meta["seed"]),
            "split": split,
            **stats_row,
            "agent_residual_freq": agent_residual_freq,
            "agent_full_exec_frac": agent_full_exec_frac,
            "dl_flag": bool(agent_residual_freq > 0.10),
            "recorded_val_vs_twap_final": recorded,
            "recorded_val_residual_freq_final": meta.get("val_residual_freq_final"),
            "normalizer_max_abs_diff_vs_refit": norm_max_abs_diff,
        }
        if split == "val" and recorded is not None:
            row["abs_diff_vs_recorded"] = abs(stats_row["mean_paired_diff_bps"] - recorded)
        results.append(row)
        del model
        gc.collect()

    # ---- free the dataset before returning (protocol §5) --------------------
    del full_train, eval_store
    if split == "val":
        del _train_sub, refit_norm
    twap_cache.clear()
    gc.collect()

    return {
        "runs_dir": runs_dir_name,
        "config": str(config_path),
        "dataset_dir": cfg.get("dataset_dir", "dataset"),
        "split": split,
        "n_steps": n_steps,
        "n_eval_requested": n_eval,
        "n_eval_indices": len(eval_indices),
        "eval_indices_head": eval_indices[:5],
        "eval_indices_tail": eval_indices[-5:],
        "n_runs": len(results),
        "peak_rss_mb_after": round(_peak_rss_mb(), 1),
        "runs": results,
    }


def _arm_summary(rows: list[dict]) -> list[dict]:
    """Per-arm (runs_dir x algo x size) pooled stats over seeds."""
    arms: dict[tuple, list[dict]] = {}
    for r in rows:
        arms.setdefault((r["runs_dir"], r["algo"], r["size_btc"]), []).append(r)
    out = []
    for (rd, algo, size), grp in sorted(arms.items()):
        means = np.array([g["mean_paired_diff_bps"] for g in grp], dtype=float)
        pooled = float(np.mean(means))
        seeds_cheaper = int(np.sum(means < 0.0))
        t_p_less = float("nan")
        if means.size > 1 and np.std(means) > 0:
            t_p_less = float(stats.ttest_1samp(means, 0.0, alternative="less").pvalue)
        out.append({
            "runs_dir": rd, "algo": algo, "size_btc": size, "n_seeds": int(means.size),
            "pooled_mean_paired_diff_bps": pooled,
            "across_seed_t_p_less": t_p_less,
            "seeds_cheaper_than_twap": seeds_cheaper,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="L2 sealed-exam evaluator (val or test)")
    ap.add_argument("--scratch-root", required=True)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--runs-dir", default=None,
                    help="single runs dir name (e.g. runs, runs_10s, runs_10s_10min)")
    ap.add_argument("--config", default=None, help="config yaml for --runs-dir")
    ap.add_argument("--all", action="store_true",
                    help="evaluate all three runs dirs, one dataset resident at a time")
    ap.add_argument("--n-eval", type=int, default=N_EVAL_DEFAULT)
    ap.add_argument("--out", required=True, help="output JSON path")
    args = ap.parse_args()

    if not args.all and (args.runs_dir is None):
        ap.error("provide --runs-dir (+ --config) or --all")

    t0 = time.perf_counter()
    if args.all:
        targets = list(CONFIG_FOR_RUNS.items())
    else:
        cfg = args.config or CONFIG_FOR_RUNS.get(args.runs_dir)
        if cfg is None:
            ap.error(f"no default config for runs-dir {args.runs_dir!r}; pass --config")
        targets = [(args.runs_dir, cfg)]

    per_dir = []
    for runs_dir_name, cfg_path in targets:
        print(f"[{runs_dir_name}] evaluating split={args.split} ...", flush=True)
        block = evaluate_runs_dir(args.scratch_root, runs_dir_name, args.split, cfg_path,
                                  n_eval=args.n_eval)
        print(f"[{runs_dir_name}] {block['n_runs']} runs; peak RSS "
              f"{block['peak_rss_mb_after']} MB", flush=True)
        per_dir.append(block)
        gc.collect()

    all_rows = [r for b in per_dir for r in b["runs"]]
    payload = {
        "split": args.split,
        "scratch_root": str(args.scratch_root),
        "n_eval": args.n_eval,
        "subset_seed": VAL_SUBSET_SEED,
        "wall_clock_s": round(time.perf_counter() - t0, 1),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
        "datasets": per_dir,
        "runs_flat": all_rows,
        "arm_summary": _arm_summary(all_rows),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}  ({len(all_rows)} runs, {payload['wall_clock_s']}s, "
          f"peak {payload['peak_rss_mb']} MB)")


if __name__ == "__main__":
    main()
