"""Per-episode cost series on blocks OTHER than the one already archived.

WHY THIS EXISTS. `per_episode_reeval.py` recovered episode-level costs for the primary
campaign on the DEVELOPMENT block only, and nothing else in the project ever stored them:
every other judgement kept per-run means and p-values. The consequence is that the minimum
detectable effect could be computed for exactly one block, and the claim that it TRANSFERS to
the blocks the verdicts were actually issued on could only be asserted, never measured. In a
study whose methodological finding is that blocks differ, that assertion is the weakest link.

WHAT IT DOES. Re-scores trained agents against both TWAP baselines on a nominated block,
saving every episode's implementation-shortfall cost. A "block" here is a contiguous range of
common-random-number episode seeds; the development block is seed0 = 5,000,000, the reserve
block 6,000,000.

NO VERDICT IS ISSUED BY THIS SCRIPT. It measures dispersion. No hypothesis is tested, no
sealed block is consumed, and no number it produces can create or destroy an edge claim.

TWO GATES, BOTH ABORTING.
  * DETERMINISM, always. The first 20 episodes of every run are scored twice and must be
    bit-identical. A non-deterministic evaluator makes every sigma meaningless.
  * INTEGRITY, when the block has a record to check against. Scoring the block a judgement was
    issued on must reproduce that judgement's per-run means to numerical identity -- the rule
    pinned in `per_episode_reeval.py`, which passed 20/20 there. On a block with no record
    there is nothing to match and the gate is skipped, explicitly and in the manifest.

Run:  PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python \
        reports/diagnostics/per_episode_multiblock.py --env reacting --block 6000000
"""
from __future__ import annotations

import argparse
import json
import re
import resource
import time
from pathlib import Path

import numpy as np

from execution.qrm.reactive_baselines import adaptive_twap, make_fixed_twap, run_episodes
from execution.qrm.step5_judgement import _core, _model_policy

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
N_EVAL = 2000
ORDER_BTC, ENV_STEPS = 25.0, 300
DET_CHECK_N = 20

# environment -> agent directory, the judgement whose block this is, and that block's seed0.
# `record_seed0` is what makes the integrity gate available: scoring THAT block must reproduce
# THAT judgement exactly. Any other block has no record and the gate is skipped explicitly.
ENVS = {
    "reacting": {"runs": "runs_primary_v3", "record": "step5_v3",
                 "record_seed0": 5_000_000, "injected": False},
    # The injected campaign's screening judgement was issued on block 18e6 over 38 runs; only
    # the 20 base-named agents are the agents of record (the rest are amendment variants),
    # matching the filter used by the RQ3 analysis.
    "injected": {"runs": "runs_signal_phaseD", "record": "step5_signal_dev",
                 "record_seed0": 18_000_000, "injected": True},
}
BASE_RUN = re.compile(r"^(ppo|dqn)_(calm|volatile)_s\d+$")


def build_env(env_name: str, regime: str):
    """The injected environment is the reacting environment plus a calibrated signal.

    It must be constructed through the SAME factory the campaign used, not re-specified here:
    the injection parameters come from a calibration file whose precedence rules
    (kernel solution > single-EMA > instantaneous) live in `load_injection_params`. Rebuilding
    those by hand would risk scoring a different instrument than the one the agents trained on
    -- which the integrity gate would catch, but only after an hour of wasted compute.
    """
    if not ENVS[env_name]["injected"]:
        return _core(S, regime, ORDER_BTC, ENV_STEPS)
    from execution.qrm.sigext_gates import load_injection_params, make_injected_factory
    factory = make_injected_factory(S, load_injection_params(S))
    return factory(S, regime, ORDER_BTC)


def load_model(path: Path, algo: str):
    from stable_baselines3 import DQN, PPO
    return (DQN if algo == "dqn" else PPO).load(str(path))


def main() -> None:
    global N_EVAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True, choices=sorted(ENVS))
    ap.add_argument("--block", required=True, type=int, help="episode-seed origin of the block")
    ap.add_argument("--label", default=None, help="output dir suffix; defaults to the seed")
    ap.add_argument("--n", type=int, default=N_EVAL,
                    help="episodes per run; ONLY for smoke tests. Any value but the default "
                         "disables the integrity gate, because a partial mean cannot equal a "
                         "full-sample record, and marks the output PROVISIONAL.")
    args = ap.parse_args()

    smoke = args.n != N_EVAL
    N_EVAL = args.n
    cfg = ENVS[args.env]
    label = args.label or f"{args.block // 1_000_000}e6"
    if smoke:
        label += f"_SMOKE{args.n}"
    out = S / f"per_episode_{args.env}_{label}"
    out.mkdir(exist_ok=True)
    seeds = list(range(args.block, args.block + N_EVAL))

    # The integrity gate is available only when this block IS the block a judgement used.
    checkable = args.block == cfg["record_seed0"] and not smoke
    judged = {}
    if checkable:
        judged = {r["run"]: r for r in
                  json.loads((S / cfg["record"] / "judgement.json").read_text())["per_run"]}

    manifest = {"env": args.env, "block_seed0": args.block, "label": label,
                "n_eval": N_EVAL, "order_btc": ORDER_BTC, "env_steps": ENV_STEPS,
                "source_runs": cfg["runs"],
                "integrity_gate": f"vs {cfg['record']}/judgement.json" if checkable
                                  else "NOT APPLICABLE -- no judgement was issued on this block",
                "determinism_gate": f"first {DET_CHECK_N} episodes scored twice, bit-identical",
                "runs": {}}
    t_all = time.time()

    for regime in ("calm", "volatile"):
        env = build_env(args.env, regime)
        arrays = {"fixed": run_episodes(env, make_fixed_twap(env), seeds)["cost_bps"],
                  "adaptive": run_episodes(env, adaptive_twap, seeds)["cost_bps"]}
        print(f"[{regime}] baselines: fixed {arrays['fixed'].mean():+.4f} | "
              f"adaptive {arrays['adaptive'].mean():+.4f}", flush=True)

        for d in sorted((S / cfg["runs"]).iterdir()):
            if not d.is_dir() or not (d / "meta.json").exists():
                continue
            meta = json.loads((d / "meta.json").read_text())
            if meta["regime"] != regime or not BASE_RUN.match(d.name):
                continue
            model = load_model(d / "model.zip", meta["algo"])
            policy = _model_policy(model)

            # ---- determinism gate ------------------------------------------------------
            head = seeds[:DET_CHECK_N]
            r1 = run_episodes(env, policy, head)["cost_bps"]
            r2 = run_episodes(env, policy, head)["cost_bps"]
            if not np.array_equal(r1, r2):
                (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
                raise SystemExit(f"NON-DETERMINISTIC on {d.name} -- aborting; every sigma "
                                 f"downstream would be meaningless")

            agent = run_episodes(env, policy, seeds)["cost_bps"]
            arrays[d.name] = agent
            rec = {"deterministic": True,
                   "mean_vs_fixed": float((agent - arrays["fixed"]).mean()),
                   "mean_vs_adaptive": float((agent - arrays["adaptive"]).mean()),
                   "sd_paired_vs_adaptive": float((agent - arrays["adaptive"]).std(ddof=1))}

            # ---- integrity gate, where a record exists ---------------------------------
            if checkable:
                j = judged[d.name]
                ok = (rec["mean_vs_fixed"] == j["mean_vs_fixed_bps"]
                      and rec["mean_vs_adaptive"] == j["mean_vs_adaptive_bps"])
                rec |= {"record_vs_fixed": j["mean_vs_fixed_bps"],
                        "record_vs_adaptive": j["mean_vs_adaptive_bps"], "exact_match": ok}
                if not ok:
                    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
                    raise SystemExit(f"INTEGRITY MISMATCH on {d.name} -- aborting per the "
                                     f"pinned rule")

            manifest["runs"][d.name] = rec
            print(f"[{regime}] {d.name}: vs_ada {rec['mean_vs_adaptive']:+.4f} "
                  f"sd {rec['sd_paired_vs_adaptive']:.4f}"
                  + (f" exact={rec['exact_match']}" if checkable else ""), flush=True)

        np.savez_compressed(out / f"{regime}.npz", **arrays)
        del arrays                      # free before the next regime; RAM stays flat
        print(f"[{regime}] saved -> {out}/{regime}.npz", flush=True)

    manifest["wall_clock_s"] = round(time.time() - t_all, 1)
    manifest["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                    / 1073741824, 2)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"DONE {args.env} block {label}: {manifest['wall_clock_s']}s, "
          f"peak RSS {manifest['peak_rss_gb']} GB")


if __name__ == "__main__":
    main()
