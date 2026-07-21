"""Per-episode cost re-evaluation (roadmap step 3; scope pinned 2026-07-17).

Deterministic replay of the 20 primary-campaign agents (`runs_primary_v3`) on the dev
block (eval_seed0 = 5,000,000, n = 2,000, the SAME CRN episode seeds as the sealed
judgement), saving EVERY episode's implementation-shortfall cost (bps) for the agent and
for both TWAP baselines. Purpose: full cost distributions + descriptive statistics
(figures F10a/F10b, table T6) — the sealed record keeps only per-run means/p-values.

INTEGRITY RULE (pinned): for every run, the recomputed mean paired differences must equal
`step5_v3/judgement.json` to numerical identity. Any mismatch aborts with a report.

Output: $S/per_episode_v3/{regime}.npz with arrays:
  fixed, adaptive              (2000,)   baseline per-episode costs
  <run_name>                   (2000,)   agent per-episode costs, one array per run
plus a JSON manifest with the integrity-check outcome per run.

Run:  PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python reports/diagnostics/per_episode_reeval.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from execution.qrm.reactive_baselines import adaptive_twap, make_fixed_twap, run_episodes
from execution.qrm.step5_judgement import EVAL_SEED0, _core, _model_policy

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
OUT = S / "per_episode_v3"
N_EVAL = 2000
ORDER_BTC, ENV_STEPS = 25.0, 300


def main() -> None:
    from stable_baselines3 import DQN, PPO

    OUT.mkdir(exist_ok=True)
    judged = {r["run"]: r for r in
              json.load(open(S / "step5_v3" / "judgement.json"))["per_run"]}
    seeds = list(range(EVAL_SEED0, EVAL_SEED0 + N_EVAL))
    manifest = {"eval_seed0": EVAL_SEED0, "n_eval": N_EVAL, "order_btc": ORDER_BTC,
                "env_steps": ENV_STEPS, "source_runs": "runs_primary_v3",
                "integrity_vs": "step5_v3/judgement.json", "runs": {}}

    for regime in ("calm", "volatile"):
        env = _core(S, regime, ORDER_BTC, ENV_STEPS)
        arrays = {
            "fixed": run_episodes(env, make_fixed_twap(env), seeds)["cost_bps"],
            "adaptive": run_episodes(env, adaptive_twap, seeds)["cost_bps"],
        }
        print(f"[{regime}] baselines done: fixed {arrays['fixed'].mean():.4f} | "
              f"adaptive {arrays['adaptive'].mean():.4f}", flush=True)
        for d in sorted((S / "runs_primary_v3").iterdir()):
            if not d.is_dir():
                continue
            meta = json.loads((d / "meta.json").read_text())
            if meta["regime"] != regime:
                continue
            algo = meta["algo"]
            model = (DQN if algo == "dqn" else PPO).load(str(d / "model.zip"))
            env = _core(S, regime, ORDER_BTC, ENV_STEPS)
            agent = run_episodes(env, _model_policy(model), seeds)["cost_bps"]
            arrays[d.name] = agent
            # ---- pinned integrity check: exact equality with the sealed record
            rec = judged[d.name]
            got_fix = float((agent - arrays["fixed"]).mean())
            got_ada = float((agent - arrays["adaptive"]).mean())
            ok = (got_fix == rec["mean_vs_fixed_bps"]) and (got_ada == rec["mean_vs_adaptive_bps"])
            manifest["runs"][d.name] = {
                "recomputed_vs_fixed": got_fix, "sealed_vs_fixed": rec["mean_vs_fixed_bps"],
                "recomputed_vs_adaptive": got_ada, "sealed_vs_adaptive": rec["mean_vs_adaptive_bps"],
                "exact_match": ok,
            }
            print(f"[{regime}] {d.name}: vs_ada {got_ada:+.4f} "
                  f"(sealed {rec['mean_vs_adaptive_bps']:+.4f}) exact={ok}", flush=True)
            if not ok:
                (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
                raise SystemExit(f"INTEGRITY MISMATCH on {d.name} — aborting per the pinned rule")
        np.savez_compressed(OUT / f"{regime}.npz", **arrays)
        print(f"[{regime}] saved {len(arrays) - 2} agent arrays + 2 baselines", flush=True)

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("ALL DONE — integrity exact for every run")


if __name__ == "__main__":
    main()
