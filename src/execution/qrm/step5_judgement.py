"""Step 5 judgement — behaviour audit FIRST, then the frozen 2,000-episode verdict.

Protocol: reports/qrm_step4_criteria.md §3-§4 (frozen 2026-07-06). Order enforced in
code: the audit (action histograms + deadline-residual reliance) runs and is WRITTEN
before any cost comparison is computed, so validity flags cannot be influenced by
results. Verdict conditions (all four required, per algo x regime):
  (i)  mean paired diff < 0 vs BOTH fixed-TWAP and adaptive-TWAP;
  (ii) Wilcoxon signed-rank p < 0.01 vs both;
  (iii) sign holds in >= 4 of 5 seeds;
  (iv) pooled mean diff vs adaptive-TWAP <= -0.05 bps (materiality floor).

Run:
    PYTHONPATH=src .venv/bin/python -m execution.qrm.step5_judgement \
        --scratch <oxford_l4> --runs <runs_reactive> --out <dir> [--n-eval 2000]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from execution.qrm.reactive_baselines import adaptive_twap, make_fixed_twap, run_episodes
from execution.qrm.reactive_env import ACTIONS, ReactiveQRMEnv

logger = logging.getLogger(__name__)

EVAL_SEED0 = 5_000_000      # disjoint from training episode seeds and curve-eval seeds
AUDIT_EPS = 200
AUDIT_ACTION_CAP = 0.90     # any single action > 90% of steps -> INVALID
AUDIT_RESIDUAL_CAP = 0.10   # forced deadline dump in > 10% of episodes -> INVALID
MATERIALITY_BPS = 0.05


def _core(scratch: Path, regime: str) -> ReactiveQRMEnv:
    return ReactiveQRMEnv(
        str(scratch / "step3g" / f"qrm_bundle_{regime}_b.npz"),
        str(scratch / "step3g" / f"move_process_{regime}_centered.npz"), order_btc=25.0)  # R2: drift-free


def _model_policy(model):
    def policy(env, obs):
        a, _ = model.predict(obs, deterministic=True)
        return int(a)
    return policy


def audit_one(scratch: Path, run_dir: Path) -> dict:
    """Action histogram + deadline reliance over AUDIT_EPS episodes. No cost stats."""
    meta = json.loads((run_dir / "meta.json").read_text())
    algo, regime = meta["algo"], meta["regime"]
    from stable_baselines3 import DQN, PPO
    model = (DQN if algo == "dqn" else PPO).load(str(run_dir / "model.zip"))
    env = _core(scratch, regime)
    counts = np.zeros(len(ACTIONS))
    residual_eps = 0
    for seed in range(EVAL_SEED0, EVAL_SEED0 + AUDIT_EPS):
        obs = env.reset(seed=seed)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            counts[int(a)] += 1
            obs, _r, done, info = env.step(int(a))
        # material = above ONE unit (the smallest voluntarily-executable amount).
        # The sub-unit tail (order 25 BTC = 44.22 units -> a 0.22-unit fragment) goes
        # to the deadline mechanism for EVERY policy incl. TWAP, by construction of
        # unit quantisation; counting it flagged 20/20 agents at 100% (2026-07-06).
        if info["deadline_residual_btc"] > env.aes1:
            residual_eps += 1
    shares = counts / counts.sum()
    top = int(np.argmax(shares))
    res_frac = residual_eps / AUDIT_EPS
    # Revised rule (criteria 4b, 2026-07-07): a dominant NON-ZERO pace is a legitimate
    # constant schedule (the TWAP baselines are single-action too); the share cap only
    # flags do-nothing dominance or when material deadline reliance also fires.
    share_flag = shares[top] > AUDIT_ACTION_CAP and (
        ACTIONS[top] == 0.0 or res_frac > AUDIT_RESIDUAL_CAP)
    valid = bool(not share_flag and res_frac <= AUDIT_RESIDUAL_CAP)
    return {"run": run_dir.name, "algo": algo, "regime": regime, "seed": meta["seed"],
            "action_shares": [round(float(s), 4) for s in shares],
            "top_action": ACTIONS[top], "top_share": float(shares[top]),
            "deadline_residual_frac": res_frac, "valid": valid}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-eval", type=int, default=2000)
    args = ap.parse_args()
    scratch, runs, out = Path(args.scratch), Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(d for d in runs.iterdir() if (d / "meta.json").exists())

    # ---- STAGE 1: audit, written to disk BEFORE any cost evaluation ----
    audit = [audit_one(scratch, d) for d in run_dirs]
    (out / "behaviour_audit.json").write_text(json.dumps(audit, indent=2))
    for a in audit:
        logger.info("AUDIT %s: top %sx @%.0f%% | residual %.1f%% | %s",
                    a["run"], a["top_action"], 100 * a["top_share"],
                    100 * a["deadline_residual_frac"],
                    "VALID" if a["valid"] else "INVALID")

    # ---- STAGE 2: baselines once per regime on the frozen eval seeds ----
    seeds = list(range(EVAL_SEED0, EVAL_SEED0 + args.n_eval))
    base = {}
    for regime in ("calm", "volatile"):
        env = _core(scratch, regime)
        base[regime] = {
            "fixed": run_episodes(env, make_fixed_twap(env), seeds)["cost_bps"],
            "adaptive": run_episodes(env, adaptive_twap, seeds)["cost_bps"],
        }
        logger.info("baseline %s: fixed %.4f bps | adaptive %.4f bps", regime,
                    base[regime]["fixed"].mean(), base[regime]["adaptive"].mean())

    # ---- STAGE 3: per-run paired evaluation + frozen verdict conditions ----
    from stable_baselines3 import DQN, PPO
    rows = []
    for d in run_dirs:
        meta = json.loads((d / "meta.json").read_text())
        algo, regime = meta["algo"], meta["regime"]
        model = (DQN if algo == "dqn" else PPO).load(str(d / "model.zip"))
        env = _core(scratch, regime)
        agent = run_episodes(env, _model_policy(model), seeds)["cost_bps"]
        dfix = agent - base[regime]["fixed"]
        dada = agent - base[regime]["adaptive"]
        rows.append({
            "run": d.name, "algo": algo, "regime": regime, "seed": meta["seed"],
            "mean_vs_fixed_bps": float(dfix.mean()),
            "mean_vs_adaptive_bps": float(dada.mean()),
            "p_fixed": float(wilcoxon(dfix).pvalue),
            "p_adaptive": float(wilcoxon(dada).pvalue),
        })
        logger.info("EVAL %s: vs fixed %+.4f (p=%.2g) | vs adaptive %+.4f (p=%.2g)",
                    d.name, dfix.mean(), rows[-1]["p_fixed"], dada.mean(),
                    rows[-1]["p_adaptive"])

    verdicts = {}
    audit_by_run = {a["run"]: a for a in audit}
    import re
    def tag_of(run_name, seed):
        m = re.search(rf"_s{seed}(_.+)?$", run_name)
        tag = (m.group(1) or "") if m else ""
        return "_v1a" if tag == "_v1aext" else tag   # criteria 4b: extensions merge
    groups = sorted({(r["algo"], r["regime"], tag_of(r["run"], r["seed"])) for r in rows})
    for algo, regime, tag in groups:
            grp = [r for r in rows if r["algo"] == algo and r["regime"] == regime
                   and tag_of(r["run"], r["seed"]) == tag]
            valid = [r for r in grp if audit_by_run[r["run"]]["valid"]]
            n_neg = sum(1 for r in valid
                        if r["mean_vs_fixed_bps"] < 0 and r["mean_vs_adaptive_bps"] < 0)
            sig = sum(1 for r in valid if r["p_fixed"] < 0.01 and r["p_adaptive"] < 0.01
                      and r["mean_vs_fixed_bps"] < 0 and r["mean_vs_adaptive_bps"] < 0)
            pooled_ada = float(np.mean([r["mean_vs_adaptive_bps"] for r in valid])) if valid else np.nan
            n_grp = len(grp)
            if n_grp >= 5:      # full protocol: the frozen §3 conditions
                edge = bool(valid and len(valid) >= 4 and n_neg >= 4 and sig >= 4
                            and pooled_ada <= -MATERIALITY_BPS)
                label = "EDGE"
            else:               # 3-seed screening: lenient escalation trigger (§5)
                edge = bool(valid and len(valid) >= 3 and n_neg >= 3 and sig >= 2
                            and pooled_ada <= -MATERIALITY_BPS)
                label = "ESCALATE"
            verdicts[f"{algo}_{regime}{tag}"] = {
                "n_valid_seeds": len(valid), "n_negative_both": n_neg,
                "n_significant_both": sig, "pooled_vs_adaptive_bps": pooled_ada,
                label: edge}
    report = {"criteria": "qrm_step4_criteria.md §3 (frozen 2026-07-06)",
              "n_eval_episodes": args.n_eval, "eval_seed0": EVAL_SEED0,
              "per_run": rows, "verdicts": verdicts}
    (out / "judgement.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
