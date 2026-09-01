"""The judgement: behaviour audit FIRST, then the frozen 2,000-episode verdict.

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
        --scratch <oxford_l4> --runs <runs_primary_v3> --out <dir> [--n-eval 2000]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon, ttest_1samp, t as tdist

from execution.qrm.reactive_baselines import adaptive_twap, make_fixed_twap, run_episodes
from execution.qrm.reactive_env import ACTIONS, ReactiveQRMEnv

logger = logging.getLogger(__name__)

EVAL_SEED0 = 5_000_000      # disjoint from training episode seeds and curve-eval seeds
AUDIT_EPS = 200
AUDIT_ACTION_CAP = 0.90     # any single action > 90% of steps -> INVALID
AUDIT_RESIDUAL_CAP = 0.10   # forced deadline dump in > 10% of episodes -> INVALID
MATERIALITY_BPS = 0.05


def _core(scratch: Path, regime: str, order_btc: float = 25.0,
          env_steps: int = 300, inject: bool = False,
          obs_pva: bool = False) -> ReactiveQRMEnv:
    # inject=True -> the CERTIFIED measured-signal env (Phase E judging); loads the
    # regime's frozen kernel so agents are judged in the SAME market they trained in
    # (their observation carries the signal feature, obs_dim +=1).
    kw = {}
    if inject:
        sol = json.loads((scratch / "signal" / "kernel_solution.json").read_text())
        kw = dict(signal_injection=True, signal_kernel=sol["regimes"][regime]["kernel"])
    if obs_pva:                                  # criteria section 8 Amendment A4
        kw["obs_price_vs_arrival"] = True
    return ReactiveQRMEnv(
        str(scratch / "step3g" / f"qrm_bundle_{regime}_b.npz"),
        str(scratch / "step3g" / f"move_process_{regime}_centered.npz"),
        order_btc=order_btc, n_steps=env_steps, **kw)  # R2: drift-free; §7 sweep params


def _model_policy(model):
    def policy(env, obs):
        a, _ = model.predict(obs, deterministic=True)
        return int(a)
    return policy


def audit_one(scratch: Path, run_dir: Path, eval_seed0: int = EVAL_SEED0,
              order_btc: float = 25.0, env_steps: int = 300, inject: bool = False,
              obs_pva: bool = False) -> dict:
    """Action histogram + deadline reliance over AUDIT_EPS episodes. No cost stats."""
    meta = json.loads((run_dir / "meta.json").read_text())
    algo, regime = meta["algo"], meta["regime"]
    from stable_baselines3 import DQN, PPO
    model = (DQN if algo == "dqn" else PPO).load(str(run_dir / "model.zip"))
    env = _core(scratch, regime, order_btc, env_steps, inject=inject, obs_pva=obs_pva)
    counts = np.zeros(len(ACTIONS))
    residual_eps = 0
    for seed in range(eval_seed0, eval_seed0 + AUDIT_EPS):
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


def _across_seed_stats(valid, n_group) -> dict:
    """INFORMATIONAL across-seed view — reported ALONGSIDE the frozen per-seed screen,
    never replacing it. Treats each valid seed's mean edge as one data point and asks
    whether they collectively point below zero (the standard, more powerful consistency
    test). This exists so a small-but-consistent edge that fails the strict per-seed rule
    can never be silently read as a null: the frozen rule can stamp EDGE/ESCALATE = False
    while THIS is significant (exactly the PPO-volatile v3 case). Changes NO frozen verdict.

    Guards the OTHER failure mode too: when the audit invalidated some seeds, this test
    runs only on the survivors, so a low p on few survivors is SURVIVORSHIP, not an edge.
    We always report n_seeds_total and set `trustworthy=False` (with a warning) whenever
    any seed in the group was dropped, so the number can never be read out of context."""
    n = len(valid)
    n_dropped = n_group - n
    if n < 2:
        return {"n_valid_seeds": n, "n_seeds_total": n_group, "trustworthy": False,
                "note": "need >=2 valid seeds for an across-seed test"}
    m_ada = np.array([r["mean_vs_adaptive_bps"] for r in valid])
    m_fix = np.array([r["mean_vs_fixed_bps"] for r in valid])
    pooled_ada = float(m_ada.mean())

    def _one_sided_less(x):                       # p that the true mean is < 0
        if np.allclose(x, x[0]):                  # zero variance -> t undefined
            return None
        return round(float(ttest_1samp(x, 0.0, alternative="less").pvalue), 4)

    se = float(m_ada.std(ddof=1) / np.sqrt(n))
    crit = float(tdist.ppf(0.975, n - 1))
    out = {
        "n_valid_seeds": n, "n_seeds_total": n_group,
        "n_cheaper_vs_adaptive": int((m_ada < 0).sum()),
        "pooled_vs_adaptive_bps": round(pooled_ada, 4),
        "pooled_vs_fixed_bps": round(float(m_fix.mean()), 4),
        "across_seed_t_p_vs_adaptive_onesided": _one_sided_less(m_ada),
        "across_seed_t_p_vs_fixed_onesided": _one_sided_less(m_fix),
        "ci95_vs_adaptive_bps": [round(pooled_ada - crit * se, 4),
                                 round(pooled_ada + crit * se, 4)],
        "trustworthy": n_dropped == 0}
    if n_dropped:
        out["warning"] = (f"{n_dropped} of {n_group} seeds were audit-invalid and dropped; "
                          f"this p is on survivors only -> SURVIVORSHIP, not a clean edge")
    return out


def _screen_verdict(rows, audit_by_run) -> dict:
    """Frozen §3 screening verdict (per algo x regime x variant-tag), unchanged.
    Each verdict also carries an INFORMATIONAL `across_seed` block (see
    _across_seed_stats) so a consistent edge failing the strict rule is never hidden."""
    import re
    def tag_of(run_name, seed):
        m = re.search(rf"_s{seed}(_.+)?$", run_name)
        tag = (m.group(1) or "") if m else ""
        return "_v1a" if tag == "_v1aext" else tag   # criteria 4b: extensions merge
    verdicts = {}
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
            label: edge,
            "across_seed": _across_seed_stats(valid, n_grp)}
    return verdicts


def _section6_verdict(rows, audit_by_run) -> dict:
    """§6 out-of-sample confirmation rule (criteria §6.4 + §6.7). Per regime, on VALID
    seeds only: PASS iff pooled cost < 0 vs BOTH benchmarks, AND across-seed one-sided
    t-test p < 0.05 vs adaptive, AND cheaper in >= 4 of the valid seeds. NO materiality
    floor (a single pre-registered replication, not the multi-agent screen). Effect size
    + 95% CI reported regardless. Volatile is the primary (headline) regime per §6.7."""
    verdicts = {}
    # group by (algo, regime) so a stray extra config can never be pooled in; the real
    # confirmation dir holds ONE config, so this is one cell per regime.
    for algo, regime in sorted({(r["algo"], r["regime"]) for r in rows}):
        key = f"{algo}_{regime}"
        grp = [r for r in rows if r["algo"] == algo and r["regime"] == regime]
        valid = [r for r in grp if audit_by_run[r["run"]]["valid"]]
        n = len(valid)
        if n < 2:
            verdicts[key] = {"n_valid_seeds": n, "PASS": False,
                             "reason": "insufficient valid seeds (<2)"}
            continue
        m_ada = np.array([r["mean_vs_adaptive_bps"] for r in valid])
        m_fix = np.array([r["mean_vs_fixed_bps"] for r in valid])
        pooled_ada, pooled_fix = float(m_ada.mean()), float(m_fix.mean())
        t_p = float(ttest_1samp(m_ada, 0.0, alternative="less").pvalue)
        try:
            w_p = float(wilcoxon(m_ada, alternative="less").pvalue)
        except ValueError:
            w_p = float("nan")
        se = float(m_ada.std(ddof=1) / np.sqrt(n))
        crit = float(tdist.ppf(0.975, n - 1))
        n_dir = int((m_ada < 0).sum())
        passed = bool(pooled_fix < 0 and pooled_ada < 0 and t_p < 0.05 and n_dir >= 4)
        verdicts[key] = {
            "n_valid_seeds": n, "n_cheaper_of_valid": n_dir,
            "pooled_vs_adaptive_bps": round(pooled_ada, 4),
            "pooled_vs_fixed_bps": round(pooled_fix, 4),
            "across_seed_t_p_onesided": round(t_p, 4),
            "across_seed_wilcoxon_p_onesided": round(w_p, 4),
            "ci95_vs_adaptive_bps": [round(pooled_ada - crit * se, 4),
                                     round(pooled_ada + crit * se, 4)],
            "is_primary_regime": regime == "volatile",
            "PASS": passed}
    return verdicts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--eval-seed0", type=int, default=EVAL_SEED0,
                    help="sealed eval block start; use 9_000_000 for the §6 confirmation")
    ap.add_argument("--mode", choices=["screen", "confirm"], default="screen",
                    help="screen = frozen §3 verdict; confirm = §6 out-of-sample rule")
    ap.add_argument("--order-btc", type=float, default=25.0,
                    help="order size for env + baselines (criteria §7 size ladder)")
    ap.add_argument("--obs-price-vs-arrival", action="store_true",
                    help="criteria section 8 Amendment A4 observation variant")
    ap.add_argument("--inject", action="store_true",
                    help="judge in the CERTIFIED injected env (Phase E); loads "
                         "signal/kernel_solution.json per regime")
    ap.add_argument("--env-steps", type=int, default=300,
                    help="episode horizon in 1s decisions (criteria §7: 600 = 10-min)")
    args = ap.parse_args()
    scratch, runs, out = Path(args.scratch), Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = sorted(d for d in runs.iterdir() if (d / "meta.json").exists())

    # ---- STAGE 1: audit, written to disk BEFORE any cost evaluation ----
    audit = [audit_one(scratch, d, args.eval_seed0, args.order_btc, args.env_steps,
                       inject=args.inject, obs_pva=args.obs_price_vs_arrival) for d in run_dirs]
    (out / "behaviour_audit.json").write_text(json.dumps(audit, indent=2))
    for a in audit:
        logger.info("AUDIT %s: top %sx @%.0f%% | residual %.1f%% | %s",
                    a["run"], a["top_action"], 100 * a["top_share"],
                    100 * a["deadline_residual_frac"],
                    "VALID" if a["valid"] else "INVALID")

    # ---- STAGE 2: baselines once per regime on the frozen eval seeds ----
    seeds = list(range(args.eval_seed0, args.eval_seed0 + args.n_eval))
    base = {}
    for regime in ("calm", "volatile"):
        env = _core(scratch, regime, args.order_btc, args.env_steps, inject=args.inject, obs_pva=args.obs_price_vs_arrival)
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
        env = _core(scratch, regime, args.order_btc, args.env_steps, inject=args.inject, obs_pva=args.obs_price_vs_arrival)
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

    audit_by_run = {a["run"]: a for a in audit}
    if args.mode == "confirm":
        verdicts = _section6_verdict(rows, audit_by_run)
        crit_ref = "qrm_step4_criteria.md §6 + §6.7 (out-of-sample confirmation)"
    else:
        verdicts = _screen_verdict(rows, audit_by_run)
        crit_ref = "qrm_step4_criteria.md §3 (frozen 2026-07-06)"
    report = {"criteria": crit_ref, "mode": args.mode,
              "n_eval_episodes": args.n_eval, "eval_seed0": args.eval_seed0,
              "order_btc": args.order_btc, "env_steps": args.env_steps,
              "per_run": rows, "verdicts": verdicts}
    (out / "judgement.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
