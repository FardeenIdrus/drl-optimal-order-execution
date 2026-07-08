"""Step 4.5 — the three pre-training gates (bands frozen in reports/qrm_step4_criteria.md).

G1  Reaction lever: matched-pair episodes (identical seed; with-order vs background-only)
    -> mean |mid displacement| at T and size-monotonicity.
G2  Effective cost-vs-size: instant execution of {1,2,5,10} units, in-sim book states vs
    real v2 books of the same regime's calibration blocks (the C5 tripwire).
G3  Benchmark sanity: TWAP completes; dump >= TWAP cost; cost non-decreasing in size.

Run:
    PYTHONPATH=src .venv/bin/python -m execution.qrm.step4_gates \
        --scratch <oxford_l4> --book-dir <book_05s_v2> --out <step4_gates.json>
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from execution.qrm.reactive_baselines import adaptive_twap, instant_dump, run_episodes
from execution.qrm.reactive_env import ACTIONS, ReactiveQRMEnv

logger = logging.getLogger(__name__)

SIZES_BTC = (5.0, 12.5, 25.0, 50.0)
PRIMARY_BTC = 25.0
UNIT_LADDER = (1, 2, 5, 10)


def _env(scratch: Path, regime: str, order_btc: float) -> ReactiveQRMEnv:
    return ReactiveQRMEnv(
        str(scratch / "step3g" / f"qrm_bundle_{regime}_b.npz"),
        str(scratch / "step3g" / f"move_process_{regime}_centered.npz"),  # R2: drift-free
        order_btc=order_btc)


# ------------------------------------------------------------------------- G1
def _mid_path(env: ReactiveQRMEnv, seed: int, trade: bool) -> float:
    """Terminal mid of one episode: adaptive TWAP if ``trade`` else pure background."""
    env.reset(seed=seed)
    done = False
    while not done:
        a = ACTIONS.index(1.0) if trade else 0
        _o, _r, done, _i = env.step(a)
        if not trade:                       # background arm must never force-complete
            env._ep.remaining_btc = 0.0
            env._ep.carry_btc = 0.0
    return env._ep.p_mid


def gate1(scratch: Path, n_pairs: int) -> Dict:
    out: Dict = {"sizes_btc": list(SIZES_BTC), "regimes": {}}
    for regime in ("calm", "volatile"):
        disp_by_size: List[float] = []
        for size in SIZES_BTC:
            env_w = _env(scratch, regime, size)
            env_o = _env(scratch, regime, size)
            d = []
            for seed in range(n_pairs):
                m_with = _mid_path(env_w, seed, trade=True)
                m_without = _mid_path(env_o, seed, trade=False)
                d.append(abs(m_with - m_without))
            disp_by_size.append(float(np.mean(d)))
            logger.info("G1 %s %.1f BTC: mean |dmid| = $%.2f", regime, size, disp_by_size[-1])
        rho = float(spearmanr(SIZES_BTC, disp_by_size).statistic)
        primary_disp = disp_by_size[SIZES_BTC.index(PRIMARY_BTC)]
        out["regimes"][regime] = {
            "mean_abs_mid_displacement_usd": disp_by_size,
            "primary_displacement_usd": primary_disp,
            "spearman_rho_vs_size": rho,
            "pass": bool(primary_disp >= 1.0 and rho > 0),
        }
    out["pass"] = all(r["pass"] for r in out["regimes"].values())
    return out


# ------------------------------------------------------------------------- G2
def _real_block_costs(book_dir: Path, hours: list, aes1: float) -> Dict[int, List[float]]:
    """Mean instant-execution cost (bps of mid) per unit-count, PER BLOCK-HOUR, from
    real v2 books: walk the ask ladder snapshot by snapshot."""
    px_cols = [f"ask_px_{i}" for i in range(1, 21)]
    sz_cols = [f"ask_sz_{i}" for i in range(1, 21)]
    per_hour: Dict[int, List[float]] = {k: [] for k in UNIT_LADDER}
    from execution.qrm.step3f import _hour_slice
    for date, hour in hours:
        g = _hour_slice(book_dir, date, [hour])
        full = pd.read_parquet(book_dir / f"{date}.parquet",
                               columns=["ts", "mid"] + px_cols + sz_cols)
        full = full[full.ts.isin(g.ts)]
        px = full[px_cols].to_numpy()
        sz = full[sz_cols].to_numpy()
        mid = full["mid"].to_numpy()
        for k in UNIT_LADDER:
            want = k * aes1
            cost = np.zeros(len(full))
            left = np.full(len(full), want)
            for lvl in range(20):
                take = np.minimum(left, np.nan_to_num(sz[:, lvl]))
                cost += take * np.nan_to_num(px[:, lvl])
                left -= take
            ok = left <= 1e-9
            bps = (cost[ok] / want - mid[ok]) / mid[ok] * 1e4
            per_hour[k].append(float(np.mean(bps)))
    return per_hour


def _sim_costs(env: ReactiveQRMEnv, n_states: int) -> Dict[int, float]:
    """Mean instant-execution cost (bps of mid) per unit-count over sampled sim states."""
    res = {k: [] for k in UNIT_LADDER}
    for seed in range(n_states):
        env.reset(seed=10_000 + seed)
        mid = env._ep.p_mid
        base_state = env._ep.state.copy()
        for k in UNIT_LADDER:
            env._ep.state = base_state.copy()
            # normalize by the ACTUAL BTC obtained (per-depth unit sizes, I3): the old
            # k*aes1 denominator manufactured phantom bps whenever fills landed deep
            cost, btc, _u = env._buy_btc(k * env.aes1, allow_overshoot=True)
            if btc >= k * env.aes1 - 1e-9:
                bps = (cost / btc - mid) / mid * 1e4
                res[k].append(bps)
        env._ep.state = base_state
    return {k: float(np.mean(v)) for k, v in res.items()}


def gate2(scratch: Path, book_dir: Path, n_states: int) -> Dict:
    out: Dict = {"unit_ladder": list(UNIT_LADDER), "regimes": {}}
    for regime in ("calm", "volatile"):
        acc = np.load(scratch / "step3g" / f"accumulators_{regime}_b.npz")
        aes1 = float(acc["aes"][0])
        hours = [(h.split(":")[0], int(h.split(":")[1])) for h in acc["hours"].tolist()]
        real_by_hour = _real_block_costs(book_dir, hours, aes1)
        sim = _sim_costs(_env(scratch, regime, PRIMARY_BTC), n_states)
        checks = {}
        for k in UNIT_LADDER:
            hour_means = np.array(real_by_hour[k])
            pooled = float(hour_means.mean())
            dispersion = float(np.median(np.abs(hour_means - pooled) / abs(pooled)))
            band = max(dispersion, 0.25)           # frozen floor (criteria §2 G2)
            rel = abs(sim[k] - pooled) / abs(pooled)
            checks[k] = {"real_pooled_bps": pooled, "sim_bps": sim[k],
                         "rel_gap": float(rel), "band": band, "pass": bool(rel <= band)}
            logger.info("G2 %s %du: real %.2f bps vs sim %.2f bps (band %.0f%%) %s",
                        regime, k, pooled, sim[k], band * 100,
                        "PASS" if rel <= band else "FAIL")
        out["regimes"][regime] = {"checks": checks,
                                  "pass": all(c["pass"] for c in checks.values())}
    out["pass"] = all(r["pass"] for r in out["regimes"].values())
    return out


# ------------------------------------------------------------------------- G3
def _driftfree_env(scratch: Path, regime: str, order_btc: float) -> ReactiveQRMEnv:
    """Env with the exogenous jumps DISABLED: costs are pure spread + impact.

    G3's cost-ordering checks compare policy mechanics whose true differences are
    ~0.1 bps; per-episode background drift is +/- several bps and enters each policy
    through different fill timings, so it does NOT cancel across policies even on
    shared seeds (diagnosed 2026-07-06: the first G3 run's 'impossible' orderings were
    exactly this noise, plus a dump baseline the 2x action cap prevented from actually
    dumping). Removing drift is legitimate here because G3 validates MECHANICS, not
    market realism (G1/G2 keep the full process)."""
    zero_mp = scratch / "step4" / "move_process_zero.npz"
    zero_mp.parent.mkdir(exist_ok=True)
    if not zero_mp.exists():
        np.savez(zero_mp, interval_s=0.5, moves=np.array([0]), probs=np.array([1.0]))
    return ReactiveQRMEnv(
        str(scratch / "step3g" / f"qrm_bundle_{regime}_b.npz"), str(zero_mp),
        order_btc=order_btc)


def gate3(scratch: Path, n_eps: int) -> Dict:
    from execution.qrm.reactive_baselines import make_fixed_twap
    out: Dict = {"regimes": {}}
    seeds = list(range(n_eps))
    for regime in ("calm", "volatile"):
        # completion: full process (drift on) — TWAP must finish in the real setting
        env = _env(scratch, regime, PRIMARY_BTC)
        fixed = run_episodes(env, make_fixed_twap(env), seeds)
        # executed_frac is now VOLUNTARY completion (criteria 4b): the sub-unit tail
        # (< 1 AES) is untradeable by construction, so full completion = within one
        # unit of the order.
        unit_frac = float(env.bundle.aes[0]) / env.order_btc
        complete = float((fixed["executed_frac"] >= 1.0 - unit_frac - 1e-9).mean())
        # ordering + monotonicity: drift-free (pure spread + impact)
        df = _driftfree_env(scratch, regime, PRIMARY_BTC)
        twap_df = run_episodes(df, adaptive_twap, seeds[:200])
        dump_costs = []
        for s in seeds[:200]:
            df.reset(seed=s)
            dump_costs.append(df.force_buy_all())      # TRUE instant dump at t=0
        dump_mean = float(np.mean(dump_costs))
        twap_mean = float(twap_df["cost_bps"].mean())
        dump_worse = bool(dump_mean >= twap_mean)
        size_means = []
        for size in SIZES_BTC:
            e = _driftfree_env(scratch, regime, size)
            r = run_episodes(e, adaptive_twap, seeds[:200])
            size_means.append(float(r["cost_bps"].mean()))
        # tolerance 0.015 bps: the <=1-unit deadline overshoot is a FIXED cost, so
        # per-BTC it taxes small orders more (~0.007 bps at 5 BTC — measured, matches
        # the mechanism arithmetic; documented quantisation granularity, ~30x below
        # the 0.2-bps scale of the wiring bug this check exists to catch)
        monotone = bool(all(size_means[i + 1] >= size_means[i] - 0.015
                            for i in range(len(size_means) - 1)))
        out["regimes"][regime] = {
            "twap_completion_rate": complete,
            "driftfree_twap_mean_bps": twap_mean,
            "driftfree_true_dump_mean_bps": dump_mean,
            "driftfree_cost_by_size_bps": size_means,
            "pass": bool(complete >= 0.99 and dump_worse and monotone),
        }
        logger.info("G3 %s: complete %.3f | driftfree twap %.3f dump %.3f | sizes %s",
                    regime, complete, twap_mean, dump_mean,
                    np.round(size_means, 3).tolist())
    out["pass"] = all(r["pass"] for r in out["regimes"].values())
    return out



# ------------------------------------------------------------- G1' / G2' (Rev 1)
UNIT_TRIPLE = (10, 25, 44)      # dose-response sizes for self-impact (units)


def gate1_rev(scratch: Path, n_seeds: int = 100) -> Dict:
    """G1' (criteria 3b): reaction = TEMPORARY impact, measured drift-free.

    (a) self-impact: 2nd identical dump >= 1.25x the 1st; (b) monotone in size;
    (c) recovery: probe cost at +30 s < at +1 s after a 44u dump."""
    out: Dict = {"regimes": {}}
    for regime in ("calm", "volatile"):
        env = _driftfree_env(scratch, regime, PRIMARY_BTC)
        second_by_size = []
        ratio_primary = None
        for units in UNIT_TRIPLE:
            d1, d2 = [], []
            for seed in range(n_seeds):
                env.reset(seed=seed)
                ep = env._ep
                mid0 = ep.p_mid

                def dump_cost(n):
                    # shortfall beyond the visible window prices at the deepest
                    # visible ask (the frozen deadline convention); normalized by
                    # ACTUAL BTC incl. the synthetic remainder (per-depth units, I3)
                    target = n * env.aes1
                    c, btc, _u = env._buy_btc(target, allow_overshoot=True)
                    if btc < target - 1e-9:
                        deepest = ep.p_ref + env.tick * (env.K - 0.5)
                        c += (target - btc) * deepest
                        btc = target
                    return (c / btc - mid0) / mid0 * 1e4

                d1.append(dump_cost(units))
                d2.append(dump_cost(units))
            second_by_size.append(float(np.mean(d2)))
            if units == 44:
                ratio_primary = float(np.mean(d2) / max(np.mean(d1), 1e-12))
        rho = float(spearmanr(UNIT_TRIPLE, second_by_size).statistic)
        rec = []
        for seed in range(50):
            env.reset(seed=seed)
            ep = env._ep
            mid0 = ep.p_mid
            env._buy_btc(44 * env.aes1, allow_overshoot=True)
            probes = []
            for step in range(30):
                for _ in range(2):
                    env._run_interval(track_flow=False)
                st = ep.state.copy()
                c, btc, _u = env._buy_btc(5 * env.aes1, allow_overshoot=True)
                ep.state = st
                probes.append((c / btc - mid0) / mid0 * 1e4
                              if btc >= 5 * env.aes1 - 1e-9 else np.nan)
            rec.append(probes)
        rec_m = np.nanmean(np.array(rec), axis=0)
        r = {"self_impact_ratio_primary": ratio_primary,
             "second_dump_bps_by_size": second_by_size,
             "spearman_rho": rho,
             "probe_bps_t1": float(rec_m[0]), "probe_bps_t30": float(rec_m[29]),
             "pass": bool(ratio_primary >= 1.25 and rho > 0 and rec_m[29] < rec_m[0])}
        logger.info("G1' %s: self-impact x%.2f | monotone rho %.2f | recovery %.3f->%.3f  %s",
                    regime, ratio_primary, rho, rec_m[0], rec_m[29],
                    "PASS" if r["pass"] else "FAIL")
        out["regimes"][regime] = r
    out["pass"] = all(r["pass"] for r in out["regimes"].values())
    return out


def gate2_rev(scratch: Path, book_dir: Path, n_states: int) -> Dict:
    """G2' (criteria 3b): the impact SLOPE (1u->10u cost-growth ratio) must be within
    25 % relative of the real ratio; the ~2.2x LEVEL gap is caveat C6 (cancels in
    paired comparisons), reported not gated."""
    out: Dict = {"regimes": {}}
    for regime in ("calm", "volatile"):
        acc = np.load(scratch / "step3g" / f"accumulators_{regime}_b.npz")
        aes1 = float(acc["aes"][0])
        hours = [(h.split(":")[0], int(h.split(":")[1])) for h in acc["hours"].tolist()]
        real_by_hour = _real_block_costs(book_dir, hours, aes1)
        real = {k: float(np.mean(v)) for k, v in real_by_hour.items()}
        sim = _sim_costs(_env(scratch, regime, PRIMARY_BTC), n_states)
        growth_real = real[10] / real[1]
        growth_sim = sim[10] / sim[1]
        rel = abs(growth_sim - growth_real) / growth_real
        level_gap = float(np.mean([real[k] / sim[k] for k in UNIT_LADDER]))
        r = {"growth_ratio_real": float(growth_real), "growth_ratio_sim": float(growth_sim),
             "rel_gap": float(rel), "level_gap_C6_realx": level_gap,
             "real_bps": real, "sim_bps": sim, "pass": bool(rel <= 0.25)}
        logger.info("G2' %s: growth sim x%.2f vs real x%.2f (gap %.0f%%) | C6 level %.1fx  %s",
                    regime, growth_sim, growth_real, rel * 100, level_gap,
                    "PASS" if r["pass"] else "FAIL")
        out["regimes"][regime] = r
    out["pass"] = all(r["pass"] for r in out["regimes"].values())
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Step-4 pre-training gates G1-G3")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--book-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pairs", type=int, default=200)
    ap.add_argument("--n-states", type=int, default=200)
    ap.add_argument("--n-eps", type=int, default=500)
    args = ap.parse_args()
    scratch, book_dir = Path(args.scratch), Path(args.book_dir)
    report = {
        "criteria": "reports/qrm_step4_criteria.md (frozen 2026-07-06; Revision 1 approved 2026-07-06)",
        "G1_reaction_lever_rev1": gate1_rev(scratch),
        "G2_cost_vs_size_rev1": gate2_rev(scratch, book_dir, args.n_states),
        "G3_benchmark_sanity_rev1": gate3(scratch, args.n_eps),
    }
    report["all_pass"] = all(report[k]["pass"] for k in
                             ("G1_reaction_lever_rev1", "G2_cost_vs_size_rev1",
                              "G3_benchmark_sanity_rev1"))
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: (v["pass"] if isinstance(v, dict) and "pass" in v else v)
                      for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
