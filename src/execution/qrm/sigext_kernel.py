"""Measured-signal extension, Amendment 3: the DERIVED multi-timescale kernel.

Registered procedure (criteria section 8, Amendment 3 + Corrections 1-2): measure the
simulator's instantaneous-imbalance autocorrelation; solve, in closed form, least-squares
gains over an exponential basis so the implied injected response matches the residual
curve (real minus endogenous) at all seven measured horizons; damped empirical refinement
iterations (max five, half-strength, early stop at 15% on the gated horizons) absorbing
feedback and tick-quantisation; offset initialised at the mean-injected-term fixed point
then solved against the MEASURED end-to-end drift (Newton, base-env stopping rule);
verification on fresh seeds; freeze constants; then the full Phase C suite runs
separately with the frozen kernel.

Zero free choices at run time: basis, sample sizes, seeds and iteration count are fixed
here and registered. The calibration target is the REAL measured curve only.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from execution.qrm.reactive_env import ReactiveQRMEnv
from execution.qrm.signal_measure import (HORIZONS_S, RegAccum, accumulate_episode,
                                          sample_episode)

logger = logging.getLogger(__name__)

BASIS_HALFLIVES_S = (0.5, 2.0, 8.0, 32.0)
RHO_LAGS = 121                       # autocorrelation lags 0..120 intervals (60 s)
RHO_EPS = 600                        # background episodes for rho, per regime
RHO_SEED_BASE = 30_300_000
REFINE_EPS = 300
OFFSET_EPS = 150
REFINE_SEED_BASE = 30_400_000
VERIFY_SEED_BASE = 30_500_000        # Correction 2 (c): fresh seeds, never calibrated on
DRIFT_SEED_BASE = 30_550_000         # Correction 2 (b): drift-null calibration seeds
MAX_REFINE_ITERS = 12                # Correction 2a: raised from 5 (was still descending)
REFINE_DAMPING = 0.5                 # Correction 2 (a): half-strength corrections
REFINE_STOP_GAP = 0.15               # Correction 2 (a): early stop, gated horizons
PLATEAU_MIN_IMPROVE = 0.01           # Correction 2a: plateau stop, <1 point over 2 iters
DRIFT_NULL_EPS = 12000               # Correction 2c: raw-drift measurement per anchor block
DRIFT_PROBE_EPS = 8000               # Correction 2c revised: 2nd point for the slope
OFFSET_PROBE_BPS = 0.015             # Correction 2c revised: probe offset (brackets the
                                     # volatile zero-crossing; large -> low slope variance)
# Correction 2c revised (2026-07-25): anchor the offset to the POOLED raw drift across
# independent blocks (the raw creep varies block-to-block by a few ticks; a single-block
# anchor under-corrects everywhere else). These blocks EXCLUDE the fairness certification
# block (FAIRNESS_SEED0 = 3,000,000) and the verify block (VERIFY_SEED_BASE = 30,500,000).
ANCHOR_SEED_BASES = (30_550_000, 31_000_000)
INTERVALS_PER_EP = 600               # injected 0.5 s intervals per episode (300 x 2)
GATED = ("1", "2", "5", "10")
EMA_TAIL = 600                       # truncation of the EMA weight tail (5 minutes)


def measure_rho(env: ReactiveQRMEnv, n_eps: int) -> Dict:
    """Autocorrelation of instantaneous S2 over background episodes (within-episode)."""
    num = np.zeros(RHO_LAGS)
    cnt = np.zeros(RHO_LAGS)
    s_sum = s_sq = s_n = 0.0
    for i in range(n_eps):
        _, s2 = sample_episode(env, RHO_SEED_BASE + i)
        x = s2[np.isfinite(s2)]
        s_sum += x.sum(); s_sq += (x * x).sum(); s_n += len(x)
        for lag in range(RHO_LAGS):
            if lag >= len(s2):
                break
            a, b = s2[:-lag] if lag else s2, s2[lag:]
            m = np.isfinite(a) & np.isfinite(b)
            num[lag] += float((a[m] * b[m]).sum())
            cnt[lag] += int(m.sum())
    mean = s_sum / s_n
    var = s_sq / s_n - mean * mean
    rho = (num / np.maximum(cnt, 1) - mean * mean) / var
    return {"rho": rho, "mean": mean, "var": var, "n_points": int(s_n)}


def response_matrix(rho: np.ndarray) -> np.ndarray:
    """A[h_idx, c]: regression slope (on instantaneous S2_t) of the cumulative injected
    move over horizon h, per unit gain of basis component c (linear-response model).

    Component driver e_c is a unit-gain EMA of S2; its covariance with S2_t at lead i is
    alpha_c * sum_j (1-alpha_c)^j * rho(|i-j|) (weights truncated at EMA_TAIL)."""
    ext = np.concatenate([rho[::-1][:-1], rho])          # rho over lags -(L-1)..(L-1)
    zero = len(rho) - 1

    def rho_at(k: int) -> float:
        k = abs(k)
        return float(ext[zero + k]) if k < len(rho) else 0.0

    horizons = list(HORIZONS_S.values())
    A = np.zeros((len(horizons), len(BASIS_HALFLIVES_S)))
    for c, hl in enumerate(BASIS_HALFLIVES_S):
        alpha = 1.0 - 2.0 ** (-0.5 / hl)
        w = alpha * (1.0 - alpha) ** np.arange(EMA_TAIL)
        cov_lead = {}
        for i in range(max(horizons)):
            cov_lead[i] = float(sum(w[j] * rho_at(i - j) for j in range(EMA_TAIL)))
        for hi, h in enumerate(horizons):
            A[hi, c] = sum(cov_lead[i] for i in range(h))
    return A


def make_env(scratch: Path, regime: str, kernel: Dict | None) -> ReactiveQRMEnv:
    kw = {}
    if kernel is not None:
        kw = dict(signal_injection=True, signal_kernel=kernel)
    return ReactiveQRMEnv(
        str(scratch / "step3g" / f"qrm_bundle_{regime}_b.npz"),
        str(scratch / "step3g" / f"move_process_{regime}_centered.npz"),
        order_btc=25.0, **kw)


def measure_curve(env: ReactiveQRMEnv, n_eps: int, seed_base: int) -> Dict:
    """Slopes regress future returns on the INSTANTANEOUS imbalance (mirroring the
    real-data measurement; sample_episode computes it from the live book). The driver
    statistics come from the environment's stored composite path (s2_path holds the
    composite in obs_norm units in kernel mode) — Amendment 3 correction 1: reading
    them from the instantaneous imbalance poisoned the offset with E[imbalance]."""
    accums = {h: RegAccum() for h in HORIZONS_S}
    comp_sum = comp_sq = comp_n = 0.0
    norm = float(env.signal_kernel.get("obs_norm", 1.0)) if env.signal_kernel else 1.0
    for i in range(n_eps):
        mids, sig = sample_episode(env, seed_base + i)
        accumulate_episode(accums, mids, sig)
        comp = env._ep.s2_path * norm            # composite delta in bps
        x = comp[np.isfinite(comp)]
        comp_sum += x.sum(); comp_sq += (x * x).sum(); comp_n += len(x)
    m = comp_sum / max(comp_n, 1)
    return {"slopes": {h: accums[h].stats()["slope"] for h in HORIZONS_S},
            "mean_delta_bps": m,
            "std_delta_bps": float(np.sqrt(max(comp_sq / max(comp_n, 1) - m * m, 0.0)))}


def _kern(g, means, offset_bps: float) -> Dict:
    """Assemble a kernel dict with the given gains, means and scalar offset (obs_norm set
    to 1.0 for the solve passes; the final obs_norm is the verified composite std)."""
    return {"halflives_s": list(BASIS_HALFLIVES_S),
            "gains_bps": [float(x) for x in g],
            "means": [float(m) for m in means], "obs_norm": 1.0,
            "offset_bps": float(offset_bps)}


def measure_drift(env: ReactiveQRMEnv, n_eps: int, seed0: int) -> Dict:
    """TOTAL end-to-end p_ref drift under a do-nothing policy -- the drift the agent
    actually faces (base + injection). Amendment 3 Correction 2c: measured UNPAIRED (the
    2b CRN pairing gave only ~30% variance reduction because the engine RNG desyncs once
    the injection perturbs the book; precision comes from episode count instead). The
    decomposition (2026-07-24) showed the injection drift is large and systematic, so it
    is easily resolved at a raised episode count, and the offset that cancels it is the
    EXACT linear correction offset += mean_bps / INTERVALS_PER_EP (1 bps of per-interval
    offset shifts p_ref by INTERVALS_PER_EP intervals * price/tick, a known slope)."""
    ticks = np.zeros(n_eps)
    bps = np.zeros(n_eps)
    for j in range(n_eps):
        env.reset(seed=seed0 + j)
        p0 = env._ep.p_ref
        done = False
        while not done:
            _o, _r, done, _i = env.step(0)
        ticks[j] = (env._ep.p_ref - p0) / env.tick
        bps[j] = (env._ep.p_ref - p0) / p0 * 1e4
    se = float(ticks.std(ddof=1) / np.sqrt(n_eps))
    return {"mean_ticks": float(ticks.mean()), "se_ticks": se,
            "t": float(ticks.mean() / se) if se > 0 else 0.0,
            "mean_bps": float(bps.mean()), "n": n_eps}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch", required=True, type=Path)
    args = ap.parse_args()
    scratch = args.scratch

    real = json.loads((scratch / "signal" / "measurement.json").read_text())
    endo = json.loads((scratch / "signal" / "endogenous_baseline.json").read_text())
    winner = real["selection"]["winner"]

    out: Dict = {"registered": {"basis_halflives_s": BASIS_HALFLIVES_S,
                                "rho_eps": RHO_EPS, "rho_seed_base": RHO_SEED_BASE,
                                "refine_eps": REFINE_EPS,
                                "refine_seed_base": REFINE_SEED_BASE,
                                "max_refine_iters": MAX_REFINE_ITERS,
                                "refine_damping": REFINE_DAMPING,
                                "refine_stop_gap": REFINE_STOP_GAP,
                                "verify_seed_base": VERIFY_SEED_BASE,
                                "drift_seed_base": DRIFT_SEED_BASE,
                                "drift_null_eps": DRIFT_NULL_EPS,
                                "drift_probe_eps": DRIFT_PROBE_EPS,
                                "offset_probe_bps": OFFSET_PROBE_BPS},
                 "regimes": {}}
    for regime in ("calm", "volatile"):
        real_c = np.array([real["results"][winner][regime]["calibrate"][h]["slope"]
                           for h in HORIZONS_S])
        endo_c = np.array([endo["endogenous"][regime][h]["slope"] for h in HORIZONS_S])
        target = real_c - endo_c                          # the residual curve

        # (a) rho
        logger.info("%s: measuring rho (%d episodes)...", regime, RHO_EPS)
        rho_info = measure_rho(make_env(scratch, regime, None), RHO_EPS)
        rho = rho_info["rho"]
        s2_mean = rho_info["mean"]

        # (b) closed-form solve
        A = response_matrix(rho)
        g, *_ = np.linalg.lstsq(A, target, rcond=None)
        logger.info("%s: analytic gains %s", regime, np.round(g, 4))

        # (c) PHASE 1 — gains iterations, offset fixed at ZERO (means-only demeaning;
        # calibration-time drift stays at the mild Amendment-1 scale). Slopes are
        # offset-invariant, so gains converge independently of the offset.
        # Correction 2 (a): damped (half-strength) corrections, cap 5, early stop when
        # every gated horizon is within 15%.
        means = [s2_mean] * len(BASIS_HALFLIVES_S)
        gated = [list(HORIZONS_S).index(h) for h in GATED]
        history: List[Dict] = []
        for it in range(MAX_REFINE_ITERS):
            kernel = {"halflives_s": list(BASIS_HALFLIVES_S),
                      "gains_bps": [float(x) for x in g],
                      "means": [float(m) for m in means], "obs_norm": 1.0,
                      "offset_bps": 0.0}
            ach = measure_curve(make_env(scratch, regime, kernel), REFINE_EPS,
                                REFINE_SEED_BASE)
            ach_curve = np.array([ach["slopes"][h] for h in HORIZONS_S])
            gaps = np.abs(ach_curve - real_c) / np.abs(real_c)
            max_gated = float(max(gaps[i] for i in gated))
            history.append({"iter": it, "gains": [float(x) for x in g],
                            "achieved": [float(x) for x in ach_curve],
                            "rel_gaps": [float(x) for x in gaps],
                            "max_gated_gap": max_gated,
                            "mean_delta_bps": ach["mean_delta_bps"],
                            "std_delta_bps": ach["std_delta_bps"]})
            logger.info("%s gains iter %d: gaps %s | max gated %.0f%% | mean delta "
                        "%+.5f bps", regime, it,
                        " ".join(f"{x*100:.0f}%" for x in gaps), max_gated * 100,
                        ach["mean_delta_bps"])
            if max_gated < REFINE_STOP_GAP:
                logger.info("%s gains converged at iter %d (max gated %.0f%% < %.0f%%)",
                            regime, it, max_gated * 100, REFINE_STOP_GAP * 100)
                break
            # Correction 2a plateau stop: data-defined convergence — halt when the max
            # gated gap improves by < 1 point over two consecutive iterations.
            if (it >= 2 and history[it - 2]["max_gated_gap"] - max_gated
                    < PLATEAU_MIN_IMPROVE):
                logger.info("%s gains PLATEAU at iter %d (%.1f%% -> %.1f%% over 2 iters)",
                            regime, it, history[it - 2]["max_gated_gap"] * 100,
                            max_gated * 100)
                break
            shortfall = target - (ach_curve - endo_c)
            dg, *_ = np.linalg.lstsq(A, shortfall, rcond=None)
            g = g + REFINE_DAMPING * dg

        # PHASE 2 — offset by EXACT 2-POINT SLOPE SOLVE (Correction 2c revised, 2026-07-25).
        # The drift is exactly linear in the offset, but the naive slope assumption (1 bps
        # of offset -> INTERVALS_PER_EP*price/tick) was WRONG: the injected move is
        # attenuated (~40%) by the fractional-carry accumulator and the book response, so
        # the fixed-step correction under-cancelled (~58%/step) and would not null volatile.
        # Instead measure the drift at TWO offsets on the calibration block, fit the (linear)
        # slope, and solve for the zero-drift offset directly. Block-transfer was confirmed
        # (2026-07-24): the injection drift is block-STABLE, so the offset solved here
        # transfers to the independent fairness certification block.
        # raw drift (offset=0) on each independent anchor block; pool the means
        raw = [measure_drift(make_env(scratch, regime, _kern(g, means, 0.0)),
                             DRIFT_NULL_EPS, sb) for sb in ANCHOR_SEED_BASES]
        raw_means = [r["mean_ticks"] for r in raw]
        pooled_raw = float(np.mean(raw_means))
        # slope from a probe on the first anchor block (block-independent mechanical constant)
        d_probe = measure_drift(make_env(scratch, regime, _kern(g, means, OFFSET_PROBE_BPS)),
                                DRIFT_PROBE_EPS, ANCHOR_SEED_BASES[0])
        slope = (d_probe["mean_ticks"] - raw[0]["mean_ticks"]) / OFFSET_PROBE_BPS
        offset = -pooled_raw / slope if slope != 0 else 0.0
        drift_history = [{"anchor_seed_base": sb, "offset_bps": 0.0, **r}
                         for sb, r in zip(ANCHOR_SEED_BASES, raw)]
        drift_history.append({"probe": True, "offset_bps": OFFSET_PROBE_BPS, **d_probe})
        drift_history.append({"pooled_raw_ticks": pooled_raw,
                              "raw_by_block": [float(x) for x in raw_means],
                              "slope_ticks_per_bps": float(slope),
                              "solved_offset_bps": float(offset)})
        logger.info("%s offset solve: raw by block %s -> pooled %+.3f | slope %.0f "
                    "ticks/bps -> offset %+.6f bps", regime,
                    " ".join(f"{x:+.2f}" for x in raw_means), pooled_raw, slope, offset)

        # PHASE 3 — final frozen kernel. Correction 2 (c): verification on FRESH seeds
        # never used during any calibration iteration; obs_norm from the verified
        # composite std so the observation/follower signal is in std units.
        kernel = {"halflives_s": list(BASIS_HALFLIVES_S),
                  "gains_bps": [float(x) for x in g],
                  "means": [float(m) for m in means], "obs_norm": 1.0,
                  "offset_bps": float(offset)}
        ver = measure_curve(make_env(scratch, regime, kernel), REFINE_EPS,
                            VERIFY_SEED_BASE)
        ver_curve = np.array([ver["slopes"][h] for h in HORIZONS_S])
        gaps = np.abs(ver_curve - real_c) / np.abs(real_c)
        # verify the solved offset actually nulls the drift, on a FRESH block
        ver_drift = measure_drift(make_env(scratch, regime, kernel), DRIFT_NULL_EPS,
                                  VERIFY_SEED_BASE)
        logger.info("%s DRIFT VERIFY (fresh block): %+.3f ticks/ep (SE %.3f, t %+.2f)",
                    regime, ver_drift["mean_ticks"], ver_drift["se_ticks"], ver_drift["t"])
        kernel["obs_norm"] = ver["std_delta_bps"] or 1.0
        out["regimes"][regime] = {
            "rho_first10": [float(x) for x in rho[:10]],
            "s2_mean": s2_mean, "history": history,
            "drift_null_history": drift_history,
            "drift_verify": ver_drift,
            "kernel": kernel,
            "verified": {"achieved": [float(x) for x in ver_curve],
                         "real": [float(x) for x in real_c],
                         "rel_gaps": [float(x) for x in gaps],
                         "mean_delta_bps": ver["mean_delta_bps"],
                         "std_delta_bps": ver["std_delta_bps"],
                         "max_gated_gap": float(max(gaps[i] for i in gated))},
        }
        logger.info("%s FROZEN: gains=%s max gated gap=%.1f%% drift %+.2f ticks", regime,
                    np.round(g, 4), out["regimes"][regime]["verified"]["max_gated_gap"] * 100,
                    ver_drift["mean_ticks"])

    (scratch / "signal" / "kernel_solution.json").write_text(json.dumps(out, indent=1))
    logger.info("wrote kernel_solution.json")


if __name__ == "__main__":
    main()
