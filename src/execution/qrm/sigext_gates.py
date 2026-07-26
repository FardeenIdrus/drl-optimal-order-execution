"""Measured-signal extension, Phase C: re-verify every environment gate with the
signal injection ON, plus the new injection-matching gate.

Registered: criteria section 8 (Phase B registration fixes the injection parameters;
the matching-gate band, +/-20% at 1-10 s horizons, was registered there before this
driver ran). Gates and bands are IDENTICAL to the originals:

- step4 gates (G1', G2', G3): re-run via the original step4_gates functions. The
  full-process environment factory (`step4_gates._env`) is swapped for the injected
  factory, so every gate that involves the background process runs with injection ON.
  The drift-free mechanics probes (`_driftfree_env`) are left as constructed by the
  original code: injection cannot affect fill/impact mechanics (the flag-OFF path is
  byte-identical to the pre-change environment, proven on golden traces), and their
  fresh outputs are recorded for completeness.
- Fairness gate: the original measurement functions (`_measure_bg_drift`,
  `_measure_pace_gradient`) on the injected environment, original sample sizes,
  seed block and thresholds; identical verdict logic.
- Injection-matching gate (NEW): the Phase A2 measurement repeated inside the
  injected environment; the total in-sim S2-to-return slope must match the real
  calibrate-split slope within 20% relative at the 1, 2, 5 and 10 s horizons,
  per regime.

Output: one JSON with every gate's full result and a single all_pass verdict.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np

from execution.qrm import step4_gates as g4
from execution.qrm.reactive_env import ReactiveQRMEnv
from execution.qrm.signal_measure import (A2_EPISODES, A2_SEED_BASE, HORIZONS_S,
                                          RegAccum, accumulate_episode, sample_episode)
from execution.qrm.step3g import (FAIRNESS_SEED0, _measure_bg_drift,
                                  _measure_pace_gradient)

logger = logging.getLogger(__name__)

MATCH_HORIZONS = ("1", "2", "5", "10")   # registered band: +/-20% relative
MATCH_REL_TOL = 0.20


def load_injection_params(scratch: Path) -> Dict[str, Dict[str, float]]:
    """Parameter precedence: Amendment-3 derived kernel (kernel_solution.json) if it
    exists; else the superseded Amendment-2 single-EMA calibration; else Amendment-1
    instantaneous parameters."""
    ks = scratch / "signal" / "kernel_solution.json"
    if ks.exists():
        k = json.loads(ks.read_text())
        return {r: {"kernel": k["regimes"][r]["kernel"]}
                for r in ("calm", "volatile")}
    kc = scratch / "signal" / "kernel_calibration.json"
    if kc.exists():
        k = json.loads(kc.read_text())
        return {r: {"residual_bps": float(k["regimes"][r]["gain"]),
                    "mean": float(k["regimes"][r]["demeaning_mean"]),
                    "ema_halflife_s": float(k["regimes"][r]["selected_halflife_s"])}
                for r in ("calm", "volatile")}
    endo = json.loads((scratch / "signal" / "endogenous_baseline.json").read_text())
    means = json.loads((scratch / "signal" / "demeaning_constants.json").read_text())
    out = {}
    for regime in ("calm", "volatile"):
        out[regime] = {
            "residual_bps": float(endo["residual"][regime]["0.5"]["residual"]),
            "mean": float(means[regime]["mean_s2_bg"]),
            "ema_halflife_s": None,
        }
    return out


def make_injected_factory(scratch: Path, params):
    def factory(_scratch: Path, regime: str, order_btc: float) -> ReactiveQRMEnv:
        return ReactiveQRMEnv(
            str(scratch / "step3g" / f"qrm_bundle_{regime}_b.npz"),
            str(scratch / "step3g" / f"move_process_{regime}_centered.npz"),
            order_btc=order_btc,
            signal_injection=True,
            **({"signal_kernel": params[regime]["kernel"]}
               if "kernel" in params[regime] else
               {"signal_residual_bps": params[regime]["residual_bps"],
                "signal_mean": params[regime]["mean"],
                "signal_ema_halflife_s": params[regime].get("ema_halflife_s")}))
    return factory


DRIFT_MAG_TOL_TICKS = 0.5   # Amendment 3 Correction 2 (d): the base-env neutralisation
                            # acceptance tolerance (step3g cmd_neutralise stopping rule)


def fairness_gate_injected(env: ReactiveQRMEnv, regime: str, n_drift: int,
                           n_grad: int) -> Dict:
    """Thresholds and verdict logic as step3g.cmd_fairness_gate. Drift is the TOTAL p_ref
    drift the agent faces, measured unpaired (Amendment 3 Correction 2c: the offset is
    solved to a precise cancellation on an independent block, so at n_drift the residual
    here is within |t|<2 without needing sub-noise-floor resolution). Criterion unchanged
    (Correction 2 (d)): PASS iff |t| < 2 (statistical zero) OR |mean| <= 0.5 ticks/ep.
    The magnitude is reported regardless; the pace-gradient exploitability clause -- the
    direct test that no timing strategy can profit from any residual -- is unchanged and
    always required."""
    d = _measure_bg_drift(env, n_drift, FAIRNESS_SEED0)
    drift_mean = float(d.mean())
    drift_se = float(d.std(ddof=1) / np.sqrt(n_drift))
    drift_t = drift_mean / drift_se if drift_se > 0 else 0.0
    drift_stat_zero = abs(drift_t) < 2.0
    drift_mag_ok = abs(drift_mean) <= DRIFT_MAG_TOL_TICKS
    drift_pass = drift_stat_zero or drift_mag_ok

    grad = _measure_pace_gradient(env, n_grad, FAIRNESS_SEED0)
    rows = []
    grad_pass = True
    for m in sorted(grad):
        mean, t, resid = grad[m]
        completing = resid < 0.02
        material_sig = bool(completing and mean <= -0.02 and t <= -2.5)
        if material_sig:
            grad_pass = False
        rows.append({"mult": m, "cost_vs_twap_bps": round(mean, 4), "t": round(t, 2),
                     "deadline_resid_frac": round(resid, 4), "completing": completing,
                     "material_sig_advantage": material_sig})
    return {"regime": regime, "n_drift": n_drift, "n_grad": n_grad,
            "seed0": FAIRNESS_SEED0,
            "background_drift_ticks_per_ep": round(drift_mean, 3),
            "background_drift_t": round(drift_t, 2),
            "background_drift_se": round(drift_se, 3),
            "background_drift_bps": round(drift_mean * env.tick / 100000.0 * 1e4, 4),
            "drift_measure": "total, unpaired (Correction 2c)",
            "drift_stat_zero": bool(drift_stat_zero),
            "drift_mag_within_tol": bool(drift_mag_ok),
            "drift_mag_tol_ticks": DRIFT_MAG_TOL_TICKS,
            "drift_criterion": "Correction 2 (d): |t|<2 OR |mean|<=0.5 ticks/ep",
            "drift_pass": bool(drift_pass),
            "pace_gradient": rows, "gradient_pass": bool(grad_pass),
            "fairness_pass": bool(drift_pass and grad_pass)}


def matching_gate(scratch: Path, factory, n_episodes: int) -> Dict:
    real = json.loads((scratch / "signal" / "measurement.json").read_text())
    winner = real["selection"]["winner"]
    out: Dict = {"signal": winner, "regimes": {}}
    for regime in ("calm", "volatile"):
        env = factory(scratch, regime, 25.0)
        accums = {h: RegAccum() for h in HORIZONS_S}
        for i in range(n_episodes):
            mids, s2 = sample_episode(env, A2_SEED_BASE + i)
            accumulate_episode(accums, mids, s2)
            if (i + 1) % 200 == 0:
                logger.info("matching %s: %d/%d", regime, i + 1, n_episodes)
        rows = {}
        ok_all = True
        for h in HORIZONS_S:
            sim = accums[h].stats()
            real_slope = real["results"][winner][regime]["calibrate"][h]["slope"]
            rel = abs(sim["slope"] - real_slope) / abs(real_slope)
            in_band = bool(rel <= MATCH_REL_TOL) if h in MATCH_HORIZONS else None
            if h in MATCH_HORIZONS and not in_band:
                ok_all = False
            rows[h] = {"sim_total_slope": sim["slope"], "sim_r2": sim["r2"],
                       "n": sim["n"], "real_calibrate_slope": real_slope,
                       "rel_gap": float(rel), "gated": h in MATCH_HORIZONS,
                       "pass": in_band}
        out["regimes"][regime] = {"horizons": rows, "pass": bool(ok_all)}
        logger.info("matching %s: %s", regime,
                    "PASS" if ok_all else "FAIL")
    out["pass"] = all(out["regimes"][r]["pass"] for r in out["regimes"])
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch", required=True, type=Path)
    ap.add_argument("--book-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--n-states", type=int, default=200)
    ap.add_argument("--n-eps", type=int, default=500)
    ap.add_argument("--n-drift", type=int, default=8000)
    ap.add_argument("--n-grad", type=int, default=3000)
    ap.add_argument("--n-match", type=int, default=A2_EPISODES)
    args = ap.parse_args()
    scratch = args.scratch
    out_path = args.out or (scratch / "signal" / "gates" / "sigext_gates.json")

    params = load_injection_params(scratch)
    logger.info("injection params: %s", params)
    factory = make_injected_factory(scratch, params)
    g4._env = factory        # full-process gates run with injection ON

    report: Dict = {
        "criteria": "reports/qrm_step4_criteria.md section 8 (Phase B/C registration)",
        "injection_params": params,
        "G1_reaction_lever_rev1": g4.gate1_rev(scratch),
        "G2_cost_vs_size_rev1": g4.gate2_rev(scratch, args.book_dir, args.n_states),
        "G3_benchmark_sanity_rev1": g4.gate3(scratch, args.n_eps),
    }
    report["fairness"] = {
        r: fairness_gate_injected(factory(scratch, r, 25.0), r,
                                  args.n_drift, args.n_grad)
        for r in ("calm", "volatile")}
    report["injection_matching"] = matching_gate(scratch, factory, args.n_match)
    report["all_pass"] = bool(
        report["G1_reaction_lever_rev1"]["pass"]
        and report["G2_cost_vs_size_rev1"]["pass"]
        and report["G3_benchmark_sanity_rev1"]["pass"]
        and all(report["fairness"][r]["fairness_pass"] for r in report["fairness"])
        and report["injection_matching"]["pass"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1))
    logger.info("wrote %s ; ALL_PASS=%s", out_path, report["all_pass"])


if __name__ == "__main__":
    main()
