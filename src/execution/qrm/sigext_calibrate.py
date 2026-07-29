"""Measured-signal extension, Amendment 2b: deterministic kernel calibration.

Registered procedure (criteria section 8, Amendment 2): per regime, per half-life in
the fixed grid, run a PROBE (300 background episodes, injection ON, probe gain = the
Amendment-1 residual, demeaning = the registered Amendment-1 mean), solve the gain
linearly so the 1 s total slope equals the real calibrate-split slope, then run a
REFINEMENT at the solved gain with the probe's measured E[driver] as the demeaning
constant (fixed-point iteration 1). Select the half-life minimising the maximum
relative gap over the gated horizons {1,2,5,10} s. Freeze (half-life, gain, mean).

The target is the REAL measured curve throughout — calibration to an external
quantity, never to any experiment outcome.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np

from execution.qrm.reactive_env import ReactiveQRMEnv
from execution.qrm.signal_measure import (HORIZONS_S, RegAccum, accumulate_episode,
                                          sample_episode)

logger = logging.getLogger(__name__)

HALFLIFE_GRID_S = (1.0, 2.0, 5.0, 10.0, 20.0)
N_CAL_EPS = 300
CAL_SEED_BASE = 30_200_000          # diagnostic-only, disjoint from all blocks
GATED = ("1", "2", "5", "10")
SELECT_H = "1"


def _make_env(scratch: Path, regime: str, gain: float, mean: float,
              halflife: float) -> ReactiveQRMEnv:
    return ReactiveQRMEnv(
        str(scratch / "step3g" / f"qrm_bundle_{regime}_b.npz"),
        str(scratch / "step3g" / f"move_process_{regime}_centered.npz"),
        order_btc=25.0, signal_injection=True, signal_residual_bps=gain,
        signal_mean=mean, signal_ema_halflife_s=halflife)


def _measure(env: ReactiveQRMEnv, n_eps: int) -> Dict:
    accums = {h: RegAccum() for h in HORIZONS_S}
    driver_sum = 0.0
    driver_n = 0
    for i in range(n_eps):
        mids, sig = sample_episode(env, CAL_SEED_BASE + i)
        accumulate_episode(accums, mids, sig)
        driver_sum += float(np.nansum(sig))
        driver_n += int(np.isfinite(sig).sum())
    return {"slopes": {h: accums[h].stats() for h in HORIZONS_S},
            "driver_mean": driver_sum / max(driver_n, 1)}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scratch", required=True, type=Path)
    ap.add_argument("--n-eps", type=int, default=N_CAL_EPS)
    args = ap.parse_args()
    scratch = args.scratch

    real = json.loads((scratch / "signal" / "measurement.json").read_text())
    endo = json.loads((scratch / "signal" / "endogenous_baseline.json").read_text())
    means0 = json.loads((scratch / "signal" / "demeaning_constants.json").read_text())
    winner = real["selection"]["winner"]

    out: Dict = {"registered": {"grid_halflives_s": HALFLIFE_GRID_S,
                                "n_eps": args.n_eps, "seed_base": CAL_SEED_BASE,
                                "gated_horizons": GATED, "select_horizon": SELECT_H},
                 "regimes": {}}
    for regime in ("calm", "volatile"):
        real_c = {h: real["results"][winner][regime]["calibrate"][h]["slope"]
                  for h in HORIZONS_S}
        endo_c = {h: endo["endogenous"][regime][h]["slope"] for h in HORIZONS_S}
        g0 = float(endo["residual"][regime]["0.5"]["residual"])
        m0 = float(means0[regime]["mean_s2_bg"])
        rows = {}
        for hl in HALFLIFE_GRID_S:
            probe = _measure(_make_env(scratch, regime, g0, m0, hl), args.n_eps)
            probe_1s = probe["slopes"][SELECT_H]["slope"]
            inj_probe = probe_1s - endo_c[SELECT_H]
            if inj_probe <= 0:
                logger.warning("%s hl=%s: probe injected contribution <= 0; skipping",
                               regime, hl)
                continue
            gain = g0 * (real_c[SELECT_H] - endo_c[SELECT_H]) / inj_probe
            mean1 = probe["driver_mean"]
            ref = _measure(_make_env(scratch, regime, gain, mean1, hl), args.n_eps)
            gaps = {h: abs(ref["slopes"][h]["slope"] - real_c[h]) / abs(real_c[h])
                    for h in HORIZONS_S}
            rows[str(hl)] = {
                "probe_1s_slope": probe_1s, "gain": gain,
                "driver_mean_iter1": mean1,
                "driver_mean_iter2": ref["driver_mean"],
                "refined_slopes": {h: ref["slopes"][h]["slope"] for h in HORIZONS_S},
                "rel_gaps": gaps,
                "max_gated_gap": max(gaps[h] for h in GATED),
            }
            logger.info("%s hl=%ss: gain=%.4f max gated gap=%.1f%%",
                        regime, hl, gain, rows[str(hl)]["max_gated_gap"] * 100)
        best = min(rows, key=lambda k: rows[k]["max_gated_gap"])
        out["regimes"][regime] = {
            "rows": rows,
            "selected_halflife_s": float(best),
            "gain": rows[best]["gain"],
            "demeaning_mean": rows[best]["driver_mean_iter2"],
            "max_gated_gap": rows[best]["max_gated_gap"],
            "real_calibrate_slopes": real_c,
        }
        logger.info("%s SELECTED: hl=%ss gain=%.4f mean=%.4f (max gated gap %.1f%%)",
                    regime, best, rows[best]["gain"],
                    rows[best]["driver_mean_iter2"],
                    rows[best]["max_gated_gap"] * 100)

    out_path = scratch / "signal" / "kernel_calibration.json"
    out_path.write_text(json.dumps(out, indent=1))
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
