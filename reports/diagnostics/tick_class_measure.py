"""Tick-class and queue-imbalance predictability, measurements (i) and (iii).

Registered protocol: reports/qrm_step4_criteria.md section 9 (2026-08-03), written BEFORE
(iii) ran. Measurement (i) was already run on 7 days on 2026-08-03 and is NOT pre-registered;
this module extends it to all 31 days and reports the 7-day subset alongside so the extension
cannot be accused of selection.

(i)   Tick size, relative tick, spread distribution -> is this venue large-tick in the
      MECHANISM sense Gould and Bonart (2016) describe (spread pinned at the minimum, so the
      price moves only when a queue empties)?
(iii) AUC of sign(forward mid return) predicted by top-of-book queue imbalance, at the seven
      registered horizons, per regime, per split.

Deliberately standalone: no existing module is edited (same discipline as the per-episode
re-scoring). Streams one day at a time; never loads the month.

NO PASS/FAIL GATE. This is measurement, not a test (registered R5).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------- registered constants (R3/R4)
SCRATCH = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
BOOK_DIR = SCRATCH / "oxford_l4" / "book_05s_v2"
LABELS = SCRATCH / "oxford_l4" / "step3g" / "regime_labels.parquet"
OUT = SCRATCH / "oxford_l4" / "tick_class" / "tick_class_measurement.json"

GRID_MS = 500
HORIZONS_S: Dict[str, int] = {          # horizon label -> forward steps on the 0.5 s grid
    "0.5": 1, "1": 2, "2": 4, "5": 10, "10": 20, "30": 60, "60": 120,
}
# (i) was measured on these 7 days first; reported alongside the full month (R1)
FIRST_SAMPLE = ("20251201", "20251206", "20251211", "20251216", "20251221",
                "20251226", "20251231")

AUC_BINS = 2001                          # imbalance lives in [-1, 1]; exact enough, RAM-flat
EPS = 1e-12


# ------------------------------------------------------------------ pure helpers (tested)
def queue_imbalance(bid_sz: np.ndarray, ask_sz: np.ndarray) -> np.ndarray:
    """S2 exactly as registered: (bid1 - ask1)/(bid1 + ask1), RAW, no normalisation.

    Undefined (NaN) where either side is swept, i.e. the denominator vanishes. Uses only
    quantities observed at t -- strictly causal by construction.
    """
    denom = bid_sz + ask_sz
    out = np.where(denom > EPS, (bid_sz - ask_sz) / np.where(denom > EPS, denom, 1.0), np.nan)
    return out


def auc_from_histograms(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(score_pos > score_neg) + 0.5 P(score_pos == score_neg), from binned counts.

    Equivalent to the Mann-Whitney statistic. Returns NaN if either class is empty.
    """
    npos, nneg = pos.sum(), neg.sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    neg_below = np.concatenate([[0.0], np.cumsum(neg)[:-1]])      # strictly-lower bins
    wins = (pos * neg_below).sum() + 0.5 * (pos * neg).sum()
    return float(wins / (npos * nneg))


def bin_index(x: np.ndarray, n_bins: int = AUC_BINS) -> np.ndarray:
    """Map imbalance in [-1, 1] to a bin index. Values are already bounded by construction."""
    z = np.clip((x + 1.0) * 0.5, 0.0, 1.0)
    return np.minimum((z * n_bins).astype(np.int64), n_bins - 1)


class AucAccum:
    """Histogram accumulator for one (regime, split, horizon) cell."""

    __slots__ = ("pos", "neg", "n_used", "n_zero")

    def __init__(self) -> None:
        self.pos = np.zeros(AUC_BINS, dtype=np.float64)
        self.neg = np.zeros(AUC_BINS, dtype=np.float64)
        self.n_used = 0
        self.n_zero = 0

    def add(self, imb: np.ndarray, fwd: np.ndarray) -> None:
        ok = np.isfinite(imb) & np.isfinite(fwd)
        self.n_zero += int((ok & (fwd == 0)).sum())
        ok &= fwd != 0                                    # no-change intervals excluded (R4)
        if not ok.any():
            return
        idx = bin_index(imb[ok])
        up = fwd[ok] > 0
        self.pos += np.bincount(idx[up], minlength=AUC_BINS).astype(np.float64)
        self.neg += np.bincount(idx[~up], minlength=AUC_BINS).astype(np.float64)
        self.n_used += int(ok.sum())

    def result(self) -> Dict[str, float]:
        return {"auc": auc_from_histograms(self.pos, self.neg),
                "n": self.n_used, "n_excluded_no_change": self.n_zero}


def infer_tick(px: np.ndarray) -> float:
    """Smallest positive gap on the observed price grid."""
    u = np.unique(px[np.isfinite(px)])
    d = np.diff(u)
    d = d[d > EPS]
    return float(np.min(d)) if d.size else float("nan")


# ---------------------------------------------------------------------------- measurement
def run() -> dict:
    labels = pd.read_parquet(LABELS)
    labels["date"] = labels["date"].astype(str)
    label_map = {(r.date, int(r.hour)): (str(r.regime), str(r.split))
                 for r in labels.itertuples()}

    days = sorted(p.stem for p in BOOK_DIR.glob("2025*.parquet"))
    logger.info("days on disk: %d", len(days))

    per_day = []
    accums: Dict[Tuple[str, str, str], AucAccum] = {}
    for regime in ("calm", "volatile"):
        for split in ("calibrate", "holdout"):
            for h in HORIZONS_S:
                accums[(regime, split, h)] = AucAccum()

    for date in days:
        df = pd.read_parquet(BOOK_DIR / f"{date}.parquet",
                             columns=["ts", "best_bid", "best_ask", "mid", "spread",
                                      "bid_sz_1", "ask_sz_1"])
        df = df.dropna(subset=["best_bid", "best_ask", "mid", "spread"])
        if df.empty:
            continue

        # ---- (i) tick and spread, whole day ----------------------------------------
        tick = infer_tick(np.concatenate([df.best_bid.values, df.best_ask.values]))
        sp_ticks = np.rint(df.spread.values / tick).astype(np.int64)
        med_mid = float(np.median(df.mid.values))
        per_day.append({
            "date": date, "n_samples": int(len(df)), "tick": tick,
            "median_mid": med_mid, "relative_tick": tick / med_mid,
            "p_spread_1_tick": float((sp_ticks == 1).mean()),
            "p_spread_le_2_ticks": float((sp_ticks <= 2).mean()),
            "mean_spread_ticks": float(sp_ticks.mean()),
            "median_spread_ticks": int(np.median(sp_ticks)),
            "mean_spread_bps": float((df.spread.values / df.mid.values * 1e4).mean()),
            "in_first_sample": date in FIRST_SAMPLE,
        })

        # ---- (iii) AUC by horizon, WITHIN each labelled hour ------------------------
        ts = pd.to_datetime(df["ts"])
        hour = ts.dt.hour.values
        imb_all = queue_imbalance(df.bid_sz_1.values.astype(float),
                                  df.ask_sz_1.values.astype(float))
        mid_all = df.mid.values.astype(float)

        for hh in np.unique(hour):
            key = label_map.get((date, int(hh)))
            if key is None:                       # unlabelled hour (743 of 744 are labelled)
                continue
            regime, split = key
            m = hour == hh
            imb_h, mid_h = imb_all[m], mid_all[m]
            for h, k in HORIZONS_S.items():
                if mid_h.size <= k:
                    continue
                fwd = mid_h[k:] - mid_h[:-k]      # within-hour only: no cross-boundary returns
                accums[(regime, split, h)].add(imb_h[:-k], fwd)
        del df

    day_df = pd.DataFrame(per_day)
    ticks = day_df.tick.unique()

    def pooled(sub: pd.DataFrame) -> dict:
        w = sub.n_samples.values.astype(float)
        return {
            "n_days": int(len(sub)), "n_samples": int(w.sum()),
            "p_spread_1_tick": float(np.average(sub.p_spread_1_tick, weights=w)),
            "p_spread_le_2_ticks": float(np.average(sub.p_spread_le_2_ticks, weights=w)),
            "mean_spread_ticks": float(np.average(sub.mean_spread_ticks, weights=w)),
            "mean_spread_bps": float(np.average(sub.mean_spread_bps, weights=w)),
            "median_mid": float(np.median(sub.median_mid)),
            "relative_tick": float(np.median(sub.relative_tick)),
        }

    out = {
        "registered": "reports/qrm_step4_criteria.md section 9 (2026-08-03)",
        "note_i_not_preregistered": (
            "Measurement (i) was run on 7 days BEFORE the registration and is reported here "
            "for the full month alongside that 7-day subset (R1). It is NOT pre-registered."),
        "no_pass_fail_gate": True,
        "tick_constant_across_days": bool(len(ticks) == 1),
        "tick_values_seen": [float(t) for t in ticks],
        "spread": {
            "full_month": pooled(day_df),
            "first_7_day_sample": pooled(day_df[day_df.in_first_sample]),
        },
        "per_day": per_day,
        "auc_by_horizon": {
            f"{regime}/{split}/{h}": accums[(regime, split, h)].result()
            for regime in ("calm", "volatile")
            for split in ("calibrate", "holdout")
            for h in HORIZONS_S
        },
        "provenance": {
            "book": str(BOOK_DIR), "labels": str(LABELS),
            "grid_ms": GRID_MS, "horizons_s": list(HORIZONS_S),
            "predictor": "(bid_sz_1 - ask_sz_1)/(bid_sz_1 + ask_sz_1), raw, NaN when swept",
            "label": "sign of forward mid change; zero-change intervals excluded",
            "boundary": "forward returns are computed WITHIN a labelled hour only",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    r = run()
    s = r["spread"]["full_month"]
    print(f"\ndays={s['n_days']}  samples={s['n_samples']:,}")
    print(f"tick constant across days: {r['tick_constant_across_days']}  {r['tick_values_seen']}")
    print(f"relative tick          : {s['relative_tick']:.3e}")
    print(f"P(spread == 1 tick)    : {s['p_spread_1_tick']:.1%}   "
          f"(7-day sample: {r['spread']['first_7_day_sample']['p_spread_1_tick']:.1%})")
    print(f"mean spread            : {s['mean_spread_ticks']:.3f} ticks  "
          f"({s['mean_spread_bps']:.4f} bps)")
    print("\nAUC by horizon (holdout):")
    for regime in ("calm", "volatile"):
        row = "  ".join(f"{h}s={r['auc_by_horizon'][f'{regime}/holdout/{h}']['auc']:.4f}"
                        for h in HORIZONS_S)
        print(f"  {regime:9s} {row}")
    print(f"\nwritten: {OUT}")
