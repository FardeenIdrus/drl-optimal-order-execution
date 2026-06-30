"""Stage 4 - feature engineering (trailing-only, no look-ahead).

Turns the per-minute book table (Stage 3) into the market-microstructure features the
RL agent observes. Five candidate signals, each a plausible driver of the execution
edge and chosen to map onto the attribution question:

    spread_bps      cost      mean intra-minute spread / mid, in basis points
    imbalance       direction (bid_depth - ask_depth) / (bid_depth + ask_depth), top-K
    recent_return   momentum  log(mid_t / mid_{t-L})
    rolling_vol     risk      sqrt(sum of intra-minute realised variance over W minutes)
    ask_depth       liquidity mean available ask size over top-K (absolute size the
                              buy-only agent consumes; the QRM paper's key finding)

These are the *market* features; the agent's full state also includes execution state
(inventory, time-remaining) supplied by the environment. Features stay in raw,
interpretable units (the network's normalisation happens in the env) so SHAP
attribution is meaningful.

No look-ahead — two guarantees:
  1. Every feature is backward-looking (current + past minutes only).
  2. The whole feature block is shifted by one minute, so the row at decision time t
     uses only information from bars up to and including t-1 (i.e. fully-completed
     minutes observable at t). This is structural and unit-tested (see tests), not a
     convention left to the environment.

Gaps: features are computed on the complete calendar-minute grid with
``min_periods = full window``; any window touching missing minutes (data gaps,
missing days, the two skip weeks) yields NaN and is flagged ``feature_valid = False``,
so a "30-minute" window never silently spans a gap. Single-snapshot minutes (no
intra-minute return, realised variance undefined) contribute zero measured variance
to the rolling-vol sum; windows are still required to be gap-free.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MS_PER_MINUTE = 60_000
TS_COL = "ts"
FEATURE_COLUMNS: List[str] = ["spread_bps", "imbalance", "recent_return", "rolling_vol", "ask_depth"]
_REQUIRED_INPUT = ["ts", "mid", "mean_spread", "realized_variance"]


def compute_features(
    minute_df: pd.DataFrame,
    imbalance_depth: int = 5,
    return_lookback: int = 5,
    vol_window: int = 30,
    bar_seconds: int = 60,
) -> pd.DataFrame:
    """Compute decision-time-aligned, look-ahead-free features from the bar table.

    Window arguments are wall-clock MINUTES and are converted to a bar count via
    ``bar_seconds`` (bars = minutes * 60 / bar_seconds), so the same economic window
    is held fixed across resolutions: a 5-minute return is 5 bars at 60s but 30 bars
    at 10s. This isolates the resolution effect (identical features, only the decision
    frequency changes) and is byte-identical at 60s (5->5, 30->30 bars).

    Args:
        minute_df: Stage 3 per-bar table (needs ts, mid, mean_spread,
            realized_variance, and mean_bid/ask_sz_{depth}).
        imbalance_depth: book depth for imbalance/ask_depth (1 or 5; the depths Stage 3
            aggregates).
        return_lookback: MINUTES L for the log return (converted to bars).
        vol_window: MINUTES W for the rolling realised volatility (converted to bars).
        bar_seconds: bar width in seconds (60 canonical; finer e.g. 10). Must divide
            each window evenly into a whole number of bars.

    Returns:
        DataFrame indexed by decision-time ``ts`` with ``mid`` and the five features,
        plus ``feature_valid``. Row t uses only bars <= t-1.
    """
    if imbalance_depth not in (1, 5):
        raise ValueError(f"imbalance_depth must be 1 or 5 (Stage 3 aggregates), got {imbalance_depth}")
    if bar_seconds <= 0 or 60 % bar_seconds != 0:
        raise ValueError(f"bar_seconds must be a positive divisor of 60, got {bar_seconds}")
    bar_ms = bar_seconds * 1000
    bars_per_min = 60 // bar_seconds
    return_lookback_bars = return_lookback * bars_per_min   # MINUTES -> bars at this resolution
    vol_window_bars = vol_window * bars_per_min
    bid_col, ask_col = f"mean_bid_sz_{imbalance_depth}", f"mean_ask_sz_{imbalance_depth}"
    required = _REQUIRED_INPUT + [bid_col, ask_col]
    missing = [c for c in required if c not in minute_df.columns]
    if missing:
        raise ValueError(f"minute_df missing required columns: {missing}")

    df = minute_df[required].dropna(subset=[TS_COL]).sort_values(TS_COL)

    # Complete calendar-bar grid so rolling windows respect true time spacing and
    # gaps become NaN rows.
    ts0, ts1 = int(df[TS_COL].min()), int(df[TS_COL].max())
    grid = np.arange(ts0, ts1 + bar_ms, bar_ms, dtype=np.int64)
    g = df.set_index(TS_COL).reindex(grid)
    present = g["mid"].notna()

    # Per-bar (backward-looking) features, NaN where the bar is absent.
    spread_bps = g["mean_spread"] / g["mid"] * 1e4
    depth_sum = g[bid_col] + g[ask_col]
    imbalance = (g[bid_col] - g[ask_col]) / depth_sum
    ask_depth = g[ask_col]
    recent_return = np.log(g["mid"] / g["mid"].shift(return_lookback_bars))

    # Rolling realised volatility from intra-bar realised variance. A present bar with
    # undefined RV (single snapshot) contributes 0; absent bars stay NaN so any window
    # spanning a gap is invalidated by min_periods.
    rv = g["realized_variance"].copy()
    rv[present & rv.isna()] = 0.0
    rolling_vol = np.sqrt(rv.rolling(vol_window_bars, min_periods=vol_window_bars).sum())

    bar = pd.DataFrame(
        {
            "mid": g["mid"],
            "spread_bps": spread_bps,
            "imbalance": imbalance,
            "recent_return": recent_return,
            "rolling_vol": rolling_vol,
            "ask_depth": ask_depth,
        },
        index=grid,
    )

    # Decision-time alignment: shift one bar so row t depends only on bars <= t-1.
    out = bar.shift(1)
    out[TS_COL] = grid
    out["feature_valid"] = out[FEATURE_COLUMNS].notna().all(axis=1)
    return out[[TS_COL, "mid"] + FEATURE_COLUMNS + ["feature_valid"]].reset_index(drop=True)


def feature_qa(features: pd.DataFrame) -> Dict[str, object]:
    """Coverage, distribution, and multicollinearity summary for the features."""
    valid = features[features["feature_valid"]]
    corr = valid[FEATURE_COLUMNS].corr().round(3)
    return {
        "rows": int(len(features)),
        "feature_valid_rows": int(len(valid)),
        "valid_pct": round(100 * len(valid) / len(features), 2) if len(features) else None,
        "describe": {c: {k: round(float(v), 6) for k, v in valid[c].describe().items()}
                     for c in FEATURE_COLUMNS},
        "correlation": {c: corr[c].to_dict() for c in FEATURE_COLUMNS},
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 4 - market-microstructure features")
    p.add_argument("--minute-parquet", required=True)
    p.add_argument("--out", required=True, help="output features parquet (scratch)")
    p.add_argument("--imbalance-depth", type=int, default=5)
    p.add_argument("--return-lookback", type=int, default=5, help="minutes for recent_return")
    p.add_argument("--vol-window", type=int, default=30, help="minutes for rolling_vol")
    p.add_argument("--bar-seconds", type=int, default=60,
                   help="bar width in seconds (default 60; finer e.g. 10). Converts the "
                        "minute windows to bars so the wall-clock window is held fixed.")
    args = p.parse_args()

    # Read only the columns the features need (ts + 4 stats + the depth pair), not the
    # full ~132-column book. At 10s the full table is ~6 GB; this subset is ~0.3 GB, so
    # the features stage fits comfortably in RAM regardless of resolution. Works whether
    # --minute-parquet is a single file or a directory of monthly parts (chunked mode).
    need = ["ts", "mid", "mean_spread", "realized_variance",
            f"mean_bid_sz_{args.imbalance_depth}", f"mean_ask_sz_{args.imbalance_depth}"]
    minute_df = pd.read_parquet(args.minute_parquet, columns=need)
    features = compute_features(
        minute_df,
        imbalance_depth=args.imbalance_depth,
        return_lookback=args.return_lookback,
        vol_window=args.vol_window,
        bar_seconds=args.bar_seconds,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out, index=False)

    params = {"imbalance_depth": args.imbalance_depth, "return_lookback": args.return_lookback,
              "vol_window": args.vol_window, "bar_seconds": args.bar_seconds}
    report = {"params": params, **feature_qa(features)}
    out.with_suffix(".qa.json").write_text(json.dumps(report, indent=2))
    logger.info("wrote %s (%d rows, %d feature-valid)", out, len(features),
                int(features["feature_valid"].sum()))
    logger.info("params: %s", params)


if __name__ == "__main__":
    main()
