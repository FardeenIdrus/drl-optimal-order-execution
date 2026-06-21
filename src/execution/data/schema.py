"""Canonical order-book schema — the source-agnostic contract.

Every source adapter decodes its raw format into a pandas DataFrame conforming to
this schema; all stages downstream of parsing depend only on these columns, never
on a provider's wire format. Swapping data sources means writing a new adapter
that produces this same shape.

Layout (one row per snapshot), with level 1 = best:
    ts                       int64    exchange time, milliseconds since Unix epoch (UTC)
    bid_px_1..N, bid_sz_1..N, bid_n_1..N    float64   bids, descending price
    ask_px_1..N, ask_sz_1..N, ask_n_1..N    float64   asks, ascending price

Absent levels (a side shorter than N) are padded with NaN. ``n`` (order count) is
stored as float64 so the NaN padding is representable.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

N_LEVELS: int = 20
TS_COL: str = "ts"
_SIDES: Tuple[str, str] = ("bid", "ask")
_FIELDS: Tuple[str, str, str] = ("px", "sz", "n")


def level_columns(side: str, field: str, n_levels: int = N_LEVELS) -> List[str]:
    """Column names for one (side, field) across all levels, best first."""
    return [f"{side}_{field}_{i}" for i in range(1, n_levels + 1)]


def canonical_columns(n_levels: int = N_LEVELS) -> List[str]:
    """Full ordered column list for the canonical book DataFrame."""
    cols: List[str] = [TS_COL]
    for side in _SIDES:
        for field in _FIELDS:
            cols += level_columns(side, field, n_levels)
    return cols


def canonical_dtypes(n_levels: int = N_LEVELS) -> Dict[str, str]:
    """Expected dtype per canonical column."""
    dtypes: Dict[str, str] = {TS_COL: "int64"}
    for side in _SIDES:
        for field in _FIELDS:
            for col in level_columns(side, field, n_levels):
                dtypes[col] = "float64"
    return dtypes


def validate(df: pd.DataFrame, n_levels: int = N_LEVELS) -> Tuple[pd.Series, Dict[str, int]]:
    """Validate book snapshots and return ``(valid_mask, counts)``.

    A snapshot is invalid if any of:
      - empty side: best bid or best ask is NaN,
      - crossed/locked: best_bid >= best_ask,
      - non-monotonic: bid prices not non-increasing, or ask prices not
        non-decreasing across present levels (should not happen if the source
        delivers sorted levels; flagged as a decode-integrity check).

    NaN-padded deep levels are ignored by the monotonicity check.
    """
    bid1, ask1 = df["bid_px_1"], df["ask_px_1"]
    empty_side = bid1.isna() | ask1.isna()
    crossed = (~empty_side) & (bid1 >= ask1)

    bid_px = df[level_columns("bid", "px", n_levels)].to_numpy()
    ask_px = df[level_columns("ask", "px", n_levels)].to_numpy()
    # NaN diffs compare False, so padding never counts as a violation.
    bid_bad = np.nansum(np.where(np.diff(bid_px, axis=1) > 0, 1.0, 0.0), axis=1) > 0
    ask_bad = np.nansum(np.where(np.diff(ask_px, axis=1) < 0, 1.0, 0.0), axis=1) > 0
    non_monotonic = pd.Series(bid_bad | ask_bad, index=df.index)

    valid = ~(empty_side | crossed | non_monotonic)
    counts = {
        "total": int(len(df)),
        "empty_side": int(empty_side.sum()),
        "crossed_or_locked": int(crossed.sum()),
        "non_monotonic": int(non_monotonic.sum()),
        "valid": int(valid.sum()),
    }
    return valid, counts
