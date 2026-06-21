"""Stage 3 - resample full-resolution snapshots to a per-minute book.

Each minute is a completed bar over ``[t, t+60s)`` indexed by its start ``t`` (UTC,
ms). A minute never spans an hour boundary, so every snapshot for a minute lives in
a single hourly file: minutes are self-contained and the parse+resample is fully
parallel per file, with nothing carried or fabricated across files.

Per minute we keep two things:
  1. The point-in-time (end-of-minute) book — the last snapshot in the bar. This is
     the book the agent observes; downstream stages apply causal lags so no future
     information leaks into a decision.
  2. High-frequency intra-minute statistics computed from ALL ~120 snapshots in the
     bar (not just the last one): realised variance (sum of squared intra-minute
     log-mid returns; Andersen & Bollerslev style), snapshot count, high/low mid,
     mean spread, and mean top-1/top-5 depth on each side. These give a far less
     noisy basis for the volatility regime label (Stage 5) and features (Stage 4)
     than a single snapshot would.

Causality note: a row is purely descriptive of its own elapsed minute; turning these
facts into agent features without look-ahead (lagging, rolling windows) is Stage 4.
A minute is ``valid`` iff it actually contained snapshots — book state is never
forward-filled or invented across gaps.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

import numpy as np
import pandas as pd

from execution.data import schema
from execution.data.parse import parse_file

logger = logging.getLogger(__name__)

MS_PER_MINUTE = 60_000
_DEPTH5 = list(range(1, 6))

# Intra-minute statistic columns added on top of the canonical book columns.
STAT_COLUMNS: List[str] = [
    "mid", "n_snapshots", "realized_variance", "hi_mid", "lo_mid",
    "mean_spread", "mean_bid_sz_1", "mean_ask_sz_1", "mean_bid_sz_5", "mean_ask_sz_5",
    "valid",
]


def minute_columns() -> List[str]:
    """Canonical book columns (ts = minute start) plus the intra-minute statistics."""
    book = [c for c in schema.canonical_columns() if c != schema.TS_COL]
    return [schema.TS_COL] + book + STAT_COLUMNS


def resample_snapshots_to_minute(snaps: pd.DataFrame) -> pd.DataFrame:
    """Aggregate validated canonical snapshots into per-minute bars.

    Args:
        snaps: canonical snapshot DataFrame (Stage 2 output), any set of minutes.

    Returns:
        One row per minute that contained snapshots, with the end-of-minute book
        and the intra-minute statistics. ``realized_variance`` is NaN for minutes
        with fewer than two snapshots (no intra-minute return is defined).
    """
    if snaps.empty:
        return pd.DataFrame(columns=minute_columns())

    s = snaps.sort_values(schema.TS_COL, kind="stable").reset_index(drop=True)
    minute = (s[schema.TS_COL] // MS_PER_MINUTE) * MS_PER_MINUTE

    bid1, ask1 = s["bid_px_1"], s["ask_px_1"]
    mid = (bid1 + ask1) / 2.0
    spread = ask1 - bid1
    # squared log-mid returns, kept only when both snapshots are in the same minute
    log_ret = np.log(mid).diff()
    same_minute = minute.eq(minute.shift())
    ret_sq = (log_ret * log_ret).where(same_minute, 0.0)
    bid_sz_5 = s[[f"bid_sz_{i}" for i in _DEPTH5]].sum(axis=1, skipna=True)
    ask_sz_5 = s[[f"ask_sz_{i}" for i in _DEPTH5]].sum(axis=1, skipna=True)

    helper = pd.DataFrame({
        "minute": minute, "mid": mid, "spread": spread, "ret_sq": ret_sq,
        "bid_sz_1": s["bid_sz_1"], "ask_sz_1": s["ask_sz_1"],
        "bid_sz_5": bid_sz_5, "ask_sz_5": ask_sz_5,
    })
    agg = helper.groupby("minute").agg(
        n_snapshots=("mid", "size"),
        realized_variance=("ret_sq", "sum"),
        hi_mid=("mid", "max"),
        lo_mid=("mid", "min"),
        mean_spread=("spread", "mean"),
        mean_bid_sz_1=("bid_sz_1", "mean"),
        mean_ask_sz_1=("ask_sz_1", "mean"),
        mean_bid_sz_5=("bid_sz_5", "mean"),
        mean_ask_sz_5=("ask_sz_5", "mean"),
    )
    # realised variance needs >= 2 snapshots; otherwise undefined.
    agg.loc[agg["n_snapshots"] < 2, "realized_variance"] = np.nan

    # End-of-minute book: the last snapshot in each minute (s is time-sorted).
    last_rows = s.groupby(minute, sort=True).tail(1)
    book_cols = [c for c in schema.canonical_columns() if c != schema.TS_COL]
    book = last_rows[book_cols].copy()
    book.index = ((last_rows[schema.TS_COL] // MS_PER_MINUTE) * MS_PER_MINUTE).to_numpy()

    out = book.join(agg)
    out[schema.TS_COL] = out.index.astype("int64")
    out["mid"] = (out["bid_px_1"] + out["ask_px_1"]) / 2.0
    out["valid"] = True
    out = out[minute_columns()].sort_values(schema.TS_COL).reset_index(drop=True)
    return out


def _parse_one(path: Union[str, Path]) -> pd.DataFrame:
    """Parse one hourly file into validated canonical snapshots (Stage 2)."""
    df, _ = parse_file(path)
    return df


def _file_sort_key(path: Union[str, Path]):
    """Chronological sort key: (date, integer hour). Hour dirs are unpadded, so a
    plain string sort would order '10' before '2' — hence the int cast."""
    parts = Path(path).parts
    return (parts[-3], int(parts[-2]))


def combine_frames_to_minute(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine a time-ordered stream of snapshot frames into one minute table.

    Carries the trailing (possibly-incomplete) minute from each frame onto the next,
    so a minute whose snapshots straddle a file boundary is aggregated exactly once
    from all of its snapshots — no duplicates, no partial bars, cross-boundary
    returns preserved. Across a data gap the carried minute is simply earlier than
    the next frame's minutes, so it is emitted in full (its only snapshots are the
    pre-gap ones) — gaps need no special handling.
    """
    out: List[pd.DataFrame] = []
    carry: pd.DataFrame | None = None
    for df in frames:
        if df is None or df.empty:
            continue
        parts = ([carry] if carry is not None and not carry.empty else []) + [df]
        snaps = pd.concat(parts, ignore_index=True).sort_values(schema.TS_COL, kind="stable")
        minute = (snaps[schema.TS_COL] // MS_PER_MINUTE) * MS_PER_MINUTE
        last_minute = minute.iat[-1]
        is_last = minute.eq(last_minute)
        carry = snaps[is_last].copy()
        complete = snaps[~is_last]
        if not complete.empty:
            out.append(resample_snapshots_to_minute(complete))
    if carry is not None and not carry.empty:
        out.append(resample_snapshots_to_minute(carry))
    if not out:
        return pd.DataFrame(columns=minute_columns())
    result = pd.concat(out, ignore_index=True)

    # Reconcile rare duplicate minutes caused by out-of-order source timestamps: a
    # stray snapshot stamped into a minute that is otherwise complete in another
    # file produces a second, sparsely-populated bar. Keep the most-populated bar
    # (max n_snapshots) for each minute and drop the artifact.
    n_before = len(result)
    result = (
        result.sort_values([schema.TS_COL, "n_snapshots"])
        .drop_duplicates(subset=schema.TS_COL, keep="last")
        .sort_values(schema.TS_COL)
        .reset_index(drop=True)
    )
    dropped = n_before - len(result)
    if dropped:
        logger.warning("dropped %d duplicate minute row(s) from out-of-order source snapshots", dropped)
    return result


def _iter_parsed(paths: Sequence[str], workers: int, batch_files: int) -> Iterable[pd.DataFrame]:
    """Yield parsed snapshot frames in time order, parsing each batch in parallel
    so memory stays bounded to one batch of raw snapshots."""
    if workers and workers > 1:
        ex = ProcessPoolExecutor(max_workers=workers)
        try:
            for i in range(0, len(paths), batch_files):
                yield from ex.map(_parse_one, paths[i:i + batch_files])
        finally:
            ex.shutdown()
    else:
        for p in paths:
            yield _parse_one(p)


def process_files(
    paths: Sequence[Union[str, Path]], workers: int = 1, batch_files: int = 48
) -> pd.DataFrame:
    """Parse + resample many hourly files into one minute table (gap- and
    boundary-correct). Files are processed in chronological order."""
    ordered = sorted((str(p) for p in paths), key=_file_sort_key)
    if not ordered:
        return pd.DataFrame(columns=minute_columns())
    return combine_frames_to_minute(_iter_parsed(ordered, workers, batch_files))


def qa_report(minute_df: pd.DataFrame, expected_minutes: int | None = None) -> Dict[str, object]:
    """Summary statistics for a minute table (coverage, integrity, activity)."""
    ts = minute_df[schema.TS_COL]
    report: Dict[str, object] = {
        "minutes_present": int(len(minute_df)),
        "expected_minutes": expected_minutes,
        "coverage_pct": (round(100 * len(minute_df) / expected_minutes, 2)
                         if expected_minutes else None),
        "duplicate_minutes": int(ts.duplicated().sum()),
        "timestamps_monotonic": bool(ts.is_monotonic_increasing),
        "realized_variance_nan": int(minute_df["realized_variance"].isna().sum()),
        "median_snapshots_per_min": float(minute_df["n_snapshots"].median()),
        "book_ordering_ok": bool((minute_df["bid_px_1"] < minute_df["ask_px_1"]).all()),
    }
    return report


def write_parquet(minute_df: pd.DataFrame, out_path: Union[str, Path]) -> Path:
    """Write the minute table to parquet (creating parent dirs)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    minute_df.to_parquet(out_path, index=False)
    return out_path


def _files_for_day(raw_dir: Union[str, Path], coin: str, day: str) -> List[str]:
    """All hourly files for one day, e.g. <raw_dir>/2025-01-01/*/BTC.lz4."""
    return sorted(glob.glob(str(Path(raw_dir) / day / "*" / f"{coin}.lz4")))


def _files_from_manifest(raw_dir: Union[str, Path], manifest_csv: str, coin: str) -> List[str]:
    """All present hourly files listed in a Stage 0 manifest, as local paths."""
    paths: List[str] = []
    with open(manifest_csv) as f:
        for row in csv.DictReader(f):
            if row["present"] == "True":
                paths.append(str(Path(raw_dir) / row["date"] / row["hour"] / f"{coin}.lz4"))
    return sorted(paths)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    p = argparse.ArgumentParser(description="Stage 3 - resample raw L2 to a per-minute book")
    p.add_argument("--raw-dir", required=True, help="root of raw .lz4 files (scratch)")
    p.add_argument("--coin", default="BTC")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--day", help="process a single day YYYY-MM-DD (validation)")
    src.add_argument("--manifest", help="process every present file in this manifest CSV")
    p.add_argument("--out", required=True, help="output parquet path (keep in scratch)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--batch-files", type=int, default=48,
                   help="files parsed in parallel per batch (bounds memory)")
    p.add_argument("--expected-minutes", type=int, default=None)
    args = p.parse_args()

    if args.day:
        files = _files_for_day(args.raw_dir, args.coin, args.day)
        expected = args.expected_minutes or 1440
    else:
        files = _files_from_manifest(args.raw_dir, args.manifest, args.coin)
        expected = args.expected_minutes

    logger.info("resampling %d files (workers=%d)", len(files), args.workers)
    minute_df = process_files(files, workers=args.workers, batch_files=args.batch_files)
    report = qa_report(minute_df, expected_minutes=expected)
    write_parquet(minute_df, args.out)
    Path(args.out).with_suffix(".qa.json").write_text(json.dumps(report, indent=2))
    logger.info("wrote %s (%d minutes)", args.out, len(minute_df))
    for key, value in report.items():
        logger.info("  %s: %s", key, value)


if __name__ == "__main__":
    main()
