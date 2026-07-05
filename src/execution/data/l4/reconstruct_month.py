"""Stage L4-2/2d - reconstruct the BTC book for the whole month and sample to parquet.

Walks the extracted hourly archives in chronological order, driving ONE persistent
:class:`~execution.data.l4.book_engine.BookEngine` across all hours/days (so the empty-
book warm-up is paid once, at the very start, not per hour). Emits a top-k snapshot every
``--cadence-ms`` of wall-clock time via :class:`~execution.data.l4.snapshot_sampler.
SnapshotSampler`, and writes one parquet file per day.

Robust to interruption: after each day it checkpoints the engine + sampler state and the
next day index, and skips days whose parquet already exists. A killed run resumes with
``--resume`` instead of replaying ~5-6h from scratch. Memory-flat: one hour of events is
streamed at a time; only the current day's rows and the live book are held.

CLI:
    PYTHONPATH=src .venv/bin/python -m execution.data.l4.reconstruct_month \
        --diffs-dir  <oxford_l4/diffs_extract>  --orders-dir <oxford_l4/orders_extract> \
        --out <oxford_l4/book_05s>  --cadence-ms 500 [--max-days 1] [--resume]
"""
from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from execution.data.l4.book_diffs_reader import (
    BID,
    REMOVE,
    BookEvent,
    attach_timestamps,
    inject_removes,
    stream_events,
)
from execution.data.l4.book_engine import BookEngine
from execution.data.l4.orders_reader import build_open_and_removes
from execution.data.l4.snapshot_sampler import Snapshot, SnapshotSampler

logger = logging.getLogger(__name__)

_CKPT = "_checkpoint.pkl"


def _hour_stream(
    diffs_dir: Path, orders_dir: Path, date: str, hour: int, coin: str
) -> Optional[Tuple[Iterator[Tuple[Optional[int], BookEvent]], List[Tuple[int, int]]]]:
    """``(timestamped stream, removes)`` for one coin-hour, or ``None`` if a file is missing.

    The hour's time-sorted terminal ``(ts_ns, oid)`` list is returned alongside the
    stream so the caller can retry removes that raced their own order's ``new``
    (see :func:`retry_late_removes`).
    """
    of = orders_dir / date / f"{coin.lower()}_{hour:02d}.data.gz"
    df = diffs_dir / date / f"ex{hour}.gz"
    if not of.exists() or not df.exists():
        logger.warning("skip %s h%02d: orders=%s diffs=%s", date, hour, of.exists(), df.exists())
        return None
    open_map, removes = build_open_and_removes(of)
    stream = inject_removes(attach_timestamps(stream_events(df, coin), open_map), removes)
    return stream, removes


def retry_late_removes(engine: BookEngine, removes: List[Tuple[int, int]]) -> int:
    """Re-apply this hour's terminal removes whose order is STILL resting; return count.

    An order can open and be cancelled within the same sub-second block with the
    cancel's status timestamp preceding the order's ``new`` line in the diff-file
    ordering. The time-merged injection then fires the remove before the ``new`` has
    applied, the remove hits an unknown oid (orphan), and the order would rest forever
    (audited 2026-07-04: 4,345 such phantoms accumulated over the month, in bursts).
    Retrying the hour's removes after the hour is fully applied clears them
    deterministically: every ``new`` of the hour has been applied by then, so a remove
    that still matches a resting order is one that raced it.
    """
    n = 0
    for _ts, oid in removes:
        if oid in engine._orders:
            engine.apply(BookEvent(oid=int(oid), side=BID, px=0.0, kind=REMOVE, sz=0.0))
            n += 1
    return n


def _row(snap: Snapshot, k: int) -> dict:
    """Flatten a snapshot to one parquet row: top of book + k levels per side."""
    row: dict = {
        "ts": snap.ts, "best_bid": snap.best_bid, "best_ask": snap.best_ask,
        "mid": snap.mid, "spread": snap.spread,
    }
    for i in range(k):
        bp, bs = snap.bids[i] if i < len(snap.bids) else (np.nan, np.nan)
        ap, as_ = snap.asks[i] if i < len(snap.asks) else (np.nan, np.nan)
        row[f"bid_px_{i+1}"], row[f"bid_sz_{i+1}"] = bp, bs
        row[f"ask_px_{i+1}"], row[f"ask_sz_{i+1}"] = ap, as_
    return row


def run(
    diffs_dir: Path, orders_dir: Path, out: Path, coin: str, cadence_ns: int,
    top_k: int, warmup_events: int, max_days: Optional[int], resume: bool,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    days = sorted(d.name for d in diffs_dir.iterdir() if d.is_dir())
    if max_days is not None:
        days = days[:max_days]

    engine = BookEngine()
    sampler = SnapshotSampler(engine, cadence_ns, top_k, warmup_events)
    start_idx = 0
    ckpt = out / _CKPT
    if resume and ckpt.exists():
        st = pickle.loads(ckpt.read_bytes())
        engine = st["engine"]
        sampler = SnapshotSampler(engine, cadence_ns, top_k, warmup_events)
        sampler.set_state(st["sampler"])
        start_idx = st["next_day_idx"]
        logger.info("resumed: next day index %d, %d live orders", start_idx, engine.n_orders)

    logger.info("days=%d (%s..%s) cadence=%dms top_k=%d warmup=%d",
                len(days), days[0], days[-1], cadence_ns // 1_000_000, top_k, warmup_events)

    for di in range(start_idx, len(days)):
        date = days[di]
        day_out = out / f"{date}.parquet"
        if day_out.exists() and not resume:
            logger.info("day %s: parquet exists, skipping", date)
            continue
        t0 = time.perf_counter()
        rows: list[dict] = []
        n_late = 0
        for hour in range(24):
            hs = _hour_stream(diffs_dir, orders_dir, date, hour, coin)
            if hs is None:
                continue
            stream, removes = hs
            for snap in sampler.feed(stream):
                rows.append(_row(snap, top_k))
            n_late += retry_late_removes(engine, removes)
        if rows:
            pd.DataFrame(rows).to_parquet(day_out, index=False)
        ckpt.write_bytes(pickle.dumps(
            {"engine": engine, "sampler": sampler.get_state(), "next_day_idx": di + 1}))
        logger.info(
            "day %s done: %d snapshots in %.0fs | live_orders=%d levels=%s orphan=%.3f%% "
            "late_removes=%d evicted=%d",
            date, len(rows), time.perf_counter() - t0, engine.n_orders,
            engine.n_levels, 100 * engine.orphan_rate, n_late, engine.n_evicted)

    logger.info("reconstruction complete -> %s", out)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Full-month BTC book reconstruction (Stage L4-2d)")
    ap.add_argument("--diffs-dir", required=True)
    ap.add_argument("--orders-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--coin", default="BTC")
    ap.add_argument("--cadence-ms", type=int, default=500)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--warmup-events", type=int, default=1_500_000, help="suppress ~first minutes")
    ap.add_argument("--max-days", type=int, default=None, help="cap days (for a quick test run)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run(
        Path(args.diffs_dir), Path(args.orders_dir), Path(args.out), args.coin,
        args.cadence_ms * 1_000_000, args.top_k, args.warmup_events, args.max_days, args.resume,
    )


if __name__ == "__main__":
    main()
