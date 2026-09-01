"""Stage L4-3 (2e) - validate the reconstructed book against the trusted L2 book.

Cross-checks the L4 reconstruction (event-exact) against the independent L2 snapshot
archive (~0.5s cadence) for the same coin-hour. At each L2 snapshot time, the L4 book
reconstructed up to that instant must carry the same best bid and best ask. Results are
reported per time-segment, so the warm-up (the diff replay starts from an empty book and
pre-existing resting orders churn out) shows up as a match rate that climbs to ~100% and
then holds -- the signature of a correct reconstruction.

The two sources are fully independent (L2 archive vs the per-order book-diff archive), so
agreement is strong evidence the reconstruction is right, not merely self-consistent.

CLI:
    PYTHONPATH=src .venv/bin/python -m execution.data.l4.validate_vs_l2 \
        --l4-diffs  <ex0.gz> --l4-orders <btc_00.data.gz> --l2 <BTC.lz4> [--segment-min 5]
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

from execution.data.l4.book_diffs_reader import attach_timestamps, stream_events
from execution.data.l4.book_engine import BookEngine
from execution.data.l4.orders_reader import build_open_ts_map
from execution.data.parse import parse_file

logger = logging.getLogger(__name__)

_NS_PER_MS = 1_000_000


def crosscheck_hour(
    l4_diffs: str, l4_orders: str, l2_lz4: str, coin: str = "BTC", segment_ns: int = 300 * 10**9
) -> dict:
    """Compare L4 top-of-book to L2 at every L2 snapshot time; tally per segment.

    Returns ``{segment_index: {n, bid_match, ask_match, both_match, sum_abs_dbid,
    sum_abs_dask}}``. A segment spans ``segment_ns`` from the first L2 snapshot.
    """
    l2df, _ = parse_file(l2_lz4)
    l2_ts = (l2df["ts"].to_numpy() * _NS_PER_MS).astype(np.int64)  # ms -> ns
    l2_bid = l2df["bid_px_1"].to_numpy()
    l2_ask = l2df["ask_px_1"].to_numpy()
    order = np.argsort(l2_ts, kind="stable")
    l2_ts, l2_bid, l2_ask = l2_ts[order], l2_bid[order], l2_ask[order]
    t0 = int(l2_ts[0])
    n_l2 = len(l2_ts)

    ts_map = build_open_ts_map(l4_orders)
    engine = BookEngine()
    seg: dict[int, dict] = {}
    j = 0
    for ts, ev in attach_timestamps(stream_events(l4_diffs, coin), ts_map):
        if ts is not None:
            while j < n_l2 and l2_ts[j] < ts:
                bb, ba = engine.best_bid(), engine.best_ask()
                if bb is not None and ba is not None:
                    s = int((l2_ts[j] - t0) // segment_ns)
                    d = seg.setdefault(
                        s, dict(n=0, bid=0, ask=0, both=0, dbid=0.0, dask=0.0)
                    )
                    mb, ma = bb == l2_bid[j], ba == l2_ask[j]
                    d["n"] += 1
                    d["bid"] += int(mb)
                    d["ask"] += int(ma)
                    d["both"] += int(mb and ma)
                    d["dbid"] += abs(bb - l2_bid[j])
                    d["dask"] += abs(ba - l2_ask[j])
                j += 1
        engine.apply(ev)
    return seg


def _print_report(seg: dict, segment_min: int) -> None:
    print(
        f"\n{'segment':>12} {'n':>7} {'bid✓%':>7} {'ask✓%':>7} {'both✓%':>7} "
        f"{'|Δbid|':>8} {'|Δask|':>8}"
    )
    for s in sorted(seg):
        d = seg[s]
        n = d["n"]
        lo = s * segment_min
        print(
            f"{lo:>4}-{lo+segment_min:<3}min {n:>7,} "
            f"{100*d['bid']/n:>7.2f} {100*d['ask']/n:>7.2f} {100*d['both']/n:>7.2f} "
            f"{d['dbid']/n:>8.3f} {d['dask']/n:>8.3f}"
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Validate L4 reconstruction vs L2 (Stage L4-3)")
    ap.add_argument("--l4-diffs", required=True, help="extracted book-diff hour (ex<H>.gz)")
    ap.add_argument("--l4-orders", required=True, help="extracted order-status hour (<coin>_<HH>.data.gz)")
    ap.add_argument("--l2", required=True, help="L2 hour file (<COIN>.lz4)")
    ap.add_argument("--coin", default="BTC")
    ap.add_argument("--segment-min", type=int, default=5)
    args = ap.parse_args()

    seg = crosscheck_hour(
        args.l4_diffs, args.l4_orders, args.l2, coin=args.coin,
        segment_ns=args.segment_min * 60 * 10**9,
    )
    _print_report(seg, args.segment_min)
    warm = [d for s, d in seg.items() if s >= 2]  # segments after the first ~10 min
    if warm:
        tn = sum(d["n"] for d in warm)
        tb = sum(d["both"] for d in warm)
        print(f"\nwarm region (>=10min): both-match {100*tb/tn:.2f}%  over {tn:,} comparisons")


if __name__ == "__main__":
    main()
