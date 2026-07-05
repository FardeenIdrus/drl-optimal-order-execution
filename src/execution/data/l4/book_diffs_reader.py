"""Stage L4-1 - stream BTC book-diff events from one hourly file.

The Oxford "Open Book" raw book-diff archive stores one gzip'd JSON-lines file per
hour (``<date>/ex<H>.gz``), with BTC / ETH / SOL interleaved. Each line is a single
change to the *visible* limit order book. This reader decompresses one hour, filters
to one coin, and yields normalised events one at a time, so the caller (Stage L4-2)
reconstructs the book hour-by-hour without ever holding a whole file in memory.

Source record shapes (see SCHEMA.md of the dataset)::

    {"oid":..., "coin":"BTC", "side":"A"|"B", "px":"90326.0",
     "raw_book_diff": "remove"}                                   # order left the book
    {..., "raw_book_diff": {"new":    {"sz":"0.54"}}}             # order placed
    {..., "raw_book_diff": {"update": {"origSz":"5.0","newSz":"4.7"}}}  # partial fill

``side`` is ``"A"`` for ask/sell and ``"B"`` for bid/buy. Prices and sizes are decimal
strings (cast to float here to match the L2 pipeline's numeric book).
"""
from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# raw_book_diff event kinds
NEW = "new"
REMOVE = "remove"
UPDATE = "update"
# a trade (market order), merged in from the trades file for QRM calibration; drives
# counting only, never the book (the diff stream already carries the depth consumption).
TRADE = "trade"

# side codes, exactly as in the source
ASK = "A"  # sell
BID = "B"  # buy


@dataclass(frozen=True, slots=True)
class BookEvent:
    """One normalised change to the visible book (a single coin, one source line).

    Attributes:
        oid: unique resting-order id; links an order across its lifecycle events.
        side: ``"A"`` ask/sell or ``"B"`` bid/buy.
        px: price level.
        kind: ``"new"``, ``"remove"`` or ``"update"``.
        sz: for ``new`` the placed size; for ``update`` the *new* remaining size
            (after a partial fill); for ``remove`` it is ``0.0`` (order leaves the book).
    """

    oid: int
    side: str
    px: float
    kind: str
    sz: float


def parse_line(obj: dict, coin: str) -> BookEvent | None:
    """Normalise one decoded JSON record into a :class:`BookEvent`.

    Returns ``None`` if the record is for a different coin or is malformed (unknown
    ``raw_book_diff`` shape). Kept separate from I/O so it is unit-testable.
    """
    if obj.get("coin") != coin:
        return None
    diff = obj.get("raw_book_diff")
    if diff == REMOVE:
        kind, sz = REMOVE, 0.0
    elif isinstance(diff, dict) and NEW in diff:
        kind, sz = NEW, float(diff[NEW]["sz"])
    elif isinstance(diff, dict) and UPDATE in diff:
        kind, sz = UPDATE, float(diff[UPDATE]["newSz"])
    else:
        return None
    return BookEvent(
        oid=int(obj["oid"]), side=obj["side"], px=float(obj["px"]), kind=kind, sz=sz
    )


def stream_events(
    path: Union[str, Path], coin: str = "BTC"
) -> Iterator[BookEvent]:
    """Yield :class:`BookEvent` for one gzip'd hourly book-diff file, filtered to ``coin``.

    Streaming and memory-flat: reads the file line-by-line, never materialising it.
    A malformed line (bad JSON or unknown diff shape) is skipped, not fatal. A
    per-file tally (lines / coin events / malformed) is logged at completion.

    Args:
        path: path to an extracted ``ex<H>.gz`` hourly book-diff file.
        coin: instrument to keep (default ``"BTC"``; the file interleaves BTC/ETH/SOL).

    Yields:
        One :class:`BookEvent` per matching, well-formed source line, in file order.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"book-diff file not found: {path}")

    lines = coin_events = malformed = 0
    with gzip.open(path, "rt") as fh:
        for line in fh:
            lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            ev = parse_line(obj, coin)
            if ev is None:
                # Either a different coin (expected, not malformed) or an unknown shape.
                if obj.get("coin") == coin:
                    malformed += 1
                continue
            coin_events += 1
            yield ev

    logger.info(
        "book-diffs %s: %d %s events / %d lines (%d malformed)",
        path.name,
        coin_events,
        coin,
        lines,
        malformed,
    )


def attach_timestamps(
    events: Iterator[BookEvent], open_ts_map: Dict[int, int]
) -> Iterator[Tuple[Optional[int], BookEvent]]:
    """Yield ``(ts_ns, event)``, timing each event from the order-status opens.

    A ``new`` event's time is its order's open timestamp (exact, from ``open_ts_map``).
    ``remove`` / ``update`` events carry no time of their own, so they inherit the most
    recent ``new`` timestamp (carry-forward). Because ``new`` events are very frequent
    (hundreds per second), this is accurate to a few milliseconds -- far finer than the
    snapshot cadence. ``ts`` is ``None`` for events before the first matched ``new`` (a
    brief warm-up the caller discards, alongside the book warm-up). A ``new`` whose oid
    is absent from the map does not advance the clock (keeps the carried value).

    The clock is **forward-only**: a ``new`` whose open timestamp is earlier than the
    current clock (a rare sub-microsecond same-block inversion, ~7e-6 of events on real
    data) does not move it backward. This guarantees the emitted ``ts`` is monotonic
    non-decreasing, which the snapshot sampler (Stage L4-2/2d) relies on to bucket
    events by time without special-casing backward blips.
    """
    current_ts: Optional[int] = None
    for ev in events:
        if ev.kind == NEW:
            ts = open_ts_map.get(ev.oid)
            if ts is not None and (current_ts is None or ts > current_ts):
                current_ts = ts
        yield current_ts, ev


def inject_removes(
    timestamped_events: Iterator[Tuple[Optional[int], BookEvent]],
    removes: List[Tuple[int, int]],
) -> Iterator[Tuple[Optional[int], BookEvent]]:
    """Merge order-status terminal removes into the timestamped book-diff stream by time.

    ``removes`` is a time-sorted ``(ts_ns, oid)`` list of every order that left the book
    (all cancel variants + full fills; from
    :func:`~execution.data.l4.orders_reader.build_open_and_removes`). Book-diffs emit
    removes only for plain cancels and most fills, so each terminal event is injected as a
    synthetic ``remove`` at its time, guaranteeing the order leaves the book. A remove for
    an oid the diffs already removed is a harmless orphan. Only ``oid`` matters for a
    remove (the engine looks up the resting order's side/price), so the synthetic event's
    side/price are placeholders.

    Both inputs are ts-ordered, so this is a standard merge; injected removes are emitted
    just before the first diff event whose timestamp is not earlier.
    """
    fi, nf = 0, len(removes)
    for ts, ev in timestamped_events:
        if ts is not None:
            while fi < nf and removes[fi][0] <= ts:
                rts, oid = removes[fi]
                fi += 1
                yield rts, BookEvent(oid=int(oid), side=BID, px=0.0, kind=REMOVE, sz=0.0)
        yield ts, ev
    while fi < nf:  # flush any removes after the last diff event
        rts, oid = removes[fi]
        fi += 1
        yield rts, BookEvent(oid=int(oid), side=BID, px=0.0, kind=REMOVE, sz=0.0)


def inject_trades(
    timestamped_events: Iterator[Tuple[Optional[int], BookEvent]],
    trades: List[Tuple[int, str, float, float]],
) -> Iterator[Tuple[Optional[int], BookEvent]]:
    """Merge trades (market orders) into the timestamped book-diff stream by time.

    ``trades`` is a time-sorted ``(ts_ns, resting_side, px, sz)`` list (from
    :func:`~execution.data.l4.trades_reader.read_trades`). Each is emitted as a
    :data:`TRADE` event at its price on the *resting* side (a buy-aggressor trade consumes
    the ask; a sell-aggressor consumes the bid). The QRM calibrator counts these as the
    authoritative market-order events (the book-diff stream omits ~1/3 of fills) and does
    NOT apply them to the book -- the diff stream's own remove/update carries the depth
    consumption, so applying here would double-count it. The event's ``sz`` carries the
    traded size (used for the quiet-clock volume anchor), still never applied to the book.
    """
    ti, nt = 0, len(trades)
    for ts, ev in timestamped_events:
        if ts is not None:
            while ti < nt and trades[ti][0] <= ts:
                tts, side, px, sz = trades[ti]
                ti += 1
                yield tts, BookEvent(oid=-1, side=side, px=px, kind=TRADE, sz=float(sz))
        yield ts, ev
    while ti < nt:
        tts, side, px, sz = trades[ti]
        ti += 1
        yield tts, BookEvent(oid=-1, side=side, px=px, kind=TRADE, sz=float(sz))
