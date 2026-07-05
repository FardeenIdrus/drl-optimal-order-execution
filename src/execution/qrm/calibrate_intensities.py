"""Stage 3c/3d - measure QRM intensities and average event size from the real book.

Drives a :class:`~execution.data.l4.book_engine.BookEngine` over the timestamped event
stream and accumulates, for each (side, depth-from-best, queue-size bucket, event-type):

    * ``counts[depth, qbucket, side, type]``  - how many such events occurred, and
    * ``time_in_state[depth, qbucket, side]`` - total time that level spent at that size,

so the intensity is ``lambda = counts / time_in_state`` (events per second per state) -- the
inputs the vendored ``qrm_core`` engine samples from. Queue sizes are expressed in units of
the **average event size** (AES) per depth (measured first), and bucketed 0..Q.

Depth is tick-distance from the current best on that side (depth 1 = at the best). BTC trades
on a $1 tick. Events better than the best (inside the spread / a new best) are booked at depth
1. Two passes over the data: :func:`measure_aes` sets the AES scale, then :func:`calibrate`
accumulates the intensities with that scale fixed.
"""
from __future__ import annotations

import logging
from typing import Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np

from execution.data.l4.book_diffs_reader import ASK, BID, NEW, TRADE, BookEvent
from execution.data.l4.book_engine import BookEngine
from execution.qrm.event_labeler import CANCEL, LIMIT, MARKET, classify_for_calibration

logger = logging.getLogger(__name__)

_SIDE_IDX = {BID: 0, ASK: 1}       # matches IntensityTable side order (bid, ask)
_TYPE_IDX = {LIMIT: 0, CANCEL: 1, MARKET: 2}


def _depth_of(engine: BookEngine, side: str, px: float, tick: float, K: int) -> Optional[int]:
    """Tick-distance depth (1..K) of ``px`` from the current best on ``side``; None if > K."""
    if side == ASK:
        best = engine.best_ask()
        if best is None:
            return None
        d = 1 if px < best else int(round((px - best) / tick)) + 1
    else:
        best = engine.best_bid()
        if best is None:
            return None
        d = 1 if px > best else int(round((best - px) / tick)) + 1
    return d if 1 <= d <= K else None


def _topk_volumes(engine: BookEngine, side: str, tick: float, K: int) -> List[float]:
    """Resting volume at depths 1..K from the best on ``side`` (0.0 where empty)."""
    if side == ASK:
        best, levels = engine.best_ask(), engine._asks
        if best is None:
            return [0.0] * K
        return [levels.get(best + i * tick, 0.0) for i in range(K)]
    best, levels = engine.best_bid(), engine._bids
    if best is None:
        return [0.0] * K
    return [levels.get(best - i * tick, 0.0) for i in range(K)]


def measure_aes(
    stream: Iterator[Tuple[Optional[int], BookEvent]], K: int, tick: float
) -> np.ndarray:
    """Average size of arriving limit orders at each depth 1..K (the AES scale, shape (K,))."""
    sums = np.zeros(K)
    cnts = np.zeros(K)
    e = BookEngine()
    for _ts, ev in stream:
        if ev.kind == NEW:
            d = _depth_of(e, ev.side, ev.px, tick, K)
            if d is not None:
                sums[d - 1] += ev.sz
                cnts[d - 1] += 1
        e.apply(ev)
    aes = sums / np.maximum(cnts, 1.0)
    logger.info("AES per depth: %s", np.round(aes, 4).tolist())
    return aes


def calibrate(
    stream: Iterator[Tuple[Optional[int], BookEvent]],
    fully_filled: Set[int],
    aes: Sequence[float],
    K: int,
    Q: int,
    tick: float,
    engine: Optional[BookEngine] = None,
    drop_crossed_every: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate ``(counts, time_in_state)`` over one event stream.

    Returns ``counts`` (K, Q+1, 2, 3) and ``time_in_state`` (K, Q+1, 2). Pass a shared
    ``engine`` to carry the book across chunks (e.g. hour to hour). The stale-crossed guard
    is applied every ``drop_crossed_every`` events so depth/queue readings stay clean.
    """
    aes = np.asarray(aes, float)
    counts = np.zeros((K, Q + 1, 2, 3))
    time_in = np.zeros((K, Q + 1, 2))
    e = engine if engine is not None else BookEngine()
    last_ts: Optional[int] = None
    n = 0
    for ts, ev in stream:
        n += 1
        if n % drop_crossed_every == 0:
            e.drop_crossed()
        # time-in-state: attribute the gap since the last event to the current top-K levels
        if ts is not None and last_ts is not None and ts > last_ts:
            dt = (ts - last_ts) / 1e9
            for side, si in _SIDE_IDX.items():
                for d, vol in enumerate(_topk_volumes(e, side, tick, K)):
                    qb = min(int(round(vol / aes[d])), Q)
                    time_in[d, qb, si] += dt
        if ev.kind == TRADE:
            # authoritative market order (from the trades file): count, do NOT apply --
            # the diff stream's own remove/update already consumes this depth.
            d = _depth_of(e, ev.side, ev.px, tick, K)
            if d is not None:
                qb = min(int(round(e._levels(ev.side).get(ev.px, 0.0) / aes[d - 1])), Q)
                counts[d - 1, qb, _SIDE_IDX[ev.side], _TYPE_IDX[MARKET]] += 1
        else:
            # book-diff event: count limit/cancel (fills skipped -> from trades), then apply
            etype = classify_for_calibration(ev, fully_filled)
            if etype is not None:
                d = _depth_of(e, ev.side, ev.px, tick, K)
                if d is not None:
                    qb = min(int(round(e._levels(ev.side).get(ev.px, 0.0) / aes[d - 1])), Q)
                    counts[d - 1, qb, _SIDE_IDX[ev.side], _TYPE_IDX[etype]] += 1
            e.apply(ev)
        if ts is not None:
            last_ts = ts
    return counts, time_in


def intensities_from(counts: np.ndarray, time_in: np.ndarray) -> np.ndarray:
    """``lambda = counts / time_in_state`` (K, Q+1, 2, 3); 0 where a state was never observed."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = counts / time_in[..., None]
    return np.nan_to_num(lam, nan=0.0, posinf=0.0, neginf=0.0)


# --------------------------------------------------- reference-frame calibration
def measure_aes_ref_frame(
    stream: Iterator[Tuple[Optional[int], BookEvent]], K: int, tick: float
) -> Tuple[np.ndarray, np.ndarray]:
    """AES per window depth + the depth-1 queue-size time profile, in the REFERENCE frame.

    First pass of the frame-consistent calibration (see :mod:`execution.qrm.ref_frame`
    for why the best-relative frame provably freezes the simulated price). Returns
    ``(aes, q1_seconds)`` where ``aes[k]`` is the mean arriving limit-order size at window
    depth k+1 and ``q1_seconds[v]`` is the time (s) the depth-1 ask queue spent holding
    ~``v`` AES units of volume (rounded; ``v`` capped at 500) -- used to choose Q from data.
    """
    from execution.qrm.ref_frame import CachedBestEngine, RefPriceTracker, depth_in_frame, window_volumes

    sums = np.zeros(K)
    cnts = np.zeros(K)
    q1_seconds = np.zeros(501)
    e = CachedBestEngine()
    tracker = RefPriceTracker(tick)
    last_ts: Optional[int] = None
    aes1_running = 1.0  # provisional AES for the q1 profile (updated as data accrues)
    n = 0
    for ts, ev in stream:
        n += 1
        if n % 512 == 0:
            e.drop_crossed()
        if tracker.p_ref is not None and ts is not None and last_ts is not None and ts > last_ts:
            v1 = window_volumes(e, tracker.p_ref, tick, ASK, 1)[0]
            q1_seconds[min(int(round(v1 / aes1_running)), 500)] += (ts - last_ts) / 1e9
        if ev.kind == NEW and tracker.p_ref is not None:
            d = depth_in_frame(tracker.p_ref, tick, ev.side, ev.px, K)
            if d is not None:
                sums[d - 1] += ev.sz
                cnts[d - 1] += 1
                if d == 1 and cnts[0] > 0:
                    aes1_running = sums[0] / cnts[0]
        if ev.kind != TRADE:
            e.apply(ev)
            tracker.update(e)
        if ts is not None:
            last_ts = ts
    aes = sums / np.maximum(cnts, 1.0)
    logger.info("ref-frame AES per depth: %s (ref moved %d times)",
                np.round(aes, 4).tolist(), tracker.n_moves)
    return aes, q1_seconds


def calibrate_ref_frame(
    stream: Iterator[Tuple[Optional[int], BookEvent]],
    fully_filled: Set[int],
    aes: Sequence[float],
    K: int,
    Q: int,
    tick: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Accumulate ``(counts, time_in_state)`` in the REFERENCE-PRICE frame.

    Identical outputs to :func:`calibrate` -- ``counts`` (K, Q+1, 2, 3) and
    ``time_in_state`` (K, Q+1, 2) -- but with depth and queue state defined at FIXED
    prices around the sticky reference (the engine's own state definition), so an empty
    depth-1 queue is an observed state and the emptying dynamics that move the price are
    measurable. Trades are counted as market orders and not applied (the diff stream
    already carries their depth consumption).
    """
    from execution.qrm.ref_frame import CachedBestEngine, RefPriceTracker, depth_in_frame, window_volumes

    aes = np.asarray(aes, float)
    counts = np.zeros((K, Q + 1, 2, 3))
    time_in = np.zeros((K, Q + 1, 2))
    e = CachedBestEngine()
    tracker = RefPriceTracker(tick)
    last_ts: Optional[int] = None
    n = 0
    for ts, ev in stream:
        n += 1
        if n % 512 == 0:
            e.drop_crossed()
        p_ref = tracker.p_ref
        # time-in-state at the FIXED window prices (empty slots are real q=0 states)
        if p_ref is not None and ts is not None and last_ts is not None and ts > last_ts:
            dt = (ts - last_ts) / 1e9
            for side, si in _SIDE_IDX.items():
                for d, vol in enumerate(window_volumes(e, p_ref, tick, side, K)):
                    time_in[d, min(int(round(vol / aes[d])), Q), si] += dt
        # count this event at its frame depth, with the queue state BEFORE it applies
        if p_ref is not None:
            if ev.kind == TRADE:
                d = depth_in_frame(p_ref, tick, ev.side, ev.px, K)
                if d is not None:
                    qb = min(int(round(e._levels(ev.side).get(ev.px, 0.0) / aes[d - 1])), Q)
                    counts[d - 1, qb, _SIDE_IDX[ev.side], _TYPE_IDX[MARKET]] += 1
            else:
                etype = classify_for_calibration(ev, fully_filled)
                if etype is not None:
                    d = depth_in_frame(p_ref, tick, ev.side, ev.px, K)
                    if d is not None:
                        qb = min(int(round(e._levels(ev.side).get(ev.px, 0.0) / aes[d - 1])), Q)
                        counts[d - 1, qb, _SIDE_IDX[ev.side], _TYPE_IDX[etype]] += 1
        if ev.kind != TRADE:
            e.apply(ev)
            tracker.update(e)
        if ts is not None:
            last_ts = ts
    return counts, time_in
