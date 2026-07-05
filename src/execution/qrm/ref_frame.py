"""Reference-price frame for QRM calibration (the fix for the frozen-price failure).

The QRM engine's state is queue volumes at FIXED prices around a sticky reference:
ask depth i lives at ``p_ref + (i - 0.5) * tick``, bid depth i at ``p_ref - (i - 0.5) *
tick`` (see ``qrm_core.engine`` / ``sampling.update_LOB``; p_ref sits on a half-tick).
The best bid/ask are simply the first NON-EMPTY queues, so **an empty depth-1 queue is a
legal, observable state** -- it is exactly the state whose dynamics move the price.

Measuring event depth relative to the *current best* (our first attempt) makes that state
unobservable by construction: the instant a level empties, "the best" relabels to the next
non-empty level, so the measurement never sees an empty first queue and the calibrated
drift points up at every queue size -> the simulated price provably freezes (diagnosed
2026-07-04: sim best-queue min 18, zero reference moves; real touch empties every ~4s).

This module provides the frame-consistent alternative:

* :class:`RefPriceTracker` -- the sticky reference, following Huang et al. (2015): the
  reference is the mid when the mid sits on a half-tick (odd spread); when the mid is on
  a tick (even spread, ambiguous) the previous reference is kept if still adjacent, else
  it steps one tick toward the mid. Built-in hysteresis, one-tick steps, mirroring the
  engine's own one-tick reference moves.
* :func:`depth_in_frame` -- an event price's depth (1..K) in the current window.
* :func:`window_volumes` -- resting volume at the K fixed window prices per side.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from execution.data.l4.book_diffs_reader import ASK, BID, NEW
from execution.data.l4.book_engine import BookEngine

logger = logging.getLogger(__name__)


class CachedBestEngine(BookEngine):
    """A :class:`BookEngine` with O(1) best-bid/ask for calibration-scale event loops.

    The reference tracker consults the best prices after *every* event; the base class
    recomputes them by scanning all price levels (~1.5k), which is infeasible across
    ~30M events. This subclass keeps a cached best per side, updated in O(1) for the
    common cases and lazily rescanned only when the cached best level itself empties
    (rare: the real touch empties ~every 4s). Behaviour is otherwise identical; a fuzz
    test asserts exact equivalence with the base engine.
    """

    _INVALID = -1.0  # sentinel: cache must be recomputed on next query

    def __init__(self) -> None:
        super().__init__()
        self._cbb: Optional[float] = None  # cached best bid (None = empty side)
        self._cba: Optional[float] = None
        self._bb_valid = True
        self._ba_valid = True

    def apply(self, ev) -> None:  # noqa: ANN001 -- BookEvent
        # a duplicate-oid NEW moves the order: its OLD price level may empty too
        prev = self._orders.get(ev.oid) if ev.kind == NEW else None
        super().apply(ev)
        if prev is not None:
            p_side, p_px, _ = prev
            if p_px != ev.px and p_px not in self._levels(p_side):
                if p_side == ASK and p_px == self._cba:
                    self._ba_valid = False
                elif p_side == BID and p_px == self._cbb:
                    self._bb_valid = False
        px = ev.px
        if ev.side == ASK:
            if self._ba_valid:
                if px not in self._asks and px == self._cba:
                    self._ba_valid = False        # cached best emptied -> lazy rescan
                elif px in self._asks and (self._cba is None or px < self._cba):
                    self._cba = px                # a new, better ask
        else:
            if self._bb_valid:
                if px not in self._bids and px == self._cbb:
                    self._bb_valid = False
                elif px in self._bids and (self._cbb is None or px > self._cbb):
                    self._cbb = px

    def drop_crossed(self) -> int:
        n = super().drop_crossed()
        if n:
            self._bb_valid = self._ba_valid = False
        return n

    def best_bid(self) -> Optional[float]:
        if not self._bb_valid:
            self._cbb = max(self._bids) if self._bids else None
            self._bb_valid = True
        return self._cbb

    def best_ask(self) -> Optional[float]:
        if not self._ba_valid:
            self._cba = min(self._asks) if self._asks else None
            self._ba_valid = True
        return self._cba


class RefPriceTracker:
    """Sticky half-tick reference price driven by the real book's mid.

    ``update(engine)`` is called after each applied event; it recomputes the mid from the
    book's best bid/ask and moves the reference **one tick at a time** toward the mid
    when the mid is unambiguous (on a half-tick) and differs from the current reference.
    When the mid is ambiguous (on a tick), the reference holds if it is one of the two
    adjacent half-ticks (hysteresis), else it steps toward the mid.
    """

    def __init__(self, tick: float) -> None:
        self.tick = float(tick)
        self.p_ref: Optional[float] = None
        self.n_moves = 0

    def _snap_half_tick(self, x: float) -> float:
        """Nearest half-tick value (…, p-0.5t, p+0.5t, …) to ``x``."""
        t = self.tick
        return (round(x / t - 0.5) + 0.5) * t

    def update(self, engine: BookEngine) -> None:
        bb, ba = engine.best_bid(), engine.best_ask()
        if bb is None or ba is None:
            return
        mid = 0.5 * (bb + ba)
        t = self.tick
        if self.p_ref is None:
            self.p_ref = self._snap_half_tick(mid)
            return
        # mid on a half-tick (odd spread in ticks): unambiguous target
        frac = (mid / t) % 1.0
        on_half = abs(frac - 0.5) < 1e-9
        if on_half:
            target = mid
        else:
            # ambiguous (mid on a tick): hold if adjacent, else step toward the mid
            if abs(mid - self.p_ref) <= 0.5 * t + 1e-9:
                return
            target = self._snap_half_tick(mid)
        # step one tick at a time (mirrors the engine's one-tick reference moves)
        while abs(target - self.p_ref) > 1e-9:
            self.p_ref += t if target > self.p_ref else -t
            self.n_moves += 1


def depth_in_frame(p_ref: float, tick: float, side: str, px: float, K: int) -> Optional[int]:
    """Depth 1..K of price ``px`` in the reference window, or ``None`` if outside.

    Ask depth i sits at ``p_ref + (i - 0.5) * tick``; bid depth i at ``p_ref - (i - 0.5) *
    tick`` -- the engine's own state layout.
    """
    if side == ASK:
        d = round((px - p_ref) / tick + 0.5)
    else:
        d = round((p_ref - px) / tick + 0.5)
    return int(d) if 1 <= d <= K else None


def window_volumes(engine: BookEngine, p_ref: float, tick: float, side: str, K: int) -> List[float]:
    """Resting volume at the K fixed window prices on ``side`` (0.0 where empty).

    Unlike a best-relative top-k, a window slot may legitimately be EMPTY -- that state
    (queue q=0) is precisely what the best-relative frame could never observe.
    """
    levels = engine._asks if side == ASK else engine._bids
    sign = 1.0 if side == ASK else -1.0
    return [levels.get(p_ref + sign * (i - 0.5) * tick, 0.0) for i in range(1, K + 1)]
