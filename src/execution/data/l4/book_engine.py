"""Stage L4-2 - reconstruct the visible limit order book from book-diff events.

Applies a stream of :class:`~execution.data.l4.book_diffs_reader.BookEvent`
(``new`` / ``remove`` / ``update``) in sequence to maintain the full resting-order
book, keyed by order id, with per-price-level size aggregates. At any point it can
emit the top of book or a top-k snapshot.

Cost: state is O(resting orders); each event is applied in O(1); a snapshot costs
O(levels log levels) and is taken only at the sampling cadence, not per event.

Warm-up (important for correctness). The archive is a *diff* stream, so orders resting
*before* our window opened have no ``new`` event we ever saw. Their later ``remove`` /
``update`` reference an ``oid`` unknown to us; these are counted as *orphans* and
ignored (we cannot know their size or price). The reconstructed book therefore matches
the true book only after a warm-up during which the pre-existing orders churn out. The
orphan counters expose this: callers discard snapshots until the orphan rate has
decayed, and Stage L4-3 confirms convergence by cross-checking against the L2 book.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from execution.data.l4.book_diffs_reader import (
    ASK,
    BID,
    NEW,
    REMOVE,
    UPDATE,
    BookEvent,
)

logger = logging.getLogger(__name__)

_EPS = 1e-9  # a level at or below this residual size is treated as empty and dropped

# (price, size) pairs, best level first
Level = Tuple[float, float]


class BookEngine:
    """In-memory reconstruction of one coin's visible limit order book.

    Bids and asks are held as ``price -> aggregate size`` dicts; a separate
    ``oid -> (side, price, size)`` map lets ``remove`` / ``update`` events adjust the
    exact size a given order contributed (so a level never drifts).
    """

    def __init__(self) -> None:
        self._orders: Dict[int, Tuple[str, float, float]] = {}  # oid -> (side, px, sz)
        self._bids: Dict[float, float] = {}  # px -> total resting size
        self._asks: Dict[float, float] = {}
        # Per-level staleness evidence for the crossing guard: last event sequence
        # number that touched each (side, px) level, plus the oids resting there so an
        # evicted level's orders leave the oid map too (no ghost resurrection later).
        self._seq = 0
        self._level_seq: Dict[Tuple[str, float], int] = {}
        self._level_oids: Dict[Tuple[str, float], set] = {}
        # Diagnostics (drive the warm-up / validation gate).
        self.n_applied = 0
        self.n_orphan_remove = 0
        self.n_orphan_update = 0
        self.n_evicted = 0

    # ---------------------------------------------------------------- apply
    def _levels(self, side: str) -> Dict[float, float]:
        return self._bids if side == BID else self._asks

    def _add(self, side: str, px: float, sz: float) -> None:
        lv = self._levels(side)
        new_sz = lv.get(px, 0.0) + sz
        if new_sz <= _EPS:
            lv.pop(px, None)
        else:
            lv[px] = new_sz

    def _touch(self, side: str, px: float, oid: int, present: bool) -> None:
        """Record activity evidence at a level and keep its oid membership in sync."""
        key = (side, px)
        if px in self._levels(side):
            self._level_seq[key] = self._seq
            members = self._level_oids.get(key)
            if members is None:
                members = self._level_oids[key] = set()
            if present:
                members.add(oid)
            else:
                members.discard(oid)
        else:  # the level emptied -> drop its bookkeeping with it
            self._level_seq.pop(key, None)
            self._level_oids.pop(key, None)

    def apply(self, ev: BookEvent) -> None:
        """Apply one event to the book, updating the per-oid map and level aggregates."""
        self.n_applied += 1
        self._seq += 1

        if ev.kind == NEW:
            prev = self._orders.get(ev.oid)
            if prev is not None:  # defensive: a repeated oid replaces its old contribution
                p_side, p_px, p_sz = prev
                self._add(p_side, p_px, -p_sz)
                self._touch(p_side, p_px, ev.oid, present=False)
            self._orders[ev.oid] = (ev.side, ev.px, ev.sz)
            self._add(ev.side, ev.px, ev.sz)
            self._touch(ev.side, ev.px, ev.oid, present=True)

        elif ev.kind == REMOVE:
            prev = self._orders.pop(ev.oid, None)
            if prev is None:
                self.n_orphan_remove += 1
                return
            side, px, sz = prev
            self._add(side, px, -sz)
            self._touch(side, px, ev.oid, present=False)

        elif ev.kind == UPDATE:
            prev = self._orders.get(ev.oid)
            if prev is None:
                self.n_orphan_update += 1
                return
            side, px, old_sz = prev
            self._add(side, px, ev.sz - old_sz)
            if ev.sz <= _EPS:  # an update to zero size means the order is gone
                self._orders.pop(ev.oid, None)
                self._touch(side, px, ev.oid, present=False)
            else:
                self._orders[ev.oid] = (side, px, ev.sz)
                self._touch(side, px, ev.oid, present=True)

    # -------------------------------------------------------------- healing
    def _evict_level(self, side: str, px: float) -> None:
        """Remove one price level entirely: its aggregate, its orders, its bookkeeping."""
        self._levels(side).pop(px, None)
        for oid in self._level_oids.pop((side, px), ()):
            self._orders.pop(oid, None)
        self._level_seq.pop((side, px), None)

    def drop_crossed(self) -> int:
        """Evict stale crossed levels, oldest evidence first; return levels dropped.

        A crossed book (best bid >= best ask) cannot be real -- the venue's matching
        engine would have executed the pair -- so one side of each crossing pair is a
        phantom left by a removal missing from the source streams. WHICH side is decided
        by evidence, not convention: the venue accepted the more recent order as RESTING
        (its ``new`` diff proves no opposing order at a crossing price still existed),
        so the level whose last activity is OLDER is the phantom and is evicted. Ties
        evict the bid (deterministic; ties are effectively impossible on real data).

        The pre-fix rule ("always heal bids first") assumed the bid side is the phantom.
        With a stale ASK stranded below a rising market that rule evicted every genuine
        bid above the phantom and froze the reconstructed top-of-book for hours at a
        time (diagnosed 2026-07-04: 117 frozen hours clustered on quiet days, e.g. the
        90187/90188 pin from Dec-12 22:08). Evicting by evidence heals the phantom side
        in both directions.

        One level is evicted per iteration (surgical), so real interior depth on the
        healthy side is never touched. Applied at snapshot time / periodically, so
        transient in-block crossings self-resolve without the guard firing.
        """
        n = 0
        while self._bids and self._asks:
            bb, ba = max(self._bids), min(self._asks)
            if bb < ba:
                break
            bid_seq = self._level_seq.get((BID, bb), 0)
            ask_seq = self._level_seq.get((ASK, ba), 0)
            if bid_seq <= ask_seq:
                self._evict_level(BID, bb)
            else:
                self._evict_level(ASK, ba)
            n += 1
        self.n_evicted += n
        return n

    # ------------------------------------------------------------- queries
    def best_bid(self) -> Optional[float]:
        return max(self._bids) if self._bids else None

    def best_ask(self) -> Optional[float]:
        return min(self._asks) if self._asks else None

    def top_of_book(
        self,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Return ``(best_bid, best_ask, mid, spread)``; components are ``None`` if a side is empty."""
        bb, ba = self.best_bid(), self.best_ask()
        if bb is None or ba is None:
            return bb, ba, None, None
        return bb, ba, (bb + ba) / 2.0, ba - bb

    def snapshot(self, k: int) -> Tuple[List[Level], List[Level]]:
        """Return the top ``k`` ``(price, size)`` levels per side, best first.

        Bids are sorted by descending price, asks by ascending price. Only computed
        at the sampling cadence, so the per-call sort is not on the hot path.
        """
        bids = sorted(self._bids.items(), key=lambda kv: -kv[0])[:k]
        asks = sorted(self._asks.items(), key=lambda kv: kv[0])[:k]
        return bids, asks

    # ---------------------------------------------------------- diagnostics
    @property
    def n_orders(self) -> int:
        """Number of live resting orders currently tracked."""
        return len(self._orders)

    @property
    def n_levels(self) -> Tuple[int, int]:
        """``(bid_levels, ask_levels)`` currently in the book."""
        return len(self._bids), len(self._asks)

    @property
    def orphan_rate(self) -> float:
        """Fraction of applied events that referenced an unknown oid (warm-up signal)."""
        if self.n_applied == 0:
            return 0.0
        return (self.n_orphan_remove + self.n_orphan_update) / self.n_applied
