"""Quiet-spell (reference-price-stable) conditioning for the QRM calibration.

Why (BUILD_PLAN "Step 3 — refinement iteration 2", 2026-07-04): the exogenous
reference-move extension owns everything *associated with price moves* — the moves
themselves AND the book re-formation at each move (`update_LOB` shift / invariant
redraw). Measuring the event intensities over ALL time therefore double-counts the
post-move re-quoting burst: the sim refills the inner book once via the redraw and
again via the inflated limit-arrival rate, which is the diagnosed cause of the
inner-slot occupancy gap (sim 14.8 % empty vs real 31.8 %).

The fix implemented here assigns each piece of real behaviour to exactly one model
component:

* the exogenous move process owns the moves and a short **guard window** after each
  move (the re-quoting burst, measured — not assumed — by :func:`measure_burst`);
* the endogenous QRM intensities are measured only over **quiet spells**: time when
  the reference price has been stable for longer than the guard.

Pre-registered decision rules (reports/qrm_3f_criteria.md §4–5, frozen before any of
this ran): exposure-normalised burst rate; burst confirmed iff the first 50 ms bin is
>= 1.5x the steady (1–3 s) rate; guard = start of the first 3 consecutive bins
<= 1.1x steady, capped at 500 ms; if no burst is confirmed the guard is NOT applied
and the double-counting diagnosis is reconsidered (the refutation branch). The
market-volume anchor is recomputed on the same quiet-time clock as the rates
(numerator and denominator from the same clock).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional, Sequence, Set, Tuple

import numpy as np

from execution.data.l4.book_diffs_reader import ASK, BID, NEW, TRADE, BookEvent
from execution.qrm.calibrate_intensities import _SIDE_IDX, _TYPE_IDX
from execution.qrm.event_labeler import MARKET, classify_for_calibration
from execution.qrm.ref_frame import (
    CachedBestEngine,
    RefPriceTracker,
    depth_in_frame,
    window_volumes,
)

logger = logging.getLogger(__name__)

# Pre-registered burst-measurement constants (criteria §4; cap revised §8/§9).
BIN_NS = 50_000_000            # 50 ms lag bins
MAX_LAG_NS = 3_000_000_000     # profile out to 3 s; one overflow bin beyond
STEADY_FROM_NS = 1_000_000_000  # steady-state rate pooled over lags 1–3 s
BURST_FACTOR = 1.5             # first bin >= 1.5x steady -> burst confirmed
GUARD_FACTOR = 1.1             # guard ends where 3 consecutive bins <= 1.1x steady
GUARD_CONSECUTIVE = 3
# Revision 1 (2026-07-05): the original 500 ms cap truncated the burst before its
# measured settle (~1.5-2 s; rate at the cap was 76/s vs 31.5/s steady), leaving
# lambda_limit inflated at low queues -> book too refilled -> spread too tight (the
# failed-gate residual). The cap now spans the measurable profile so the settle rule
# above decides the guard on its own; the burst evidence predates the failed gate.
GUARD_CAP_NS = MAX_LAG_NS
INNER_DEPTHS = (1, 2)          # burst measured on limit arrivals at window depths 1–2
Q_PROFILE_MIN_LAG_NS = 500_000_000  # q1 exposure profile lag floor (fixed pre-pass:
#                                     the guard itself is only known after the pass)

# Real-run warm-up (criteria §5); tests pass zeros.
WARMUP_NS = 300_000_000_000    # 5 min of stream time
WARMUP_EVENTS = 100_000


@dataclass
class BurstProfile:
    """Exposure-normalised limit-arrival profile vs time since the last reference move."""

    bin_ns: int
    n_bins: int                      # regular bins; index n_bins is the overflow bin
    arrivals_inner: np.ndarray       # (n_bins+1,) NEW counts at INNER_DEPTHS, both sides
    exposure_s: np.ndarray           # (n_bins+1,) seconds observed at each lag bin
    size_sums: np.ndarray            # (K, n_bins+1) arriving limit sizes per depth/lag
    size_counts: np.ndarray          # (K, n_bins+1)
    q1_seconds: np.ndarray           # (501,) depth-1 ask exposure by AES-rounded size,
                                     # restricted to lag >= Q_PROFILE_MIN_LAG_NS
    n_moves: int = 0
    total_s: float = 0.0

    def rates(self) -> np.ndarray:
        """Arrivals/second at each lag bin (NaN where a bin was never observed)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.arrivals_inner / self.exposure_s

    def steady_rate(self) -> float:
        lo = STEADY_FROM_NS // self.bin_ns
        a = self.arrivals_inner[lo:self.n_bins].sum()
        e = self.exposure_s[lo:self.n_bins].sum()
        return float(a / e) if e > 0 else float("nan")


def decide_guard(profile: BurstProfile) -> Tuple[bool, int]:
    """Apply the pre-registered rule: ``(burst_confirmed, guard_ns)``.

    Refutation branch: ``(False, 0)`` — no guard is applied and the double-counting
    diagnosis must be reconsidered (criteria §4).
    """
    steady = profile.steady_rate()
    rates = profile.rates()
    if not np.isfinite(steady) or steady <= 0 or not np.isfinite(rates[0]):
        logger.warning("burst measurement inconclusive (steady=%s)", steady)
        return False, 0
    confirmed = rates[0] >= BURST_FACTOR * steady
    if not confirmed:
        logger.info("burst REFUTED: first-bin %.1f/s < %.1fx steady %.1f/s",
                    rates[0], BURST_FACTOR, steady)
        return False, 0
    cap_bin = min(GUARD_CAP_NS // profile.bin_ns, profile.n_bins)
    guard_bin = cap_bin
    for b in range(profile.n_bins - GUARD_CONSECUTIVE + 1):
        window = rates[b:b + GUARD_CONSECUTIVE]
        if np.all(np.isfinite(window)) and np.all(window <= GUARD_FACTOR * steady):
            guard_bin = b
            break
    guard_ns = int(min(guard_bin * profile.bin_ns, GUARD_CAP_NS))
    logger.info("burst CONFIRMED: first-bin %.1f/s vs steady %.1f/s -> guard %d ms",
                rates[0], steady, guard_ns // 1_000_000)
    return True, guard_ns


def quiet_aes(profile: BurstProfile, guard_ns: int) -> np.ndarray:
    """Mean arriving limit-order size per depth over quiet lags (bins >= guard)."""
    b0 = guard_ns // profile.bin_ns
    sums = profile.size_sums[:, b0:].sum(axis=1)
    cnts = profile.size_counts[:, b0:].sum(axis=1)
    return sums / np.maximum(cnts, 1.0)


def choose_q(q1_seconds: np.ndarray, coverage: float = 0.95, floor: int = 30) -> int:
    """Smallest Q covering ``coverage`` of quiet depth-1 ask exposure; >= ``floor``."""
    tot = q1_seconds.sum()
    if tot <= 0:
        return floor
    cum = np.cumsum(q1_seconds) / tot
    q = int(np.searchsorted(cum, coverage))
    return max(q, floor)


@dataclass
class _Walker:
    """Shared walk state: engine, reference tracker, warm-up, and the move clock."""

    tick: float
    warmup_ns: int = WARMUP_NS
    warmup_events: int = WARMUP_EVENTS
    engine: CachedBestEngine = field(default_factory=CachedBestEngine)
    tracker: RefPriceTracker = None  # type: ignore[assignment]
    t_start: Optional[int] = None
    t_move: Optional[int] = None     # ts of the last reference move
    last_ts: Optional[int] = None
    n_events: int = 0

    def __post_init__(self) -> None:
        self.tracker = RefPriceTracker(self.tick)

    @property
    def warmed(self) -> bool:
        return (
            self.t_move is not None
            and self.tracker.p_ref is not None
            and self.n_events >= self.warmup_events
            and self.last_ts is not None
            and self.t_start is not None
            and self.last_ts - self.t_start >= self.warmup_ns
        )

    def advance(self, ts: Optional[int], ev: BookEvent) -> None:
        """Apply the event and update the reference/move clock (call AFTER measuring)."""
        self.n_events += 1
        if self.n_events % 512 == 0:
            self.engine.drop_crossed()
        if ev.kind != TRADE:
            moves_before = self.tracker.n_moves
            self.engine.apply(ev)
            self.tracker.update(self.engine)
            if self.tracker.n_moves != moves_before and ts is not None:
                self.t_move = ts
        if ts is not None:
            if self.t_start is None:
                self.t_start = ts
            self.last_ts = ts


def _add_exposure(expo: np.ndarray, l0: int, l1: int, bin_ns: int, n_bins: int) -> None:
    """Distribute the lag interval [l0, l1) ns across bins (overflow bin = index n_bins)."""
    if l1 <= l0:
        return
    b = min(l0 // bin_ns, n_bins)
    while l0 < l1:
        hi = (b + 1) * bin_ns if b < n_bins else l1
        step = min(l1, hi) - l0
        expo[b] += step / 1e9
        l0 += step
        if b < n_bins:
            b = min(b + 1, n_bins)


def measure_burst(
    stream: Iterator[Tuple[Optional[int], BookEvent]],
    K: int,
    tick: float,
    warmup_ns: int = WARMUP_NS,
    warmup_events: int = WARMUP_EVENTS,
) -> BurstProfile:
    """Pass A: exposure-normalised post-move arrival profile + quiet AES + Q profile.

    Trades in the stream are ignored here (arrival sizes and lags come from ``new``
    events; the trade clock adds nothing to the lag exposure that the diff clock does
    not already provide).
    """
    n_bins = MAX_LAG_NS // BIN_NS
    prof = BurstProfile(
        bin_ns=BIN_NS, n_bins=n_bins,
        arrivals_inner=np.zeros(n_bins + 1), exposure_s=np.zeros(n_bins + 1),
        size_sums=np.zeros((K, n_bins + 1)), size_counts=np.zeros((K, n_bins + 1)),
        q1_seconds=np.zeros(501),
    )
    w = _Walker(tick, warmup_ns, warmup_events)
    aes1_running = 1.0
    q_min_lag = Q_PROFILE_MIN_LAG_NS
    for ts, ev in stream:
        if w.warmed and ts is not None and w.t_move is not None:
            # exposure since the last event, in lag-since-move coordinates
            if w.last_ts is not None and ts > w.last_ts:
                l0, l1 = w.last_ts - w.t_move, ts - w.t_move
                _add_exposure(prof.exposure_s, l0, l1, BIN_NS, n_bins)
                if l1 > q_min_lag:  # quiet q1 exposure for the Q rule
                    v1 = window_volumes(w.engine, w.tracker.p_ref, tick, ASK, 1)[0]
                    dt = (l1 - max(l0, q_min_lag)) / 1e9
                    prof.q1_seconds[min(int(round(v1 / aes1_running)), 500)] += dt
            if ev.kind == NEW:
                lag = ts - w.t_move
                b = min(lag // BIN_NS, n_bins)
                d = depth_in_frame(w.tracker.p_ref, tick, ev.side, ev.px, K)
                if d is not None:
                    prof.size_sums[d - 1, b] += ev.sz
                    prof.size_counts[d - 1, b] += 1
                    if d in INNER_DEPTHS:
                        prof.arrivals_inner[b] += 1
                    if d == 1 and prof.size_counts[0].sum() > 0:
                        aes1_running = prof.size_sums[0].sum() / prof.size_counts[0].sum()
        w.advance(ts, ev)
    prof.n_moves = w.tracker.n_moves
    prof.total_s = float(prof.exposure_s.sum())
    logger.info("burst profile: %.0fs exposure, %d ref moves, steady %.2f/s, first-bin %.2f/s",
                prof.total_s, prof.n_moves, prof.steady_rate(), prof.rates()[0])
    return prof


@dataclass
class QuietStats:
    """Same-clock bookkeeping accumulated alongside the quiet-spell calibration."""

    quiet_s: float = 0.0
    total_s: float = 0.0
    quiet_trade_btc: float = 0.0
    total_trade_btc: float = 0.0
    ask1_zero_s: float = 0.0        # UNCONDITIONED time the ask depth-1 slot is empty
    n_moves: int = 0
    n_events_counted: int = 0

    def market_units_target(self, aes1: float) -> float:
        """Trades-file anchor on the quiet clock: AES units consumed per quiet second."""
        if self.quiet_s <= 0 or aes1 <= 0:
            return 0.0
        return self.quiet_trade_btc / self.quiet_s / aes1


def calibrate_quiet(
    stream: Iterator[Tuple[Optional[int], BookEvent]],
    fully_filled: Set[int],
    aes: Sequence[float],
    K: int,
    Q: int,
    tick: float,
    guard_ns: int,
    warmup_ns: int = WARMUP_NS,
    warmup_events: int = WARMUP_EVENTS,
) -> Tuple[np.ndarray, np.ndarray, QuietStats]:
    """Pass B: reference-frame calibration restricted to quiet spells.

    Identical accumulators to
    :func:`~execution.qrm.calibrate_intensities.calibrate_ref_frame` — ``counts``
    (K, Q+1, 2, 3) and ``time_in_state`` (K, Q+1, 2) — but exposure time and event
    counts accrue only while the reference price has been stable for > ``guard_ns``.
    The event that itself triggers a move IS counted (it arrives during quiet time;
    counts stay consistent with exposure — criteria §5). Trade volume is accumulated
    on the same quiet clock for the market anchor, and the unconditioned ask depth-1
    empty time is tracked for the M3 gate target.
    """
    aes = np.asarray(aes, float)
    counts = np.zeros((K, Q + 1, 2, 3))
    time_in = np.zeros((K, Q + 1, 2))
    qs = QuietStats()
    w = _Walker(tick, warmup_ns, warmup_events)
    for ts, ev in stream:
        p_ref = w.tracker.p_ref
        if w.warmed and ts is not None and p_ref is not None and w.t_move is not None:
            # ---- time attribution over the gap since the last event
            if w.last_ts is not None and ts > w.last_ts:
                dt = (ts - w.last_ts) / 1e9
                qs.total_s += dt
                vols = {s: window_volumes(w.engine, p_ref, tick, s, K) for s in (BID, ASK)}
                if round(vols[ASK][0] / aes[0]) == 0:
                    qs.ask1_zero_s += dt
                q_start = max(w.last_ts, w.t_move + guard_ns)
                if ts > q_start:  # the quiet portion of this gap
                    dtq = (ts - q_start) / 1e9
                    qs.quiet_s += dtq
                    for side, si in _SIDE_IDX.items():
                        for d, vol in enumerate(vols[side]):
                            time_in[d, min(int(round(vol / aes[d])), Q), si] += dtq
            # ---- event counting (quiet events only)
            in_quiet = ts - w.t_move >= guard_ns
            if ev.kind == TRADE:
                qs.total_trade_btc += ev.sz
                if in_quiet:
                    qs.quiet_trade_btc += ev.sz
                    d = depth_in_frame(p_ref, tick, ev.side, ev.px, K)
                    if d is not None:
                        qb = min(int(round(w.engine._levels(ev.side).get(ev.px, 0.0) / aes[d - 1])), Q)
                        counts[d - 1, qb, _SIDE_IDX[ev.side], _TYPE_IDX[MARKET]] += 1
                        qs.n_events_counted += 1
            elif in_quiet:
                etype = classify_for_calibration(ev, fully_filled)
                if etype is not None:
                    d = depth_in_frame(p_ref, tick, ev.side, ev.px, K)
                    if d is not None:
                        qb = min(int(round(w.engine._levels(ev.side).get(ev.px, 0.0) / aes[d - 1])), Q)
                        counts[d - 1, qb, _SIDE_IDX[ev.side], _TYPE_IDX[etype]] += 1
                        qs.n_events_counted += 1
        w.advance(ts, ev)
    qs.n_moves = w.tracker.n_moves
    logger.info(
        "quiet calibration: %.0fs total, %.0fs quiet (%.1f%%), %d events counted, "
        "%d moves, trades %.0f BTC total / %.0f quiet, ask1 empty %.1f%% (uncond.)",
        qs.total_s, qs.quiet_s, 100 * qs.quiet_s / max(qs.total_s, 1e-9),
        qs.n_events_counted, qs.n_moves, qs.total_trade_btc, qs.quiet_trade_btc,
        100 * qs.ask1_zero_s / max(qs.total_s, 1e-9),
    )
    return counts, time_in, qs
