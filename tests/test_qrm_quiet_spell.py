"""Unit tests for quiet-spell conditioning (Stage 3f refinement).

Run with:  PYTHONPATH=src pytest tests/test_qrm_quiet_spell.py
"""
import numpy as np

from execution.data.l4.book_diffs_reader import ASK, BID, NEW, REMOVE, TRADE, BookEvent
from execution.qrm.quiet_spell import (
    BIN_NS,
    BurstProfile,
    _add_exposure,
    calibrate_quiet,
    choose_q,
    decide_guard,
    measure_burst,
    quiet_aes,
)

S = 1_000_000_000  # 1 s in ns


def _new(oid, side, px, sz=1.0):
    return BookEvent(oid=oid, side=side, px=px, kind=NEW, sz=sz)


def _trade(side, px, sz):
    return BookEvent(oid=0, side=side, px=px, kind=TRADE, sz=sz)


def _base_stream():
    """Build a stream whose reference moves 100.5 -> 101.5 at t = 4 s.

    Sequence: bid 100 / ask 101 sets p_ref = 100.5; the ask steps to 102 (mid 101,
    ambiguous -> ref holds); a new bid at 101 makes mid 101.5 -> ref moves one tick.
    """
    return [
        (0 * S, _new(1, BID, 100.0)),
        (1 * S, _new(2, ASK, 101.0)),
        (2 * S, _new(3, ASK, 102.0)),
        (3 * S, BookEvent(oid=2, side=ASK, px=101.0, kind=REMOVE, sz=0.0)),
        (4 * S, _new(4, BID, 101.0)),          # <- reference move fires here (t_move = 4 s)
    ]


def test_calibrate_quiet_guard_excludes_post_move_events_and_time():
    guard = 1 * S
    stream = _base_stream() + [
        (int(4.5 * S), _new(5, ASK, 103.0)),   # lag 0.5 s < guard -> NOT counted
        (int(4.6 * S), _trade(ASK, 102.0, 0.7)),  # in guard -> total volume only
        (5 * S, _new(6, BID, 99.0)),           # lag exactly = guard -> counted (>=)
        (6 * S, _new(7, BID, 100.0)),          # lag 2 s -> counted
        (7 * S, _trade(ASK, 102.0, 0.5)),      # quiet trade -> counted + quiet volume
    ]
    counts, time_in, qs = calibrate_quiet(
        iter(stream), fully_filled=set(), aes=[1.0, 1.0, 1.0], K=3, Q=10, tick=1.0,
        guard_ns=guard, warmup_ns=0, warmup_events=0,
    )
    # events: bid@99 (depth 3), bid@100 (depth 2), one quiet market at depth 1 = 3 counted
    assert qs.n_events_counted == 3
    assert counts[2, :, 0, 0].sum() == 1       # limit, depth 3, bid side
    assert counts[1, :, 0, 0].sum() == 1       # limit, depth 2, bid side
    assert counts[0, :, 1, 2].sum() == 1       # market, depth 1, ask side
    assert counts[:, :, :, :].sum() == 3       # the in-guard ask@103 was NOT counted
    # trade volume clocks: total sees both trades, quiet only the post-guard one
    assert qs.total_trade_btc == 1.2 and qs.quiet_trade_btc == 0.5
    # time: warmed span is [4 s, 7 s] = 3 s total; quiet portion is [5 s, 7 s] = 2 s
    assert abs(qs.total_s - 3.0) < 1e-9
    assert abs(qs.quiet_s - 2.0) < 1e-9
    # exposure landed in time_in only during quiet seconds
    assert abs(time_in.sum() - 2.0 * 2 * 3) < 1e-6  # 2 s x 2 sides x K=3 depths


def test_calibrate_quiet_guard_zero_counts_everything_after_first_move():
    stream = _base_stream() + [
        (int(4.5 * S), _new(5, ASK, 103.0)),   # with guard 0 this IS counted
        (6 * S, _new(6, BID, 99.0)),
    ]
    counts, _t, qs = calibrate_quiet(
        iter(stream), set(), [1.0, 1.0, 1.0], K=3, Q=10, tick=1.0,
        guard_ns=0, warmup_ns=0, warmup_events=0,
    )
    assert qs.n_events_counted == 2
    assert abs(qs.quiet_s - qs.total_s) < 1e-9  # no guard -> all warmed time is quiet


def test_add_exposure_distributes_across_bins_and_overflow():
    expo = np.zeros(4)  # 3 regular bins + overflow
    _add_exposure(expo, 0, 3 * BIN_NS, BIN_NS, 3)
    assert np.allclose(expo, [BIN_NS / 1e9] * 3 + [0.0])
    _add_exposure(expo, int(2.5 * BIN_NS), 10 * BIN_NS, BIN_NS, 3)
    assert abs(expo[2] - 1.5 * BIN_NS / 1e9) < 1e-12       # half of bin 2 added
    assert abs(expo[3] - 7 * BIN_NS / 1e9) < 1e-12         # rest into overflow


def test_measure_burst_bins_arrivals_by_lag():
    stream = _base_stream() + [
        (4 * S + 120_000_000, _new(5, ASK, 102.0, sz=2.0)),  # lag 120 ms -> bin 2, depth 1
        (4 * S + 130_000_000, _new(6, ASK, 103.0, sz=4.0)),  # lag 130 ms -> bin 2, depth 2
        (6 * S, _new(7, BID, 99.0)),                          # lag 2 s, depth 3 (not inner)
    ]
    prof = measure_burst(iter(stream), K=3, tick=1.0, warmup_ns=0, warmup_events=0)
    assert prof.arrivals_inner[2] == 2          # both inner-depth arrivals in bin 2
    assert prof.arrivals_inner.sum() == 2       # depth-3 arrival is not "inner"
    assert prof.size_sums[0, 2] == 2.0 and prof.size_sums[1, 2] == 4.0
    assert prof.size_sums[2].sum() == 1.0       # depth-3 size recorded for AES
    assert prof.n_moves == 1
    assert prof.exposure_s.sum() > 0


def test_decide_guard_confirms_and_places_guard():
    n = 60
    prof = BurstProfile(
        bin_ns=BIN_NS, n_bins=n,
        arrivals_inner=np.concatenate([[100, 60, 30], np.full(n - 2, 10.0)]),
        exposure_s=np.ones(n + 1),
        size_sums=np.zeros((1, n + 1)), size_counts=np.zeros((1, n + 1)),
        q1_seconds=np.zeros(501),
    )
    confirmed, guard = decide_guard(prof)   # steady = 10/s; first bin 100/s >= 15/s
    assert confirmed
    assert guard == 3 * BIN_NS              # bins 0-2 elevated; bins 3+ at steady


def test_decide_guard_refutation_branch():
    n = 60
    flat = BurstProfile(
        bin_ns=BIN_NS, n_bins=n,
        arrivals_inner=np.full(n + 1, 10.0), exposure_s=np.ones(n + 1),
        size_sums=np.zeros((1, n + 1)), size_counts=np.zeros((1, n + 1)),
        q1_seconds=np.zeros(501),
    )
    confirmed, guard = decide_guard(flat)
    assert not confirmed and guard == 0     # no burst -> NO guard (pre-registered)


def test_quiet_aes_uses_only_post_guard_bins():
    n = 60
    prof = BurstProfile(
        bin_ns=BIN_NS, n_bins=n,
        arrivals_inner=np.zeros(n + 1), exposure_s=np.ones(n + 1),
        size_sums=np.zeros((2, n + 1)), size_counts=np.zeros((2, n + 1)),
        q1_seconds=np.zeros(501),
    )
    prof.size_sums[0, 0], prof.size_counts[0, 0] = 100.0, 10.0   # burst bin: big orders
    prof.size_sums[0, 5], prof.size_counts[0, 5] = 10.0, 10.0    # quiet bin: 1.0 avg
    aes = quiet_aes(prof, guard_ns=2 * BIN_NS)
    assert aes[0] == 1.0                    # burst bin excluded from the AES scale


def test_choose_q_covers_95pct_with_floor():
    q1 = np.zeros(501)
    q1[:100] = 1.0                          # uniform exposure over sizes 0..99
    assert choose_q(q1) == 94               # 95% of mass
    q1 = np.zeros(501)
    q1[:5] = 1.0
    assert choose_q(q1) == 30               # floor applies
