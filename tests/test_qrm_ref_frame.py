"""Unit tests for the reference-price frame (the frozen-price fix).

Run with:  PYTHONPATH=src pytest tests/test_qrm_ref_frame.py
"""
import random

from execution.data.l4.book_diffs_reader import ASK, BID, NEW, REMOVE, UPDATE, BookEvent
from execution.data.l4.book_engine import BookEngine
from execution.qrm.calibrate_intensities import calibrate_ref_frame
from execution.qrm.ref_frame import (
    CachedBestEngine,
    RefPriceTracker,
    depth_in_frame,
    window_volumes,
)


def _ev(oid, side, px, kind=NEW, sz=1.0):
    return BookEvent(oid=oid, side=side, px=px, kind=kind, sz=sz)


# ------------------------------------------------------------ RefPriceTracker
def test_tracker_sets_ref_on_half_tick_mid():
    e = CachedBestEngine()
    e.apply(_ev(1, BID, 100.0))
    e.apply(_ev(2, ASK, 101.0))   # spread 1 tick -> mid 100.5 (half-tick)
    t = RefPriceTracker(1.0)
    t.update(e)
    assert t.p_ref == 100.5


def test_tracker_holds_on_ambiguous_mid_hysteresis():
    e = CachedBestEngine()
    e.apply(_ev(1, BID, 100.0))
    e.apply(_ev(2, ASK, 101.0))
    t = RefPriceTracker(1.0)
    t.update(e)                    # ref = 100.5
    e.apply(_ev(2, ASK, 101.0, kind=REMOVE, sz=0.0))
    e.apply(_ev(3, ASK, 102.0))    # spread 2 -> mid 101.0 (on-tick, ambiguous)
    t.update(e)
    assert t.p_ref == 100.5        # adjacent -> hold (hysteresis)


def test_tracker_steps_one_tick_toward_distant_mid():
    e = CachedBestEngine()
    e.apply(_ev(1, BID, 100.0))
    e.apply(_ev(2, ASK, 101.0))
    t = RefPriceTracker(1.0)
    t.update(e)                    # 100.5
    # market gaps up 3 ticks: bid 103, ask 104 -> mid 103.5
    e.apply(_ev(1, BID, 100.0, kind=REMOVE, sz=0.0))
    e.apply(_ev(2, ASK, 101.0, kind=REMOVE, sz=0.0))
    e.apply(_ev(3, BID, 103.0))
    e.apply(_ev(4, ASK, 104.0))
    t.update(e)
    assert t.p_ref == 103.5 and t.n_moves == 3   # stepped tick-by-tick


# ------------------------------------------------------- frame depth / windows
def test_depth_in_frame_fixed_prices():
    # ref 100.5: ask depths at 101,102,... bid depths at 100,99,...
    assert depth_in_frame(100.5, 1.0, ASK, 101.0, 5) == 1
    assert depth_in_frame(100.5, 1.0, ASK, 103.0, 5) == 3
    assert depth_in_frame(100.5, 1.0, BID, 100.0, 5) == 1
    assert depth_in_frame(100.5, 1.0, BID, 96.0, 5) == 5
    assert depth_in_frame(100.5, 1.0, BID, 95.0, 5) is None    # beyond K
    assert depth_in_frame(100.5, 1.0, ASK, 100.0, 5) is None   # wrong side of ref


def test_window_volumes_shows_empty_slots():
    e = CachedBestEngine()
    e.apply(_ev(1, ASK, 102.0, sz=2.0))   # nothing at 101 -> depth-1 slot EMPTY
    vols = window_volumes(e, 100.5, 1.0, ASK, 3)
    assert vols == [0.0, 2.0, 0.0]        # q=0 at depth 1 is observable (the fix)


# ------------------------------------------------- CachedBestEngine equivalence
def test_cached_best_engine_fuzz_equivalence():
    rng = random.Random(7)
    base, cached = BookEngine(), CachedBestEngine()
    live = {}
    next_oid = 1
    for step in range(4000):
        r = rng.random()
        if r < 0.5 or not live:                       # new order
            side = ASK if rng.random() < 0.5 else BID
            px = float(rng.randint(95, 105))
            ev = _ev(next_oid, side, px, NEW, sz=round(rng.uniform(0.1, 3.0), 2))
            live[next_oid] = (side, px)
            next_oid += 1
        elif r < 0.8:                                 # remove a live order
            oid = rng.choice(list(live))
            side, px = live.pop(oid)
            ev = _ev(oid, side, px, REMOVE, sz=0.0)
        else:                                         # partial-fill update
            oid = rng.choice(list(live))
            side, px = live[oid]
            new_sz = round(rng.uniform(0.0, 2.0), 2)
            ev = _ev(oid, side, px, UPDATE, sz=new_sz)
            if new_sz <= 1e-9:
                live.pop(oid)
        base.apply(ev)
        cached.apply(ev)
        assert cached.best_bid() == base.best_bid(), f"bid mismatch at step {step}"
        assert cached.best_ask() == base.best_ask(), f"ask mismatch at step {step}"
        if step % 500 == 499:                          # exercise the guard path too
            base.drop_crossed()
            cached.drop_crossed()
            assert cached.best_bid() == base.best_bid()
            assert cached.best_ask() == base.best_ask()


# ------------------------------------------- ref-frame calibration observes q=0
def test_calibrate_ref_frame_observes_empty_queue_state():
    # book: bid 100 (1.0), ask 102 (1.0) -> ref settles at 100.5 or 101.5; the slot
    # between best bid and best ask is EMPTY and must accumulate q=0 time-in-state.
    stream = [
        (1_000_000_000, _ev(1, BID, 100.0, NEW, 1.0)),
        (2_000_000_000, _ev(2, ASK, 102.0, NEW, 1.0)),
        (12_000_000_000, _ev(3, ASK, 102.0, NEW, 1.0)),   # 10s later: clock advances
    ]
    counts, time_in = calibrate_ref_frame(
        iter(stream), fully_filled=set(), aes=[1.0] * 3, K=3, Q=10, tick=1.0
    )
    # ask side (index 1): SOME depth spent time at q=0 (the empty slot inside the spread)
    assert time_in[:, 0, 1].sum() > 0.0
    # and the resting ask level accumulated time at q=1
    assert time_in[:, 1, 1].sum() > 0.0