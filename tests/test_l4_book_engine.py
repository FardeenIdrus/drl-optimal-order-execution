"""Unit tests for Stage L4-1/L4-2: book-diff reader and book reconstruction engine.

Run with:  PYTHONPATH=src pytest tests/test_l4_book_engine.py
"""
import gzip
import json

from execution.data.l4.book_diffs_reader import (
    ASK,
    BID,
    NEW,
    REMOVE,
    UPDATE,
    BookEvent,
    parse_line,
    stream_events,
)
from execution.data.l4.book_engine import BookEngine
from execution.data.l4.snapshot_sampler import SnapshotSampler, iter_snapshots


# ------------------------------------------------------------------ reader
def test_parse_line_new_remove_update():
    new = parse_line(
        {"oid": 1, "coin": "BTC", "side": "B", "px": "100.0",
         "raw_book_diff": {"new": {"sz": "2.5"}}}, "BTC")
    assert new == BookEvent(oid=1, side=BID, px=100.0, kind=NEW, sz=2.5)

    rem = parse_line(
        {"oid": 1, "coin": "BTC", "side": "B", "px": "100.0", "raw_book_diff": "remove"}, "BTC")
    assert rem == BookEvent(oid=1, side=BID, px=100.0, kind=REMOVE, sz=0.0)

    upd = parse_line(
        {"oid": 1, "coin": "BTC", "side": "A", "px": "101.0",
         "raw_book_diff": {"update": {"origSz": "5.0", "newSz": "4.7"}}}, "BTC")
    assert upd == BookEvent(oid=1, side=ASK, px=101.0, kind=UPDATE, sz=4.7)


def test_parse_line_filters_other_coin_and_bad_shape():
    assert parse_line({"oid": 1, "coin": "ETH", "side": "B", "px": "1",
                       "raw_book_diff": "remove"}, "BTC") is None
    assert parse_line({"oid": 1, "coin": "BTC", "side": "B", "px": "1",
                       "raw_book_diff": {"weird": {}}}, "BTC") is None


def test_stream_events_reads_gzip_and_filters(tmp_path):
    p = tmp_path / "ex0.gz"
    lines = [
        {"oid": 1, "coin": "BTC", "side": "B", "px": "100.0", "raw_book_diff": {"new": {"sz": "1.0"}}},
        {"oid": 2, "coin": "ETH", "side": "B", "px": "50.0", "raw_book_diff": {"new": {"sz": "9.0"}}},
        {"oid": 1, "coin": "BTC", "side": "B", "px": "100.0", "raw_book_diff": "remove"},
    ]
    with gzip.open(p, "wt") as fh:
        fh.write("\n".join(json.dumps(x) for x in lines) + "\n")
    evs = list(stream_events(p, coin="BTC"))
    assert [e.kind for e in evs] == [NEW, REMOVE]  # ETH filtered out
    assert evs[0].oid == 1 and evs[0].sz == 1.0


def test_stream_events_skips_malformed_json(tmp_path):
    p = tmp_path / "ex1.gz"
    good = json.dumps({"oid": 1, "coin": "BTC", "side": "B", "px": "100.0",
                       "raw_book_diff": {"new": {"sz": "1.0"}}})
    with gzip.open(p, "wt") as fh:
        fh.write(good + "\n{ not json }\n\n")
    evs = list(stream_events(p, coin="BTC"))
    assert len(evs) == 1


# ------------------------------------------------------------------ engine
def _new(oid, side, px, sz):
    return BookEvent(oid=oid, side=side, px=px, kind=NEW, sz=sz)


def test_new_builds_levels_and_top_of_book():
    e = BookEngine()
    e.apply(_new(1, BID, 100.0, 2.0))
    e.apply(_new(2, BID, 99.0, 1.0))
    e.apply(_new(3, ASK, 101.0, 3.0))
    bb, ba, mid, spread = e.top_of_book()
    assert (bb, ba, mid, spread) == (100.0, 101.0, 100.5, 1.0)
    assert e.n_levels == (2, 1) and e.n_orders == 3


def test_multiple_orders_same_level_aggregate():
    e = BookEngine()
    e.apply(_new(1, ASK, 101.0, 1.0))
    e.apply(_new(2, ASK, 101.0, 2.5))
    bids, asks = e.snapshot(5)
    assert asks == [(101.0, 3.5)]  # both orders sum into one level
    assert e.n_levels == (0, 1)


def test_remove_subtracts_exact_size_and_clears_empty_level():
    e = BookEngine()
    e.apply(_new(1, ASK, 101.0, 1.0))
    e.apply(_new(2, ASK, 101.0, 2.0))
    e.apply(BookEvent(oid=1, side=ASK, px=101.0, kind=REMOVE, sz=0.0))
    assert e.snapshot(5)[1] == [(101.0, 2.0)]  # only order 2 remains
    e.apply(BookEvent(oid=2, side=ASK, px=101.0, kind=REMOVE, sz=0.0))
    assert e.n_levels == (0, 0) and e.best_ask() is None  # level fully cleared


def test_update_partial_fill_then_remove():
    e = BookEngine()
    e.apply(_new(1, BID, 100.0, 5.0))
    # partial fill: remaining size drops to 3.0
    e.apply(BookEvent(oid=1, side=BID, px=100.0, kind=UPDATE, sz=3.0))
    assert e.snapshot(5)[0] == [(100.0, 3.0)]
    # remove must subtract the CURRENT (reduced) size, not the original
    e.apply(BookEvent(oid=1, side=BID, px=100.0, kind=REMOVE, sz=0.0))
    assert e.n_levels == (0, 0) and e.n_orders == 0


def test_drop_crossed_evicts_stale_bid_when_market_falls_below_it():
    # The Dec-16 pathology: a bid whose removal went missing, market falls below it.
    e = BookEngine()
    e.apply(_new(1, BID, 105.0, 1.0))   # becomes stale (removal missing)
    e.apply(_new(2, BID, 100.0, 1.0))   # fresh real bid
    e.apply(_new(3, ASK, 101.0, 1.0))   # fresh real ask BELOW the stale bid -> crossed
    assert e.best_bid() == 105.0
    dropped = e.drop_crossed()
    assert dropped == 1                 # only the phantom level removed
    assert e.best_bid() == 100.0 and e.best_ask() == 101.0  # real levels intact


def test_drop_crossed_evicts_stale_ask_when_market_rises_above_it():
    # The Dec-12 22:08 freeze pathology: a stale ASK stranded below a rising market.
    # The pre-fix "heal bids first" rule evicted every REAL bid here and pinned the
    # top-of-book at the phantom pair for hours (117 frozen hours across the month).
    e = BookEngine()
    e.apply(_new(1, ASK, 90188.0, 1.0))     # becomes stale (removal missing)
    e.apply(_new(2, BID, 90187.0, 1.0))     # old bid below it
    for i, px in enumerate([90250.0, 90251.0, 90252.0]):
        e.apply(_new(10 + i, ASK, px + 3.0, 1.0))  # fresh real asks above (90253-90255)
        e.apply(_new(20 + i, BID, px, 1.0))        # fresh real bids -> crossed vs stale ask
    dropped = e.drop_crossed()
    assert dropped == 1                       # ONLY the stale ask evicted
    assert e.best_bid() == 90252.0            # real bids survive
    assert e.best_ask() == 90253.0            # real asks survive
    # the stale ask's order left the oid map too: a late event for it is an orphan
    e.apply(BookEvent(oid=1, side=ASK, px=90188.0, kind=UPDATE, sz=0.5))
    assert e.n_orphan_update == 1
    assert 90188.0 not in e._asks             # no ghost resurrection of the level


def test_drop_crossed_evicts_multiple_stale_levels():
    # Several stale asks left behind as the market rose: all evicted, real depth intact.
    e = BookEngine()
    for i, px in enumerate([200.0, 201.0, 202.0]):
        e.apply(_new(i, ASK, px, 1.0))       # stale asks (removals missing)
    e.apply(_new(10, BID, 250.0, 1.0))       # fresh real bid far above
    e.apply(_new(11, ASK, 251.0, 1.0))       # fresh real ask
    dropped = e.drop_crossed()
    assert dropped == 3
    assert e.best_bid() == 250.0 and e.best_ask() == 251.0


def test_drop_crossed_recency_beats_side_convention():
    # A resting NEW accepted by the venue proves the older opposing level was already
    # gone: the newest order survives regardless of which side it is on.
    e = BookEngine()
    e.apply(_new(1, ASK, 101.0, 1.0))   # older ask
    e.apply(_new(2, BID, 105.0, 2.0))   # NEWER resting bid above it -> ask is the phantom
    assert e.drop_crossed() == 1
    assert e.best_bid() == 105.0 and e.best_ask() is None


def test_drop_crossed_noop_when_book_is_clean():
    e = BookEngine()
    e.apply(_new(1, BID, 100.0, 1.0))
    e.apply(_new(2, ASK, 101.0, 1.0))
    assert e.drop_crossed() == 0
    assert e.n_levels == (1, 1)


def test_update_to_zero_removes_order():
    # a partial-fill update that brings remaining size to 0 must drop the order entirely
    e = BookEngine()
    e.apply(_new(1, ASK, 101.0, 2.0))
    e.apply(BookEvent(oid=1, side=ASK, px=101.0, kind=UPDATE, sz=0.0))
    assert e.n_orders == 0 and e.n_levels == (0, 0)
    assert e.best_ask() is None


def test_orphan_remove_and_update_counted_and_ignored():
    e = BookEngine()
    e.apply(BookEvent(oid=999, side=BID, px=100.0, kind=REMOVE, sz=0.0))
    e.apply(BookEvent(oid=888, side=ASK, px=101.0, kind=UPDATE, sz=1.0))
    assert e.n_orphan_remove == 1 and e.n_orphan_update == 1
    assert e.n_levels == (0, 0)  # unknown oids never touch the book
    assert e.orphan_rate == 1.0


def test_snapshot_orders_and_truncates_to_k():
    e = BookEngine()
    for i, px in enumerate([98.0, 100.0, 99.0]):
        e.apply(_new(i, BID, px, 1.0))
    for i, px in enumerate([103.0, 101.0, 102.0], start=10):
        e.apply(_new(i, ASK, px, 1.0))
    bids, asks = e.snapshot(2)
    assert [p for p, _ in bids] == [100.0, 99.0]   # descending, top 2
    assert [p for p, _ in asks] == [101.0, 102.0]  # ascending, top 2


def test_new_duplicate_oid_replaces_contribution():
    # defensive path: a repeated oid should not double-count into the level
    e = BookEngine()
    e.apply(_new(1, ASK, 101.0, 1.0))
    e.apply(_new(1, ASK, 101.0, 4.0))  # same oid re-placed
    assert e.snapshot(5)[1] == [(101.0, 4.0)]
    assert e.n_orders == 1


# ---------------------------------------------------------------- sampler
def test_iter_snapshots_emits_on_cadence_grid():
    # cadence 10ns; events at ts 5, 12, 25 raise the best bid over time
    stream = [
        (5, _new(1, BID, 100.0, 1.0)),
        (12, _new(2, BID, 101.0, 1.0)),
        (25, _new(3, BID, 102.0, 1.0)),
    ]
    snaps = list(iter_snapshots(iter(stream), BookEngine(), cadence_ns=10, top_k=5))
    # boundaries at 10 and 20 fall before the last event; each shows the book as of then
    assert [s.ts for s in snaps] == [10, 20]
    assert snaps[0].best_bid == 100.0  # only event@5 applied by boundary 10
    assert snaps[1].best_bid == 101.0  # events@5,12 applied by boundary 20


def test_iter_snapshots_respects_warmup_and_gaps():
    stream = [
        (5, _new(1, BID, 100.0, 1.0)),
        (12, _new(2, BID, 101.0, 1.0)),
        (95, _new(3, BID, 102.0, 1.0)),  # big gap: boundaries 20..90 all emit
    ]
    snaps = list(iter_snapshots(iter(stream), BookEngine(), cadence_ns=10, warmup_events=2))
    # warmup_events=2 suppresses boundary 10 (only 1 event applied); 20..90 emit (8 boundaries)
    assert [s.ts for s in snaps] == [20, 30, 40, 50, 60, 70, 80, 90]
    assert all(s.best_bid == 101.0 for s in snaps)  # book unchanged across the quiet gap


def test_sampler_state_carries_across_feed_calls():
    # feeding two chunks through one persistent sampler must equal feeding all at once
    a = [(5, _new(1, BID, 100.0, 1.0)), (12, _new(2, BID, 101.0, 1.0))]
    b = [(25, _new(3, BID, 102.0, 1.0)), (33, _new(4, BID, 103.0, 1.0))]
    whole = list(iter_snapshots(iter(a + b), BookEngine(), cadence_ns=10))

    s = SnapshotSampler(BookEngine(), cadence_ns=10)
    split = list(s.feed(iter(a))) + list(s.feed(iter(b)))
    assert [(x.ts, x.best_bid) for x in split] == [(x.ts, x.best_bid) for x in whole]

    # get_state / set_state round-trips onto a rebuilt sampler (checkpoint/resume path)
    st = s.get_state()
    s2 = SnapshotSampler(BookEngine(), cadence_ns=10)
    s2.set_state(st)
    assert s2.get_state() == st
