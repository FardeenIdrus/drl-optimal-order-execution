"""Unit tests for Stage 3: minute resampling + intra-minute statistics.

Run with:  PYTHONPATH=src pytest tests/test_resample.py
"""
import json
import math

import numpy as np

from execution.data import resample, schema
from execution.data.sources import hyperliquid_l2 as hl


def _snap(ts, bid, ask, bsz=1.0, asz=1.0):
    return json.dumps(
        {"raw": {"data": {"coin": "BTC", "time": ts,
                          "levels": [[{"px": str(bid), "sz": str(bsz), "n": 1}],
                                     [{"px": str(ask), "sz": str(asz), "n": 1}]]}}}
    )


def _snaps_df(lines):
    df, _ = hl.decode("\n".join(lines))
    return df


def test_minute_columns_shape():
    cols = resample.minute_columns()
    assert cols[0] == schema.TS_COL
    # ts + 120 book columns + 11 stat columns
    assert len(cols) == 1 + (2 * 3 * schema.N_LEVELS) + len(resample.STAT_COLUMNS)


def test_resample_buckets_and_stats():
    # minute 0: three snapshots; minute 60000: one snapshot
    lines = [
        _snap(0, 100, 101),          # mid 100.5, spread 1
        _snap(30_000, 100, 102),     # mid 101.0, spread 2
        _snap(59_000, 101, 103),     # mid 102.0, spread 2  (last in minute 0)
        _snap(60_000, 200, 201),     # mid 200.5  (minute 60000)
    ]
    out = resample.resample_snapshots_to_minute(_snaps_df(lines))

    assert list(out[schema.TS_COL]) == [0, 60_000]
    r0 = out.iloc[0]
    assert r0["n_snapshots"] == 3
    assert r0["hi_mid"] == 102.0 and r0["lo_mid"] == 100.5
    # end-of-minute (point-in-time) book is the LAST snapshot of the minute
    assert r0["bid_px_1"] == 101.0 and r0["ask_px_1"] == 103.0 and r0["mid"] == 102.0
    assert math.isclose(r0["mean_spread"], (1 + 2 + 2) / 3, rel_tol=1e-9)

    expected_rv = math.log(101 / 100.5) ** 2 + math.log(102 / 101) ** 2
    assert math.isclose(r0["realized_variance"], expected_rv, rel_tol=1e-9)
    assert bool(r0["valid"]) is True


def test_single_snapshot_minute_has_nan_realized_variance():
    out = resample.resample_snapshots_to_minute(_snaps_df([_snap(60_000, 200, 201)]))
    assert len(out) == 1
    assert out.iloc[0]["n_snapshots"] == 1
    assert np.isnan(out.iloc[0]["realized_variance"])
    assert out.iloc[0]["mid"] == 200.5


def test_empty_input_returns_empty_with_columns():
    empty = resample.resample_snapshots_to_minute(_snaps_df([]))
    assert empty.empty and list(empty.columns) == resample.minute_columns()


def test_realized_variance_does_not_cross_minute_boundary():
    # one snapshot per minute -> no intra-minute returns anywhere -> all RV NaN
    lines = [_snap(0, 100, 101), _snap(60_000, 110, 111), _snap(120_000, 90, 91)]
    out = resample.resample_snapshots_to_minute(_snaps_df(lines))
    assert len(out) == 3
    assert out["realized_variance"].isna().all()


def test_combine_stitches_minute_split_across_frames():
    # Minute 0 is split across two "files": frame A holds its first two snapshots,
    # frame B holds its third snapshot plus the next minute. The carry must combine
    # minute 0 into ONE row (n=3), not two partial rows.
    frame_a = _snaps_df([_snap(0, 100, 101), _snap(30_000, 100, 102)])
    frame_b = _snaps_df([_snap(59_000, 101, 103), _snap(60_000, 200, 201)])
    out = resample.combine_frames_to_minute([frame_a, frame_b])

    assert list(out[schema.TS_COL]) == [0, 60_000]          # no duplicate minute 0
    assert out.iloc[0]["n_snapshots"] == 3                   # all three combined
    assert out.iloc[0]["bid_px_1"] == 101.0                 # end-of-minute book = last snap
    # cross-boundary return (30s->59s snapshot) is included in realised variance
    expected_rv = math.log(101 / 100.5) ** 2 + math.log(102 / 101) ** 2
    assert math.isclose(out.iloc[0]["realized_variance"], expected_rv, rel_tol=1e-9)
    assert out.iloc[1]["n_snapshots"] == 1


def test_combine_dedupes_out_of_order_minute_keeping_fuller_bar():
    # A stray snapshot for minute 0 arrives in a later frame, after minute 0 was
    # already emitted in full. Result must keep ONE minute-0 row, the fuller one.
    full_min0 = _snaps_df([_snap(0, 100, 101), _snap(30_000, 100, 102)])
    min1 = _snaps_df([_snap(60_000, 110, 111)])
    stray_min0 = _snaps_df([_snap(1_000, 50, 51)])  # out-of-order timestamp
    out = resample.combine_frames_to_minute([full_min0, min1, stray_min0])

    assert list(out[schema.TS_COL]) == [0, 60_000]            # no duplicate minute 0
    assert int(out.iloc[0]["n_snapshots"]) == 2               # kept the fuller bar
    assert out.iloc[0]["bid_px_1"] == 100.0                   # not the stray (bid 50)
