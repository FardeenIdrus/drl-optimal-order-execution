"""Unit tests for Stage 3: minute resampling + intra-minute statistics.

Run with:  PYTHONPATH=src pytest tests/test_resample.py
"""
import json
import math

import numpy as np
import pytest

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


# --- finer-resolution (parametrised bar width) ------------------------------------

def test_validate_bar_ms_accepts_hour_divisors_rejects_others():
    for ok in (60_000, 30_000, 10_000, 5_000):                # 60/30/10/5 s divide 3600 s
        assert resample.validate_bar_ms(ok) == ok
    for bad in (7_000, 0, -10_000, 3_600_001):                # non-divisor / non-positive
        with pytest.raises(ValueError):
            resample.validate_bar_ms(bad)


def test_resample_buckets_into_10s_bars():
    # At bar_ms=10_000 the same hour splits into 10s bars: snapshots at 0/5s/9s fall
    # in bar 0; the snapshot at 12s falls in bar 10_000. The 60s default would merge
    # all four into one bar -- this asserts the width actually drives the bucketing.
    lines = [
        _snap(0, 100, 101),          # bar 0
        _snap(5_000, 100, 102),      # bar 0
        _snap(9_000, 101, 103),      # bar 0  (last in bar 0)
        _snap(12_000, 200, 201),     # bar 10_000
    ]
    out = resample.resample_snapshots_to_minute(_snaps_df(lines), bar_ms=10_000)

    assert list(out[schema.TS_COL]) == [0, 10_000]
    r0 = out.iloc[0]
    assert r0["n_snapshots"] == 3
    # end-of-bar (point-in-time) book is the LAST snapshot of the bar
    assert r0["bid_px_1"] == 101.0 and r0["ask_px_1"] == 103.0 and r0["mid"] == 102.0
    # realised variance accumulates only the intra-bar returns (two, within bar 0)
    expected_rv = math.log(101 / 100.5) ** 2 + math.log(102 / 101) ** 2
    assert math.isclose(r0["realized_variance"], expected_rv, rel_tol=1e-9)
    assert out.iloc[1]["n_snapshots"] == 1


def test_combine_stitches_10s_bar_split_across_frames():
    # A single 10s bar (bar 0) is split across two "files": frame A holds 0s/5s,
    # frame B holds 9s plus the next bar (12s). The carry must combine bar 0 into ONE
    # row (n=3) at the 10s width, mirroring the 60s carry test.
    frame_a = _snaps_df([_snap(0, 100, 101), _snap(5_000, 100, 102)])
    frame_b = _snaps_df([_snap(9_000, 101, 103), _snap(12_000, 200, 201)])
    out = resample.combine_frames_to_minute([frame_a, frame_b], bar_ms=10_000)

    assert list(out[schema.TS_COL]) == [0, 10_000]           # no duplicate bar 0
    assert out.iloc[0]["n_snapshots"] == 3                   # all three combined
    assert out.iloc[0]["bid_px_1"] == 101.0                  # end-of-bar book = last snap
    assert out.iloc[1]["n_snapshots"] == 1


def test_process_files_rejects_bad_bar_width(tmp_path):
    # The public entry point guards the invariant before doing any work.
    with pytest.raises(ValueError):
        resample.process_files([], bar_ms=7_000)


# --- Path A: memory-safe monthly chunking (must equal the single-pass output) ------

# Epochs at exact UTC month starts so snapshots land in known months/hours.
_JAN = 1_704_067_200_000   # 2024-01-01 00:00:00 UTC
_FEB = 1_706_745_600_000   # 2024-02-01 00:00:00 UTC


def _write_lz4(path, lines):
    import lz4.frame
    path.parent.mkdir(parents=True, exist_ok=True)
    with lz4.frame.open(path, "wb") as fh:
        fh.write(("\n".join(lines)).encode())


def test_group_paths_by_month_orders_and_buckets():
    paths = [
        "/raw/2024-02-03/9/BTC.lz4", "/raw/2024-01-15/10/BTC.lz4",
        "/raw/2024-01-15/2/BTC.lz4", "/raw/2024-02-01/0/BTC.lz4",
    ]
    groups = resample._group_paths_by_month(paths)
    assert list(groups.keys()) == ["2024-01", "2024-02"]                 # chronological months
    # within a month, hours are int-sorted (2 before 10), not string-sorted
    assert groups["2024-01"] == ["/raw/2024-01-15/2/BTC.lz4", "/raw/2024-01-15/10/BTC.lz4"]


def test_chunked_resample_equals_single_pass(tmp_path):
    # Two months of real lz4 files; the chunked (per-month) build must reproduce the
    # single-pass build byte-for-byte (after the re-sort every downstream stage does).
    raw = tmp_path / "raw"
    f_jan = raw / "2024-01-01" / "0" / "BTC.lz4"
    f_feb = raw / "2024-02-01" / "0" / "BTC.lz4"
    _write_lz4(f_jan, [_snap(_JAN + 0, 100, 101), _snap(_JAN + 5_000, 100, 102),
                       _snap(_JAN + 9_000, 101, 103), _snap(_JAN + 12_000, 104, 105)])
    _write_lz4(f_feb, [_snap(_FEB + 0, 200, 201), _snap(_FEB + 5_000, 200, 202),
                       _snap(_FEB + 12_000, 203, 204)])
    paths = [str(f_jan), str(f_feb)]

    single = resample.process_files(paths, bar_ms=10_000).sort_values(schema.TS_COL).reset_index(drop=True)

    out_dir = tmp_path / "minute_10s"
    parts = resample.resample_chunked(paths, out_dir, bar_ms=10_000)
    assert sorted(p.name for p in parts) == ["part_2024-01.parquet", "part_2024-02.parquet"]
    import pandas as pd
    chunked = pd.read_parquet(out_dir).sort_values(schema.TS_COL).reset_index(drop=True)

    pd.testing.assert_frame_equal(single, chunked, check_like=True)


def test_chunked_drops_cross_month_boundary_stray(tmp_path):
    # The real-data failure: a Feb file carries an out-of-order snapshot timestamped in
    # the last Jan bar, which (un-trimmed) would emit that Jan bar in BOTH parts -> a
    # duplicate ts that breaks the features reindex. Trimming each part to its month
    # must leave the boundary bar exactly once (in Jan's part).
    import pandas as pd
    jan_bar = _FEB - 10_000                       # 2024-01-31 23:59:50 bar (last Jan 10s bar)
    raw = tmp_path / "raw"
    f_jan = raw / "2024-01-31" / "23" / "BTC.lz4"
    f_feb = raw / "2024-02-01" / "0" / "BTC.lz4"
    _write_lz4(f_jan, [_snap(jan_bar + 0, 100, 101), _snap(jan_bar + 5_000, 100, 102)])
    _write_lz4(f_feb, [_snap(jan_bar + 1_000, 100, 103),          # STRAY: Jan ts in a Feb file
                       _snap(_FEB + 0, 200, 201), _snap(_FEB + 5_000, 200, 202)])
    paths = [str(f_jan), str(f_feb)]

    resample.resample_chunked(paths, tmp_path / "m", bar_ms=10_000)
    out = pd.read_parquet(tmp_path / "m").sort_values(schema.TS_COL).reset_index(drop=True)
    assert out[schema.TS_COL].duplicated().sum() == 0                  # the bug is fixed
    assert list(out[schema.TS_COL]) == [jan_bar, _FEB]                 # boundary bar appears ONCE
    # the kept bar is Jan's real one (2 snaps); the Feb stray was trimmed, not merged
    assert int(out[out[schema.TS_COL] == jan_bar]["n_snapshots"].iloc[0]) == 2


def test_qa_from_parts_matches_single_pass_qa(tmp_path):
    raw = tmp_path / "raw"
    f_jan = raw / "2024-01-01" / "0" / "BTC.lz4"
    _write_lz4(f_jan, [_snap(_JAN + 0, 100, 101), _snap(_JAN + 5_000, 100, 102),
                       _snap(_JAN + 12_000, 104, 105)])
    paths = [str(f_jan)]
    out_dir = tmp_path / "minute_10s"
    resample.resample_chunked(paths, out_dir, bar_ms=10_000)
    single = resample.process_files(paths, bar_ms=10_000)
    qa_parts = resample.qa_report_from_parts(out_dir)
    qa_single = resample.qa_report(single)
    assert qa_parts["minutes_present"] == qa_single["minutes_present"]
    assert qa_parts["duplicate_minutes"] == 0 and qa_parts["book_ordering_ok"] is True
