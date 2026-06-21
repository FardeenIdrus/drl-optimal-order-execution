"""Unit tests for Stage 2: canonical schema, Hyperliquid decode, and validation.

Run with:  PYTHONPATH=src pytest tests/test_parse.py
"""
import json

import numpy as np

from execution.data import schema
from execution.data.sources import hyperliquid_l2 as hl


def _lvl(px, sz, n=1):
    return {"px": str(px), "sz": str(sz), "n": n}


def _snapshot(ts, bids, asks):
    return json.dumps(
        {
            "time": "2025-01-01T00:00:00",
            "ver_num": 1,
            "raw": {"channel": "l2Book", "data": {"coin": "BTC", "time": ts, "levels": [bids, asks]}},
        }
    )


def test_canonical_columns_shape():
    cols = schema.canonical_columns()
    assert cols[0] == schema.TS_COL
    # ts + (bid/ask) x (px/sz/n) x N
    assert len(cols) == 1 + 2 * 3 * schema.N_LEVELS
    assert "bid_px_1" in cols and "ask_n_20" in cols


def test_decode_maps_levels_and_pads():
    bids = [_lvl(100.0, 2.0, 3), _lvl(99.0, 1.0, 1)]
    asks = [_lvl(101.0, 0.5, 1)]
    df, n_malformed = hl.decode(_snapshot(1_700_000_000_000, bids, asks))
    assert n_malformed == 0 and len(df) == 1
    r = df.iloc[0]
    assert r["ts"] == 1_700_000_000_000
    assert (r["bid_px_1"], r["bid_sz_1"], r["bid_n_1"]) == (100.0, 2.0, 3.0)
    assert r["bid_px_2"] == 99.0 and r["ask_px_1"] == 101.0
    # levels beyond those present are NaN-padded
    assert np.isnan(r["bid_px_3"]) and np.isnan(r["ask_px_2"])
    assert df["ts"].dtype == "int64"


def test_decode_skips_malformed_lines():
    good = _snapshot(1, [_lvl(100, 1)], [_lvl(101, 1)])
    df, n_malformed = hl.decode(good + "\n{ not valid json }\n\n")
    assert len(df) == 1 and n_malformed == 1


def test_validate_flags_crossed_and_empty():
    rows = [
        _snapshot(1, [_lvl(100, 1)], [_lvl(101, 1)]),  # valid
        _snapshot(2, [_lvl(102, 1)], [_lvl(101, 1)]),  # crossed: bid >= ask
        _snapshot(3, [], [_lvl(101, 1)]),              # empty bid side
    ]
    df, _ = hl.decode("\n".join(rows))
    valid, counts = schema.validate(df)
    assert counts["valid"] == 1
    assert counts["crossed_or_locked"] == 1
    assert counts["empty_side"] == 1
    assert bool(valid.iloc[0]) and not bool(valid.iloc[1]) and not bool(valid.iloc[2])


def test_validate_flags_non_monotonic():
    # bid_px_2 > bid_px_1 violates descending order
    df, _ = hl.decode(_snapshot(1, [_lvl(100, 1), _lvl(101, 1)], [_lvl(102, 1)]))
    valid, counts = schema.validate(df)
    assert counts["non_monotonic"] == 1 and not bool(valid.iloc[0])
