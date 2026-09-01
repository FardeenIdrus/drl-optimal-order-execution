"""Unit tests for Stage 6: materialisation, decision-time alignment, no look-ahead.

Run with:  PYTHONPATH=src pytest tests/test_dataset.py
"""
import pandas as pd
import pytest

from execution.data import dataset as D
from execution.data.regimes import EPISODE_MS, MS_PER_MINUTE

BASE = (1_700_000_000_000 // EPISODE_MS) * EPISODE_MS  # align to a 30-min boundary


def _minute_df(idx_lo=-1, idx_hi=61):
    """One row per minute; ask_px_1 = 1000 + minute_index so a book is traceable."""
    rows = []
    for idx in range(idx_lo, idx_hi):
        apx1 = 1000.0 + idx
        d = {"ts": BASE + idx * MS_PER_MINUTE}
        for i in range(1, 21):
            d[f"ask_px_{i}"] = apx1 + (i - 1)
            d[f"ask_sz_{i}"] = 1.0
        d["bid_px_1"] = apx1 - 1.0
        d["bid_sz_1"] = 1.0
        d["mid"] = apx1 - 0.5
        rows.append(d)
    return pd.DataFrame(rows)


def _features_df(n=60):
    ts = [BASE + i * MS_PER_MINUTE for i in range(n)]
    return pd.DataFrame({"ts": ts, "spread_bps": 0.2, "imbalance": 0.1,
                         "recent_return": 0.0, "rolling_vol": 0.001, "ask_depth": 5.0})


def _episodes_df():
    return pd.DataFrame({
        "start_ts": [BASE, BASE + EPISODE_MS], "episode_id": [0, 1],
        "regime": ["calm", "volatile"], "split": ["train", "test"],
    })


def test_materialise_counts_and_steps():
    out = D.materialise(_features_df(), _minute_df(), _episodes_df())
    assert len(out) == 60
    assert set(out["episode_id"]) == {0, 1}
    assert (out.groupby("episode_id")["step"].size() == 30).all()
    assert list(out[out.episode_id == 0]["step"]) == list(range(30))


def test_book_is_previous_minute():
    out = D.materialise(_features_df(), _minute_df(), _episodes_df())
    # decision at episode 0 step 0 (ts=BASE, minute index 0) -> book is minute index -1
    row0 = out[(out.episode_id == 0) & (out.step == 0)].iloc[0]
    assert row0["ask_px_1"] == 999.0  # 1000 + (-1)
    # decision at ts = BASE + 10 min (index 10) -> book is minute index 9
    r10 = out[out.ts == BASE + 10 * MS_PER_MINUTE].iloc[0]
    assert r10["ask_px_1"] == 1009.0


def test_no_look_ahead():
    feats, eps = _features_df(), _episodes_df()
    out1 = D.materialise(feats, _minute_df(), eps)

    m2 = _minute_df()
    m2.loc[m2.ts == BASE + 5 * MS_PER_MINUTE, "ask_px_1"] = 9999.0  # perturb minute 5
    out2 = D.materialise(feats, m2, eps)

    # decision at ts = BASE+5min uses minute 4 -> unchanged
    a = out1[out1.ts == BASE + 5 * MS_PER_MINUTE]["ask_px_1"].iloc[0]
    b = out2[out2.ts == BASE + 5 * MS_PER_MINUTE]["ask_px_1"].iloc[0]
    assert a == b == 1004.0
    # decision at ts = BASE+6min uses minute 5 -> reflects the perturbation
    assert out2[out2.ts == BASE + 6 * MS_PER_MINUTE]["ask_px_1"].iloc[0] == 9999.0


def test_integrity_and_temporal_order():
    out = D.materialise(_features_df(), _minute_df(), _episodes_df())
    rep = D.check_integrity(out)
    assert rep["nan_in_core_fields"] == 0
    assert rep["book_ordering_ok"] is True
    assert rep["train_rows"] == 30 and rep["test_rows"] == 30
    train, test = out[out.split == "train"], out[out.split == "test"]
    assert train["ts"].max() < test["ts"].min()


# --- finer-resolution (10s bars: 180 steps/episode, one-bar lag = 10s) -------------

def _minute_df_10s(idx_lo=-1, idx_hi=360):
    """One row per 10s bar; ask_px_1 = 1000 + bar_index so a book is traceable."""
    rows = []
    for idx in range(idx_lo, idx_hi):
        apx1 = 1000.0 + idx
        d = {"ts": BASE + idx * 10_000}
        for i in range(1, 21):
            d[f"ask_px_{i}"] = apx1 + (i - 1)
            d[f"ask_sz_{i}"] = 1.0
        d["bid_px_1"], d["bid_sz_1"], d["mid"] = apx1 - 1.0, 1.0, apx1 - 0.5
        rows.append(d)
    return pd.DataFrame(rows)


def _features_df_10s(n=360):
    ts = [BASE + i * 10_000 for i in range(n)]
    return pd.DataFrame({"ts": ts, "spread_bps": 0.2, "imbalance": 0.1,
                         "recent_return": 0.0, "rolling_vol": 0.001, "ask_depth": 5.0})


def test_materialise_at_10s_resolution():
    # 360 bars = two 30-min episodes of 180 bars; _episodes_df() spans the same 2 buckets.
    out = D.materialise(_features_df_10s(360), _minute_df_10s(), _episodes_df(), bar_seconds=10)
    assert (out.groupby("episode_id")["step"].size() == 180).all()
    assert list(out[out.episode_id == 0]["step"]) == list(range(180))   # step // bar_ms, not //60s
    # fill book is the PREVIOUS 10s bar (lag = one 10s bar, NOT 60s):
    row0 = out[(out.episode_id == 0) & (out.step == 0)].iloc[0]
    assert row0["ask_px_1"] == 999.0                       # bar 0 trades against bar -1
    r5 = out[out.ts == BASE + 5 * 10_000].iloc[0]
    assert r5["ask_px_1"] == 1004.0                        # bar 5 trades against bar 4
    D.check_integrity(out, bars_per_episode=180)           # 180-step check must pass


def test_materialise_rejects_bad_bar_seconds():
    with pytest.raises(ValueError):
        D.materialise(_features_df(), _minute_df(), _episodes_df(), bar_seconds=7)


def test_materialise_short_horizon_10min_at_10s():
    # Option-2 lever: a 10-minute horizon at 10s = 60 steps/episode (vs 180 at 30 min).
    EP10 = 10 * MS_PER_MINUTE
    base = (1_700_000_000_000 // EP10) * EP10
    n = 120                                                  # two 10-min episodes of 60 bars
    feats = pd.DataFrame({"ts": [base + i * 10_000 for i in range(n)], "spread_bps": 0.2,
                          "imbalance": 0.1, "recent_return": 0.0, "rolling_vol": 0.001, "ask_depth": 5.0})
    rows = []
    for idx in range(-1, n):
        apx1 = 1000.0 + idx
        d = {"ts": base + idx * 10_000}
        for i in range(1, 21):
            d[f"ask_px_{i}"], d[f"ask_sz_{i}"] = apx1 + (i - 1), 1.0
        d["bid_px_1"], d["bid_sz_1"], d["mid"] = apx1 - 1.0, 1.0, apx1 - 0.5
        rows.append(d)
    minute = pd.DataFrame(rows)
    eps = pd.DataFrame({"start_ts": [base, base + EP10], "episode_id": [0, 1],
                        "regime": ["calm", "volatile"], "split": ["train", "test"]})
    out = D.materialise(feats, minute, eps, bar_seconds=10, episode_minutes=10)
    assert (out.groupby("episode_id")["step"].size() == 60).all()      # 60 steps, not 180
    assert list(out[out.episode_id == 0]["step"]) == list(range(60))
    D.check_integrity(out, bars_per_episode=60)                          # 60-step check passes
