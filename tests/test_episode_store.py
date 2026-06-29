"""Unit tests for the EpisodeStore loader (Phase 2 env).

Run with:  PYTHONPATH=src .venv/bin/pytest tests/test_episode_store.py
The real-data test is skipped automatically if the scratch dataset is absent.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from execution.data.features import FEATURE_COLUMNS
from execution.env.episode_store import EpisodeStore
from execution.env.normalize import FeatureNormalizer

# Dataset lives outside the repo (gitignored scratch dir, one level above code/).
_DS = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/dataset")
TEST_PARQUET = _DS / "test.parquet"
TRAIN_PARQUET = _DS / "train.parquet"


def _toy_frame(n_ep: int, n_steps: int, n_levels: int) -> pd.DataFrame:
    """A minimal but schema-faithful dataset: ascending ask ladder, unit sizes."""
    rows = []
    for e in range(n_ep):
        for s in range(n_steps):
            row = {
                "episode_id": e,
                "step": s,
                "regime": "calm" if e % 2 == 0 else "volatile",
                "mid": 100.0 + e,  # distinct arrival per episode
            }
            for lv in range(1, n_levels + 1):
                row[f"ask_px_{lv}"] = 100.0 + e + lv      # ascending
                row[f"ask_sz_{lv}"] = 1.0
            for f in FEATURE_COLUMNS:
                row[f] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_reshape_and_shapes(tmp_path):
    df = _toy_frame(n_ep=4, n_steps=5, n_levels=20)
    p = tmp_path / "toy.parquet"
    df.to_parquet(p)
    store = EpisodeStore.from_parquet(p, n_steps=5)

    assert store.n_episodes == 4
    assert store.n_levels == 20
    assert store.mid.shape == (4, 5)
    assert store.ask_px.shape == (4, 5, 20)
    assert store.features.shape == (4, 5, len(FEATURE_COLUMNS))
    # episode 2 -> mid 102, ask L1 = 100+2+1 = 103
    assert store.mid[2, 0] == pytest.approx(102.0)
    assert store.ask_px[2, 0, 0] == pytest.approx(103.0)
    assert list(store.regime) == ["calm", "volatile", "calm", "volatile"]


def test_ask_ladder_is_ascending(tmp_path):
    df = _toy_frame(n_ep=2, n_steps=3, n_levels=20)
    p = tmp_path / "toy.parquet"
    df.to_parquet(p)
    store = EpisodeStore.from_parquet(p, n_steps=3)
    diffs = np.diff(store.ask_px, axis=2)
    assert (diffs > 0).all()


def test_rejects_wrong_row_count(tmp_path):
    df = _toy_frame(n_ep=2, n_steps=5, n_levels=20).iloc[:-1]  # drop one row
    p = tmp_path / "bad.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError, match="multiple of n_steps"):
        EpisodeStore.from_parquet(p, n_steps=5)


def test_handles_unsorted_input(tmp_path):
    df = _toy_frame(n_ep=3, n_steps=4, n_levels=20).sample(frac=1.0, random_state=0)
    p = tmp_path / "shuffled.parquet"
    df.to_parquet(p)
    store = EpisodeStore.from_parquet(p, n_steps=4)  # must sort internally
    assert store.n_episodes == 3
    assert store.ask_px[1, 0, 0] == pytest.approx(102.0)  # ep1 L1 = 100+1+1


def _varying_feature_store(n_ep=20, n_steps=3, n_levels=4):
    """Store whose features equal the episode index (so split distributions differ)."""
    E, T, L = n_ep, n_steps, n_levels
    regime = np.array(["calm" if e % 2 == 0 else "volatile" for e in range(E)], dtype=object)
    mid = np.full((E, T), 100.0)
    px = np.tile(101.0 + np.arange(L), (E, T, 1))
    sz = np.ones((E, T, L))
    feats = np.zeros((E, T, len(FEATURE_COLUMNS)))
    for e in range(E):
        feats[e, :, :] = float(e)
    return EpisodeStore(np.arange(E), regime, mid, px, sz, feats,
                        tuple(FEATURE_COLUMNS), T)


def test_chrono_split_sizes_disjoint_and_tail():
    full = _varying_feature_store(n_ep=10)
    train_sub, val = full.chrono_split(0.2)
    assert (train_sub.n_episodes, val.n_episodes) == (8, 2)
    assert train_sub.n_episodes + val.n_episodes == full.n_episodes
    # val is the chronological TAIL (last episodes); no overlap.
    assert list(val.episode_ids) == [8, 9]
    assert set(train_sub.episode_ids).isdisjoint(val.episode_ids)
    assert train_sub.episode_ids.max() < val.episode_ids.min()


def test_chrono_split_keeps_at_least_one_each():
    full = _varying_feature_store(n_ep=4)
    train_sub, val = full.chrono_split(0.001)   # rounds toward 0 -> clamp to 1 val
    assert val.n_episodes == 1 and train_sub.n_episodes == 3


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_chrono_split_rejects_bad_frac(bad):
    with pytest.raises(ValueError):
        _varying_feature_store().chrono_split(bad)


def test_normalizer_on_train_sub_differs_from_full():
    # The leakage-free choice: fit the normalizer on the training sub-split only.
    full = _varying_feature_store(n_ep=20)
    train_sub, _ = full.chrono_split(0.3)
    n_full = FeatureNormalizer.fit(full.features, full.feature_names)
    n_sub = FeatureNormalizer.fit(train_sub.features, train_sub.feature_names)
    assert not np.allclose(n_full.center, n_sub.center)   # excludes recent episodes


@pytest.mark.skipif(not (TRAIN_PARQUET.exists() and TEST_PARQUET.exists()),
                    reason="scratch dataset not present")
def test_chrono_split_real_data_is_leakage_free():
    train = EpisodeStore.from_parquet(TRAIN_PARQUET)
    test = EpisodeStore.from_parquet(TEST_PARQUET)
    train_sub, val = train.chrono_split(0.15)
    assert train_sub.n_episodes + val.n_episodes == train.n_episodes
    # chronological + leakage-free: train_sub < val < test by (chronological) episode_id
    assert train_sub.episode_ids.max() < val.episode_ids.min()
    assert val.episode_ids.max() < test.episode_ids.min()


@pytest.mark.skipif(not TEST_PARQUET.exists(), reason="scratch dataset not present")
def test_loads_real_test_dataset():
    store = EpisodeStore.from_parquet(TEST_PARQUET)
    assert store.n_episodes == 6551
    assert store.n_steps == 30
    assert store.n_levels == 20
    assert set(np.unique(store.regime)) <= {"calm", "volatile"}
    # book ordering holds on level-1 across all rows (bid < ask handled upstream)
    finite = np.isfinite(store.ask_px[:, :, 0])
    assert finite.all()
    # ask ladder ascending on the first 5 levels (ignoring trailing gaps)
    diffs = np.diff(store.ask_px[:, :, :5], axis=2)
    assert (diffs[np.isfinite(diffs)] > 0).all()
