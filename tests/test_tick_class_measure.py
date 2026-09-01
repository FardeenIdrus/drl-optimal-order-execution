"""Integrity gates for the tick-class measurement (criteria section 9).

Registered gates: causality, determinism, planted-signal recovery. All must pass before any
number from tick_class_measure is quoted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reports" / "diagnostics"))
from tick_class_measure import (  # noqa: E402
    AucAccum,
    auc_from_histograms,
    bin_index,
    infer_tick,
    queue_imbalance,
)


# ------------------------------------------------------------------ predictor definition
def test_imbalance_matches_registered_formula():
    bid = np.array([3.0, 1.0, 0.0, 2.0])
    ask = np.array([1.0, 3.0, 4.0, 2.0])
    got = queue_imbalance(bid, ask)
    np.testing.assert_allclose(got, [0.5, -0.5, -1.0, 0.0])


def test_imbalance_is_nan_when_both_sides_swept():
    got = queue_imbalance(np.array([0.0]), np.array([0.0]))
    assert np.isnan(got[0])


def test_imbalance_is_bounded():
    rng = np.random.default_rng(0)
    bid, ask = rng.random(10_000) * 50, rng.random(10_000) * 50
    got = queue_imbalance(bid, ask)
    assert np.nanmax(np.abs(got)) <= 1.0 + 1e-12


def test_predictor_is_causal_elementwise():
    """The predictor at t is a function of t only: perturbing any later row cannot change it."""
    rng = np.random.default_rng(1)
    bid, ask = rng.random(200) + 0.1, rng.random(200) + 0.1
    base = queue_imbalance(bid, ask)
    bid2, ask2 = bid.copy(), ask.copy()
    bid2[100:] += 5.0
    ask2[100:] *= 0.1
    after = queue_imbalance(bid2, ask2)
    np.testing.assert_array_equal(base[:100], after[:100])


# ------------------------------------------------------------------------------ AUC maths
def test_auc_perfect_separation_is_one():
    pos = np.zeros(10)
    pos[9] = 5.0
    neg = np.zeros(10)
    neg[0] = 5.0
    assert auc_from_histograms(pos, neg) == pytest.approx(1.0)


def test_auc_identical_distributions_is_half():
    pos = np.array([1.0, 1.0, 1.0])
    neg = np.array([1.0, 1.0, 1.0])
    assert auc_from_histograms(pos, neg) == pytest.approx(0.5)


def test_auc_reversed_is_complement():
    pos = np.array([3.0, 1.0, 0.0])
    neg = np.array([0.0, 1.0, 3.0])
    a = auc_from_histograms(pos, neg)
    b = auc_from_histograms(neg, pos)
    assert a + b == pytest.approx(1.0)


def test_auc_empty_class_is_nan():
    assert np.isnan(auc_from_histograms(np.zeros(4), np.array([1.0, 0, 0, 0])))


def test_auc_matches_sklearn_style_reference():
    """Histogram AUC equals the rank-based Mann-Whitney value on the same data."""
    rng = np.random.default_rng(7)
    n = 20_000
    y = rng.random(n) < 0.5
    score = np.where(y, rng.normal(0.4, 1.0, n), rng.normal(-0.4, 1.0, n))
    score = np.clip(score, -1, 1)

    acc = AucAccum()
    acc.add(score, np.where(y, 1.0, -1.0))
    got = acc.result()["auc"]

    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    s_sorted = score[order]
    i = 0
    while i < n:                                   # average ranks within ties
        j = i
        while j + 1 < n and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    npos, nneg = y.sum(), (~y).sum()
    ref = (ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    assert got == pytest.approx(ref, abs=2e-3)     # binning tolerance only


# ------------------------------------------------------------- planted-signal recovery
def test_planted_signal_is_recovered():
    """A series where imbalance predicts direction with known strength recovers a high AUC;
    a series where it carries no information recovers ~0.5."""
    rng = np.random.default_rng(11)
    n = 200_000
    imb = rng.uniform(-1, 1, n)

    informative = np.sign(imb + rng.normal(0, 0.25, n))
    acc = AucAccum()
    acc.add(imb, informative)
    assert acc.result()["auc"] > 0.90

    noise = np.sign(rng.normal(0, 1, n))
    acc2 = AucAccum()
    acc2.add(imb, noise)
    assert abs(acc2.result()["auc"] - 0.5) < 0.01


def test_no_change_intervals_are_excluded_not_counted():
    imb = np.array([0.5, -0.5, 0.9, -0.9])
    fwd = np.array([1.0, -1.0, 0.0, 0.0])
    acc = AucAccum()
    acc.add(imb, fwd)
    r = acc.result()
    assert r["n"] == 2
    assert r["n_excluded_no_change"] == 2


def test_nan_predictor_rows_are_dropped():
    imb = np.array([np.nan, 0.5, -0.5])
    fwd = np.array([1.0, 1.0, -1.0])
    acc = AucAccum()
    acc.add(imb, fwd)
    assert acc.result()["n"] == 2


# ------------------------------------------------------------------------- determinism
def test_accumulation_is_order_independent_and_deterministic():
    rng = np.random.default_rng(3)
    imb = rng.uniform(-1, 1, 5_000)
    fwd = np.sign(imb + rng.normal(0, 0.5, 5_000))

    one = AucAccum()
    one.add(imb, fwd)
    chunked = AucAccum()
    for s in range(0, 5_000, 137):
        chunked.add(imb[s:s + 137], fwd[s:s + 137])
    assert one.result()["auc"] == pytest.approx(chunked.result()["auc"])
    assert one.result()["n"] == chunked.result()["n"]

    again = AucAccum()
    again.add(imb, fwd)
    assert again.result()["auc"] == one.result()["auc"]


def test_bin_index_is_monotone_and_in_range():
    x = np.linspace(-1, 1, 5_000)
    idx = bin_index(x)
    assert idx.min() >= 0 and idx.max() <= 2000
    assert np.all(np.diff(idx) >= 0)


# --------------------------------------------------------------------------- tick helper
def test_infer_tick_finds_the_grid_step():
    px = np.array([100.0, 101.0, 103.0, 107.0, 100.0])
    assert infer_tick(px) == pytest.approx(1.0)


def test_infer_tick_ignores_nan():
    px = np.array([100.0, np.nan, 100.5, 101.0])
    assert infer_tick(px) == pytest.approx(0.5)
