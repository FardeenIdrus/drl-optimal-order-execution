"""Unit tests for impact calibration.

Covers the OLS-through-origin coefficient recovery, the square-root fit gated on
ADV, and the relative-daily-vol helper. The deadline-residual *pricing* (continuity
and shape) is tested in test_real_data_env.py where the env applies it.

Run with:  PYTHONPATH=src .venv/bin/pytest tests/test_calibration.py
"""
import numpy as np
import pytest

from execution.data.features import FEATURE_COLUMNS
from execution.env.calibration import (
    _ols_through_origin,
    calibrate_temporary_impact,
    estimate_relative_daily_vol,
)
from execution.env.episode_store import EpisodeStore


def make_store(n_ep=4, n_steps=4, n_levels=12, mid=100.0):
    """Convex ladder: level i priced mid+i, size 1 -> cost(q) = sum_{i<=q} i."""
    E, T, L = n_ep, n_steps, n_levels
    midA = np.full((E, T), mid)
    px = np.tile(mid + 1.0 + np.arange(L), (E, T, 1))
    sz = np.ones((E, T, L))
    feats = np.zeros((E, T, len(FEATURE_COLUMNS)))
    regime = np.array(["calm" if e % 2 == 0 else "volatile" for e in range(E)], dtype=object)
    return EpisodeStore(np.arange(E), regime, midA, px, sz, feats, tuple(FEATURE_COLUMNS), T)


def test_ols_through_origin_recovers_known_coef():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    coef, r2 = _ols_through_origin(x, 0.37 * x)   # exact linear-through-origin
    assert coef == pytest.approx(0.37)
    assert r2 == pytest.approx(1.0)


def test_sqrt_fit_present_with_adv_absent_without():
    store = make_store()
    with_adv = calibrate_temporary_impact(store, [1, 2, 3, 4], adv=1000.0, n_samples=40, seed=1)
    no_adv = calibrate_temporary_impact(store, [1, 2, 3, 4], n_samples=40, seed=1)
    assert np.isfinite(with_adv.sqrt_coef) and with_adv.sqrt_coef > 0.0
    assert np.isnan(no_adv.sqrt_coef)
    # Same sampled fills (same seed) -> eta is identical whether or not ADV is given.
    assert with_adv.eta == pytest.approx(no_adv.eta)


def test_sqrt_coef_scales_with_adv():
    # frac_cost ~ sqrt_coef * sqrt(q/V): for fixed fills, sqrt_coef ∝ sqrt(V).
    store = make_store()
    a = calibrate_temporary_impact(store, [1, 2, 3, 4], adv=1000.0, n_samples=40, seed=3)
    b = calibrate_temporary_impact(store, [1, 2, 3, 4], adv=4000.0, n_samples=40, seed=3)
    assert b.sqrt_coef / a.sqrt_coef == pytest.approx(2.0, rel=1e-6)   # sqrt(4000/1000)=2


def test_calibrate_rejects_nonpositive_adv():
    with pytest.raises(ValueError):
        calibrate_temporary_impact(make_store(), [1, 2, 3], adv=0.0, n_samples=10)


def test_relative_daily_vol_positive_when_varied_zero_when_flat():
    flat = make_store(mid=100.0)
    assert estimate_relative_daily_vol(flat) == pytest.approx(0.0)
    varied = make_store()
    varied.mid[:] = np.array([100.0, 101.0, 100.0, 101.0])
    assert estimate_relative_daily_vol(varied) > 0.0


def test_relative_daily_vol_annualises_by_bar_width():
    # periods/day = 86_400/bar_seconds (1440 at 60s, 8640 at 10s). For the SAME store
    # (same per-bar std), the only difference is the annualisation factor, so the 10s
    # estimate must exceed the 60s one by exactly sqrt(8640/1440)=sqrt(6).
    s = make_store()
    s.mid[:] = np.array([100.0, 101.0, 100.0, 101.0])
    v60 = estimate_relative_daily_vol(s, bar_seconds=60)
    v10 = estimate_relative_daily_vol(s, bar_seconds=10)
    assert v10 / v60 == pytest.approx(np.sqrt(6.0), rel=1e-12)
    # default is the 60s case (back-compat with the rest of the suite)
    assert estimate_relative_daily_vol(s) == pytest.approx(v60, rel=1e-12)


def test_relative_daily_vol_is_resolution_invariant_for_diffusive_path():
    # The physical invariance the fix restores: a finer bar has a sqrt(6)-smaller
    # per-bar move, but sqrt(6)-more periods/day, so the recovered DAILY vol is the
    # SAME. (Same mean_mid in both, so the relative normalisation is identical.)
    coarse = make_store()
    coarse.mid[:] = np.array([100.0, 101.0, 100.0, 101.0])       # std(diff) = 1
    a = 0.5 / np.sqrt(6.0)
    fine = make_store()
    fine.mid[:] = np.array([100.5 - a, 100.5 + a, 100.5 - a, 100.5 + a])  # std(diff)=1/sqrt6, same mean
    d_coarse = estimate_relative_daily_vol(coarse, bar_seconds=60)
    d_fine = estimate_relative_daily_vol(fine, bar_seconds=10)
    assert d_fine == pytest.approx(d_coarse, rel=1e-9)


def test_relative_daily_vol_rejects_nonpositive_bar_seconds():
    with pytest.raises(ValueError):
        estimate_relative_daily_vol(make_store(), bar_seconds=0)
