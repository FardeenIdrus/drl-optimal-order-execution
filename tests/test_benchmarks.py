"""Unit tests for the core execution benchmarks (Phase 3).

Run with:  PYTHONPATH=src .venv/bin/pytest tests/test_benchmarks.py
"""
import numpy as np
import pytest

from execution.data.features import FEATURE_COLUMNS
from execution.env.benchmarks import (
    AlmgrenChriss,
    FixedLiquidityParticipation,
    TWAP,
)
from execution.env.calibration import calibrate_temporary_impact, estimate_volatility
from execution.env.episode_store import EpisodeStore
from execution.env.real_data_env import RealDataExecutionEnv


def make_store(n_ep=2, n_steps=4, n_levels=10, mid=100.0):
    """Synthetic store: ascending unit-size ladder (mid+1, mid+2, ...)."""
    E, T, L = n_ep, n_steps, n_levels
    midA = np.full((E, T), float(mid))
    ask_px = np.tile(mid + np.arange(1, L + 1, dtype=float), (E, T, 1))
    ask_sz = np.ones((E, T, L), float)
    feats = np.zeros((E, T, len(FEATURE_COLUMNS)))
    regime = np.array(["calm" if e % 2 == 0 else "volatile" for e in range(E)], dtype=object)
    return EpisodeStore(np.arange(E), regime, midA, ask_px, ask_sz, feats,
                        tuple(FEATURE_COLUMNS), T)


# ----------------------------------------------------------------- Almgren-Chriss
def test_ac_kappa_zero_is_exactly_twap():
    ac = AlmgrenChriss.from_kappa(30.0, 30, kappa=0.0)
    assert ac.child_orders.sum() == pytest.approx(30.0)
    assert np.allclose(ac.child_orders, 30.0 / 30)        # uniform == TWAP


def test_ac_urgency_zero_is_twap():
    ac = AlmgrenChriss.from_urgency(30.0, 30, kappa_t=0.0)
    assert np.allclose(ac.child_orders, 1.0)


def test_ac_front_loads_with_urgency():
    ac = AlmgrenChriss.from_urgency(30.0, 30, kappa_t=3.0)
    assert ac.child_orders.sum() == pytest.approx(30.0)
    assert ac.child_orders[0] > 30.0 / 30                 # first slice bigger than TWAP
    assert ac.child_orders[0] > ac.child_orders[-1]       # front-loaded
    assert np.all(np.diff(ac.child_orders) <= 1e-9)       # monotone non-increasing


def test_ac_schedule_sums_to_inventory_for_various_kappa():
    for k in [0.0, 0.05, 0.1, 0.3]:
        ac = AlmgrenChriss.from_kappa(12.5, 20, kappa=k)
        assert ac.child_orders.sum() == pytest.approx(12.5)
        assert len(ac.child_orders) == 20


def test_kappa_from_lambda_monotone_and_validates():
    f = AlmgrenChriss.kappa_from_lambda
    assert f(0.0, 1.0, 1.0) == 0.0
    assert f(1.0, 2.0, 1.0) > f(1.0, 1.0, 1.0)            # higher sigma
    assert f(2.0, 1.0, 1.0) > f(1.0, 1.0, 1.0)            # higher lambda
    with pytest.raises(ValueError):
        f(1.0, 1.0, 0.0)


def test_ac_drives_env_and_fully_executes():
    store = make_store(n_ep=1, n_steps=4, n_levels=20)
    env = RealDataExecutionEnv(store, initial_inventory=4.0, test_mode=True)
    ac = AlmgrenChriss.from_urgency(4.0, 4, kappa_t=2.0)
    env.set_absolute_quantity(True)
    env.reset(options={"episode_index": 0})
    ac.reset(env)
    done = False
    while not done:
        _, _, done, _, info = env.step(ac.action())
    assert info["inventory"] == pytest.approx(0.0)


# ----------------------------------------------------- fixed liquidity participation
def test_participation_uses_current_top_level_depth():
    store = make_store(n_ep=1, n_steps=3, n_levels=5)     # unit size per level
    env = RealDataExecutionEnv(store, initial_inventory=10.0)
    env.set_absolute_quantity(True)
    env.reset(options={"episode_index": 0})
    pol = FixedLiquidityParticipation(fraction=0.5, levels=1)
    pol.reset(env)
    assert pol.action() == pytest.approx(0.5)             # 0.5 * top-1 depth (1.0)


def test_participation_sums_depth_over_levels():
    store = make_store(n_ep=1, n_steps=3, n_levels=5)
    env = RealDataExecutionEnv(store, initial_inventory=10.0)
    env.set_absolute_quantity(True)
    env.reset(options={"episode_index": 0})
    pol = FixedLiquidityParticipation(fraction=0.5, levels=3)
    pol.reset(env)
    assert pol.action() == pytest.approx(1.5)             # 0.5 * top-3 depth (3.0)


def test_participation_validates_fraction():
    with pytest.raises(ValueError):
        FixedLiquidityParticipation(fraction=0.0)
    with pytest.raises(ValueError):
        FixedLiquidityParticipation(fraction=1.5)


# ------------------------------------------------------------------- calibration
def test_estimate_volatility_zero_when_flat_positive_when_varied():
    flat = make_store(mid=100.0)
    assert estimate_volatility(flat) == pytest.approx(0.0)
    varied = make_store()
    varied.mid[:] = np.array([100.0, 101.0, 100.0, 101.0])  # within-episode swings
    assert estimate_volatility(varied) > 0.0


def test_calibrate_temporary_impact_fits_convex_cost():
    # ladder cost(q) = sum_{i=1}^q i = q(q+1)/2 -> well-approximated by eta*q^2
    store = make_store(n_ep=4, n_steps=4, n_levels=12)
    cal = calibrate_temporary_impact(store, q_grid=[1, 2, 3, 4, 5], n_samples=40, seed=1)
    assert cal.eta > 0.0
    assert 0.9 < cal.r2_linear <= 1.0


def test_calibrate_impact_per_regime_runs():
    store = make_store(n_ep=4, n_steps=4, n_levels=12)
    cal_calm = calibrate_temporary_impact(store, q_grid=[1, 2, 3], n_samples=20,
                                          regime="calm", seed=2)
    assert cal_calm.eta > 0.0
