"""Unit tests for the Amendment A3 comparator machinery (execution.qrm.ac_vwap)."""
import numpy as np
import pytest

from execution.qrm.ac_vwap import (KAPPA_T_GRID, _remaining_fraction, make_ac,
                                   make_schedule_policy, make_vwap_expected)
from execution.qrm.reactive_baselines import ACTIONS


class _StubEp:
    def __init__(self, step_idx, remaining):
        self.step_idx = step_idx
        self.remaining_btc = remaining


class _StubEnv:
    n_steps = 300
    order_btc = 25.0

    def __init__(self, step_idx, remaining):
        self._ep = _StubEp(step_idx, remaining)


def test_remaining_fraction_endpoints_and_monotone():
    for kT in KAPPA_T_GRID:
        r = _remaining_fraction(kT, 300)
        assert r[0] == pytest.approx(1.0)
        assert r[-1] == pytest.approx(0.0, abs=1e-12)
        assert np.all(np.diff(r) < 1e-15)          # strictly non-increasing


def test_kappa_zero_is_uniform():
    r = _remaining_fraction(0.0, 300)
    assert np.allclose(r, 1.0 - np.arange(301) / 300.0)


def test_front_loading_monotone_in_kappa():
    executed_half = [1.0 - _remaining_fraction(kT, 300)[150] for kT in KAPPA_T_GRID]
    assert all(a < b for a, b in zip(executed_half, executed_half[1:]))
    # kT=0 executes exactly half by halftime; urgency executes more
    assert executed_half[0] == pytest.approx(0.5)
    assert executed_half[-1] > 0.85


def test_vwap_expected_schedule_is_uniform_line():
    pol_v = make_vwap_expected(order_btc=25.0, T=300)
    pol_0 = make_ac(0.0, order_btc=25.0, T=300)
    # identical decisions across a sweep of states (both track the uniform line)
    for j in (0, 1, 57, 150, 298):
        for rem_frac in (1.0, 0.9, 0.5, 0.2, 0.01):
            env = _StubEnv(j, 25.0 * rem_frac)
            assert pol_v(env, None) == pol_0(env, None)


def test_on_schedule_policy_picks_one_x():
    # exactly on the uniform line at step j -> remaining = X*(T-j)/T -> action 1.0x
    pol = make_ac(0.0, order_btc=25.0, T=300)
    for j in (0, 10, 150, 299):
        env = _StubEnv(j, 25.0 * (300 - j) / 300.0)
        assert ACTIONS[pol(env, None)] == 1.0


def test_rate_request_delivers_the_schedule_slice():
    # the registered emulation: requested quantity ~ the schedule's own slice, whatever
    # the drift (the env's carry handles granularity; NO line-correction)
    pol = make_ac(0.0, order_btc=25.0, T=300)
    slice_uniform = 25.0 / 300.0
    j = 150
    for rem in (25.0 * 0.5 * 1.4, 25.0 * 0.5, 25.0 * 0.5 * 0.6):   # behind/on/ahead
        env = _StubEnv(j, rem)
        pace = rem / (300 - j)
        a = ACTIONS[pol(env, None)]
        # chosen action is the grid's best approximation of the slice
        best = min(ACTIONS, key=lambda x: abs(x * pace - slice_uniform))
        assert a == best


def test_kappa_zero_matches_fixed_twap_policy_exactly():
    from execution.qrm.reactive_baselines import make_fixed_twap
    pol0 = make_ac(0.0, order_btc=25.0, T=300)
    for j in (0, 1, 57, 150, 298):
        for rem_frac in (1.0, 0.9, 0.5, 0.2, 0.01):
            env = _StubEnv(j, 25.0 * rem_frac)
            pol_fx = make_fixed_twap(env)
            assert pol0(env, None) == pol_fx(env, None)


def test_urgent_ac_frontloads_relative_to_uniform():
    pol4 = make_ac(4.0, order_btc=25.0, T=300)
    env = _StubEnv(0, 25.0)                 # start of episode, full order
    assert ACTIONS[pol4(env, None)] > 1.0   # urgency requests a bigger early slice


def test_schedule_policy_zero_when_done():
    pol = make_schedule_policy(_remaining_fraction(1.0, 300), 25.0)
    env = _StubEnv(100, 0.0)
    assert ACTIONS[pol(env, None)] == 0.0
