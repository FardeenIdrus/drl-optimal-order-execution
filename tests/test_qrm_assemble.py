"""Unit tests for Stage 3e: assembling the calibration into the engine format.

Run with:  PYTHONPATH=src pytest tests/test_qrm_assemble.py
"""
import numpy as np

from execution.qrm.assemble import (
    assemble,
    build_rate_array,
    empirical_invariant,
    invariant_distribution,
    scale_market_to_rate,
)


def test_build_rate_array_transposes_to_engine_order():
    lam = np.arange(1 * 3 * 2 * 3, dtype=float).reshape(1, 3, 2, 3)  # (K,Q+1,side,type)
    r = build_rate_array(lam)
    assert r.shape == (2, 1, 3, 3)                      # (side,K,Q+1,type)
    assert r[0, 0, 1, 2] == lam[0, 1, 0, 2]             # value preserved through transpose


def test_invariant_distribution_is_valid_and_uniform_when_rho_one():
    # limit rate == cancel+market at every queue -> rho=1 -> uniform stationary dist
    K, Q1 = 1, 5
    lam = np.zeros((K, Q1, 3))
    lam[:, :, 0] = 2.0   # limit
    lam[:, :, 1] = 1.0   # cancel
    lam[:, :, 2] = 1.0   # market  (cancel+market = 2 == limit -> rho=1)
    pi = invariant_distribution(lam)
    assert pi.shape == (K, Q1)
    assert np.allclose(pi.sum(axis=1), 1.0)
    assert np.allclose(pi[0], 1.0 / Q1)                # uniform


def test_empirical_invariant_normalises_time_weights():
    time_in = np.zeros((1, 3, 2))
    time_in[0, :, 1] = [30.0, 60.0, 10.0]   # ask depth-1: 30% empty, 60% q1, 10% q2
    inv_bid, inv_ask = empirical_invariant(time_in)
    assert np.allclose(inv_ask[0], [0.3, 0.6, 0.1])
    assert inv_bid[0, 0] == 1.0             # never-observed side -> degenerate at q=0
    assert np.allclose(inv_bid.sum(axis=1), 1.0)


def test_scale_market_to_rate_hits_target():
    K, Q1 = 1, 3
    lam = np.zeros((K, Q1, 2, 3))
    lam[:, :, :, 2] = 2.0                       # market rate 2 in every state
    inv_b = np.array([[0.5, 0.5, 0.0]])
    inv_a = np.array([[0.0, 1.0, 0.0]])
    # expected = sum(pi * lam_M) over both sides = (0.5+0.5)*2 + 1.0*2 = 4.0
    scaled, f = scale_market_to_rate(lam, inv_b, inv_a, target_units_per_s=8.0)
    assert abs(f - 2.0) < 1e-12
    assert np.allclose(scaled[:, :, :, 2], 4.0)
    assert np.allclose(scaled[:, :, :, :2], lam[:, :, :, :2])   # limit/cancel untouched


def test_assemble_empirical_invariant_and_market_anchor():
    K, Q = 2, 4
    counts = np.zeros((K, Q + 1, 2, 3))
    time_in = np.ones((K, Q + 1, 2))
    counts[:, :, :, 0] = 5.0
    counts[:, :, :, 1] = 3.0
    counts[:, :, :, 2] = 2.0
    b = assemble(counts, time_in, aes=[0.3, 0.3], tick=1.0,
                 invariant="empirical", market_target_units_per_s=10.0)
    b.check()
    # empirical invariant of uniform time_in is uniform
    assert np.allclose(b.inv_ask, 1.0 / (Q + 1))
    # expected market rate under uniform pi: sum over 2 sides x K x (Q+1) states of pi*lam
    pi_lam = (1.0 / (Q + 1)) * b.rate_int_all[:, :, :, 2]
    assert abs(pi_lam.sum() - 10.0) < 1e-9


def test_assemble_produces_checked_bundle():
    K, Q = 3, 8
    counts = np.zeros((K, Q + 1, 2, 3))
    time_in = np.zeros((K, Q + 1, 2))
    # populate a plausible birth-death pattern so distributions are well-defined
    for i in range(K):
        for q in range(Q + 1):
            time_in[i, q, :] = 1.0
            counts[i, q, :, 0] = 5.0   # limit
            counts[i, q, :, 1] = 3.0   # cancel
            counts[i, q, :, 2] = 2.0   # market
    b = assemble(counts, time_in, aes=[0.3, 0.3, 0.3], tick=1.0)
    b.check()  # raises if malformed
    assert b.rate_int_all.shape == (2, K, Q + 1, 3)
    assert b.K == K and b.Q == Q
    assert (b.rate_int_all >= 0).all()
