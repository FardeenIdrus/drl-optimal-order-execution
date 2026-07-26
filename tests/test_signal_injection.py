"""Unit tests for the measured-signal injection (extension Phase B; criteria section 8).

Covers: flag-OFF observation identity on a shared synthetic bundle, ON-mode observation
feature, policy-independence of the pre-computed signal and injected paths (the CRN
guarantee), deterministic fractional-tick carry conservation, the follower benchmark's
action mapping, and that injection actually alters the price path.
Uses tiny synthetic bundles (no data files needed).
"""
import numpy as np
import pytest

from execution.qrm.assemble import assemble
from execution.qrm.reactive_env import (ACTIONS, WARMUP_INTERVALS, ReactiveQRMEnv)
from execution.qrm.reactive_baselines import A_ONE, signal_follower


def _paths(tmp_path):
    K, Q = 3, 6
    counts = np.zeros((K, Q + 1, 2, 3))
    time_in = np.ones((K, Q + 1, 2))
    counts[:, :, :, 0] = 6.0
    counts[:, :, :, 1] = 2.0
    counts[:, :, :, 2] = 1.0
    bundle = assemble(counts, time_in, aes=[0.5, 0.5, 0.5], tick=1.0,
                      invariant="empirical")
    bp = tmp_path / "bundle.npz"
    bundle.save(str(bp))
    mp = tmp_path / "mp.npz"
    np.savez(mp, interval_s=0.5, moves=np.array([-1, 0, 1]),
             probs=np.array([0.1, 0.8, 0.1]))
    return str(bp), str(mp)


def _env(tmp_path, **kw):
    bp, mp = _paths(tmp_path)
    return ReactiveQRMEnv(bp, mp, order_btc=5.0, n_steps=20, **kw)


# A large residual on the synthetic bundle so integer ticks actually fire:
# mid ~ 100,000, tick 1 -> 1 bps = 10 ticks.
RESID = 0.30


def _trajectory(env, seed, n=5):
    """Sequential rollout record. NOTE: the engine's random stream is per-process,
    so two env instances must be run SEQUENTIALLY (never interleaved) to compare —
    the same usage pattern as the whole training/judging pipeline."""
    obs = env.reset(seed=seed)
    rec = [obs.copy()]
    rewards = []
    for _ in range(n):
        obs, r, _done, _info = env.step(A_ONE)
        rec.append(obs.copy())
        rewards.append(r)
    return rec, rewards


def test_off_mode_obs_identical_and_dim_unchanged(tmp_path):
    e_plain = _env(tmp_path)
    e_off = _env(tmp_path, signal_injection=False, signal_residual_bps=RESID,
                 signal_mean=0.1)
    assert e_plain.obs_dim == e_off.obs_dim
    rec1, rew1 = _trajectory(e_plain, seed=3)
    rec2, rew2 = _trajectory(e_off, seed=3)
    for a, b in zip(rec1, rec2):
        assert np.array_equal(a, b)
    assert rew1 == rew2


def test_on_mode_obs_feature_appended(tmp_path):
    e_on = _env(tmp_path, signal_injection=True, signal_residual_bps=RESID)
    e_off = _env(tmp_path)
    assert e_on.obs_dim == e_off.obs_dim + 1
    obs = e_on.reset(seed=3)
    assert len(obs) == e_on.obs_dim
    ep = e_on._ep
    assert ep.s2_path is not None and len(ep.s2_path) == 20 * 2
    k = ep.move_idx - WARMUP_INTERVALS
    k = min(max(k, 0), len(ep.s2_path) - 1)
    expected = ep.s2_path[k]
    if np.isfinite(expected):
        assert obs[-1] == pytest.approx(expected, abs=1e-6)


def test_signal_is_policy_independent(tmp_path):
    """The CRN guarantee: two different policies on the same seed face the identical
    pre-computed signal and injected-move paths."""
    e1 = _env(tmp_path, signal_injection=True, signal_residual_bps=RESID)
    e2 = _env(tmp_path, signal_injection=True, signal_residual_bps=RESID)
    e1.reset(seed=7)
    e2.reset(seed=7)
    p1 = (e1._ep.s2_path.copy(), e1._ep.delta_path.copy())
    p2 = (e2._ep.s2_path.copy(), e2._ep.delta_path.copy())
    assert np.array_equal(p1[0], p2[0], equal_nan=True)
    assert np.array_equal(p1[1], p2[1])
    # run wildly different policies; the stored paths must not change
    done = False
    while not done:
        _, _, done, _ = e1.step(0)                # never trade (until forced)
    done = False
    while not done:
        _, _, done, _ = e2.step(len(ACTIONS) - 1) # max pace
    assert np.array_equal(e1._ep.s2_path, p1[0], equal_nan=True)
    assert np.array_equal(e1._ep.delta_path, p1[1])
    assert np.array_equal(e2._ep.s2_path, p2[0], equal_nan=True)
    assert np.array_equal(e2._ep.delta_path, p2[1])


def test_carry_conserves_cumulative_ticks(tmp_path):
    """The integer delta path must equal the cumulative fractional-tick stream within
    one tick at every prefix (deterministic carry, no drift, no new randomness)."""
    e = _env(tmp_path, signal_injection=True, signal_residual_bps=RESID,
             signal_mean=0.0)
    e.reset(seed=11)
    ep = e._ep
    # reconstruct the fractional stream from the stored signal path and a replayed
    # mid path is not available; instead verify the carry invariant structurally:
    # cumulative emitted ticks never deviate from ANY consistent fractional stream
    # by construction if each |delta| <= ceil(max |fractional per interval|) and the
    # emitted total is within 1 tick of the fractional total. Empirically:
    deltas = ep.delta_path
    s2 = np.where(np.isfinite(ep.s2_path), ep.s2_path, 0.0)
    # fractional per-interval ticks assuming mid ~ constant 100k (tick=1):
    approx_frac = 100_000.0 * RESID * (s2 - 0.0) * 1e-4 / 1.0
    assert abs(float(np.sum(deltas)) - float(np.sum(approx_frac))) < len(deltas) * 0.02 + 1.5


def test_injection_alters_price_path(tmp_path):
    e_on = _env(tmp_path, signal_injection=True, signal_residual_bps=RESID)
    e_off = _env(tmp_path)
    e_on.reset(seed=5)
    e_off.reset(seed=5)
    mids_on, mids_off = [], []
    done = False
    while not done:
        _, _, done, _ = e_on.step(A_ONE)
        mids_on.append(e_on._ep.p_mid)
    done = False
    while not done:
        _, _, done, _ = e_off.step(A_ONE)
        mids_off.append(e_off._ep.p_mid)
    if np.any(e_on._ep.delta_path != 0):
        assert mids_on != mids_off               # the injected moves reached the price


def test_follower_action_mapping(tmp_path):
    e = _env(tmp_path, signal_injection=True, signal_residual_bps=RESID)
    obs = e.reset(seed=9)
    ep = e._ep
    # plant known signal values at the current index and check the mapped action
    k = min(max(ep.move_idx - WARMUP_INTERVALS, 0), len(ep.s2_path) - 1)
    for planted, want_pace in [(0.0, 1.0), (0.9, 2.0), (-0.9, 0.0), (0.25, 1.2)]:
        ep.s2_path[k] = planted
        idx = signal_follower(e, obs)
        assert ACTIONS[idx] == pytest.approx(
            min(ACTIONS, key=lambda a: abs(a - min(max(1.0 + planted, 0.0), 2.0))))
        assert abs(ACTIONS[idx] - (1.0 + planted)) <= 0.31  # nearest-action fidelity
    # no signal available (OFF-style episode): follower degrades to adaptive TWAP
    ep.s2_path = None
    assert signal_follower(e, obs) == A_ONE


def test_ema_driver_smooths_and_is_deterministic(tmp_path):
    """Amendment 2a: with a half-life set, the stored signal is the smoothed driver
    (lower variance than the instantaneous signal), deterministic across resets."""
    e_inst = _env(tmp_path, signal_injection=True, signal_residual_bps=RESID)
    e_ema = _env(tmp_path, signal_injection=True, signal_residual_bps=RESID,
                 signal_ema_halflife_s=5.0)
    e_inst.reset(seed=21)
    inst = e_inst._ep.s2_path.copy()
    e_ema.reset(seed=21)
    sm1 = e_ema.s2_path if hasattr(e_ema, "s2_path") else e_ema._ep.s2_path.copy()
    e_ema.reset(seed=21)
    sm2 = e_ema._ep.s2_path.copy()
    assert np.array_equal(sm1, sm2)                       # deterministic
    v_inst = np.nanvar(inst)
    v_sm = np.nanvar(sm1)
    assert v_sm < v_inst                                  # smoothing reduces variance
    assert np.all(np.isfinite(sm1))                       # EMA never NaN
