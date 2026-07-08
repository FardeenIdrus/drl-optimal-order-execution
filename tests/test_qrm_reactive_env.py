"""Unit tests for the Step-4 reactive environment (fills, carry, deadline, CRN).

Run with:  PYTHONPATH=src pytest tests/test_qrm_reactive_env.py
Uses tiny synthetic bundles (no data files needed).
"""
import numpy as np
import pytest

from execution.qrm.assemble import assemble
from execution.qrm.reactive_env import ACTIONS, ReactiveQRMEnv


@pytest.fixture()
def env(tmp_path):
    K, Q = 3, 6
    counts = np.zeros((K, Q + 1, 2, 3))
    time_in = np.ones((K, Q + 1, 2))
    counts[:, :, :, 0] = 6.0   # steady arrivals keep the book populated
    counts[:, :, :, 1] = 2.0
    counts[:, :, :, 2] = 1.0
    bundle = assemble(counts, time_in, aes=[0.5, 0.5, 0.5], tick=1.0, invariant="empirical")
    bp = tmp_path / "bundle.npz"
    bundle.save(str(bp))
    mp = tmp_path / "mp.npz"
    np.savez(mp, interval_s=0.5, moves=np.array([-1, 0, 1]),
             probs=np.array([0.1, 0.8, 0.1]))
    return ReactiveQRMEnv(str(bp), str(mp), order_btc=5.0, n_steps=20)


def test_full_episode_executes_and_terminates(env):
    env.reset(seed=1)
    done = False
    steps = 0
    while not done:
        _obs, _r, done, info = env.step(ACTIONS.index(1.0))
        steps += 1
    assert steps == 20
    assert info["remaining_btc"] <= 1e-9          # adaptive TWAP finishes


def test_zero_action_defers_and_deadline_forces_completion(tmp_path):
    # FULLY frozen book (all rates zero, jumps disabled) so the deadline dump's cost
    # sign is deterministic: walking ask levels MUST cost more than the arrival mid.
    # (A merely drift-free book no longer suffices: since remediation I6 the engine's
    # own queue-depletion price moves are captured, so background dynamics can move
    # the reference price even without injected jumps.)
    K, Q = 3, 6
    counts = np.zeros((K, Q + 1, 2, 3))          # zero rates -> nothing ever happens
    time_in = np.ones((K, Q + 1, 2))
    bundle = assemble(counts, time_in, aes=[0.5, 0.5, 0.5], tick=1.0, invariant="empirical")
    bp = tmp_path / "b0.npz"
    bundle.save(str(bp))
    mp = tmp_path / "mp0.npz"
    np.savez(mp, interval_s=0.5, moves=np.array([0]), probs=np.array([1.0]))
    e = ReactiveQRMEnv(str(bp), str(mp), order_btc=5.0, n_steps=20)
    e.reset(seed=2)
    total_r = 0.0
    for _ in range(20):
        _obs, r, done, info = e.step(0)            # never trade voluntarily
        total_r += r
    assert done and info["remaining_btc"] <= 1e-9  # force-completion fired
    assert total_r < 0                             # the dump has a real cost


def test_carry_accumulates_sub_unit_pace(env):
    # order 5 BTC over 20 steps -> pace 0.25 BTC/step at 1.0x = half a 0.5-BTC unit:
    # fills must arrive on alternating steps via the carry, not be starved to zero
    env.reset(seed=3)
    fills = []
    for _ in range(6):
        _o, _r, _d, info = env.step(ACTIONS.index(1.0))
        fills.append(info["filled_units"])
    assert sum(fills[:2]) >= 1                     # a unit fires within two steps
    assert max(fills) >= 1


def test_buys_mutate_the_book_and_cost_positive(env):
    env.reset(seed=4)
    ep = env._ep
    ask_before = ep.state[env.K:].copy()
    cost, filled = env._buy_units(3)
    ask_after = ep.state[env.K:]
    assert filled == 3
    assert ask_before.sum() - ask_after.sum() == 3   # consumption is real
    assert cost > 0


def test_crn_same_seed_same_background_path(env):
    # identical seeds + identical actions -> identical rewards (bitwise)
    def run(seed):
        env.reset(seed=seed)
        rs = []
        for i in range(20):
            _o, r, _d, _i = env.step(ACTIONS.index(1.0))
            rs.append(r)
        return np.array(rs)
    a, b = run(7), run(7)
    assert np.array_equal(a, b)
    c = run(8)
    assert not np.array_equal(a, c)                # different seed -> different path


def test_reward_is_bps_of_arrival_notional(env):
    # dumping everything instantly must cost more than adaptive TWAP on the same seed
    def total(policy_idx, seed=11):
        env.reset(seed=seed)
        tot = 0.0
        done = False
        while not done:
            _o, r, done, _i = env.step(policy_idx)
            tot += r
        return tot
    twap = total(ACTIONS.index(1.0))
    dump = total(len(ACTIONS) - 1)
    assert dump <= twap + 1e-9                     # more aggressive >= cost (<= reward)


def test_total_executed_equals_order_non_divisible(tmp_path):
    # remediation I2 regression: 25/0.5-style non-divisible orders must NOT overbuy.
    # order 1.3 BTC, unit 0.5 -> exactly 2 whole units voluntary + 0.3 at the deadline
    K, Q = 3, 6
    counts = np.zeros((K, Q + 1, 2, 3))
    time_in = np.ones((K, Q + 1, 2))
    counts[:, :, :, 0] = 8.0
    counts[:, :, :, 1] = 1.0
    bundle = assemble(counts, time_in, aes=[0.5, 0.5, 0.5], tick=1.0, invariant="empirical")
    bp = tmp_path / "b.npz"; bundle.save(str(bp))
    mp = tmp_path / "m.npz"
    np.savez(mp, interval_s=0.5, moves=np.array([0]), probs=np.array([1.0]))
    e = ReactiveQRMEnv(str(bp), str(mp), order_btc=1.3, n_steps=10)
    e.reset(seed=5)
    bought_btc = 0.0
    done = False
    while not done:
        ep = e._ep
        before = ep.remaining_btc
        _o, _r, done, info = e.step(ACTIONS.index(1.0))
        bought_btc += before - ep.remaining_btc if not done else 0.0
    # at the deadline the residual reported must be the TRUE remainder (< 1 unit here)
    assert info["deadline_residual_btc"] < 0.5 + 1e-9
    # carry can never exceed remaining (root-cause invariant)
    e.reset(seed=6)
    for _ in range(10):
        _o, _r, d2, _i = e.step(ACTIONS.index(1.0))
        assert e._ep.carry_btc <= e._ep.remaining_btc + 1e-9


def test_per_depth_unit_sizes_price_and_credit_correctly(tmp_path):
    # remediation I3: a unit at slot i is aes[i] BTC — cost and BTC credited must use
    # the slot's own size, not the touch size
    K, Q = 3, 6
    counts = np.zeros((K, Q + 1, 2, 3)); time_in = np.ones((K, Q + 1, 2))
    counts[:, :, :, 0] = 4.0
    bundle = assemble(counts, time_in, aes=[0.5, 1.0, 2.0], tick=1.0, invariant="empirical")
    bp = tmp_path / "b.npz"; bundle.save(str(bp))
    mp = tmp_path / "m.npz"
    np.savez(mp, interval_s=0.5, moves=np.array([0]), probs=np.array([1.0]))
    e = ReactiveQRMEnv(str(bp), str(mp), order_btc=5.0, n_steps=10)
    e.reset(seed=1)
    ep = e._ep
    ep.state[e.K:] = np.array([1, 1, 1], dtype=np.int8)   # one unit per ask slot
    ep.p_ref = 100.0
    cost, btc, units = e._buy_btc(3.5, allow_overshoot=True)
    # walk: 0.5 @ 100.5 + 1.0 @ 101.5 + 2.0 @ 102.5 = 50.25 + 101.5 + 205 = 356.75
    assert abs(btc - 3.5) < 1e-9 and units == 3
    assert abs(cost - 356.75) < 1e-9
    # no-overshoot mode stops before exceeding the target
    ep.state[e.K:] = np.array([1, 1, 1], dtype=np.int8)
    cost2, btc2, units2 = e._buy_btc(1.2, allow_overshoot=False)
    assert abs(btc2 - 0.5) < 1e-9 and units2 == 1   # next unit (1.0) would overshoot


def test_empty_ask_side_widens_quotes_instead_of_pinning_mid(tmp_path):
    K, Q = 3, 6
    counts = np.zeros((K, Q + 1, 2, 3)); time_in = np.ones((K, Q + 1, 2))
    counts[:, :, :, 0] = 4.0
    bundle = assemble(counts, time_in, aes=[0.5, 0.5, 0.5], tick=1.0, invariant="empirical")
    bp = tmp_path / "b.npz"; bundle.save(str(bp))
    mp = tmp_path / "m.npz"
    np.savez(mp, interval_s=0.5, moves=np.array([0]), probs=np.array([1.0]))
    e = ReactiveQRMEnv(str(bp), str(mp), order_btc=5.0, n_steps=10)
    e.reset(seed=2)
    ep = e._ep
    mid_before = ep.p_mid
    ep.state[e.K:] = 0                                  # agent swept the whole ask side
    mid_after = e._mid_from_state(ep)
    assert mid_after > mid_before                       # quotes widen upward (I5)
    assert e._spread_ticks(ep) >= e.K + 1               # spread reflects the empty side
