"""Unit tests for RealDataExecutionEnv (Phase 2, Step 1).

Covers fill/IS accounting, reward sign, termination, determinism, positional
(no off-by-one) book access, the observation vector, and the §4 sanity checks
(instant liquidation = worst cost; do-nothing = force-execution penalty;
TWAP fully executes). Uses small synthetic stores with hand-computable costs.

Run with:  PYTHONPATH=src .venv/bin/pytest tests/test_real_data_env.py
"""
import numpy as np
import pytest

from execution.data.features import FEATURE_COLUMNS
from execution.env.episode_store import EpisodeStore
from execution.env.normalize import FeatureNormalizer
from execution.env.real_data_env import RealDataExecutionEnv


def make_store(n_ep=2, n_steps=3, n_levels=3, mid=100.0,
               px=(101.0, 102.0, 103.0), sz=(1.0, 1.0, 1.0)):
    """Flat store: every step of every episode has the same ladder."""
    E, T, L = n_ep, n_steps, n_levels
    midA = np.full((E, T), mid)
    ask_px = np.tile(np.asarray(px, float), (E, T, 1))
    ask_sz = np.tile(np.asarray(sz, float), (E, T, 1))
    feats = np.zeros((E, T, len(FEATURE_COLUMNS)))
    regime = np.array(["calm" if e % 2 == 0 else "volatile" for e in range(E)], dtype=object)
    return EpisodeStore(np.arange(E), regime, midA, ask_px, ask_sz, feats,
                        tuple(FEATURE_COLUMNS), T)


def drive(env, quantities, ep=0):
    """Drive the env in absolute-quantity mode with an explicit per-step list."""
    env.set_absolute_quantity(True)
    env.reset(options={"episode_index": ep})
    info = None
    for q in quantities:
        _, reward, done, _, info = env.step(q)
        if done:
            break
    return info


def test_spaces_and_arrival():
    env = RealDataExecutionEnv(make_store(), initial_inventory=1.5)
    assert env.action_space.n == 7   # default TWAP-pace grid (0, .5, .8, 1, 1.2, 1.5, 2)
    assert env.observation_space.shape == (2,)
    obs, info = env.reset(options={"episode_index": 1})
    assert info["arrival_price"] == pytest.approx(100.0)
    assert obs[0] == pytest.approx(1.0)   # inventory_norm at start
    assert obs[1] == pytest.approx(1.0)   # time_norm at start


def test_twap_even_split_fully_executes_known_is():
    # 0.5 BTC/step @101 vs arrival 100 -> 0.5 cost/step, 3 steps -> 1.5 USD IS.
    env = RealDataExecutionEnv(make_store(), initial_inventory=1.5, test_mode=True)
    info = drive(env, [0.5, 0.5, 0.5])
    assert info["terminal"] is True
    assert env.n_filled == pytest.approx(1.5)
    assert info["inventory"] == pytest.approx(0.0)
    assert info["realized_cost"] == pytest.approx(1.5)
    assert info["is_bps"] == pytest.approx(1e4 * 1.5 / (100.0 * 1.5))  # 100 bps


def test_reward_is_negative_implementation_shortfall():
    env = RealDataExecutionEnv(make_store(), initial_inventory=1.5)
    env.set_absolute_quantity(True)
    env.reset(options={"episode_index": 0})
    _, reward, _, _, _ = env.step(0.5)        # buy 0.5 @101, arrival 100
    assert reward == pytest.approx(-0.5)       # cost is positive -> reward negative


def test_reward_bps_mode_scales_to_basis_points():
    env = RealDataExecutionEnv(make_store(), initial_inventory=1.5, reward_bps=True)
    env.set_absolute_quantity(True)
    env.reset(options={"episode_index": 0})
    _, reward, _, _, _ = env.step(0.5)         # step_cost 0.5 USD; notional 100*1.5
    assert reward == pytest.approx(-0.5 / (100.0 * 1.5) * 1e4)   # bps of arrival notional


def test_leftover_penalty_in_bps_mode():
    # Do nothing (training mode): terminal penalty = -leftover_bps * fraction left.
    env = RealDataExecutionEnv(make_store(), initial_inventory=1.5, reward_bps=True,
                               leftover_penalty_bps=100.0)
    env.set_absolute_quantity(True)
    env.reset(options={"episode_index": 0})
    cum = 0.0
    for _ in range(3):
        _, r, done, _, _ = env.step(0.0)
        cum += r
        if done:
            break
    assert cum == pytest.approx(-100.0)        # full order left -> -100 bps, no fills


def test_instant_liquidation_is_worse_than_twap():
    inv = 1.5
    twap = drive(RealDataExecutionEnv(make_store(), initial_inventory=inv, test_mode=True),
                 [0.5, 0.5, 0.5])
    instant = drive(RealDataExecutionEnv(make_store(), initial_inventory=inv, test_mode=True),
                    [1.5, 0.0, 0.0])
    # instant walks 1@101 + 0.5@102 = 2.0 USD IS > TWAP 1.5
    assert instant["realized_cost"] == pytest.approx(2.0)
    assert instant["realized_cost"] > twap["realized_cost"]


def test_do_nothing_force_executes_and_penalises():
    inv = 1.5
    env = RealDataExecutionEnv(make_store(), initial_inventory=inv, final_penalty=1.0,
                               test_mode=True)
    info = drive(env, [0.0, 0.0, 0.0])
    # force-exec at last book: 1@101 + 0.5@102 = 2.0 realized IS
    assert info["realized_cost"] == pytest.approx(2.0)
    assert info["inventory"] == pytest.approx(0.0)   # completed in test_mode
    # reward carried the leftover penalty (-final_penalty * inv)
    assert env.cum_reward < -inv + 1e-9


def _residual_env(**kw):
    return RealDataExecutionEnv(
        make_store(), initial_inventory=4.0, test_mode=True,
        residual_coef_by_regime={"calm": 0.5, "volatile": 0.5}, adv_btc=1000.0, **kw)


def test_per_step_excess_carries_no_midstep_penalty():
    # Request beyond this step's depth (3 BTC) -> partial fill, remainder carries,
    # NO penalty mid-episode (the residual is a deadline-only concept).
    env = _residual_env()
    env.set_absolute_quantity(True)
    env.reset(options={"episode_index": 0})
    _, _, done, _, info = env.step(10.0)         # depth is 3 BTC, inventory 4
    assert not done                               # step 1 of 3, inventory remains
    assert info["inventory"] == pytest.approx(1.0)    # 4 - 3 filled, carried
    assert info["residual_btc"] == pytest.approx(0.0)  # residual only fires at the deadline


def test_deadline_residual_logged_and_priced_with_sqrt():
    # Order 4 BTC, visible depth 3 BTC, do nothing -> 1 BTC residual beyond depth.
    env = _residual_env()
    info = drive(env, [0.0, 0.0, 0.0])
    assert info["residual_btc"] == pytest.approx(1.0)
    assert info["residual_bps"] > 0.0
    assert info["inventory"] == pytest.approx(0.0)
    # full-book walk (1@101+1@102+1@103 vs mid 100 = 6) + residual anchor + sqrt term
    expected = 6.0 + 1.0 * (103 - 100) + 1.0 * 103 * 0.5 * float(np.sqrt(1.0 / 1000.0))
    assert info["realized_cost"] == pytest.approx(expected)


def test_residual_cost_continuous_at_boundary():
    # As residual -> 0 the residual cost -> 0: continuous with the within-book fill.
    env = _residual_env()
    env.reset(options={"episode_index": 0})
    px = np.array([110.0, 111.0, np.nan])
    assert abs(env._residual_cost_for(1e-9, px)) < 1e-6


def test_residual_sqrt_term_scales_as_R_to_the_1p5():
    env = _residual_env()
    env.reset(options={"episode_index": 0})
    px = np.array([110.0, 111.0, np.nan])
    env.arrival_price = 111.0                     # anchor term = 0 -> isolate the sqrt term
    c1 = env._residual_cost_for(2.0, px)
    c2 = env._residual_cost_for(8.0, px)          # 4x residual -> 4^1.5 = 8x
    assert c2 / c1 == pytest.approx(8.0)


def test_residual_without_calibration_is_anchor_only():
    env = RealDataExecutionEnv(make_store(), initial_inventory=4.0, test_mode=True)
    env.reset(options={"episode_index": 0})
    px = np.array([110.0, 111.0, np.nan])
    env.arrival_price = 100.0
    assert env._residual_cost_for(2.0, px) == pytest.approx(2.0 * (111.0 - 100.0))


def test_positional_book_access_no_off_by_one():
    # distinct level-1 price per step: 101, 111, 121
    store = make_store(n_ep=1, n_steps=3, n_levels=3)
    px = store.ask_px.copy()
    for t in range(3):
        px[0, t] = [101 + 10 * t, 102 + 10 * t, 103 + 10 * t]
    store = EpisodeStore(store.episode_ids, store.regime, store.mid, px, store.ask_sz,
                         store.features, store.feature_names, store.n_steps)
    env = RealDataExecutionEnv(store, initial_inventory=1.5, test_mode=True)
    info = drive(env, [0.5, 0.5, 0.5])
    # 0.5*(101-100) + 0.5*(111-100) + 0.5*(121-100) = 0.5 + 5.5 + 10.5 = 16.5
    assert info["realized_cost"] == pytest.approx(16.5)


def test_inventory_cap_prevents_overbuy():
    env = RealDataExecutionEnv(make_store(sz=(10.0, 10.0, 10.0)), initial_inventory=1.0,
                               test_mode=True)
    info = drive(env, [5.0])   # request 5 but only 1.0 inventory
    assert env.n_filled == pytest.approx(1.0)
    assert info["inventory"] == pytest.approx(0.0)


def test_termination_after_n_steps_and_step_after_done_raises():
    env = RealDataExecutionEnv(make_store(), initial_inventory=3.0, test_mode=True)
    env.set_absolute_quantity(True)
    env.reset(options={"episode_index": 0})
    done = False
    for _ in range(3):
        _, _, done, _, _ = env.step(0.0)
    assert done
    with pytest.raises(RuntimeError):
        env.step(0.0)


def test_step_before_reset_raises():
    env = RealDataExecutionEnv(make_store(), initial_inventory=1.0)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_determinism_same_episode_same_trajectory():
    a = drive(RealDataExecutionEnv(make_store(), initial_inventory=1.5, test_mode=True),
              [0.4, 0.6, 0.5])
    b = drive(RealDataExecutionEnv(make_store(), initial_inventory=1.5, test_mode=True),
              [0.4, 0.6, 0.5])
    assert a["realized_cost"] == pytest.approx(b["realized_cost"])


def test_action_is_fraction_of_remaining_inventory():
    # action 1 (frac 0.25) of inventory 4.0 -> 1.0 BTC, NOT 0.25 * L1 size (1.0).
    env = RealDataExecutionEnv(make_store(sz=(10.0, 10.0, 10.0)), initial_inventory=4.0,
                               actions=(0.0, 0.25, 0.5, 1.0), action_mode="inventory_fraction")
    env.reset(options={"episode_index": 0})
    _, _, _, _, info = env.step(1)
    assert info["filled"] == pytest.approx(1.0)      # 0.25 * 4.0
    assert info["inventory"] == pytest.approx(3.0)   # not 4.0 - 0.25*L1


def test_large_action_walks_multiple_levels():
    # frac 0.5 of inventory 4.0 = 2.0 BTC; L1 size is 1.0 so the request walks
    # L1@101 + L2@102 -> ask_depth is now actionable (a fill past level 1).
    env = RealDataExecutionEnv(make_store(sz=(1.0, 1.0, 1.0)), initial_inventory=4.0,
                               actions=(0.0, 0.25, 0.5, 1.0), action_mode="inventory_fraction")
    env.reset(options={"episode_index": 0})
    _, reward, _, _, info = env.step(2)              # action 2 -> frac 0.5
    assert info["filled"] == pytest.approx(2.0)
    assert info["levels_walked"] == 2
    assert reward == pytest.approx(-3.0)             # 101 + 102 - 2*100


def test_under_pace_then_clear_fully_executes():
    # Under-pace early, then a large late action clears the remainder on its own
    # (full self-execution is reachable without force-completion).
    env = RealDataExecutionEnv(make_store(sz=(10.0, 10.0, 10.0)), initial_inventory=4.0,
                               actions=(0.0, 0.25, 0.5, 1.0), action_mode="inventory_fraction")
    env.reset(options={"episode_index": 0})
    env.step(1)                                      # 0.25 * 4.0   = 1.0 -> inv 3.0
    env.step(1)                                      # 0.25 * 3.0   = 0.75 -> inv 2.25
    _, _, done, _, info = env.step(3)                # 1.0  * 2.25 -> inv 0
    assert done
    assert info["inventory"] == pytest.approx(0.0)
    assert env.n_filled == pytest.approx(4.0)


def test_twap_multiple_1x_matches_twap_pace():
    # mult=1 requests remaining/steps_left = the TWAP slice each step.
    env = RealDataExecutionEnv(make_store(sz=(10.0, 10.0, 10.0)), initial_inventory=3.0,
                               actions=(0.0, 1.0, 2.0))   # default mode = twap_pace_multiple
    env.reset(options={"episode_index": 0})
    _, _, _, _, info = env.step(1)               # 1x of 3.0 over 3 steps = 1.0
    assert info["filled"] == pytest.approx(1.0)
    assert info["inventory"] == pytest.approx(2.0)
    _, _, _, _, info = env.step(1)               # 1x of 2.0 over 2 steps left = 1.0
    assert info["filled"] == pytest.approx(1.0)


def test_twap_multiple_2x_front_loads_and_reaches_deeper():
    store = make_store(n_levels=5, px=(101.0, 102.0, 103.0, 104.0, 105.0),
                       sz=(1.0, 1.0, 1.0, 1.0, 1.0))
    env = RealDataExecutionEnv(store, initial_inventory=6.0, actions=(0.0, 1.0, 2.0))
    env.reset(options={"episode_index": 0})
    _, _, _, _, info = env.step(2)               # 2x of 6.0 over 3 steps = 4.0 requested
    assert info["filled"] == pytest.approx(4.0)  # vs 1x would request only 2.0
    assert info["levels_walked"] == 4            # walks past L1 -> ask_depth reachable
    assert info["inventory"] == pytest.approx(2.0)


def test_twap_multiple_1x_fully_executes():
    env = RealDataExecutionEnv(make_store(sz=(10.0, 10.0, 10.0)), initial_inventory=3.0,
                               actions=(0.0, 1.0, 2.0), test_mode=True)
    env.reset(options={"episode_index": 0})
    done = False
    while not done:
        _, _, done, _, info = env.step(1)        # always 1x
    assert info["inventory"] == pytest.approx(0.0)
    assert env.n_filled == pytest.approx(3.0)


def test_twap_multiple_1x_reproduces_twap_benchmark():
    # The always-1x policy == the TWAP benchmark on a full-fill book (same IS).
    from execution.env.benchmarks import TWAP
    store = make_store(sz=(10.0, 10.0, 10.0))
    a = RealDataExecutionEnv(store, initial_inventory=3.0, actions=(0.0, 1.0, 2.0),
                             test_mode=True)
    a.reset(options={"episode_index": 0})
    done = False
    while not done:
        _, _, done, _, ia = a.step(1)
    b = RealDataExecutionEnv(store, initial_inventory=3.0, test_mode=True)
    b.set_absolute_quantity(True)
    b.reset(options={"episode_index": 0})
    twap = TWAP(3.0, 3)
    twap.reset(b)
    done = False
    while not done:
        _, _, done, _, ib = b.step(twap.action())
    assert ia["realized_cost"] == pytest.approx(ib["realized_cost"])


def test_front_load_when_deep_beats_uniform_pacing():
    # Step 0: deep & cheap (ask 100.5, size 5). Step 1: thin & expensive (ask 105).
    # Arrival mid = 100. Front-loading into the cheap deep book beats spreading the
    # order into the later expensive book -> ask_depth is now actionable (the edge).
    E, T, L = 1, 2, 3
    mid = np.full((E, T), 100.0)
    ask_px = np.zeros((E, T, L))
    ask_sz = np.full((E, T, L), 5.0)
    ask_px[0, 0] = [100.5, 100.6, 100.7]   # cheap, deep
    ask_px[0, 1] = [105.0, 105.1, 105.2]   # expensive
    feats = np.zeros((E, T, len(FEATURE_COLUMNS)))
    store = EpisodeStore(np.arange(E), np.array(["calm"], dtype=object), mid, ask_px,
                         ask_sz, feats, tuple(FEATURE_COLUMNS), T)

    def run(action_seq):
        env = RealDataExecutionEnv(store, initial_inventory=2.0,
                                   actions=(0.0, 0.5, 1.0), action_mode="inventory_fraction",
                                   test_mode=True)
        env.reset(options={"episode_index": 0})
        info = None
        for a in action_seq:
            _, _, done, _, info = env.step(a)
            if done:
                break
        return info["realized_cost"]

    front = run([2])           # frac 1.0 at step 0 -> buy all 2.0 in the deep cheap book
    uniform = run([1, 2])      # frac 0.5 then 1.0 -> 1.0 cheap + 1.0 expensive
    assert front == pytest.approx(1.0)     # 2.0 * (100.5 - 100)
    assert uniform == pytest.approx(5.5)   # 1.0*(100.5-100) + 1.0*(105-100)
    assert front < uniform


def test_obs_includes_features_when_requested():
    store = make_store()
    store.features[:] = 0.0
    store.features[0, 0, 1] = 0.77            # imbalance at ep0 step0
    env = RealDataExecutionEnv(store, initial_inventory=1.5, obs_features=tuple(FEATURE_COLUMNS))
    assert env.observation_space.shape == (2 + len(FEATURE_COLUMNS),)
    obs, _ = env.reset(options={"episode_index": 0})
    assert obs.shape == (7,)
    assert obs[2 + 1] == pytest.approx(0.77)   # imbalance is the 2nd feature (raw, no normalizer)


def test_obs_features_are_positional_no_off_by_one():
    store = make_store(n_ep=1, n_steps=3, n_levels=3)
    feats = store.features.copy()
    for t in range(3):
        feats[0, t, 1] = 10.0 * t            # imbalance distinct per step: 0, 10, 20
    store = EpisodeStore(store.episode_ids, store.regime, store.mid, store.ask_px,
                         store.ask_sz, feats, store.feature_names, store.n_steps)
    env = RealDataExecutionEnv(store, initial_inventory=3.0, obs_features=tuple(FEATURE_COLUMNS))
    env.set_absolute_quantity(True)
    obs, _ = env.reset(options={"episode_index": 0})
    assert obs[2 + 1] == pytest.approx(0.0)   # step 0
    obs, *_ = env.step(0.0)
    assert obs[2 + 1] == pytest.approx(10.0)  # step 1 (not peeking at a later step)
    obs, *_ = env.step(0.0)
    assert obs[2 + 1] == pytest.approx(20.0)  # step 2


def test_obs_applies_train_fit_normalizer():
    store = make_store()
    feats = store.features.copy()
    feats[0, 0] = [0.5, 0.2, 0.001, 0.003, 8.0]   # one realistic-ish feature row
    store = EpisodeStore(store.episode_ids, store.regime, store.mid, store.ask_px,
                         store.ask_sz, feats, store.feature_names, store.n_steps)
    norm = FeatureNormalizer.fit(feats, store.feature_names)
    env = RealDataExecutionEnv(store, initial_inventory=1.5,
                               obs_features=tuple(FEATURE_COLUMNS), normalizer=norm)
    obs, _ = env.reset(options={"episode_index": 0})
    expected = norm.transform(feats[0, 0]).astype("float32")
    assert np.allclose(obs[2:], expected, atol=1e-5)


def test_normalizer_feature_mismatch_raises():
    store = make_store()
    bad = FeatureNormalizer.fit(np.ones((5, 3)), ("a", "b", "c"))
    with pytest.raises(ValueError, match="normalizer"):
        RealDataExecutionEnv(store, initial_inventory=1.0,
                             obs_features=tuple(FEATURE_COLUMNS), normalizer=bad)
