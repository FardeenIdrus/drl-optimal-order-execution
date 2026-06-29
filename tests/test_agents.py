"""Unit tests for the Phase-5 agents (Step 5): DQN factory, policy adapter, and a
single-seed training smoke that exercises the full env -> SB3 -> callback -> eval
path on a tiny synthetic dataset.

Run with:  PYTHONPATH=src .venv/bin/pytest tests/test_agents.py
"""
import json

import numpy as np
import pytest
from stable_baselines3 import DQN

from execution.agents.build import derive_exploration_fraction, make_dqn
from execution.agents.policy import SB3Policy
from execution.agents.train import TrainSetup, train_one_seed
from execution.data.features import FEATURE_COLUMNS
from execution.env.episode_store import EpisodeStore
from execution.env.normalize import FeatureNormalizer
from execution.env.real_data_env import RealDataExecutionEnv

_TINY_DQN = dict(net_arch=[16, 16], activation="LeakyReLU", learning_rate=1e-3,
                 buffer_size=2000, batch_size=32, learning_starts=100, gamma=0.99,
                 train_freq=10, gradient_steps=1, target_update_interval=100, n_steps=1,
                 exploration_initial_eps=1.0, exploration_final_eps=0.05,
                 exploration_fraction=0.5)


def _synth_store(n_ep, n_steps=30, n_levels=20, seed=0):
    rng = np.random.default_rng(seed)
    mid = 100.0 + rng.normal(0, 0.2, (n_ep, n_steps)).cumsum(axis=1)
    px = mid[..., None] + 0.5 + np.arange(n_levels) * 0.1
    sz = np.full((n_ep, n_steps, n_levels), 3.0)
    feats = rng.normal(0, 1, (n_ep, n_steps, len(FEATURE_COLUMNS)))
    regime = np.array(["calm" if e % 2 == 0 else "volatile" for e in range(n_ep)], dtype=object)
    return EpisodeStore(np.arange(n_ep), regime, mid, px, sz, feats,
                        tuple(FEATURE_COLUMNS), n_steps)


def _env(store):
    norm = FeatureNormalizer.fit(store.features, store.feature_names)
    return RealDataExecutionEnv(store, initial_inventory=10.0,
                                obs_features=tuple(FEATURE_COLUMNS), normalizer=norm)


def test_make_dqn_mirrors_hyperparameters():
    model = make_dqn(_env(_synth_store(4)), _TINY_DQN, seed=0, device="cpu")
    assert isinstance(model, DQN)
    assert model.batch_size == 32
    assert model.gamma == pytest.approx(0.99)
    assert model.n_steps == 1
    assert model.policy_kwargs["net_arch"] == [16, 16]


def test_derive_exploration_fraction_budget_relative():
    ef2 = derive_exploration_fraction(2_000_000, 40_000, 0.25)
    assert ef2 == pytest.approx((40_000 + 0.25 * (2_000_000 - 40_000)) / 2_000_000)
    assert ef2 * 2_000_000 > 40_000                 # anneal always ends after warmup
    ef6 = derive_exploration_fraction(6_000_000, 40_000, 0.25)
    assert abs(ef2 - ef6) < 0.02                    # ~constant across budgets


@pytest.mark.parametrize("total,ls,af", [(1000, 2000, 0.25), (2_000_000, 40_000, 0.0)])
def test_derive_exploration_rejects_bad_inputs(total, ls, af):
    with pytest.raises(ValueError):
        derive_exploration_fraction(total, ls, af)


def test_make_dqn_derives_exploration_from_budget():
    hp = dict(_TINY_DQN)
    hp.pop("exploration_fraction")
    hp["exploration_anneal_frac"] = 0.25
    model = make_dqn(_env(_synth_store(4)), hp, seed=0, device="cpu", total_timesteps=10_000)
    assert model.exploration_fraction == pytest.approx(
        derive_exploration_fraction(10_000, hp["learning_starts"], 0.25))


def test_sb3policy_returns_int_action_in_range():
    env = _env(_synth_store(4))
    model = make_dqn(env, _TINY_DQN, seed=0, device="cpu")
    pol = SB3Policy(model)
    obs, _ = env.reset(options={"episode_index": 0})
    a = pol.action(obs)
    assert isinstance(a, int) and 0 <= a < env.action_space.n


def test_train_one_seed_smoke(tmp_path):
    cfg = {
        "env": {"n_steps": 30, "actions": [0.0, 1.0, 2.0], "initial_inventory": 10.0,
                "action_mode": "twap_pace_multiple"},
        "obs": {"features": list(FEATURE_COLUMNS)},
        "agent": {"leftover_penalty_bps": 100.0, "dqn": _TINY_DQN},
    }
    train_sub = _synth_store(20, seed=0)
    val = _synth_store(8, seed=1)
    norm = FeatureNormalizer.fit(train_sub.features, train_sub.feature_names)
    setup = TrainSetup(train_sub, val, norm, {"calm": 0.005, "volatile": 0.005}, 19000.0)

    meta = train_one_seed(setup, size=10.0, cfg=cfg, seed=0, total_timesteps=1500,
                          out_dir=tmp_path, eval_freq=500, n_val_eval=8)

    assert (tmp_path / "model.zip").exists()
    assert (tmp_path / "curve.csv").exists()
    assert (tmp_path / "normalizer.json").exists()
    assert json.loads((tmp_path / "meta.json").read_text())["seed"] == 0
    assert meta["val_vs_twap_final"] is not None        # curve produced a final point
    assert meta["train_time_s"] >= 0.0
