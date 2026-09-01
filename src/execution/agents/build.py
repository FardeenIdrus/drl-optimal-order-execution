"""SB3 agent factories. Mirrors the QRM scaffold's DQN architecture and
hyperparameters (arXiv 2511.15262), wrapped around our ``RealDataExecutionEnv``.

The hyperparameters live in ``configs/experiment.yaml`` (``agent.dqn``) so a run is
fully reproducible from config; only the activation function is mapped from a name
here. PPO is built by the same factory module.
"""
from __future__ import annotations

from typing import Any

import torch.nn as nn
from stable_baselines3 import DQN, PPO

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "LeakyReLU": nn.LeakyReLU,
    "ReLU": nn.ReLU,
    "Tanh": nn.Tanh,
}


def derive_exploration_fraction(total_timesteps: int, learning_starts: int,
                                anneal_frac: float) -> float:
    """Budget-relative epsilon-anneal window.

    SB3 measures ``exploration_fraction`` against *total* timesteps with the clock
    starting at step 0, so a fixed value tuned to a different budget can decay
    epsilon before ``learning_starts`` (it did: QRM's 0.03 finished annealing at
    15k steps, inside the 40k warmup, leaving the agent near-greedy throughout
    learning). Instead, anneal across the warmup PLUS ``anneal_frac`` of the
    post-warmup learning phase, so epsilon is still high when learning begins and
    the window scales with the budget (no re-reconciliation at 3M/6M).

    This is a standard anneal convention (~10-50% of training) expressed
    budget-relative, not a first-principles derivation; the resulting value is
    recorded per run. With ``learning_starts`` small vs the budget it lands near
    ~0.25-0.27 across 1M-6M.
    """
    if not 0.0 < anneal_frac <= 1.0:
        raise ValueError("anneal_frac must be in (0, 1]")
    if total_timesteps <= learning_starts:
        raise ValueError("total_timesteps must exceed learning_starts")
    anneal_end = learning_starts + anneal_frac * (total_timesteps - learning_starts)
    return anneal_end / total_timesteps


def make_dqn(env: Any, hp: dict, *, seed: int, device: str = "cpu",
             total_timesteps: int | None = None) -> DQN:
    """Build an SB3 DQN mirroring the QRM hyperparameters in ``hp``.

    ``exploration_fraction`` is taken from ``hp`` if given explicitly; otherwise it
    is derived from ``exploration_anneal_frac`` and ``total_timesteps`` via
    :func:`derive_exploration_fraction` (the budget-relative rule). ``activation``
    is a name resolved via ``_ACTIVATIONS`` (default LeakyReLU, the QRM choice).
    ``device`` defaults to cpu (the probe showed MPS ~9x slower for this net).
    """
    activation = _ACTIVATIONS[hp.get("activation", "LeakyReLU")]
    policy_kwargs = dict(net_arch=list(hp["net_arch"]), activation_fn=activation)

    if "exploration_fraction" in hp:
        exploration_fraction = hp["exploration_fraction"]
    elif "exploration_anneal_frac" in hp:
        if total_timesteps is None:
            raise ValueError("total_timesteps required to derive exploration_fraction")
        exploration_fraction = derive_exploration_fraction(
            total_timesteps, hp["learning_starts"], hp["exploration_anneal_frac"])
    else:
        raise ValueError("hp needs exploration_fraction or exploration_anneal_frac")

    return DQN(
        "MlpPolicy",
        env,
        learning_rate=hp["learning_rate"],
        buffer_size=hp["buffer_size"],
        batch_size=hp["batch_size"],
        learning_starts=hp["learning_starts"],
        gamma=hp["gamma"],
        train_freq=hp["train_freq"],
        gradient_steps=hp["gradient_steps"],
        target_update_interval=hp["target_update_interval"],
        n_steps=hp["n_steps"],
        exploration_initial_eps=hp["exploration_initial_eps"],
        exploration_final_eps=hp["exploration_final_eps"],
        exploration_fraction=exploration_fraction,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
        verbose=0,
    )


def make_ppo(env: Any, hp: dict, *, seed: int, device: str = "cpu") -> PPO:
    """Build an SB3 PPO that is a like-for-like counterpart to the DQN.

    Same observation, reward (bps), action grid, split, and order size as the DQN,
    and deliberately the SAME ``[30x5]`` LeakyReLU trunk and ``gamma``, so the
    DQN-vs-PPO comparison isolates the *algorithm* rather than the architecture.
    PPO is on-policy: it has no replay buffer and no epsilon-greedy schedule, and
    explores via its stochastic policy plus a small entropy bonus (``ent_coef``),
    which also guards against the premature collapse to a constant TWAP-pace policy
    seen in some DQN seeds. The hyperparameters are new (the QRM scaffold had only
    DQN); they are the well-established SB3 defaults with ``gamma`` matched to the
    DQN and a small positive ``ent_coef``, all set in config for reproducibility.

    Note for SHAP attribution: PPO has no ``replay_buffer``, so attribution states must
    be sourced from a policy rollout, not ``model.replay_buffer``.
    """
    activation = _ACTIVATIONS[hp.get("activation", "LeakyReLU")]
    policy_kwargs = dict(net_arch=list(hp["net_arch"]), activation_fn=activation)
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=hp["learning_rate"],
        n_steps=hp["n_steps"],
        batch_size=hp["batch_size"],
        n_epochs=hp["n_epochs"],
        gamma=hp["gamma"],
        gae_lambda=hp["gae_lambda"],
        clip_range=hp["clip_range"],
        ent_coef=hp["ent_coef"],
        vf_coef=hp.get("vf_coef", 0.5),
        max_grad_norm=hp.get("max_grad_norm", 0.5),
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
        verbose=0,
    )
