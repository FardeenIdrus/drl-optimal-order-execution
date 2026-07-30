"""Step 5 — train DQN/PPO inside the reactive QRM (protocol: qrm_step4_criteria.md §4).

Reuses the L2 track's agent builders (same [30x5] LeakyReLU trunk, same hyperparameter
discipline) so the L2-vs-QRM comparison isolates the ENVIRONMENT, not the architecture.
gamma follows the locked fixed-rate scaling rule (BUILD_PLAN "Discount factor"):
gamma = 0.995^(1/60) ~ 0.99992 per 1 s step (the L2 value was per 60 s step).

Run bundle per (algo, regime, seed): model.zip + meta.json + eval curve CSV.
    PYTHONPATH=src .venv/bin/python -m execution.qrm.train_reactive \
        --scratch <oxford_l4> --algo ppo --regime calm --seed 0 --out <runs_dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:  # gymnasium is pinned in the venv (L2 track dependency)
    import gymnasium as gym
except ImportError:  # pragma: no cover
    raise SystemExit("gymnasium required")

from execution.qrm.reactive_env import ACTIONS, ReactiveQRMEnv

logger = logging.getLogger(__name__)

GAMMA_1S = 0.995 ** (1.0 / 60.0)      # fixed-rate discount scaling (locked rule)
TRAIN_STEPS = 2_000_000               # criteria §4
EVAL_EVERY = 100_000
N_EVAL_CURVE = 200


class GymReactive(gym.Env):
    """Gymnasium adapter around :class:`ReactiveQRMEnv` (SB3-compatible)."""

    metadata = {"render_modes": []}

    def __init__(self, core: ReactiveQRMEnv, train_seed_base: int = 0,
                 reward_scale: float = 1.0) -> None:
        self.core = core
        self._base = int(train_seed_base)
        self._episode = 0
        # V2 (criteria §5): multiplies the TRAINING signal only; evaluation always
        # reads unscaled costs from the core env, so the objective is unchanged.
        self.reward_scale = float(reward_scale)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                                shape=(core.obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(core.n_actions)

    def reset(self, *, seed: Optional[int] = None, options=None):
        ep_seed = seed if seed is not None else self._base + self._episode
        self._episode += 1
        return self.core.reset(seed=int(ep_seed)), {}

    def step(self, action):
        obs, r, done, info = self.core.step(int(action))
        return obs, r * self.reward_scale, done, False, info


def _load_kernel(scratch: Path, regime: str) -> dict:
    """The certified frozen injection kernel for a regime (Phase D/E; criteria §8)."""
    sol = json.loads((scratch / "signal" / "kernel_solution.json").read_text())
    return sol["regimes"][regime]["kernel"]


def _core(scratch: Path, regime: str, order_btc: float,
          env_steps: int = 300, kernel: Optional[dict] = None,
          obs_pva: bool = False) -> ReactiveQRMEnv:
    # kernel is None -> the boundary-null (no-signal) env; a kernel -> the CERTIFIED
    # measured-signal injected env (Phase E). The observation gains the signal feature
    # (obs_dim +=1) automatically when injection is ON.
    kw = dict(signal_injection=True, signal_kernel=kernel) if kernel is not None else {}
    if obs_pva:                                  # criteria section 8 Amendment A4
        kw["obs_price_vs_arrival"] = True
    return ReactiveQRMEnv(
        str(scratch / "step3g" / f"qrm_bundle_{regime}_b.npz"),
        str(scratch / "step3g" / f"move_process_{regime}_centered.npz"),
        order_btc=order_btc, n_steps=env_steps, **kw)  # R2: drift-free; §7 horizon variant


def eval_paired_vs_adaptive(scratch: Path, regime: str, order_btc: float,
                            model, n_eps: int, seed0: int = 1_000_000,
                            env_steps: int = 300, kernel: Optional[dict] = None,
                            obs_pva: bool = False) -> dict:
    """CRN-paired mean cost difference (model − adaptive-TWAP), bps; negative = better.
    In Phase E `kernel` is the certified injected kernel, so agent and benchmark are
    scored in the SAME injected market (the shadow pass makes the signal path
    policy-independent, so CRN pairing holds exactly)."""
    from execution.qrm.reactive_baselines import adaptive_twap, run_episodes
    core = _core(scratch, regime, order_btc, env_steps, kernel=kernel, obs_pva=obs_pva)
    seeds = list(range(seed0, seed0 + n_eps))
    base = run_episodes(core, adaptive_twap, seeds)

    def policy(env, obs):
        a, _ = model.predict(obs, deterministic=True)
        return int(a)
    agent = run_episodes(core, policy, seeds)
    diff = agent["cost_bps"] - base["cost_bps"]
    return {"mean_diff_bps": float(diff.mean()),
            "executed_frac": float(agent["executed_frac"].mean()),
            "n": n_eps}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    ap.add_argument("--regime", choices=["calm", "volatile"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--order-btc", type=float, default=25.0)
    ap.add_argument("--env-steps", type=int, default=300,
                    help="episode horizon in 1s decisions (criteria §7: 600 = 10-min variant)")
    ap.add_argument("--steps", type=int, default=TRAIN_STEPS)
    ap.add_argument("--net-arch", default="30,30,30,30,30",
                    help="comma-separated layer widths (criteria §5 V1a/V1b)")
    ap.add_argument("--reward-scale", type=float, default=1.0,
                    help="training-signal multiplier (criteria §5 V2)")
    ap.add_argument("--tag", default="", help="run-dir suffix for variant runs")
    ap.add_argument("--lr", type=float, default=None, help="override learning rate (V3)")
    ap.add_argument("--ent-coef", type=float, default=None, help="override PPO entropy (V4)")
    ap.add_argument("--ppo-n-steps", type=int, default=None, help="override PPO rollout (V5)")
    ap.add_argument("--dqn-final-eps", type=float, default=None, help="DQN D1")
    ap.add_argument("--dqn-anneal-frac", type=float, default=None, help="DQN D1")
    ap.add_argument("--dqn-train-freq", type=int, default=None,
                    help="override DQN update interval in env steps (criteria §7.7 Part E d3)")
    ap.add_argument("--dqn-batch-size", type=int, default=None,
                    help="override DQN batch size (criteria §7.7 Part E d3)")
    ap.add_argument("--obs-price-vs-arrival", action="store_true",
                    help="criteria section 8 Amendment A4: add price-vs-arrival (bps) to the "
                         "observation (obs_dim +1). OFF by default; the registered campaigns "
                         "used the original observation.")
    ap.add_argument("--log-learning", action="store_true",
                    help="task register item 7: write SB3's per-update learning diagnostics "
                         "(value loss, explained variance, entropy, approx KL, clip fraction) "
                         "to <run_dir>/progress.csv. The registered campaigns trained with no "
                         "logger, so these trajectories were never captured and CANNOT be "
                         "recovered from a saved model.zip. OFF by default: attaching a logger "
                         "must not change any completed run. It touches no RNG stream, so a "
                         "run repeated with this flag is expected to reproduce its original "
                         "curve exactly -- which the item-7 launcher GATES on before the "
                         "trajectories are attributed to the agents of record.")
    ap.add_argument("--inject", action="store_true",
                    help="train + eval in the CERTIFIED measured-signal injected env "
                         "(Phase D/E; loads signal/kernel_solution.json for the regime)")
    args = ap.parse_args()
    # audit A1: a variant --tag MUST start with "_" or step5_judgement's tag_of regex
    # (`_s{seed}(_.+)?$`) silently pools the variant into the base group, corrupting its cell.
    assert not args.tag or args.tag.startswith("_"), \
        f"--tag must start with '_' (got {args.tag!r})"
    scratch = Path(args.scratch)
    run_dir = Path(args.out) / f"{args.algo}_{args.regime}_s{args.seed}{args.tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    net_arch = [int(x) for x in args.net_arch.split(",")]

    from execution.agents.build import make_dqn, make_ppo
    kernel = _load_kernel(scratch, args.regime) if args.inject else None
    if args.inject:
        logger.info("%s %s s%d: INJECTED env (offset %+.6f bps)", args.algo, args.regime,
                    args.seed, kernel["offset_bps"])
    core = _core(scratch, args.regime, args.order_btc, args.env_steps, kernel=kernel,
                 obs_pva=args.obs_price_vs_arrival)
    env = GymReactive(core, train_seed_base=args.seed * 10_000_000,
                      reward_scale=args.reward_scale)
    # the L2 track's locked hyperparameters (configs/experiment*.yaml agent blocks),
    # with ONLY gamma re-scaled to the 1 s step (the locked fixed-rate rule)
    if args.algo == "dqn":
        hp = {"net_arch": net_arch, "activation": "LeakyReLU",
              "learning_rate": 1e-4, "buffer_size": 1_000_000, "batch_size": 1024,
              "learning_starts": 40_000, "gamma": GAMMA_1S, "train_freq": 100,
              "gradient_steps": 1, "target_update_interval": 1000, "n_steps": 1,
              "exploration_initial_eps": 1.0, "exploration_final_eps": 0.01,
              "exploration_anneal_frac": 0.25}
        if args.lr is not None:
            hp["learning_rate"] = args.lr
        if args.dqn_final_eps is not None:
            hp["exploration_final_eps"] = args.dqn_final_eps
        if args.dqn_anneal_frac is not None:
            hp["exploration_anneal_frac"] = args.dqn_anneal_frac
        if args.dqn_train_freq is not None:
            hp["train_freq"] = args.dqn_train_freq
        if args.dqn_batch_size is not None:
            hp["batch_size"] = args.dqn_batch_size
        model = make_dqn(env, hp, seed=args.seed, total_timesteps=args.steps)
    else:
        hp = {"net_arch": net_arch, "activation": "LeakyReLU",
              "learning_rate": 3e-4, "n_steps": 2048, "batch_size": 64,
              "n_epochs": 10, "gamma": GAMMA_1S, "gae_lambda": 0.95,
              "clip_range": 0.2, "ent_coef": 0.01}
        if args.lr is not None:
            hp["learning_rate"] = args.lr
        if args.ent_coef is not None:
            hp["ent_coef"] = args.ent_coef
        if args.ppo_n_steps is not None:
            hp["n_steps"] = args.ppo_n_steps
        model = make_ppo(env, hp, seed=args.seed)

    if args.log_learning:
        # Standard SB3 CSV logger. PPO dumps once per rollout (n_steps env steps), DQN once
        # per log_interval episodes, so the file is the per-update learning trajectory.
        from stable_baselines3.common.logger import configure as sb3_configure
        model.set_logger(sb3_configure(str(run_dir), ["csv"]))

    # Remediation I8: ONE learn() call for the whole budget. The old 100k-chunk loop
    # made SB3 recompute the epsilon anneal against each chunk, so every DQN run
    # trained at the epsilon floor from (before) the first gradient update. The curve
    # is now recorded by a callback inside the single call.
    from stable_baselines3.common.callbacks import BaseCallback

    curve = []
    t0 = time.perf_counter()

    class CurveCallback(BaseCallback):
        def _on_step(self) -> bool:
            if self.num_timesteps % EVAL_EVERY == 0:
                r = eval_paired_vs_adaptive(scratch, args.regime, args.order_btc,
                                            self.model, N_EVAL_CURVE,
                                            env_steps=args.env_steps, kernel=kernel,
                                            obs_pva=args.obs_price_vs_arrival)
                eps = getattr(self.model, "exploration_rate", None)
                curve.append({"steps": int(self.num_timesteps), **r,
                              **({"eps": float(eps)} if eps is not None else {})})
                logger.info("%s %s s%d @%dk: vs adaptive %.4f bps%s",
                            args.algo, args.regime, args.seed,
                            self.num_timesteps // 1000, r["mean_diff_bps"],
                            f", eps {eps:.3f}" if eps is not None else "")
            return True

    model.learn(total_timesteps=args.steps, callback=CurveCallback(), progress_bar=False)
    model.save(str(run_dir / "model.zip"))
    (run_dir / "curve.json").write_text(json.dumps(curve, indent=2))
    (run_dir / "meta.json").write_text(json.dumps({
        "algo": args.algo, "regime": args.regime, "seed": args.seed,
        "order_btc": args.order_btc, "env_steps": args.env_steps,
        "steps": args.steps, "gamma": GAMMA_1S,
        "injected": bool(args.inject),
        "obs_price_vs_arrival": bool(args.obs_price_vs_arrival),
        "log_learning": bool(args.log_learning),
        "signal_offset_bps": (kernel["offset_bps"] if kernel is not None else None),
        "net_arch": net_arch, "reward_scale": args.reward_scale, "tag": args.tag,
        "overrides": {"lr": args.lr, "ent_coef": args.ent_coef,
                      "ppo_n_steps": args.ppo_n_steps,
                      "dqn_final_eps": args.dqn_final_eps,
                      "dqn_anneal_frac": args.dqn_anneal_frac,
                      "dqn_train_freq": args.dqn_train_freq,
                      "dqn_batch_size": args.dqn_batch_size},
        "wall_s": time.perf_counter() - t0,
        "criteria": "reports/qrm_step4_criteria.md"}, indent=2))
    logger.info("run complete -> %s", run_dir)


if __name__ == "__main__":
    main()
