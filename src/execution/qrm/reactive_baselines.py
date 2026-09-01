"""Baseline policies run through the SAME reacting environment.

Every baseline runs through :class:`~execution.qrm.reactive_env.ReactiveQRMEnv`
step-for-step (same fills, same reaction, same episode randomness), so agent-vs-
baseline comparisons are apples to apples with common random numbers.

* **Adaptive TWAP** = the always-1.0x action: pace = remaining/steps-left each second
  (self-correcting; the L2 track's fairness anchor).
* **Fixed TWAP** = order/n_steps BTC per second regardless of fills so far, with its
  own fractional-unit carry (the classic even schedule).
* **Instant dump** = 2.0x (the max multiple) every step until done — the most
  aggressive representable policy, the G3 cost-ordering reference.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from execution.qrm.reactive_env import ACTIONS, ReactiveQRMEnv

A_ZERO = 0
A_ONE = ACTIONS.index(1.0)
A_MAX = len(ACTIONS) - 1


def adaptive_twap(env: ReactiveQRMEnv, _obs: np.ndarray) -> int:
    return A_ONE


def instant_dump(env: ReactiveQRMEnv, _obs: np.ndarray) -> int:
    return A_MAX


def signal_follower(env: ReactiveQRMEnv, _obs: np.ndarray) -> int:
    """Naive signal-follower (measured-signal extension, secondary benchmark; criteria
    section 8 Phase B registration): each decision, the pace multiple nearest to
    1 + S2_bg, where S2_bg is the pre-computed background signal at the current
    interval. Nothing is tuned: the mapping is fixed, the sign follows the measured
    (positive) slope — imbalance toward the bid predicts a rise, so trade faster
    ahead of it. Same completion semantics as every other policy (the environment's
    own deadline logic applies)."""
    ep = env._ep
    if ep is None or ep.s2_path is None:
        return A_ONE                              # no signal available: adaptive TWAP
    from execution.qrm.reactive_env import WARMUP_INTERVALS
    k = min(max(ep.move_idx - WARMUP_INTERVALS, 0), len(ep.s2_path) - 1)
    v = ep.s2_path[k]
    s2 = float(v) if np.isfinite(v) else env.signal_mean
    want = min(max(1.0 + s2, 0.0), 2.0)
    return int(np.argmin([abs(a - want) for a in ACTIONS]))


def make_fixed_twap(env: ReactiveQRMEnv) -> Callable[[ReactiveQRMEnv, np.ndarray], int]:
    """Fixed even slices: emulated via the pace-multiple action closest to the fixed
    slice each step (exact early on; when behind/ahead the multiple compensates so the
    *cumulative* schedule tracks the fixed line — within one unit by the carry)."""
    fixed_btc = env.order_btc / env.n_steps

    def policy(e: ReactiveQRMEnv, _obs: np.ndarray) -> int:
        ep = e._ep
        steps_left = e.n_steps - ep.step_idx
        adaptive_pace = ep.remaining_btc / max(steps_left, 1)
        if adaptive_pace <= 1e-12:
            return A_ZERO
        want = fixed_btc / adaptive_pace          # multiple that reproduces the fixed slice
        idx = int(np.argmin([abs(a - want) for a in ACTIONS]))
        return idx

    return policy


def run_episodes(
    env: ReactiveQRMEnv, policy: Callable, seeds: List[int]
) -> Dict[str, np.ndarray]:
    """Run one policy over the given episode seeds; per-episode totals."""
    cost_bps = np.zeros(len(seeds))
    executed = np.zeros(len(seeds))
    for j, seed in enumerate(seeds):
        obs = env.reset(seed=seed)
        done = False
        tot = 0.0
        while not done:
            obs, r, done, info = env.step(policy(env, obs))
            tot += r
        cost_bps[j] = -tot                        # positive = cost in bps
        # criteria 4b: voluntary completion (post-finalize remaining is always 0)
        executed[j] = 1.0 - info["deadline_residual_btc"] / env.order_btc
    return {"cost_bps": cost_bps, "executed_frac": executed}
