"""Adapt a trained SB3 model to the benchmark-policy interface.

``execution.eval.runner.run_episodes`` drives any object with ``reset(env)`` /
``action(obs, info)`` and an ``absolute`` flag. Wrapping the SB3 model in that
interface lets the agent be evaluated through the *same* loop and fill engine as
TWAP / Almgren-Chriss -- apples-to-apples for the generalization gate and
the paired evaluation.
"""
from __future__ import annotations

from typing import Any


class SB3Policy:
    """A trained SB3 model as a run_episodes-compatible policy (deterministic)."""

    absolute = False  # discrete fraction-of-inventory actions, not absolute BTC

    def __init__(self, model: Any) -> None:
        self.model = model

    def reset(self, env: Any = None) -> None:  # noqa: ARG002 -- parity with benchmarks
        pass

    def action(self, obs: Any, info: dict | None = None) -> int:  # noqa: ARG002
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)
