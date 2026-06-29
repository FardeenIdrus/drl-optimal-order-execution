"""Execution benchmarks for the real-data environment.

All benchmarks drive the env in absolute-quantity mode (``policy.absolute is
True``): ``action()`` returns a BTC child-order size, and the env walks the *same*
real ask ladder used for the RL agent, so every strategy is priced on identical
ground.

Policy protocol (shared with the evaluation runner):
    * ``absolute``           -- class attr; True if ``action()`` returns a BTC quantity.
    * ``reset(env)``         -- per-episode hook (stateful/liquidity-aware policies use it).
    * ``action(obs, info)``  -- returns the child-order size for the current step.

Core benchmarks (full period, L2 only): TWAP, Almgren-Chriss, FixedLiquidity-
Participation. A true volume-weighted benchmark is NOT included here because the
L2 feed carries no traded volume; that is a separate, test-period-only supplement.
"""
from __future__ import annotations

import numpy as np


class TWAP:
    """Time-weighted average price: an equal child order at every decision step."""

    absolute = True

    def __init__(self, initial_inventory: float, n_steps: int) -> None:
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        self.initial_inventory = float(initial_inventory)
        self.n_steps = int(n_steps)
        self._slice = self.initial_inventory / self.n_steps

    def reset(self, env=None) -> None:  # noqa: ARG002 - stateless; hook for protocol parity
        pass

    def action(self, obs=None, info=None) -> float:
        return self._slice


class AlmgrenChriss:
    """Almgren-Chriss optimal-execution schedule (deterministic child orders).

    The continuous-time solution trades inventory along
        x(t) = X0 * sinh(kappa (T - t)) / sinh(kappa T),
    where ``kappa`` is the trading urgency: larger kappa front-loads execution
    (risk-averse), and ``kappa -> 0`` recovers the linear TWAP schedule. ``kappa``
    is derived from risk-aversion lambda, volatility sigma, and temporary-impact
    eta via ``kappa = sqrt(lambda * sigma^2 / eta)``; sigma and eta are calibrated
    from the real book (see ``calibration.py``).

    AC assumes linear temporary impact and arithmetic Brownian prices, which our
    real, convex LOB does not satisfy: it is a *theoretical reference*, not a
    like-for-like optimum (Hendricks & Wilcox 2014; Macri & Lillo 2024).
    """

    absolute = True

    def __init__(self, child_orders) -> None:
        self.child_orders = np.asarray(child_orders, dtype=float)

    def reset(self, env=None) -> None:  # noqa: ARG002
        self._k = 0

    def action(self, obs=None, info=None) -> float:
        q = self.child_orders[self._k] if self._k < len(self.child_orders) else 0.0
        self._k += 1
        return float(q)

    @classmethod
    def from_kappa(cls, initial_inventory: float, n_steps: int, kappa: float,
                   horizon: float | None = None) -> "AlmgrenChriss":
        """Build the schedule from an urgency ``kappa`` (units 1/time)."""
        X0 = float(initial_inventory)
        N = int(n_steps)
        T = float(horizon) if horizon else float(N)
        t = np.arange(N + 1) * (T / N)                  # t_0 .. t_N
        if abs(kappa) < 1e-12:
            x = X0 * (T - t) / T                         # linear limit == TWAP
        else:
            x = X0 * np.sinh(kappa * (T - t)) / np.sinh(kappa * T)
        x[0], x[-1] = X0, 0.0                            # exact endpoints
        return cls(-np.diff(x))                          # child_j = x_{j-1} - x_j

    @classmethod
    def from_urgency(cls, initial_inventory: float, n_steps: int,
                     kappa_t: float) -> "AlmgrenChriss":
        """Build from the dimensionless urgency ``kappa*T`` (0 -> TWAP)."""
        return cls.from_kappa(initial_inventory, n_steps, kappa_t / float(n_steps),
                              horizon=float(n_steps))

    @staticmethod
    def kappa_from_lambda(lam: float, sigma: float, eta: float) -> float:
        """kappa = sqrt(lambda * sigma^2 / eta), with calibrated sigma, eta."""
        if eta <= 0:
            raise ValueError("eta (temporary impact) must be positive")
        return float(np.sqrt(max(lam, 0.0) * sigma ** 2 / eta))


class FixedLiquidityParticipation:
    """Trade a fixed fraction of currently-available ask depth each step.

    A liquidity-responsive baseline (renamed from the repo's "POPV"): it
    participates in *visible book depth*, NOT traded volume, so it is explicitly
    not a percentage-of-volume strategy. Any inventory left at the deadline is
    force-completed by the env.
    """

    absolute = True

    def __init__(self, fraction: float = 0.5, levels: int = 1) -> None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = float(fraction)
        self.levels = int(levels)

    def reset(self, env) -> None:
        self.env = env

    def action(self, obs=None, info=None) -> float:
        return self.fraction * self.env.ask_liquidity(self.levels)
