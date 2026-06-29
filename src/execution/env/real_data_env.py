"""Real-data order-execution environment (replaces the QRM simulator as book source).

A Gymnasium environment for executing a buy meta-order over a fixed number of
decision steps against *replayed* real Hyperliquid L2 books (the frozen Stage-6
datasets). Each step walks the real ask ladder for the chosen quantity
(:func:`execution.env.fills.walk_ask_ladder`); the reward is the negative
per-step implementation shortfall versus the episode's arrival (decision) price.

Design decisions (defensible, for the methodology chapter)
----------------------------------------------------------
* **Replay, not simulation.** The book at each step is fixed historical data.
  Direct/temporary impact (our order walking real visible depth) is modelled;
  indirect/permanent impact (the book reacting to us, refilling, others
  responding) is *not*, by construction of replay. This is the
  counterfactual-feedback limitation stated in HANDOVER, made explicit rather
  than hidden. The QRM book-rewrite machinery (``_write_batch`` / ``update_LOB``)
  is therefore intentionally dropped.
* **Arrival price = mid as-of the episode's first decision** (Perold
  implementation-shortfall benchmark).
* **Continuous BTC quantities** (a divisible asset), unlike the QRM integer
  shares: no rounding.
* **Fill mechanic adapted from QRM** (arXiv 2511.15262), reading our explicit
  20-level ladder instead of the simulator's uniform tick grid.

The ``reward`` carries a terminal penalty on un-executed inventory (a training
signal). ``realized_cost`` is the honest implementation shortfall in USD used for
evaluation; in ``test_mode`` it force-completes any residual against the last
visible book so the full meta-order is always accounted for.
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from execution.env.episode_store import EpisodeStore
from execution.env.fills import QTY_EPS, walk_ask_ladder
from execution.env.normalize import FeatureNormalizer


class RealDataExecutionEnv(gym.Env):
    """Execute a buy meta-order against replayed real L2 books."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        store: EpisodeStore,
        *,
        initial_inventory: float,
        final_penalty: float = 1.0,
        actions: tuple[float, ...] = (0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0),
        action_mode: str = "twap_pace_multiple",
        obs_features: tuple[str, ...] = (),
        normalizer: FeatureNormalizer | None = None,
        test_mode: bool = False,
        residual_coef_by_regime: dict[str, float] | None = None,
        adv_btc: float | None = None,
        reward_bps: bool = False,
        leftover_penalty_bps: float = 100.0,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        if initial_inventory <= 0:
            raise ValueError("initial_inventory must be positive")
        unknown = set(obs_features) - set(store.feature_names)
        if unknown:
            raise ValueError(f"obs_features not in store: {sorted(unknown)}")
        if normalizer is not None and tuple(normalizer.feature_names) != tuple(store.feature_names):
            raise ValueError("normalizer.feature_names must match store.feature_names")

        self.store = store
        self.initial_inventory = float(initial_inventory)
        self.final_penalty = float(final_penalty)
        self.actions = tuple(float(a) for a in actions)
        if action_mode not in ("twap_pace_multiple", "inventory_fraction"):
            raise ValueError(f"unknown action_mode: {action_mode!r}")
        self.action_mode = action_mode
        self.obs_features = tuple(obs_features)
        # Train-fit transform applied to the feature block of the observation. When
        # None the raw features pass through (intended for tests/diagnostics only;
        # real training/eval MUST pass the train-fit normalizer, see runner).
        self.normalizer = normalizer
        self.test_mode = bool(test_mode)
        # Square-root deadline-residual pricing (Setup Step 2): per-regime
        # sqrt_coef (= Y*sigma) and ADV (V). When unset, the residual falls back
        # to the deepest-visible anchor only (no extrapolation beyond the book).
        self.residual_coef_by_regime = residual_coef_by_regime
        self.adv_btc = float(adv_btc) if adv_btc else None
        # Reward scaling (Setup Step 5): when True the per-step reward is the
        # negative implementation shortfall in BPS of arrival notional (well-
        # conditioned for DQN/PPO and the SAME objective as the is_bps eval
        # metric), and the terminal leftover penalty is in bps. This flag changes
        # only the learning signal -- realized_cost / is_bps (the eval accounting)
        # are unaffected.
        self.reward_bps = bool(reward_bps)
        self.leftover_penalty_bps = float(leftover_penalty_bps)
        self.n_steps = store.n_steps
        self._feat_idx = [store.feature_names.index(f) for f in self.obs_features]

        self.action_space = spaces.Discrete(len(self.actions))
        obs_dim = 2 + len(self.obs_features)  # [inventory, time, *features]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        # Benchmark hook: when True, step() reads the action as an absolute BTC
        # quantity (used by TWAP/VWAP) rather than a discrete fraction index.
        self._absolute_quantity = False
        self._ep: int | None = None

    # ----------------------------------------------------------------- lifecycle
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "episode_index" in options:
            ep = int(options["episode_index"])
            if not 0 <= ep < self.store.n_episodes:
                raise IndexError(f"episode_index {ep} out of range [0,{self.store.n_episodes})")
            self._ep = ep
        else:
            self._ep = int(self._rng.integers(self.store.n_episodes))

        self._step = 0
        self.inventory = self.initial_inventory
        self.arrival_price = float(self.store.mid[self._ep, 0])
        self.realized_cost = 0.0   # cumulative implementation shortfall (USD)
        self.cum_reward = 0.0
        self.n_filled = 0.0
        self._residual_btc = 0.0   # BTC force-completed beyond visible depth (deadline)
        self._residual_cost = 0.0  # USD cost of that beyond-depth residual
        return self._obs(), self._info(filled=0.0, fill=None, terminal=False)

    def set_absolute_quantity(self, flag: bool) -> None:
        """Benchmark hook: interpret ``step(action)`` as an absolute BTC quantity."""
        self._absolute_quantity = bool(flag)

    def ask_liquidity(self, levels: int = 1) -> float:
        """Available ask size (BTC) at the current decision step, top ``levels``.

        Used by liquidity-responsive benchmarks to size a child order off the same
        book they will trade against (the current step, not a future one).
        """
        if self._ep is None:
            raise RuntimeError("ask_liquidity() called before reset()")
        step = min(self._step, self.n_steps - 1)
        sz = self.store.ask_sz[self._ep, step, :levels]
        sz = sz[np.isfinite(sz) & (sz > 0.0)]
        return float(sz.sum())

    # ---------------------------------------------------------------- transition
    def step(self, action: Any):
        if self._ep is None:
            raise RuntimeError("step() called before reset()")
        if self._step >= self.n_steps:
            raise RuntimeError("step() after episode end; call reset()")

        px = self.store.ask_px[self._ep, self._step]
        sz = self.store.ask_sz[self._ep, self._step]

        qty = min(self._quantity_for(action), self.inventory)
        fill = walk_ask_ladder(px, sz, qty)

        step_cost = fill.cash - self.arrival_price * fill.filled  # IS contribution (USD)
        reward = self._reward(step_cost)
        self.inventory -= fill.filled
        self.realized_cost += step_cost
        self.n_filled += fill.filled

        self._step += 1
        done = self._step >= self.n_steps or self.inventory <= QTY_EPS
        if done:
            reward += self._finalize()
        self.cum_reward += reward

        return self._obs(), float(reward), bool(done), False, self._info(fill.filled, fill, done)

    # ------------------------------------------------------------------ internals
    def _reward(self, step_cost: float) -> float:
        """Per-step learning signal: negative IS, in bps of notional or raw USD."""
        if self.reward_bps:
            notional = self.arrival_price * self.initial_inventory
            return -step_cost / notional * 1e4 if notional > 0 else 0.0
        return -step_cost

    def _leftover_penalty(self) -> float:
        """Terminal penalty on un-executed inventory (a finish-the-order signal).

        In bps mode the full order left unexecuted costs ``leftover_penalty_bps``,
        scaled by the fraction remaining -- strong enough to dominate the typical
        single-digit-bps IS so the agent learns to finish, in the same units as
        the reward. In raw-USD mode it is the legacy ``final_penalty`` per BTC.
        """
        if self.reward_bps:
            return -self.leftover_penalty_bps * (self.inventory / self.initial_inventory)
        return -self.final_penalty * self.inventory

    def _quantity_for(self, action: Any) -> float:
        """Map a discrete action to a requested buy quantity (BTC).

        Two modes (``action_mode``):

        * **twap_pace_multiple** (canonical) -- ``qty = mult * inventory /
          steps_left``. The ``mult = 1`` action requests exactly the TWAP slice
          (``remaining / steps_left``) at every step, so the policy can reproduce
          TWAP and is interpretable as *deviation* from it; ``mult > 1``
          front-loads (walks deeper into the book when it pays), ``mult < 1``
          delays. The order is always finishable (near the deadline
          ``steps_left -> 1`` so ``mult >= 1`` clears the remainder).
        * **inventory_fraction** (superseded Step-0 mode, retained for ablation)
          -- ``qty = frac * inventory``. Coarse grids here cannot represent TWAP's
          per-step pace (1/steps_left), which is why it was superseded.

        Both decouple throughput from L1 depth (a large request walks multiple
        ladder levels, making ``ask_depth`` actionable). Benchmarks bypass this
        via the absolute-quantity hook.
        """
        if self._absolute_quantity:
            return max(0.0, float(action))
        a = self.actions[int(action)]
        if self.action_mode == "twap_pace_multiple":
            steps_left = self.n_steps - self._step          # >= 1 (step() guards end)
            return a * self.inventory / steps_left
        return a * self.inventory                           # inventory_fraction

    def _finalize(self) -> float:
        """Terminal handling for any un-executed inventory.

        Returns the reward adjustment (a penalty on leftover inventory). In
        ``test_mode`` the residual is force-completed against the last visible
        book and folded into ``realized_cost`` so evaluation always reflects the
        full meta-order.
        """
        if self.inventory <= QTY_EPS:
            return 0.0
        adj = self._leftover_penalty()  # training signal: finish the order
        if self.test_mode:
            last = self._step - 1
            px = self.store.ask_px[self._ep, last]
            fill = walk_ask_ladder(px, self.store.ask_sz[self._ep, last], self.inventory)
            cost = fill.cash - self.arrival_price * fill.filled
            residual = self.inventory - fill.filled
            if residual > QTY_EPS:  # beyond visible depth: square-root residual
                cost += self._residual_cost_for(residual, px)
            self.realized_cost += cost
            self.n_filled += self.inventory
            self.inventory = 0.0
        return adj

    def _residual_cost_for(self, residual: float, px: np.ndarray) -> float:
        """USD cost of clearing ``residual`` BTC left beyond visible depth at the deadline.

        Anchored at the deepest visible ask ``worst`` so the cost curve is
        continuous with the within-book fill (the anchor term -> 0 as residual ->
        0), PLUS the square-root-law slippage for depth beyond the book
        (regime-specific ``sqrt_coef`` = Y*sigma; ADV = V). Both terms vanish as
        residual -> 0, so there is no jump at the depth boundary. When no
        calibration was supplied the anchor term alone is used. Logged for
        transparency / sizing diagnostics.
        """
        finite = px[np.isfinite(px)]
        worst = float(finite[-1]) if finite.size else self.arrival_price
        cost = residual * (worst - self.arrival_price)            # continuous anchor
        coef = self._residual_coef()
        if coef is not None and np.isfinite(coef) and self.adv_btc:
            cost += residual * worst * coef * float(np.sqrt(residual / self.adv_btc))
        self._residual_btc = residual
        self._residual_cost = cost
        return cost

    def _residual_coef(self) -> float | None:
        """Per-regime square-root coefficient (Y*sigma) for the current episode."""
        if self.residual_coef_by_regime is None:
            return None
        return self.residual_coef_by_regime.get(str(self.store.regime[self._ep]))

    def _obs(self) -> np.ndarray:
        inv_norm = 2.0 * self.inventory / self.initial_inventory - 1.0
        steps_left = self.n_steps - self._step
        time_norm = 2.0 * steps_left / self.n_steps - 1.0
        vec = [inv_norm, time_norm]
        if self._feat_idx:
            row = min(self._step, self.n_steps - 1)
            feats = self.store.features[self._ep, row]            # raw, full feature row
            if self.normalizer is not None:
                feats = self.normalizer.transform(feats)          # train-fit transform
            vec.extend(np.asarray(feats)[self._feat_idx].tolist())
        return np.asarray(vec, dtype=np.float32)

    @property
    def is_bps(self) -> float:
        """Realized implementation shortfall in basis points of arrival notional."""
        notional = self.arrival_price * self.initial_inventory
        return 1e4 * self.realized_cost / notional if notional > 0 else float("nan")

    def _info(self, filled: float, fill, terminal: bool) -> dict:
        return {
            "episode_index": self._ep,
            "episode_id": int(self.store.episode_ids[self._ep]),
            "regime": str(self.store.regime[self._ep]),
            "step": self._step,
            "filled": filled,
            "inventory": self.inventory,
            "arrival_price": self.arrival_price,
            "realized_cost": self.realized_cost,
            "is_bps": self.is_bps,
            "residual_btc": self._residual_btc,
            "residual_bps": (1e4 * self._residual_cost / (self.arrival_price * self.initial_inventory)
                             if self.arrival_price * self.initial_inventory > 0 else 0.0),
            "levels_walked": fill.levels_walked if fill else 0,
            "exhausted": fill.exhausted if fill else False,
            "reward_cum": self.cum_reward,
            "terminal": terminal,
        }
