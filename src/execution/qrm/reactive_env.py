"""Step 4.2 — the reactive execution environment (agent inside the calibrated QRM).

The agent executes a buy meta-order against the LIVING calibrated book: its market
buys consume simulator queue units, the book refills at the calibrated state-dependent
rates, and eating through the best queue moves the price — the endogenous impact
channel the L2 replay could not represent. Design locked in BUILD_PLAN "STEP 4 — PLAN
LOCKED (user-approved 06/07/2026)"; numeric criteria frozen in
reports/qrm_step4_criteria.md BEFORE this file was written.

Key mechanics (each with a locked rationale):

* **Cadence:** one agent decision per second = 2 simulator intervals of 0.5 s.
* **Action:** TWAP-pace multiple m in {0, 0.5, 0.8, 1, 1.2, 1.5, 2}; the desired
  quantity is m * remaining/steps_left BTC, pushed through a **fractional-unit carry**
  (sub-unit amounts accrue until >= 1 AES unit) so slow pacing survives the unit floor.
  The always-1.0x policy is adaptive TWAP — the built-in fairness anchor.
* **Fills:** a buy of n units walks the ask window outward from the best non-empty
  slot, one unit per queue position, priced at that slot's price. Consumption mutates
  the simulator state directly — that IS the reaction channel.
* **Common random numbers:** the per-episode seed pre-draws the ENTIRE exogenous
  jump path at reset, so every policy evaluated on the same seed faces the identical
  jump sequence (the ~90 %-of-variance component cancels exactly in paired
  comparisons). The endogenous event stream is seed-aligned too but necessarily
  diverges after the first differing action (state-dependent rates) — stated, not
  hidden.
* **Reward:** negative per-step implementation shortfall vs the arrival mid, in bps of
  the arrival notional (the L2 track's locked reward, unchanged).
* **Deadline:** unexecuted remainder at the last step is force-bought through the
  CURRENT book; remainder beyond the visible window prices at the deepest visible ask
  (stated, conservative; rare at the primary size).

The vendored engine is reused as a library; never edited.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from execution.qrm.assemble import QRMBundle, load_bundle
from execution.qrm.exo_ref_sim import MoveProcess, _seed_numba
from execution.qrm.vendored import add_vendored_path

logger = logging.getLogger(__name__)

ACTIONS = (0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0)   # TWAP-pace multiples (L2 grid)
INTERVALS_PER_DECISION = 2                        # 1 s decision / 0.5 s sim tick
WARMUP_INTERVALS = 60                             # 30 s background before the agent starts
FLOW_WINDOW_DECISIONS = 30                        # trailing 30 s for the flow features


@dataclass
class EpisodeState:
    """Everything mutable within one episode."""

    state: np.ndarray            # int8 queue units, [bid_1..K, ask_1..K]
    p_ref: float
    p_mid: float
    arrival_mid: float = 0.0
    remaining_btc: float = 0.0
    carry_btc: float = 0.0       # fractional-unit accumulator
    step_idx: int = 0
    move_path: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    move_idx: int = 0
    last_fill_units: int = 0
    cum_fill_units: int = 0
    flow_market_units: List[int] = field(default_factory=list)   # per decision
    flow_signed_units: List[int] = field(default_factory=list)
    s2_path: Optional[np.ndarray] = None      # shadow background signal per interval
    delta_path: Optional[np.ndarray] = None   # injected ticks per interval (pre-computed)


class ReactiveQRMEnv:
    """Gymnasium-style env: a buy meta-order inside the calibrated reactive QRM."""

    def __init__(
        self,
        bundle_path: str,
        move_process_path: str,
        order_btc: float = 25.0,
        n_steps: int = 300,
        theta_reinit: float = 0.1,
        max_events_per_interval: int = 4000,
        signal_injection: bool = False,
        signal_residual_bps: float = 0.0,
        signal_mean: float = 0.0,
        signal_ema_halflife_s: Optional[float] = None,
        signal_kernel: Optional[dict] = None,
    ) -> None:
        add_vendored_path()
        from qrm_core.engine import simulate_QRM_jit          # noqa: PLC0415
        from qrm_core.sampling import sample_stationary_lob, update_LOB  # noqa: PLC0415

        self._sim = simulate_QRM_jit
        self._sample_lob = sample_stationary_lob
        self._update_lob = update_LOB
        self.bundle: QRMBundle = load_bundle(bundle_path)
        mp = np.load(move_process_path)
        self.move_process = MoveProcess(
            interval_s=float(mp["interval_s"]), moves=mp["moves"], probs=mp["probs"])
        self.order_btc = float(order_btc)
        self.n_steps = int(n_steps)
        self.theta_reinit = float(theta_reinit)
        self.max_events = int(max_events_per_interval)
        self.K = self.bundle.K
        self.tick = self.bundle.tick
        self.aes1 = float(self.bundle.aes[0])      # BTC per queue unit at the touch
        self.signal_injection = bool(signal_injection)
        self.signal_residual_bps = float(signal_residual_bps)
        self.signal_mean = float(signal_mean)
        # Amendment 2: persistence-matched driver. None = instantaneous S2 (Amendment-1
        # behaviour, kept for tests); a half-life in seconds smooths the driver with a
        # per-interval EMA so its persistence can match the real book's.
        self.signal_ema_halflife_s = signal_ema_halflife_s
        if signal_ema_halflife_s is None:
            self._ema_alpha = None
        else:
            self._ema_alpha = 1.0 - 2.0 ** (-0.5 / float(signal_ema_halflife_s))
        # Amendment 3: derived multi-timescale kernel. Keys: halflives_s (list),
        # gains_bps (list, bps per unit demeaned component), means (list),
        # obs_norm (std of the composite driver; observation/follower units).
        self.signal_kernel = signal_kernel
        if signal_kernel is not None:
            self._k_alphas = [1.0 - 2.0 ** (-0.5 / float(h))
                              for h in signal_kernel["halflives_s"]]
            self._k_gains = [float(g) for g in signal_kernel["gains_bps"]]
            self._k_means = [float(m) for m in signal_kernel["means"]]
            self._k_norm = float(signal_kernel.get("obs_norm", 1.0)) or 1.0
            self._k_offset = float(signal_kernel.get("offset_bps", 0.0))
        self.obs_dim = 2 + 2 * self.K + 1 + 2 + 2  # inv, time, queues, spread, fills, flow
        if self.signal_injection:
            self.obs_dim += 1                      # the pre-computed background signal
        self.n_actions = len(ACTIONS)
        self._ep: Optional[EpisodeState] = None
        self.n_sampler_stalls = 0

    # ------------------------------------------------------------- lifecycle
    def reset(self, seed: int) -> np.ndarray:
        """Start one episode. With signal injection ON, a background-only SHADOW pass
        runs first (same seed, injection applied self-consistently to its own
        evolution) to pre-compute the policy-independent signal and injected-move
        paths; the episode is then re-initialised identically and consumes them."""
        obs = self._init_episode(seed)
        if self.signal_injection:
            s2_path, delta_path = self._shadow_pass()
            obs = self._init_episode(seed)         # identical fresh start (re-seeded)
            assert self._ep is not None
            self._ep.s2_path = s2_path
            self._ep.delta_path = delta_path
            obs = self._observe()
        return obs

    def _shadow_pass(self) -> Tuple[np.ndarray, np.ndarray]:
        """Background-only counterfactual of the CURRENT episode: no agent, injection
        applied to its own evolution (self-consistent). Returns per-interval
        (S2_bg, injected ticks) for the post-warm-up intervals."""
        ep = self._ep
        assert ep is not None
        n = self.n_steps * INTERVALS_PER_DECISION
        s2_path = np.zeros(n)
        delta_path = np.zeros(n, dtype=int)
        carry = 0.0
        ema = self.signal_mean                     # neutral start (Amendment 2a)
        emas = list(self._k_means) if self.signal_kernel is not None else None
        for i in range(n):
            inst = self._s2_bg(ep)
            if self.signal_kernel is not None:     # Amendment 3: composite driver
                if np.isfinite(inst):
                    for c, a in enumerate(self._k_alphas):
                        emas[c] += a * (inst - emas[c])
                delta_bps = float(sum(
                    g * (emas[c] - self._k_means[c])
                    for c, g in enumerate(self._k_gains))) - self._k_offset
                s2_path[i] = delta_bps / self._k_norm   # driver in std units
            elif self._ema_alpha is None:
                s2_path[i] = inst
                s2 = s2_path[i] if np.isfinite(s2_path[i]) else self.signal_mean
                delta_bps = self.signal_residual_bps * (s2 - self.signal_mean)
            else:
                if np.isfinite(inst):
                    ema += self._ema_alpha * (inst - ema)
                s2_path[i] = ema                   # the smoothed driver IS the signal
                s2 = s2_path[i] if np.isfinite(s2_path[i]) else self.signal_mean
                delta_bps = self.signal_residual_bps * (s2 - self.signal_mean)
            carry += ep.p_mid * delta_bps * 1e-4 / self.tick
            d = int(carry)                         # deterministic fractional carry
            carry -= d
            delta_path[i] = d
            self._run_interval(track_flow=False, extra_move=d)
        return s2_path, delta_path

    def _s2_bg(self, ep: EpisodeState) -> float:
        """Depth imbalance of the first non-empty level per side, sizes in BTC
        (mirrors the registered Phase A2 definition); NaN while a side is swept."""
        bi, ai = self._best_slots(ep)
        if bi >= self.K or ai >= self.K:
            return float("nan")
        b = float(ep.state[bi]) * float(self.bundle.aes[bi])
        a = float(ep.state[self.K + ai]) * float(self.bundle.aes[ai])
        return (b - a) / (b + a) if (b + a) > 0 else float("nan")

    def _init_episode(self, seed: int) -> np.ndarray:
        """Seed ALL randomness, pre-draw the jump path, warm up (pre-change body)."""
        rng = np.random.default_rng(seed)
        _seed_numba(seed)   # the vendored njit dynamics use numba's per-thread RNG.
        # NOTE (remediation I10): the global np.random stream is deliberately NOT
        # touched — SB3's DQN epsilon coin-flips read it, and reseeding per episode
        # entangled exploration with episode indices.
        n_intervals = (self.n_steps * INTERVALS_PER_DECISION) + WARMUP_INTERVALS + 8
        move_path = np.array(
            [self.move_process.sample(rng) for _ in range(n_intervals)], dtype=int)

        state = np.empty(2 * self.K, np.int8)
        state[: self.K] = self._sample_lob(self.bundle.inv_bid, np.empty((0,), np.int8))
        state[self.K:] = self._sample_lob(self.bundle.inv_ask, np.empty((0,), np.int8))
        ep = EpisodeState(state=state, p_ref=100_000.0, p_mid=0.0,
                          move_path=move_path)
        ep.p_mid = self._mid_from_state(ep)
        self._ep = ep
        for _ in range(WARMUP_INTERVALS):          # background-only warm-up
            self._run_interval(track_flow=False)
        ep.arrival_mid = ep.p_mid
        ep.remaining_btc = self.order_btc
        ep.step_idx = 0
        return self._observe()

    # ------------------------------------------------------------ mechanics
    def _best_slots(self, ep: EpisodeState) -> Tuple[int, int]:
        """(bid_slot, ask_slot) of the first non-empty queue per side; an EMPTY side
        reads as the window edge K (one past the deepest slot) instead of argmax's
        silent 0 (remediation I5: a swept side must widen the quoted book, not pin
        the mid at a stale price)."""
        bid = ep.state[: self.K]
        ask = ep.state[self.K:]
        bid_i = int(np.argmax(bid > 0)) if bid.any() else self.K
        ask_i = int(np.argmax(ask > 0)) if ask.any() else self.K
        return bid_i, ask_i

    def _mid_from_state(self, ep: EpisodeState) -> float:
        bid_i, ask_i = self._best_slots(ep)
        return 0.5 * ((ep.p_ref + self.tick * (ask_i + 0.5))
                      + (ep.p_ref - self.tick * (bid_i + 0.5)))

    def _spread_ticks(self, ep: EpisodeState) -> float:
        bid_i, ask_i = self._best_slots(ep)
        return float(bid_i + ask_i + 1)

    def _run_interval(self, track_flow: bool, extra_move: Optional[int] = None
                      ) -> Tuple[int, int]:
        """One 0.5 s slice: endogenous dynamics, then the pre-drawn exogenous move
        (plus, with signal injection ON, the pre-computed injected component).

        ``extra_move``: None -> resolved from the episode's delta_path (actual
        episodes); an int -> used directly (the shadow pass, which computes it on
        the fly). With injection OFF it resolves to 0 and the arithmetic is
        byte-identical to the pre-change environment.

        Returns (market_units, signed_units) of BACKGROUND trades in the interval
        (buy-aggressor = +, sell-aggressor = -) for the flow features.
        """
        ep = self._ep
        assert ep is not None
        mkt = signed = 0
        try:
            (times, p_mids, _pr, sides, _d, events, _r, states) = self._sim(
                0.0, ep.p_mid, ep.p_ref, ep.state, self.bundle.rate_int_all, self.tick,
                0.7, self.theta_reinit, self.move_process.interval_s,
                self.bundle.inv_bid, self.bundle.inv_ask, self.max_events,
                self.bundle.aes, np.nan)
            if len(times):
                ep.state = states[-1].copy()
                ep.p_mid = float(p_mids[-1])
                # remediation I6: keep the engine's evolved reference price. The old
                # code discarded it while keeping the shifted state, silently slipping
                # the price frame and erasing endogenous (incl. agent-caused) moves.
                ep.p_ref = float(_pr[-1])
                if track_flow:
                    is_mkt = events == 3
                    mkt = int(is_mkt.sum())
                    # engine side coding is 1=bid, 2=ask (remediation I4): an ask-side
                    # market order consumes asks = BUY aggressor (+), bid-side = sell (-)
                    signed = int((sides[is_mkt] == 2).sum() - (sides[is_mkt] == 1).sum())
        except ValueError:
            self.n_sampler_stalls += 1             # cornered sampler: book carries over
        dm = int(ep.move_path[ep.move_idx])
        if extra_move is not None:
            dm += int(extra_move)
        elif self.signal_injection and ep.delta_path is not None:
            k = ep.move_idx - WARMUP_INTERVALS
            if 0 <= k < len(ep.delta_path):
                dm += int(ep.delta_path[k])
        ep.move_idx += 1
        if abs(dm) == 1:
            try:
                ep.p_mid, ep.p_ref, ep.state, _ = self._update_lob(
                    self.K, ep.p_ref, ep.state, 1 if dm > 0 else -1, 1.0,
                    self.theta_reinit, self.tick,
                    self.bundle.inv_bid, self.bundle.inv_ask, self.bundle.aes)
            except ValueError:
                self._reform(dm)
        elif dm != 0:
            self._reform(dm)
        ep.p_mid = self._mid_from_state(ep)
        return mkt, signed

    def _reform(self, dm: int) -> None:
        ep = self._ep
        assert ep is not None
        ep.p_ref += dm * self.tick
        ep.state[: self.K] = self._sample_lob(self.bundle.inv_bid, np.empty((0,), np.int8))
        ep.state[self.K:] = self._sample_lob(self.bundle.inv_ask, np.empty((0,), np.int8))

    def _buy_btc(self, target_btc: float, allow_overshoot: bool) -> Tuple[float, float, int]:
        """Market-buy toward ``target_btc``: walk the ask window outward consuming whole
        queue units, where a unit at slot i is aes[i] BTC (the engine's own semantics —
        remediation I3; the old flat-aes1 conversion over-credited depth-2 fills by up
        to 47% in the volatile regime). With ``allow_overshoot=False`` the walk stops
        BEFORE exceeding the target (mid-episode buys: the shortfall stays banked);
        with True it completes past the target by at most one unit (deadline only).
        Returns (cost_usd, btc_filled, units_filled). Consumption mutates the queues
        directly — the reaction channel."""
        ep = self._ep
        assert ep is not None
        cost = 0.0
        btc = 0.0
        units = 0
        while btc < target_btc - 1e-12:
            ask = ep.state[self.K:]
            nz = np.nonzero(ask > 0)[0]
            if len(nz) == 0:                        # visible ask side exhausted
                break
            i = int(nz[0])
            unit_btc = float(self.bundle.aes[i])
            px = ep.p_ref + self.tick * (i + 0.5)
            want = target_btc - btc
            max_units_here = int(ask[i])
            if allow_overshoot:
                n = min(max_units_here, int(np.ceil(want / unit_btc)))
            else:
                n = min(max_units_here, int(want / unit_btc))
                if n == 0:
                    break                            # next whole unit would overshoot
            cost += n * unit_btc * px
            ep.state[self.K + i] -= n
            btc += n * unit_btc
            units += n
        ep.p_mid = self._mid_from_state(ep)
        return cost, btc, units

    def _buy_units(self, n_units: int) -> Tuple[float, int]:
        """Compatibility wrapper for the gate scripts: buy ~n_units x aes1 BTC with
        per-depth pricing/units (remediation I3); returns (cost_usd, touch_units_equiv)."""
        cost, btc, _u = self._buy_btc(n_units * self.aes1, allow_overshoot=True)
        return cost, int(round(btc / self.aes1))

    # ------------------------------------------------------------------ step
    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, dict]:
        ep = self._ep
        assert ep is not None
        steps_left = self.n_steps - ep.step_idx
        pace_btc = ACTIONS[action_idx] * ep.remaining_btc / max(steps_left, 1)
        # remediation I2 (root cause): the bank may only hold UNCOMMITTED remainder —
        # capping by `remaining` alone let cumulative desire exceed the order.
        ep.carry_btc += max(0.0, min(pace_btc, ep.remaining_btc - ep.carry_btc))
        cost = 0.0
        btc_filled = 0.0
        filled = 0
        if ep.carry_btc >= float(self.bundle.aes[0]) - 1e-12:
            cost, btc_filled, filled = self._buy_btc(ep.carry_btc, allow_overshoot=False)
            ep.carry_btc -= btc_filled
            ep.remaining_btc = max(0.0, ep.remaining_btc - btc_filled)
        ep.last_fill_units = filled
        ep.cum_fill_units += filled
        # background market runs for the decision's two intervals
        mkt = signed = 0
        for _ in range(INTERVALS_PER_DECISION):
            m, s = self._run_interval(track_flow=True)
            mkt += m
            signed += s
        ep.flow_market_units.append(mkt)
        ep.flow_signed_units.append(signed)
        ep.step_idx += 1

        done = ep.step_idx >= self.n_steps
        is_usd = cost - btc_filled * ep.arrival_mid
        deadline_residual_btc = 0.0
        if done and ep.remaining_btc > 1e-9:
            # remediation I2: the unexecuted quantity is `remaining` ALONE (the bank
            # is a sub-account of it); the old remaining+carry re-bought the bank and
            # ceil'd a phantom unit -> paced policies over-executed every episode.
            deadline_residual_btc = ep.remaining_btc
            is_usd += self._finalize()
        reward = -is_usd / (ep.arrival_mid * self.order_btc) * 1e4  # bps of arrival notional
        info = {"filled_units": filled, "remaining_btc": ep.remaining_btc,
                "spread_ticks": self._spread_ticks(ep),
                "deadline_residual_btc": deadline_residual_btc}
        return self._observe(), float(reward), done, info

    def _finalize(self) -> float:
        """Deadline rule: force-buy the TRUE remainder through the CURRENT book
        (whole units, per-depth sizes, overshoot <= one unit — remediation I2/I3);
        anything beyond the visible window prices at the deepest visible ask
        (frozen rule, documented as generous)."""
        ep = self._ep
        assert ep is not None
        leftover_btc = ep.remaining_btc
        cost, btc, units = self._buy_btc(leftover_btc, allow_overshoot=True)
        short_btc = max(0.0, leftover_btc - btc)
        if short_btc > 1e-12:
            deepest = ep.p_ref + self.tick * (self.K - 0.5)
            cost += short_btc * deepest
            btc += short_btc
        ep.cum_fill_units += units
        ep.remaining_btc = ep.carry_btc = 0.0
        ep.last_fill_units = units
        return cost - btc * ep.arrival_mid

    # ----------------------------------------------------------- utilities
    def force_buy_all(self) -> float:
        """Execute the ENTIRE remaining order right now (true instant dump).

        Not reachable through the agent's action grid (capped at 2.0x pace) — used by
        the G3 mechanics gate and diagnostics only. Walks the current book; anything
        beyond the visible window prices at the deepest visible ask (the same frozen
        deadline rule). Returns the implementation shortfall in bps of arrival
        notional (positive = cost)."""
        ep = self._ep
        assert ep is not None
        is_usd = self._finalize()
        return float(is_usd / (ep.arrival_mid * self.order_btc) * 1e4)

    # ------------------------------------------------------------- observe
    def _observe(self) -> np.ndarray:
        ep = self._ep
        assert ep is not None
        w = FLOW_WINDOW_DECISIONS
        mkt = float(sum(ep.flow_market_units[-w:]))
        signed = float(sum(ep.flow_signed_units[-w:]))
        obs = np.concatenate([
            [ep.remaining_btc / self.order_btc,
             1.0 - ep.step_idx / self.n_steps],
            np.log1p(ep.state.astype(np.float32)),          # 2K queue sizes
            [self._spread_ticks(ep) / self.K,
             float(ep.last_fill_units), ep.cum_fill_units * self.aes1 / self.order_btc,
             np.log1p(mkt), np.sign(signed) * np.log1p(abs(signed))],
        ]).astype(np.float32)
        if self.signal_injection:
            if ep.s2_path is not None:
                k = min(max(ep.move_idx - WARMUP_INTERVALS, 0), len(ep.s2_path) - 1)
                v = ep.s2_path[k]
                sig = float(v) if np.isfinite(v) else self.signal_mean
            else:
                sig = self.signal_mean
            obs = np.concatenate([obs, np.array([sig], dtype=np.float32)])
        return obs
