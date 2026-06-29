"""Ask-ladder fill engine for real-data order execution.

A buy meta-order is executed by walking *up* the ask side of a real Hyperliquid
L2 book, consuming liquidity level by level. This module isolates that mechanic
as a pure, side-effect-free function so it can be unit-tested independently of
the RL environment.

The algorithm is the one used by the QRM scaffold
(``market_environment.MarketEnvironment.step``, arXiv 2511.15262). Here it is
adapted to read an explicit, non-uniform price ladder (real ``ask_px_*`` /
``ask_sz_*`` columns) rather than the simulator's uniform tick grid, and to use
continuous BTC quantities rather than integer shares.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Quantities are in BTC (continuous). Anything at or below this is treated as
# zero, so float dust cannot accumulate across the steps of an episode.
QTY_EPS: float = 1e-9


@dataclass(frozen=True)
class Fill:
    """Outcome of walking the ask ladder for a requested buy quantity."""

    filled: float       # BTC actually bought (<= requested)
    cash: float         # total USD paid for ``filled``
    levels_walked: int  # number of ask levels actually consumed
    exhausted: bool     # True iff the ladder ran out before the request was met

    @property
    def vwap(self) -> float:
        """Volume-weighted average execution price, or NaN if nothing filled."""
        return self.cash / self.filled if self.filled > QTY_EPS else float("nan")


def walk_ask_ladder(ask_px: np.ndarray, ask_sz: np.ndarray, qty: float) -> Fill:
    """Execute a market buy of ``qty`` BTC against an ask ladder.

    Walks levels in ascending-price order, taking ``min(remaining, level_size)``
    at each until the request is met or the ladder is exhausted. Levels with a
    non-finite price/size or a non-positive size are skipped (a real snapshot can
    carry gaps at deeper levels).

    Args:
        ask_px: ask prices, ascending, shape ``(L,)``.
        ask_sz: ask sizes per level (BTC), shape ``(L,)``.
        qty:    requested buy quantity (BTC), ``>= 0``.

    Returns:
        A :class:`Fill` with the filled quantity, cash paid, levels consumed, and
        whether the ladder was exhausted before the request was satisfied.

    Raises:
        ValueError: if ``ask_px`` and ``ask_sz`` have different shapes.
    """
    if ask_px.shape != ask_sz.shape:
        raise ValueError(f"ladder shape mismatch: px{ask_px.shape} vs sz{ask_sz.shape}")
    if qty <= QTY_EPS:
        return Fill(filled=0.0, cash=0.0, levels_walked=0, exhausted=False)

    remaining = float(qty)
    cash = 0.0
    filled = 0.0
    levels_walked = 0
    for px, sz in zip(ask_px, ask_sz):
        px = float(px)
        sz = float(sz)
        if not np.isfinite(px) or not np.isfinite(sz) or sz <= 0.0:
            continue
        take = remaining if remaining < sz else sz
        cash += take * px
        filled += take
        remaining -= take
        levels_walked += 1
        if remaining <= QTY_EPS:
            break

    return Fill(
        filled=filled,
        cash=cash,
        levels_walked=levels_walked,
        exhausted=remaining > QTY_EPS,
    )
