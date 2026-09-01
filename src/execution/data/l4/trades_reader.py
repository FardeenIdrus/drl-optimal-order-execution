"""Read BTC trades (market orders) from one hourly trades file.

The Oxford trades archive stores one gzip'd JSON-lines file per hour (``<date>/<H>.gz``,
all coins interleaved). Each line is one trade with an aggressor ``side`` ("A" = sell
aggressor, "B" = buy aggressor), price, size, and counterparties. For the QRM these are
the authoritative **market-order** events (the book-diff stream omits ~1/3 of fills).

A buy-aggressor trade consumes liquidity on the **ask** side; a sell-aggressor consumes
the **bid** side -- so the "resting side" we return is the opposite of the aggressor.
"""
from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

from execution.data.l4.book_diffs_reader import ASK, BID

logger = logging.getLogger(__name__)


def read_trades(path: Union[str, Path], coin: str = "BTC") -> List[Tuple[int, str, float, float]]:
    """Time-sorted ``(ts_ns, resting_side, px, sz)`` for every ``coin`` trade in the hour.

    ``resting_side`` is :data:`~execution.data.l4.book_diffs_reader.ASK` for a buy-aggressor
    trade (side "B") and ``BID`` for a sell-aggressor ("A"): the side whose depth was consumed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"trades file not found: {path}")
    out: List[Tuple[int, str, float, float]] = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("coin") != coin:
                continue
            ts = int(np.datetime64(r["time"], "ns").astype("int64"))
            resting = ASK if r["side"] == "B" else BID  # aggressor hits the opposite side
            out.append((ts, resting, float(r["px"]), float(r["sz"])))
    out.sort(key=lambda x: x[0])
    logger.info("trades %s: %d %s trades", path.name, len(out), coin)
    return out
