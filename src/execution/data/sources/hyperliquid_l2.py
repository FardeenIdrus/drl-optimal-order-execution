"""Hyperliquid L2 source adapter (native AWS archive).

Provider-specific access to ``s3://hyperliquid-archive``. Layout:

    market_data/<YYYYMMDD>/<hour>/l2Book/<COIN>.lz4   (hour is 0-23, not zero-padded)

Each ``.lz4`` is one hour of LZ4-compressed NDJSON l2Book snapshots (20 levels per
side). The bucket is requester-pays, so every request passes ``--request-payer
requester``. Listing/fetching is done via the AWS CLI to avoid a boto3 dependency.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from execution.data import schema

BUCKET = "hyperliquid-archive"
AWS_BIN = shutil.which("aws") or "/opt/homebrew/bin/aws"


def object_key(date: str, hour: int, coin: str = "BTC") -> str:
    """Return the S3 key for one hourly l2Book file. ``date`` is ``YYYYMMDD``."""
    return f"market_data/{date}/{hour}/l2Book/{coin}.lz4"


def list_coin_hours(date: str, coin: str = "BTC", retries: int = 1) -> Dict[int, int]:
    """List which hourly l2Book files exist for ``coin`` on ``date``.

    Returns a mapping ``{hour: size_bytes}`` for every hour present (0-23). An empty
    dict means no data for that date. One paginated ``list-objects-v2`` per date,
    filtered server-response-side to the coin's l2Book files.
    """
    query = f"Contents[?contains(Key, 'l2Book/{coin}.lz4')].[Key, Size]"
    cmd = [
        AWS_BIN, "s3api", "list-objects-v2",
        "--bucket", BUCKET,
        "--prefix", f"market_data/{date}/",
        "--request-payer", "requester",
        "--query", query,
        "--output", "text",
    ]
    last_err = ""
    for _ in range(retries + 1):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            out: Dict[int, int] = {}
            for line in proc.stdout.splitlines():
                line = line.strip()
                if not line or line == "None":
                    continue
                key, size = line.split("\t")
                # key = market_data/<date>/<hour>/l2Book/<coin>.lz4 -> hour at index 2
                hour = int(key.split("/")[2])
                out[hour] = int(size)
            return out
        last_err = proc.stderr.strip()
    raise RuntimeError(f"list failed for {date}: {last_err}")


def decode(ndjson_text: str, n_levels: int = schema.N_LEVELS) -> Tuple[pd.DataFrame, int]:
    """Decode Hyperliquid l2Book NDJSON into the canonical book DataFrame.

    This is the only place Hyperliquid's wire format is interpreted. Each input
    line is a JSON object of the form::

        {"time": <iso>, "ver_num": 1,
         "raw": {"channel": "l2Book",
                 "data": {"coin": "BTC", "time": <ms>,
                          "levels": [[ {px, sz, n}, ... bids ], [ ... asks ]]}}}

    Returns ``(df, n_malformed)`` where ``df`` conforms to :mod:`schema` and
    ``n_malformed`` is the count of lines that could not be parsed (skipped).
    """
    columns = schema.canonical_columns(n_levels)
    records = []
    n_malformed = 0
    for line in ndjson_text.splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)["raw"]["data"]
            bids, asks = data["levels"][0], data["levels"][1]
            row = {schema.TS_COL: int(data["time"])}
            for side, side_levels in (("bid", bids), ("ask", asks)):
                for i in range(n_levels):
                    px = sz = n = np.nan
                    if i < len(side_levels):
                        lvl = side_levels[i]
                        px, sz, n = float(lvl["px"]), float(lvl["sz"]), float(lvl["n"])
                    row[f"{side}_px_{i + 1}"] = px
                    row[f"{side}_sz_{i + 1}"] = sz
                    row[f"{side}_n_{i + 1}"] = n
            records.append(row)
        except (KeyError, ValueError, TypeError, IndexError, json.JSONDecodeError):
            n_malformed += 1

    df = pd.DataFrame.from_records(records, columns=columns)
    if not df.empty:
        df[schema.TS_COL] = df[schema.TS_COL].astype("int64")
    return df, n_malformed
