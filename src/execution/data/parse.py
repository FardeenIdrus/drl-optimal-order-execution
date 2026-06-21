"""Stage 2 - parse raw L2 files into the canonical book schema.

Decompresses one hourly ``.lz4`` file, decodes it via the source adapter, validates
the snapshots, and returns a clean canonical DataFrame. Streaming and stateless:
one file in, one DataFrame out, so the caller (Stage 3) processes files one at a
time and the full-resolution history is never materialised on disk.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Tuple, Union

import pandas as pd

from execution.data import schema
from execution.data.sources import hyperliquid_l2 as hl

logger = logging.getLogger(__name__)

_LZ4_BIN = shutil.which("lz4") or "/opt/homebrew/bin/lz4"


def _decompress(path: Path) -> str:
    """Decompress an LZ4 file to text. Prefer python-lz4; fall back to the lz4 CLI."""
    try:
        import lz4.frame  # type: ignore

        with lz4.frame.open(path, "rb") as fh:
            return fh.read().decode("utf-8")
    except ImportError:
        proc = subprocess.run([_LZ4_BIN, "-dc", str(path)], capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"lz4 decompress failed for {path}: "
                f"{proc.stderr.decode('utf-8', 'replace')[:200]}"
            )
        return proc.stdout.decode("utf-8")


def parse_file(
    path: Union[str, Path], validate: bool = True
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Parse one hourly ``.lz4`` file into a validated canonical book DataFrame.

    Args:
        path: path to a ``<COIN>.lz4`` hourly file.
        validate: if True, drop snapshots that fail :func:`schema.validate`.

    Returns:
        ``(df, stats)``. ``df`` conforms to :mod:`schema`. ``stats`` records the
        file, malformed-line count, and (when validated) the validation breakdown.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"raw L2 file not found: {path}")

    text = _decompress(path)
    df, n_malformed = hl.decode(text)
    stats: Dict[str, object] = {
        "file": str(path),
        "snapshots_decoded": int(len(df)),
        "malformed_lines": int(n_malformed),
    }

    if validate and not df.empty:
        valid_mask, counts = schema.validate(df)
        df = df[valid_mask].reset_index(drop=True)
        stats["validation"] = counts
        stats["snapshots_valid"] = int(len(df))

    logger.info(
        "parsed %s: %d valid snapshots (%d malformed lines)",
        path.name,
        stats.get("snapshots_valid", stats["snapshots_decoded"]),
        n_malformed,
    )
    return df, stats
