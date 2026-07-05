"""Locate and import the vendored QRM engine (`qrm_optimal_execution`) as a library.

The vendored repo lives beside this one (``code/qrm_optimal_execution``); it is reused
read-only (Option-A boundary: we never edit it). This helper makes its ``qrm_core``
importable from our package and tests without a hard-coded absolute path.
"""
from __future__ import annotations

import sys
from pathlib import Path


def add_vendored_path() -> Path:
    """Append the vendored repo's ``src`` to ``sys.path``; return it. Idempotent."""
    # this file: <repo>/src/execution/qrm/vendored.py  ->  <code>/ is repo.parent
    code_dir = Path(__file__).resolve().parents[4]
    vend = code_dir / "qrm_optimal_execution" / "src"
    if not vend.is_dir():
        raise FileNotFoundError(f"vendored QRM repo not found at {vend}")
    s = str(vend)
    if s not in sys.path:
        sys.path.append(s)
    return vend
