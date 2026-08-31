"""Locate and import the vendored QRM engine (`qrm_optimal_execution`) as a library.

The engine is vendored inside this repository (``qrm_optimal_execution/``) and is
reused read-only; it is never edited. This helper makes its ``qrm_core`` importable
from our package and tests without a hard-coded absolute path. A side-by-side
checkout beside the repository is accepted as a fallback.
"""
from __future__ import annotations

import sys
from pathlib import Path


def add_vendored_path() -> Path:
    """Append the vendored engine's ``src`` to ``sys.path``; return it. Idempotent."""
    # this file: <repo>/src/execution/qrm/vendored.py  ->  parents[3] is the repo root
    repo = Path(__file__).resolve().parents[3]
    candidates = (repo / "qrm_optimal_execution" / "src",          # tracked, in-repo copy
                  repo.parent / "qrm_optimal_execution" / "src")   # side-by-side checkout
    for vend in candidates:
        if vend.is_dir():
            s = str(vend)
            if s not in sys.path:
                sys.path.append(s)
            return vend
    raise FileNotFoundError(
        f"vendored QRM engine not found; looked in: {', '.join(map(str, candidates))}")
