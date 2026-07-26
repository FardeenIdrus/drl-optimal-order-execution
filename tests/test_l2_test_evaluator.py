"""Unit tests for the L2 sealed-exam evaluator (data-free parts only).

These cover the deterministic scaffolding -- the fixed-subset construction (which
MUST match execution.agents.train), run discovery, and the paired-difference
statistics -- without loading any parquet, so they are fast and touch no dataset
(and, by construction, never any test.parquet).
"""
from __future__ import annotations

import json

import numpy as np

from execution.eval.test_evaluator import (
    N_EVAL_DEFAULT,
    VAL_SUBSET_SEED,
    _eval_indices,
    _paired_stats,
    discover_runs,
)


def test_eval_indices_match_training_construction():
    # Mirror execution.agents.train.train_one_seed exactly.
    n_episodes, n_eval = 1000, 400
    rng = np.random.default_rng(12345)
    expected = sorted(rng.choice(n_episodes, size=min(n_eval, n_episodes),
                                 replace=False).tolist())
    assert _eval_indices(n_episodes, n_eval) == expected
    assert VAL_SUBSET_SEED == 12345 and N_EVAL_DEFAULT == 400


def test_eval_indices_caps_at_population():
    idx = _eval_indices(50, 400)
    assert len(idx) == 50
    assert idx == sorted(set(idx))
    assert min(idx) >= 0 and max(idx) < 50


def test_paired_stats_signs_and_counts():
    diff = np.array([-1.0, -2.0, -3.0, -4.0])   # agent strictly cheaper
    s = _paired_stats(diff)
    assert s["n_episodes"] == 4
    assert s["mean_paired_diff_bps"] == -2.5
    assert s["wilcoxon_p_less"] < s["wilcoxon_p_two_sided"] + 1e-12
    assert np.isfinite(s["std_paired_diff_bps"])


def test_paired_stats_all_zero_diff_is_safe():
    # All-zero paired diffs never occur for a real agent over 400 episodes, but the
    # stat helper must not crash on the degenerate input (scipy may return nan/1.0).
    s = _paired_stats(np.zeros(10))
    assert s["mean_paired_diff_bps"] == 0.0
    assert isinstance(s["wilcoxon_p_two_sided"], float)


def test_discover_runs_filters_incomplete(tmp_path):
    good = tmp_path / "ppo_size96.57_seed0"
    good.mkdir()
    for f in ("model.zip", "normalizer.json", "meta.json"):
        (good / f).write_text("{}" if f.endswith(".json") else "x")
    incomplete = tmp_path / "dqn_size96.57_seed1"
    incomplete.mkdir()
    (incomplete / "curve.csv").write_text("x")     # no model/normalizer/meta
    (tmp_path / "notes.txt").write_text("x")        # not a run dir
    found = [p.name for p in discover_runs(tmp_path)]
    assert found == ["ppo_size96.57_seed0"]
