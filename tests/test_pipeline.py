"""Unit tests for Stage 7: config resolution, stage planning, skip logic, QA render.

Run with:  PYTHONPATH=src pytest tests/test_pipeline.py
"""
from pathlib import Path

from execution import pipeline as P
from execution.data import qa_report as Q

CFG = {
    "coin": "BTC", "start": "2024-01-01", "end": "2025-12-31",
    "skip_ranges": [["2025-10-17", "2025-10-22"]],
    "features": {"imbalance_depth": 5, "return_lookback": 5, "vol_window": 30},
    "regimes": {"episode_minutes": 30, "test_frac": 0.2, "buffer_episodes": 1, "n_regimes": 2},
    "compute": {"pull_workers": 16, "resample_workers": 8, "resample_batch_files": 96},
    "artifacts": {
        "manifest_dir": "manifest", "raw_dir": "raw_l2/BTC",
        "minute_parquet": "minute/m.parquet", "features_parquet": "features/f.parquet",
        "episodes_parquet": "episodes/e.parquet", "dataset_dir": "dataset",
    },
}


def test_resolve_paths():
    paths = P.resolve_paths(CFG, "/scratch")
    assert paths["manifest_csv"] == Path("/scratch/manifest/BTC_2024-01-01_2025-12-31_manifest.csv")
    assert paths["raw_dir"] == Path("/scratch/raw_l2/BTC")
    assert paths["train_parquet"] == Path("/scratch/dataset/train.parquet")


def test_plan_stages_order_and_argv():
    paths = P.resolve_paths(CFG, "/scratch")
    stages = P.plan_stages(CFG, paths)
    assert [s.name for s in stages] == P.STAGE_ORDER
    assert "execution.data.manifest" in stages[0].argv
    assert "execution.data.dataset" in stages[-1].argv
    # parameters are threaded from the config into the module invocations
    minute = next(s for s in stages if s.name == "minute")
    assert "--batch-files" in minute.argv and "96" in minute.argv
    feats = next(s for s in stages if s.name == "features")
    assert "--vol-window" in feats.argv and "30" in feats.argv


def test_is_satisfied(tmp_path):
    out = tmp_path / "out.parquet"
    stage = P.Stage("x", out, argv=[])
    assert not P.is_satisfied(stage)
    out.write_text("x")
    assert P.is_satisfied(stage)


def test_is_satisfied_count_check(tmp_path):
    raw = tmp_path / "raw"
    (raw / "d" / "0").mkdir(parents=True)
    stage = P.Stage("pull", raw, argv=[], count_check=2)
    (raw / "d" / "0" / "BTC.lz4").write_text("x")
    assert not P.is_satisfied(stage)         # only 1 of 2 files
    (raw / "d" / "1").mkdir(parents=True)
    (raw / "d" / "1" / "BTC.lz4").write_text("x")
    assert P.is_satisfied(stage)             # 2 of 2


def test_qa_render_has_sections():
    report = {
        "scope": {"coin": "BTC", "start": "2024-01-01", "end": "2025-12-31", "skip_ranges": []},
        "stage0_source_coverage": {"hours_present": 100, "hours_expected": 100, "coverage_pct": 100.0},
        "stage1_pull": {"raw_files": 100, "expected": 100, "byte_exact": True},
        "stage3_minute": {"minutes_present": 99}, "stage4_features": {"valid_pct": 95.0},
        "stage5_regimes": {"n_episodes": 10, "counts_by_split_regime": {"test": {"calm": 2, "volatile": 1}}},
        "stage6_datasets": {"train_rows": 8, "test_rows": 2, "train_before_test": True},
        "determinism": "deterministic", "limitations": ["minute resolution"],
    }
    md = Q.render_markdown(report)
    assert "# Data Pipeline" in md
    assert "coverage gate" in md.lower() and "Limitation" in md
