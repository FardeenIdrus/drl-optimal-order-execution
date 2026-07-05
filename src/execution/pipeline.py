"""Phase 1 pipeline orchestrator — reproduce the whole data pipeline from one config.

Runs Stages 0-6 in order by invoking each stage's tested module entry point (so there
is no duplicated stage logic to drift). Idempotent: a stage whose output already exists
is skipped, so re-running never re-pulls 15 GB or re-parses the archive. The pipeline is
deterministic, so the same raw data + config reproduce identical outputs.

Usage:
    PYTHONPATH=src python -m execution.pipeline --config configs/pipeline.yaml \
        --scratch-root <scratch> [--force] [--from manifest] [--to dataset]
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

logger = logging.getLogger(__name__)

STAGE_ORDER = ["manifest", "pull", "minute", "features", "regimes", "dataset"]


@dataclass
class Stage:
    name: str
    output: Path            # existence of this path means the stage is satisfied
    argv: List[str]         # module invocation (without the python executable)
    count_check: int = 0    # if >0, "satisfied" also requires this many *.lz4 under output


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: dict, scratch_root: str) -> Dict[str, Path]:
    """Resolve all artifact paths under the scratch root."""
    root = Path(scratch_root)
    a = cfg["artifacts"]
    stem = f"{cfg['coin']}_{cfg['start']}_{cfg['end']}"
    manifest_dir = root / a["manifest_dir"]
    return {
        "manifest_dir": manifest_dir,
        "manifest_csv": manifest_dir / f"{stem}_manifest.csv",
        "coverage_json": manifest_dir / f"{stem}_coverage.json",
        "raw_dir": root / a["raw_dir"],
        "minute_parquet": root / a["minute_parquet"],
        "features_parquet": root / a["features_parquet"],
        "episodes_parquet": root / a["episodes_parquet"],
        "dataset_dir": root / a["dataset_dir"],
        "train_parquet": root / a["dataset_dir"] / "train.parquet",
    }


def _present_hours(paths: Dict[str, Path]) -> int:
    """Expected raw-file count = present hours from the Stage 0 coverage report."""
    if paths["coverage_json"].exists():
        return int(json.loads(paths["coverage_json"].read_text()).get("hours_present", 0))
    return 0


def plan_stages(cfg: dict, paths: Dict[str, Path]) -> List[Stage]:
    """Build the ordered stage list (output to check + module argv)."""
    coin = cfg["coin"]
    feat, reg, comp = cfg["features"], cfg["regimes"], cfg["compute"]
    bar_seconds = cfg.get("bar_seconds", 60)   # 60s canonical; finer (e.g. 10) for the resolution study
    chunk_by = cfg.get("resample_chunk_by", "none")   # 'month' = memory-safe resample for fine bars
    episode_minutes = reg.get("episode_minutes", 30)  # 30-min horizon; shorter (e.g. 10) for the SNR study
    return [
        Stage("manifest", paths["manifest_csv"], [
            "-m", "execution.data.manifest", "--coin", coin,
            "--start", cfg["start"], "--end", cfg["end"], "--out", str(paths["manifest_dir"])]),
        Stage("pull", paths["raw_dir"], [
            "-m", "execution.data.pull", "--manifest", str(paths["manifest_csv"]),
            "--dest", str(paths["raw_dir"]), "--coin", coin,
            "--workers", str(comp["pull_workers"])], count_check=_present_hours(paths)),
        Stage("minute", paths["minute_parquet"], [
            "-m", "execution.data.resample", "--raw-dir", str(paths["raw_dir"]), "--coin", coin,
            "--manifest", str(paths["manifest_csv"]), "--out", str(paths["minute_parquet"]),
            "--bar-seconds", str(bar_seconds), "--chunk-by", chunk_by,
            "--workers", str(comp["resample_workers"]),
            "--batch-files", str(comp["resample_batch_files"])]),
        Stage("features", paths["features_parquet"], [
            "-m", "execution.data.features", "--minute-parquet", str(paths["minute_parquet"]),
            "--out", str(paths["features_parquet"]),
            "--imbalance-depth", str(feat["imbalance_depth"]),
            "--return-lookback", str(feat["return_lookback"]),
            "--vol-window", str(feat["vol_window"]),
            "--bar-seconds", str(bar_seconds)]),
        Stage("regimes", paths["episodes_parquet"], [
            "-m", "execution.data.regimes", "--features-parquet", str(paths["features_parquet"]),
            "--minute-parquet", str(paths["minute_parquet"]),
            "--out", str(paths["episodes_parquet"]), "--test-frac", str(reg["test_frac"]),
            "--buffer-episodes", str(reg["buffer_episodes"]), "--n-regimes", str(reg["n_regimes"]),
            "--bar-seconds", str(bar_seconds), "--episode-minutes", str(episode_minutes)]),
        Stage("dataset", paths["train_parquet"], [
            "-m", "execution.data.dataset", "--features-parquet", str(paths["features_parquet"]),
            "--minute-parquet", str(paths["minute_parquet"]),
            "--episodes-parquet", str(paths["episodes_parquet"]),
            "--out-dir", str(paths["dataset_dir"]),
            "--bar-seconds", str(bar_seconds), "--episode-minutes", str(episode_minutes)]),
    ]


def is_satisfied(stage: Stage) -> bool:
    """True if the stage's output already exists (so it can be skipped)."""
    if not stage.output.exists():
        return False
    if stage.count_check:
        return len(list(stage.output.rglob("*.lz4"))) >= stage.count_check
    return True


def run(config_path: str, scratch_root: str, force: bool = False,
        from_stage: str = STAGE_ORDER[0], to_stage: str = STAGE_ORDER[-1]) -> None:
    cfg = load_config(config_path)
    paths = resolve_paths(cfg, scratch_root)
    stages = plan_stages(cfg, paths)
    lo, hi = STAGE_ORDER.index(from_stage), STAGE_ORDER.index(to_stage)

    for stage in stages:
        if not (lo <= STAGE_ORDER.index(stage.name) <= hi):
            continue
        if not force and is_satisfied(stage):
            logger.info("[skip] %s — output exists (%s)", stage.name, stage.output)
            continue
        logger.info("[run ] %s", stage.name)
        proc = subprocess.run([sys.executable, *stage.argv])
        if proc.returncode != 0:
            raise RuntimeError(f"stage '{stage.name}' failed (exit {proc.returncode})")
    logger.info("pipeline complete (%s -> %s)", from_stage, to_stage)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Phase 1 data-pipeline orchestrator")
    p.add_argument("--config", required=True)
    p.add_argument("--scratch-root", required=True, help="root for data artifacts (outside the repo)")
    p.add_argument("--force", action="store_true", help="re-run stages even if outputs exist")
    p.add_argument("--from", dest="from_stage", default=STAGE_ORDER[0], choices=STAGE_ORDER)
    p.add_argument("--to", dest="to_stage", default=STAGE_ORDER[-1], choices=STAGE_ORDER)
    args = p.parse_args()
    run(args.config, args.scratch_root, args.force, args.from_stage, args.to_stage)


if __name__ == "__main__":
    main()
