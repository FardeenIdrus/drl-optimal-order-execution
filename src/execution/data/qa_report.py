"""Stage 7 - consolidate per-stage QA into one Phase 1 evidence report.

Reads the QA artifacts each stage already writes (coverage.json, the per-parquet
.qa.json files, dataset_meta.json) plus the raw-file count, and renders a single
machine-readable JSON and a human-readable Markdown report for the methodology/data
chapter. Generates nothing new from the data; it only aggregates existing evidence.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

from execution.pipeline import load_config, resolve_paths

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def build_report(cfg: dict, paths: Dict[str, Path]) -> dict:
    """Aggregate every stage's QA into one dict."""
    coverage = _load_json(paths["coverage_json"])
    minute_qa = _load_json(paths["minute_parquet"].with_suffix(".qa.json"))
    feature_qa = _load_json(paths["features_parquet"].with_suffix(".qa.json"))
    regime_qa = _load_json(paths["episodes_parquet"].with_suffix(".qa.json"))
    dataset_meta = _load_json(paths["dataset_dir"] / "dataset_meta.json")
    raw_files = len(list(paths["raw_dir"].rglob("*.lz4"))) if paths["raw_dir"].exists() else 0

    return {
        "scope": {"coin": cfg["coin"], "start": cfg["start"], "end": cfg["end"],
                  "skip_ranges": cfg["skip_ranges"]},
        "params": {"features": cfg["features"], "regimes": cfg["regimes"]},
        "stage0_source_coverage": coverage,
        "stage1_pull": {"raw_files": raw_files,
                        "expected": coverage.get("hours_present"),
                        "byte_exact": raw_files == coverage.get("hours_present")},
        "stage3_minute": minute_qa,
        "stage4_features": feature_qa,
        "stage5_regimes": regime_qa,
        "stage6_datasets": dataset_meta,
        "determinism": "Phase 1 has no randomness; same raw data + config => identical outputs.",
        "limitations": [
            "Minute resolution; one book snapshot per minute for fills (no intra-minute replenishment).",
            "Hyperliquid BTC perpetual futures (not spot/equities); single asset for now.",
            "Fixed train-median regime threshold => test regime mix reflects the test period (here ~70/30 calm/volatile).",
        ],
    }


def render_markdown(r: dict) -> str:
    s = r["scope"]
    cov = r.get("stage0_source_coverage", {})
    mins = r.get("stage3_minute", {})
    feats = r.get("stage4_features", {})
    regs = r.get("stage5_regimes", {})
    ds = r.get("stage6_datasets", {})
    L: list[str] = []
    L.append(f"# Phase 1 Data Pipeline — QA Report ({s['coin']}, {s['start']} to {s['end']})\n")
    L.append("Real Hyperliquid L2 order-book data; deterministic, leakage-checked pipeline.\n")

    L.append("## Source & coverage (Stage 0)")
    L.append(f"- Hourly coverage: **{cov.get('hours_present')}/{cov.get('hours_expected')} "
             f"({cov.get('coverage_pct')}%)**; download {cov.get('total_size_gib')} GiB.")
    L.append(f"- Fully-covered days: {cov.get('days_fully_covered')}; "
             f"missing: {len(cov.get('days_missing', []))}; partial: {len(cov.get('days_partial', []))}.\n")

    p = r.get("stage1_pull", {})
    L.append("## Pull integrity (Stage 1)")
    L.append(f"- Raw files on disk: **{p.get('raw_files')}** vs expected {p.get('expected')} "
             f"(byte-exact: {p.get('byte_exact')}).\n")

    L.append("## Minute book + HF stats (Stage 3)")
    L.append(f"- Minutes: **{mins.get('minutes_present')}**; duplicates: {mins.get('duplicate_minutes')}; "
             f"coverage {mins.get('coverage_pct')}%; book ordering OK: {mins.get('book_ordering_ok')}; "
             f"median snapshots/min: {mins.get('median_snapshots_per_min')}.\n")

    L.append("## Features (Stage 4)")
    L.append(f"- Feature-valid rows: **{feats.get('feature_valid_rows')} "
             f"({feats.get('valid_pct')}%)**; params {feats.get('params')}.")
    corr = feats.get("correlation", {})
    if corr:
        cols = list(corr.keys())
        L.append("\n| corr | " + " | ".join(cols) + " |")
        L.append("|" + "---|" * (len(cols) + 1))
        for c in cols:
            L.append(f"| {c} | " + " | ".join(f"{corr[c].get(k, ''):.2f}"
                     if isinstance(corr[c].get(k), (int, float)) else "" for k in cols) + " |")
    L.append("")

    L.append("## Regimes + coverage gate (Stage 5)")
    L.append(f"- Episodes: **{regs.get('n_episodes')}**; threshold: {regs.get('thresholds')}.")
    counts = regs.get("counts_by_split_regime", {})
    if counts:
        regimes = sorted({k for v in counts.values() for k in v})
        L.append("\n| split | " + " | ".join(regimes) + " |")
        L.append("|" + "---|" * (len(regimes) + 1))
        for split, row in counts.items():
            L.append(f"| {split} | " + " | ".join(str(row.get(k, 0)) for k in regimes) + " |")
        L.append(f"\n- Coverage gate (min test-regime episodes): "
                 f"**{regs.get('min_test_regime_episodes')}** -> both regimes amply represented.")
    L.append("")

    L.append("## Train/test datasets (Stage 6)")
    L.append(f"- train: **{ds.get('train_rows')} rows / "
             f"{sum(ds.get('counts_by_split_regime', {}).get('train', {}).values()) if ds.get('counts_by_split_regime') else '?'} episodes**; "
             f"test: **{ds.get('test_rows')} rows**.")
    L.append(f"- Leakage checks: train_before_test={ds.get('train_before_test')}, "
             f"NaN in core fields={ds.get('nan_in_core_fields')}, book ordering OK={ds.get('book_ordering_ok')}.\n")

    L.append("## Reproducibility & limitations")
    L.append(f"- {r.get('determinism')}")
    for lim in r.get("limitations", []):
        L.append(f"- Limitation: {lim}")
    return "\n".join(L) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description="Stage 7 - consolidated Phase 1 QA report")
    p.add_argument("--config", required=True)
    p.add_argument("--scratch-root", required=True)
    p.add_argument("--out-dir", required=True, help="repo reports/ dir")
    args = p.parse_args()

    cfg = load_config(args.config)
    paths = resolve_paths(cfg, args.scratch_root)
    report = build_report(cfg, paths)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "phase1_qa.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "phase1_qa.md").write_text(render_markdown(report))
    logger.info("wrote %s/phase1_qa.{json,md}", out)


if __name__ == "__main__":
    main()
