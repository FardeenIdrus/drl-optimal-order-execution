"""Phase F archive: bring the results archive up to date and checksum everything.

WHY. `results_archive/` is the version-controlled evidence layer -- the provenance appendix
cites paths relative to it, and the claim made in its README is that every number in the
report traces to a file inside it. As of 2026-07-31 that claim was FALSE for everything after
mid-July: the entire measured-signal extension, the observation amendments, the comparator
study, the frozen-replay sealed exam and the inversion analysis were all missing. This script
closes that gap and writes a checksum manifest so byte-exactness is verifiable rather than
asserted.

WHAT IS ARCHIVED, AND WHAT IS NOT. Scored evaluations, certification artifacts and diagnostic
outputs are archived in full -- they are small and they are the source of truth. Trained agents
are archived as `meta.json` + `curve.json` only, WITHOUT `model.zip`: the weights are ~19 MB
across the extension campaigns, are not cited by any number in the report, and are
regenerable from the recorded configuration and seed (training is bit-reproducible; see
Amendment A4 verification). The configuration and the training curve ARE cited, so they stay.
This exclusion is stated in the README rather than left implicit.

Idempotent: re-running refreshes anything whose source is newer and rewrites the manifest.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARCHIVE = REPO / "results_archive"
S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
L4 = S / "oxford_l4"

# Scored evaluations + certification: copied whole (JSON, small, cited).
WHOLE_DIRS = [
    (L4 / "step5_signal_dev", "qrm/step5_signal_dev"),
    (L4 / "step5_signal_sealed", "qrm/step5_signal_sealed"),
    (L4 / "step5_signal_reserve", "qrm/step5_signal_reserve"),
    (L4 / "step5_signal_curveblock", "qrm/step5_signal_curveblock"),
    (L4 / "step5_signal_ceiling21e6", "qrm/step5_signal_ceiling21e6"),
    (L4 / "step5_comparators", "qrm/step5_comparators"),
    (L4 / "step5_signal_obsfix", "qrm/step5_signal_obsfix"),
    (L4 / "step5_signal_obsfix_var", "qrm/step5_signal_obsfix_var"),
    (L4 / "step5_signal_obsfix_dqn", "qrm/step5_signal_obsfix_dqn"),
    (L4 / "step5_primary_v3_obsfix", "qrm/step5_primary_v3_obsfix"),
    (L4 / "signal", "qrm/signal"),
    (L4 / "step5_a5_armB_freshseeds", "qrm/step5_a5_armB_freshseeds"),
    (L4 / "step5_a5_armA_sameagents", "qrm/step5_a5_armA_sameagents"),
    (L4 / "rq3_attribution", "qrm/rq3_attribution"),
    (S / "l2_test_results", "l2/sealed_exam"),
]

# Trained agents: configuration + training curve only, no weights (see module docstring).
RUN_DIRS = [
    (L4 / "runs_signal_phaseD", "qrm/runs_signal_phaseD"),
    (L4 / "runs_signal_obsfix", "qrm/runs_signal_obsfix"),
    (L4 / "runs_signal_obsfix_var", "qrm/runs_signal_obsfix_var"),
    (L4 / "runs_signal_obsfix_dqn", "qrm/runs_signal_obsfix_dqn"),
    (L4 / "runs_primary_v3_obsfix", "qrm/runs_primary_v3_obsfix"),
    (L4 / "runs_a5_volatile_freshseeds", "qrm/runs_a5_volatile_freshseeds"),
    (L4 / "runs_signal_logged", "qrm/runs_signal_logged"),
    (L4 / "runs_signal_logged_dqn", "qrm/runs_signal_logged_dqn"),
]
RUN_KEEP = {"meta.json", "curve.json", "progress.csv"}   # progress.csv IS the
# deliverable for the logged re-runs; excluding it would archive the runs without
# the learning trajectories they exist to provide.


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_if_newer(src: Path, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def main() -> None:
    if not L4.exists():
        sys.exit(f"source not found: {L4}")
    copied = skipped = 0
    missing_sources = []

    for src, rel in WHOLE_DIRS:
        if not src.exists():
            missing_sources.append(str(src))
            continue
        for p in sorted(src.rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                if copy_if_newer(p, ARCHIVE / rel / p.relative_to(src)):
                    copied += 1
                else:
                    skipped += 1

    for src, rel in RUN_DIRS:
        if not src.exists():
            missing_sources.append(str(src))
            continue
        for run in sorted(d for d in src.iterdir() if d.is_dir()):
            for name in sorted(RUN_KEEP):
                p = run / name
                if p.exists():
                    if copy_if_newer(p, ARCHIVE / rel / run.name / name):
                        copied += 1
                    else:
                        skipped += 1

    if missing_sources:
        print("WARNING -- these sources do not exist and were skipped:")
        for m in missing_sources:
            print(f"    {m}")

    # ---- checksum manifest over the WHOLE archive, not just today's additions ----
    files = sorted(p for p in ARCHIVE.rglob("*")
                   if p.is_file() and p.name not in {"CHECKSUMS.sha256", "archive_phase_f.py"}
                   and not p.name.startswith("."))
    lines = [f"{sha256(p)}  {p.relative_to(ARCHIVE)}" for p in files]
    (ARCHIVE / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n")
    total_mb = sum(p.stat().st_size for p in files) / 1e6

    print(f"archived: {copied} file(s) copied, {skipped} already current")
    print(f"manifest: {len(files)} files, {total_mb:.1f} MB -> results_archive/CHECKSUMS.sha256")
    print("verify with:  cd results_archive && shasum -a 256 -c CHECKSUMS.sha256 | grep -v OK$")


if __name__ == "__main__":
    main()
