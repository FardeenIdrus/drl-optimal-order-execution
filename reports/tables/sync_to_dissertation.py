"""Copy the table sources the dissertation cites into the dissertation LaTeX repo.

Same rationale as reports/figures/sync_to_dissertation.py: the dissertation is a SEPARATE
repo, so a table that has not been copied across cannot be \\input by the document and every
\\ref to it renders as "??". Only the tables the chapter actually cites are copied -- the
rest stay in the analysis repo until they are needed, so the dissertation repo does not
accumulate files nobody references.

Tables land in <dissertation>/tables/ and are included with, e.g.
    \\begin{table}[htbp]\\centering
    \\input{tables/t3_sealed_confirmations.tex}
    \\caption{...}\\label{tab:t3}
    \\end{table}
The .tex files contain ONLY the tabular/longtable environment -- no float, no caption -- so
captions and labels live in the chapter where they can be read alongside the prose.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "reports" / "tables"
DEST = REPO.parent / "Idrus_Fardeen_MSc_Dissertation" / "tables"

# Tables cited by the Results chapter (reports/results_chapter/results.tex), plus the
# per-run tables the appendix carries. Keep this list in step with the chapter.
WANTED = [
    # Data chapter
    "d1a_sources.tex",                 # what was collected, one row per stream
    "d1b_instrument.tex",              # what the instrument is like, and the scale of an order
    "d2_pipeline.tex",                 # processing steps and their checks, snapshot record
    "d2b_perorder.tex",                # processing steps and their checks, per-order record
    "d3_partitions.tex",               # how the data was divided, all three panels
    "d3a_versions.tex",                # panel A alone, section 3.4 paragraph 1
    "d3c_december.tex",                # panel C alone, section 3.4 paragraph 2
    "d3b_seedsets.tex",                # panel B alone, section 3.4 paragraph 3
    # main body
    "t3_sealed_confirmations.tex",     # the two sealed reactive confirmations
    "t4_robustness_grid.tex",          # size x deadline grid
    "t5_env_validation.tex",           # environment validation gates
    "t7_l2_summary.tex",               # frozen-replay arms, validation + sealed exam
    "ts4_sigext_exploiter.tex",        # exploiter ceiling
    "ts6_sigext_sealed.tex",           # sealed exhibit, injected environment
    "ts7_sigext_ceiling.tex",          # attainable-edge ceiling, one-shot confirmed
    "ts8_sigext_comparators.tex",      # AC + oracle VWAP, both environments
    "ts9_sigext_a4_observation.tex",   # observation specification, injected track
    "t13_primary_observation.tex",     # observation specification, PRIMARY track (A4.3)
    # appendix
    "t1_primary_campaign.tex",
    "t2_tuning_selection.tex",
    "t6_descriptive_stats.tex",
    "t8_hyperparameters.tex",
    "t9_l2_perrun.tex",
    "t10_tuning_perrun.tex",
    "t11_dqn_probe.tex",
    "t12_ladder.tex",
    "ts1_sigext_certification.tex",
    "ts2_sigext_dev_verdicts.tex",
    "ts3_sigext_per_run.tex",
    "ts5_sigext_base_env.tex",
]


def main() -> None:
    if not DEST.parent.exists():
        sys.exit(f"dissertation repo not found at {DEST.parent}")
    DEST.mkdir(parents=True, exist_ok=True)

    missing = [n for n in WANTED if not (SRC / n).exists()]
    if missing:
        sys.exit("these tables are listed as wanted but do not exist -- build them first:\n  "
                 + "\n  ".join(missing))

    copied = updated = 0
    for name in WANTED:
        src, dst = SRC / name, DEST / name
        if not dst.exists():
            copied += 1
        elif src.stat().st_mtime > dst.stat().st_mtime:
            updated += 1
        else:
            continue
        shutil.copy2(src, dst)

    print(f"synced -> {DEST}")
    print(f"  {len(WANTED)} tables ({copied} new, {updated} refreshed)")
    extra = sorted(p.name for p in DEST.glob("*.tex") if p.name not in WANTED)
    if extra:
        print(f"  {len(extra)} file(s) present but not on the wanted list -- NOT deleted:")
        for e in extra:
            print(f"      {e}")


if __name__ == "__main__":
    main()
