"""Copy every current figure into the dissertation LaTeX repo's figures/ folder.

WHY THIS EXISTS. The analysis repo builds figures under reports/figures/**; the dissertation
is a SEPARATE repo and LaTeX's \\includegraphics can only see files inside its own figures/
folder. A figure that has not been copied across cannot appear in the dissertation at all.
That copy was last done by hand on 2026-07-21 and by 2026-07-30 the dissertation was missing
ELEVEN figures, including the two most important ones (the three-environment comparison and
the frozen-replay inversion mechanism). Hand-copying is how that happened; this script is the
fix. Run it after any figure rebuild.

NAMING. Files are copied under their build names, which are unique across the whole tree
(checked below -- a collision raises rather than silently overwriting). The frozen-replay
figures were renamed 2026-07-30 from an `l1_`/`l2_`/`l4_` scheme in which "l2" meant the TRACK
in one filename and the FIGURE NUMBER in another, and in which the L2 sealed-exam figure was
called `l4_exam_inversion`. They now carry a `frozen_` prefix naming the track descriptively.

The dissertation renumbers figures as Figure 1..N in reading order; these are source names,
not presentation names.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO / "reports" / "figures"
DEST = (REPO.parent / "Idrus_Fardeen_MSc_Dissertation" / "figures")

# Every directory holding figures of record. Untracked draft-figure directories are
# deliberately excluded: they are not sources of truth.
SRC_DIRS = [
    SRC_ROOT / "data",
    SRC_ROOT / "qrm",
    SRC_ROOT / "l2",
    SRC_ROOT / "sigext" / "main_body",
    SRC_ROOT / "sigext" / "appendix",
    SRC_ROOT / "methodology",
]


def main() -> None:
    if not DEST.parent.exists():
        sys.exit(f"dissertation repo not found at {DEST.parent}")
    DEST.mkdir(parents=True, exist_ok=True)

    found: dict[str, Path] = {}
    for d in SRC_DIRS:
        if not d.exists():
            print(f"  WARNING: source dir missing, skipped: {d}")
            continue
        for p in sorted(d.glob("*.pdf")):
            if p.name in found:
                sys.exit(f"NAME COLLISION: {p.name} in both {found[p.name].parent} and {d}. "
                         f"Rename one before syncing -- silently overwriting a figure in the "
                         f"dissertation is exactly the failure this script exists to prevent.")
            found[p.name] = p

    before = {p.name for p in DEST.glob("*.pdf")}
    copied = updated = 0
    for name, src in found.items():
        dst = DEST / name
        if not dst.exists():
            copied += 1
        elif src.stat().st_mtime > dst.stat().st_mtime:
            updated += 1
        else:
            continue
        shutil.copy2(src, dst)

    stale = sorted(before - set(found))
    print(f"synced -> {DEST}")
    print(f"  {len(found)} figures of record   ({copied} new, {updated} refreshed)")
    if stale:
        print(f"  {len(stale)} file(s) in the dissertation with NO current source "
              f"(renamed or retired upstream) -- NOT deleted, check before removing:")
        for s in stale:
            print(f"      {s}")


if __name__ == "__main__":
    main()
