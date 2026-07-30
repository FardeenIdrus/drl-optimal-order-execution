#!/bin/zsh
# Render the LaTeX table fragments to viewable PDFs for meetings/screen-sharing.
# The .tex fragments remain the source of truth for the dissertation; these are
# read-only renders. Produces:
#   rendered/<id>.pdf                  one PDF per table (show a single table)
#   rendered/ALL_TABLES.pdf            every table, one per page (flip through)
# Usage:  zsh reports/tables/render_tables.sh
set -e
cd "$(dirname "$0")"
LATEX=/Library/TeX/texbin/pdflatex
CROP=/Library/TeX/texbin/pdfcrop
OUT=rendered
TMP=$(mktemp -d)
mkdir -p "$OUT"

# wide tables need landscape
is_wide() { [[ "$1" == "ts8_sigext_comparators" ]] }

preamble() {
  print -r -- '\documentclass[11pt]{article}'
  print -r -- '\usepackage{booktabs}'
  print -r -- '\usepackage{float}'
  print -r -- '\usepackage{pdflscape}'
  print -r -- "\\usepackage[margin=$1]{geometry}"
  print -r -- '\pagestyle{empty}'
  print -r -- '\begin{document}'
}

TABLES=(ts1_sigext_certification ts2_sigext_dev_verdicts ts3_sigext_per_run \
        ts4_sigext_exploiter ts5_sigext_base_env ts6_sigext_sealed \
        ts7_sigext_ceiling ts8_sigext_comparators)

# ---- individual PDFs ----
for t in $TABLES; do
  [[ -f "$t.tex" ]] || { print "skip $t (missing)"; continue }
  f="$TMP/$t.tex"
  if is_wide "$t"; then
    { preamble "0.5in"; print -r -- '\begin{landscape}'; \
      sed 's/\[htbp\]/[H]/' "$t.tex"; print -r -- '\end{landscape}'; \
      print -r -- '\end{document}' } > "$f"
  else
    { preamble "0.9in"; sed 's/\[htbp\]/[H]/' "$t.tex"; \
      print -r -- '\end{document}' } > "$f"
  fi
  if (cd "$TMP" && $LATEX -interaction=nonstopmode -halt-on-error "$t.tex" >/dev/null 2>&1); then
    # trim to content so the table fills the screen when shared
    if $CROP --margins 14 "$TMP/$t.pdf" "$TMP/${t}_c.pdf" >/dev/null 2>&1; then
      cp "$TMP/${t}_c.pdf" "$OUT/$t.pdf"
    else
      cp "$TMP/$t.pdf" "$OUT/$t.pdf"
    fi
    print "  rendered $OUT/$t.pdf"
  else
    print "  FAILED $t (see $TMP/$t.log)"
  fi
done

# ---- combined PDF, one table per page ----
ALL="$TMP/ALL_TABLES.tex"
{
  preamble "0.7in"
  for t in $TABLES; do
    [[ -f "$t.tex" ]] || continue
    print -r -- "\\section*{\\texttt{${t//_/\\_}}}"
    if is_wide "$t"; then
      print -r -- '\begin{landscape}'; sed 's/\[htbp\]/[H]/' "$t.tex"; print -r -- '\end{landscape}'
    else
      sed 's/\[htbp\]/[H]/' "$t.tex"
    fi
    print -r -- '\clearpage'
  done
  print -r -- '\end{document}'
} > "$ALL"
(cd "$TMP" && $LATEX -interaction=nonstopmode -halt-on-error ALL_TABLES.tex >/dev/null 2>&1) \
  && cp "$TMP/ALL_TABLES.pdf" "$OUT/ALL_TABLES.pdf" && print "  rendered $OUT/ALL_TABLES.pdf" \
  || print "  FAILED combined (see $TMP/ALL_TABLES.log)"
rm -rf "$TMP"
