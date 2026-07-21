# 00 — DOCUMENTATION MAP (START HERE)

The single entry point. Lists every planning/logging document, what each is for, where the raw
results live, and the update protocol. Read this, then the LIVE doc. Updated 2026-07-16.

## Read this for the CURRENT state
**`reports/qrm_step5_remediation.md`** = THE LIVE WORKING DOC. Current state, full per-seed result
tables, the RUN-DIRECTORY MANIFEST (which results to cite), and the REMAINING ROADMAP (next steps).
If you read one file, read this.

## The frozen rules (pre-registration — never change after the fact)
**`reports/qrm_step4_criteria.md`** = the judging bar (§3), the tuning table (§5), the
confirmation protocol + BOTH verdicts (§6-§6.8 v3a FAIL; §6.10-§6.11 remedial v1b FAIL — family
closed), the process audit (§6.9), the sweep/grid protocol (§7, §7.7 Parts A+B), the if-edge
ladder + its executed verdict (§7.5/§7.5a/§7.5b — both grid triggers died on the reserve block),
the DQN probe registration + outcome (Part D), the d3 update-rhythm variant (Part E), and the
cadence check spec (Part C). Every change is dated with a rationale, logged BEFORE affected runs.

## The write-up argument bank (dissertation-bound reasoning)
**`reports/writeup_arguments.md`** = every INTERPRETIVE argument the report needs, with full
reasoning + evidence pointers: signal structure + scope (A), null-not-vacuous (B),
testing-soundness defenses (C), spurious-edge anatomy (D, + D4 size-response), two-track
complementarity (E), future work (F), contribution framing (G), limitations prose (H),
selection-metric analysis (I), the measured-signal extension design (J), benchmark
justification — adaptive vs normal TWAP (K), the grid findings + ladder outcome (L, L5),
the three-layer evaluation methodology discussion (M), and the DQN systematic-collapse
finding + measured Q-value diagnostic (N, N5 + correction). ADD TO IT at every major finding.

## Results pack (proto results chapter + meeting materials, 2026-07-17)
**`reports/results_pack/`** = results_pack.pdf (full dissertation-standard draft, NOT frozen),
meeting_pack.pdf (every figure + every table with bullet points, for live walkthroughs),
single-file `*_overleaf.tex` copies of both (paste into Overleaf with the figures/ folder),
and talking_points.md (meeting script incl. the since-last-meeting run ledger).

## Figures + tables tracker
**`reports/figures/FIGURES_TABLES_MANIFEST.md`** = every figure/table built, pending, or candidate
(comprehensive-first; curation only at report assembly). Generation scripts live in
`reports/figures/`.

## L2 sealed-test pre-registration (current; not yet executed)
**`reports/l2_test_protocol.md`** = the pre-registered protocol (2026-07-14) for the L2 track's
one-shot sealed test-set evaluation: the verified inventory of all 51 trained L2 agents, the
missing-coverage map, the paired-evaluation rules, multiplicity handling, and the RAM procedure.

## Supporting / episode log (current but narrow)
**`reports/qrm_prelaunch_audit_2026-07-08.md`** = the pre-launch audit and the drift-fix episode
(findings, the fix, and a chronological work log). Mostly closed.

## Entry points for a brand-new chat (pointers + history, not the live detail)
- **`PLANS/BUILD_PLAN (10).md`** — implementation plan. TOP block = current-state pointer; body =
  append-only history (older L2 / Step-3 content, marked superseded).
- **`PLANS/HANDOVER (14).md`** — session handover. TOP block = current state; below = history.
- **`PLANS/LITERATURE (13).md`** — reading / citation reference (not a status doc).

## Historical / superseded (do NOT treat as current)
- **`reports/phase1_qa.md`** — L2 Phase-1 QA (old L2 track).
- **`reports/qrm_3f_criteria.md`** — Step-3f calibration-gate criteria (closed; calibration done).
- **`../../HANDOVER .md`** (parent dir) — early proposal-stage handover, SUPERSEDED (stamped).

## Where the RAW RESULTS live (OUTSIDE the repo)
**NEW 2026-07-21: `results_archive/` (IN the repo) is the frozen, citable snapshot of the
complete evidence layer** — all judgement/audit JSONs (incl. SUPERSEDED_step5_v2 for the
drift figure), gates, fairness, calibration bundles, per-episode arrays, every run's
meta/curve/model for BOTH tracks (1,165 files, 56 MB, checksum-verified). The results pack's
provenance appendix now cites results_archive/-relative paths. Scratch (below) remains the
WORKING record that scripts read; the archive is append-only.
QRM/L4 track: `Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/`. That folder's own
**`RESULTS_MANIFEST.md` is the AUTHORITATIVE, kept-current list of which folders are results of
record vs superseded** — do not rely on any hand-copied folder list elsewhere (including here);
read the manifest. Highlights: `step5_*` = scored numbers (judgement.json + behaviour_audit.json
are the source of truth for every cited number), `runs_*` = trained agents; superseded folders
carry a `SUPERSEDED_` prefix and must never be cited. L2 track raw results live one level up in
`scratch_hyperliquid/` (`runs/`, `runs_10s/`, `runs_10s_10min/`, `dataset*/`), mapped in
`reports/l2_test_protocol.md` (inventory + fill-in status) and the FIGURES manifest's L2 data map.

## UPDATE PROTOCOL (who updates what, when)
1. **Every run boundary** (a campaign / tuning / gate finishes): append the FULL per-seed result
   table to `qrm_step5_remediation.md`; if the "current" folder changed, update its RUN-DIRECTORY
   MANIFEST, the scratch `RESULTS_MANIFEST.md`, and the `qrm-results-locations` memory.
2. **Any change to the judging rules/protocol:** log it in `qrm_step4_criteria.md` with a DATE +
   rationale, BEFORE the affected run.
3. **Major state change:** refresh the TOP blocks of BUILD_PLAN + HANDOVER, and this index.
4. Auto-loaded every session: the memory files (rigor, qrm-results-locations, log-full-result-sets).

## One-line mental model
Rules (criteria) are fixed up front. The live doc (remediation) tracks state + results + next steps.
BUILD_PLAN/HANDOVER are the front door for a new chat. The rest is history or data docs.
