# 00 — DOCUMENTATION MAP (START HERE)

The single entry point. Lists every planning/logging document, what each is for, where the raw
results live, and the update protocol. Read this, then the LIVE doc. Updated 2026-07-09.

## Read this for the CURRENT state
**`reports/qrm_step5_remediation.md`** = THE LIVE WORKING DOC. Current state, full per-seed result
tables, the RUN-DIRECTORY MANIFEST (which results to cite), and the REMAINING ROADMAP (next steps).
If you read one file, read this.

## The frozen rules (pre-registration — never change after the fact)
**`reports/qrm_step4_criteria.md`** = the judging bar (§3), the tuning table (§5), the
out-of-sample confirmation protocol (§6 + §6.7), the confirmation VERDICT (§6.8 = FAIL), and the
post-verdict process audit (§6.9). Every change is dated with a rationale.

## The write-up argument bank (dissertation-bound reasoning)
**`reports/writeup_arguments.md`** = every INTERPRETIVE argument the report needs, with full
reasoning + evidence pointers: environment signal structure + scope statement (A), why the null
is not vacuous (B), testing-soundness defenses (C), the anatomy of the spurious edge (D),
L2+QRM two-track complementarity (E), named future work (F), contribution framing (G),
limitations prose (H). ADD TO IT at every major finding; the write-up chapters draw from here.

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
`Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4/`. That folder has its own
**`RESULTS_MANIFEST.md`** listing current vs superseded runs, plus `README.md` + `SCHEMA.md` (data
docs). CURRENT = `step5_selection_v3/` (tuning + selection numbers of record, 2026-07-10; winner
ppo_volatile_v3a) + `step5_v3/` (primary-campaign numbers) + `runs_primary_v3/` + `runs_tuning_v3/`
(the agents). `step5_tuning_v3/` = sealed 60-run screen stage record (numbers reproduced inside
step5_selection_v3). All superseded folders are RENAMED with a `SUPERSEDED_` prefix (2026-07-10,
contents untouched) — anything so prefixed must never be cited.

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
