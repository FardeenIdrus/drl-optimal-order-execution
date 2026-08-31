# Injected-signal figures

Figures for the injected-environment experiments, built by `make_sigext_figures.py` from
archived result records only; no script here runs a new evaluation.

- `main_body/` holds the figures that carry the core argument: the injection is certified
  against the venue measurement, the agents' result is a null, and a non-learning
  signal-reading rule captures the available saving.
- `appendix/` holds the supporting figures: training curves, the base-versus-injected
  comparison, kernel structure, and the risk-return frontier.

Each figure is written as PDF (vector) and PNG (preview). Every number traces to a
source-of-record file archived under `results_archive/`; the governing decision rules are in
`reports/qrm_step4_criteria.md`.
