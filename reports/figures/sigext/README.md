# Measured-signal extension — figure tiers

Built by `make_sigext_figures.py` from frozen result JSONs only. Every number traces to a
source-of-record file (live doc `qrm_step5_remediation.md`, addenda G/H/I/J/K/L).

## `main_body/` — the three figures that carry the argument

These are the ones that earn main-text space. Together they establish, in order: the
instrument is valid, the result is a null, and the null is a *learnability* failure.

| figure | the claim it makes | why it belongs in the main body |
|---|---|---|
| `s1_injection_fidelity` | The injected predictability reproduces the real venue signal at every gated horizon | Pre-empts the first question any examiner asks: was the signal realistic? Without this the whole experiment is dismissible |
| `s3_dev_campaign_forest` | All 38 agents, every configuration, no material edge | The primary result, with full seed scatter so it cannot be read as one unlucky run |
| `s4_exploiter_vs_agents` | A one-line rule captures 0.15–0.49 bps; the agents capture ~0 | **The decisive figure of the extension.** Signal present, capturable, not learned |

`s4` is the single most important figure in the extension and arguably the most
publication-relevant in the dissertation: it converts "RL did not beat TWAP" into "RL
failed to learn something demonstrably learnable", which is a far stronger and more
interesting claim.

## `appendix/` — supporting evidence and technical detail

Not lesser work; lesser *load*. Each answers a follow-up rather than carrying the argument.

| figure | what it supports |
|---|---|
| `s5_base_vs_injected` | The injection amplified the capturable edge ×5.7 (calm) / ×4.1 (volatile) rather than creating it — carries the honest correction to the earlier "no prediction channel" claim |
| `s6_training_curves` | No agent trends toward an edge across the full 2M-step budget |
| `s8_kernel_structure` | Kernel gains by timescale and the measured signal persistence the injection reproduces |

`s5` is the most promotable of these: if the Results chapter has room for a fourth
main-body figure, it is the candidate, because it states precisely what the instrument does.

## Not built by design

`s2_three_environment` (frozen replay → reactive → reactive + signal) requires the **L2
sealed exam** numbers. Building it from L2 *validation* numbers next to two sealed results
would reproduce exactly the selection bias this dissertation documents, so it is left unbuilt
until the sealed exam runs. Caption caveat to carry when it is built: the three environments
differ on more than one axis; the controlled contrast is environments 2 → 3, where only the
signal changed. Environment 1 is a qualitative anchor, not a matched comparison.

`s7_policy_sensitivity` (per-seed signal response, showing seeds learned opposite directions)
is described in the live doc addendum (I) but not yet built as a figure; it is the natural
mechanism exhibit for the Discussion if that section needs one.

## Palette note

matplotlib's default colour cycle is **not** used beyond two series: its red/green
(colour-blind ΔE 3.9) and green/orange (ΔE 0.7) pairs are indistinguishable to colour-blind
readers. The house blue/red pair passes (ΔE 21.1) and is retained; 3–4 category figures use
the validated Okabe-Ito subset.
