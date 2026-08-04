"""Minimum detectable effect for EVERY environment that issues a null, and whether it transfers.

WHAT THIS ADDS TO `power_analysis.py`, WHICH IT DOES NOT REPLACE. That script established the
MDE for one campaign on one block: the reacting simulator's primary campaign, development
block. Its statistics are correct and are imported here unchanged. Two gaps remained, and an
examiner is entitled to both:

  1. SCOPE. The chapter issues nulls in THREE environments -- recorded order books, the
     reacting simulator, and the injected-signal simulator -- and only one of them was ever
     power-licensed. A chapter whose answer to "the null is not evidenced" is a power analysis
     covering a third of its nulls has not answered it.

  2. TRANSFER. The MDE was computed on development-block dispersion; the verdicts were issued
     on sealed blocks. In a chapter whose Part C finding is that BLOCKS DIFFER, assuming a
     dispersion measured on one block applies to another is the weakest link in the argument.
     It was assumed because it could not be measured: no other block had per-episode data.
     Now they do.

WHAT IS IMPORTED, NOT REWRITTEN. `power_one_sided_t` (exact non-central t), `mde` (brentq),
`mde_interval` (chi-square interval on sigma) and `seeds_needed` come from `power_analysis.py`
untouched. So does the CRN-aware variance decomposition, reimplemented here identically and
cross-checked against that script's stored output for the one cell both compute. The maths was
debugged once -- notably a first version that reported "episode noise explains 253% of the
across-seed variance", which is impossible and came from assuming independent episode noise
under common random numbers. It is not being re-derived.

A CORRECTION THIS SCRIPT MAKES. `power_analysis.py` applies CEILING = {calm: 0.313,
volatile: 0.625} to the REACTING cells. Those ceilings were measured in the INJECTED
environment, where a signal was deliberately added. As an internal sanity anchor on the power
function that is harmless. As a sentence in the report -- "unit power against the opportunity"
-- it silently crosses environments. Here the ceiling is per-environment and is None where no
ceiling was ever measured, so the report cannot quote one that does not exist.

Sources (absolute):
  .../oxford_l4/per_episode_v3/{calm,volatile}.npz                 reacting, development 5e6
  .../oxford_l4/per_episode_reacting_{6e6,30e6}/{calm,volatile}.npz reacting, reserve + fresh
  .../oxford_l4/per_episode_injected_{18e6,31e6}/{calm,volatile}.npz injected, dev + fresh
  .../oxford_l4/per_episode_frozen/{val,test}.npz                   recorded books, both splits
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from power_analysis import (ALPHA, MATERIALITY, mde, mde_interval,  # noqa: E402
                            power_one_sided_t, seeds_needed)

S = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid/oxford_l4")
SCRATCH_L2 = S.parent
OUT = S / "power_analysis_multiblock.json"

# A ceiling is a MEASURED attainable edge. It exists only where one was measured.
CEILING = {"injected": {"calm": 0.313, "volatile": 0.625},
           "reacting": {"calm": None, "volatile": None},
           "frozen": {"pooled": None}}

SIM_CELL = re.compile(r"^(ppo|dqn)_(calm|volatile)_s\d+$")


def sim_cells(z, adaptive):
    """Reacting/injected: a cell is (algo, regime); seeds are the s0..s4 within it."""
    out = {}
    for k in z.files:
        m = SIM_CELL.match(k)
        if m:
            out.setdefault((m.group(1), m.group(2)), []).append(k)
    return {c: sorted(v) for c, v in out.items()}


def frozen_cells(z, _adaptive):
    """Frozen replay: a cell is (algo, size) -- the registered ARM -- and seeds are its five.

    The naming is `dqn_size193.13_seed0`, not the simulator's convention. Grouping this wrong
    is the exact failure that once silently deleted half the RQ3 sample by pattern-matching on
    a substring, so the pattern is anchored and anything unmatched is reported, not dropped.
    """
    pat = re.compile(r"^(ppo|dqn)_size([0-9.]+)_seed(\d+)$")
    out, unmatched = {}, []
    for k in z.files:
        m = pat.match(k)
        (out.setdefault((m.group(1), f"size{m.group(2)}"), []).append(k)
         if m else unmatched.append(k))
    if unmatched:
        raise SystemExit(f"unmatched frozen run names, refusing to guess: {unmatched}")
    return {c: sorted(v) for c, v in out.items()}


def analyse_block(env: str, block: str, npz: Path, valid: dict[str, bool],
                  paired: bool) -> list[dict]:
    """One block of one environment -> one record per cell.

    `paired=True`  arrays are agent COSTS and the benchmark is the 'adaptive' array.
    `paired=False` arrays are ALREADY agent-minus-TWAP differences (the frozen dumper
                   stores differences, because that is what its record is stated in).
    """
    z = np.load(npz)
    adaptive = z["adaptive"] if paired else None
    cells = sim_cells(z, adaptive) if paired else frozen_cells(z, adaptive)
    n_ep = len(z[next(iter(cells.values()))[0]])
    out = []

    for (algo, group), keys in sorted(cells.items()):
        rows = []
        for k in keys:
            d = (z[k] - adaptive) if paired else z[k]
            rows.append({"run": k, "valid": valid.get(k, True), "mean": float(d.mean()),
                         "sd_episode": float(d.std(ddof=1))})
        keep = [r for r in rows if r["valid"]]
        n = len(keep)
        base = {"env": env, "block": block, "algo": algo, "group": group,
                "n_valid": n, "n_total": len(rows), "n_episodes_per_seed": int(n_ep)}
        if n < 2:
            out.append(base | {"note": "fewer than two audit-valid seeds; "
                                       "no across-seed test exists"})
            continue

        means = np.array([r["mean"] for r in keep])
        sd_across = float(means.std(ddof=1))
        D = np.array([(z[r["run"]] - adaptive) if paired else z[r["run"]] for r in keep])
        rho = float(np.corrcoef(D)[np.triu_indices(n, 1)].mean())
        sd_ep_mean = float(np.mean([r["sd_episode"] for r in keep]))
        var_idio = (1.0 - rho) * sd_ep_mean ** 2 / n_ep
        ceil = CEILING.get(env, {}).get(group if env == "frozen" else group)

        out.append(base | {
            "observed_pooled_bps": float(means.mean()),
            "sd_episode_mean_bps": sd_ep_mean,
            "sd_across_seed_bps": sd_across,
            "inter_seed_corr": rho,
            "var_share_from_episode_noise": (min(var_idio / sd_across ** 2, 1.0)
                                             if sd_across > 0 else float("nan")),
            "sd_policy_only_bps": float(np.sqrt(max(sd_across ** 2 - var_idio, 0.0))),
            "mde_80_bps": mde(sd_across, n, 0.80),
            "mde_95_bps": mde(sd_across, n, 0.95),
            **mde_interval(sd_across, n),
            "power_at_materiality_0p05": power_one_sided_t(MATERIALITY, sd_across, n),
            "power_at_ceiling": (power_one_sided_t(ceil, sd_across, n)
                                 if ceil is not None else None),
            "ceiling_bps": ceil,
            "seeds_needed_for_0p05_at_80pct": seeds_needed(sd_across, MATERIALITY, 0.80),
            "quotable": n >= 3,
            "per_seed": keep,
        })
    return out


def cross_block(cells: list[dict]) -> list[dict]:
    """THE NEW MEASUREMENT: does dispersion -- and therefore the MDE -- transfer across blocks?

    For each (environment, cell) seen on more than one block, report episode-level sigma on
    each block and the ratio of the largest to the smallest. A ratio near 1 licenses quoting
    one MDE for the environment; a ratio far from 1 means the MDE is a property of the block
    and every claim-bearing block must be measured separately. The answer is reported whichever
    way it comes out.
    """
    by = {}
    for c in cells:
        if "sd_episode_mean_bps" in c:
            by.setdefault((c["env"], c["algo"], c["group"]), {})[c["block"]] = c
    out = []
    for (env, algo, group), blocks in sorted(by.items()):
        if len(blocks) < 2:
            continue
        sds = {b: v["sd_episode_mean_bps"] for b, v in blocks.items()}
        mdes = {b: v["mde_80_bps"] for b, v in blocks.items()}
        lo, hi = min(sds.values()), max(sds.values())
        out.append({"env": env, "algo": algo, "group": group,
                    "sd_episode_by_block": sds, "mde_80_by_block": mdes,
                    "sd_ratio_max_over_min": hi / lo if lo > 0 else float("inf"),
                    "sd_spread_pct": 100.0 * (hi - lo) / lo if lo > 0 else float("inf")})
    return out


def main() -> None:
    audit_sim = {r["run"]: bool(r.get("valid", True))
                 for r in json.loads((S / "step5_v3" / "behaviour_audit.json").read_text())}
    audit_inj_path = S / "step5_signal_dev" / "behaviour_audit.json"
    audit_inj = ({r["run"]: bool(r.get("valid", True))
                  for r in json.loads(audit_inj_path.read_text())}
                 if audit_inj_path.exists() else {})

    jobs = [
        ("reacting", "dev_5e6", S / "per_episode_v3", audit_sim, True),
        ("reacting", "reserve_6e6", S / "per_episode_reacting_6e6", audit_sim, True),
        ("reacting", "fresh_30e6", S / "per_episode_reacting_30e6", audit_sim, True),
        ("injected", "dev_18e6", S / "per_episode_injected_18e6", audit_inj, True),
        ("injected", "fresh_31e6", S / "per_episode_injected_31e6", audit_inj, True),
    ]

    cells = []
    for env, block, d, audit, paired in jobs:
        for regime in ("calm", "volatile"):
            p = d / f"{regime}.npz"
            if not p.exists():
                print(f"SKIP (missing) {env}/{block}/{regime}: {p}")
                continue
            cells += analyse_block(env, block, p, audit, paired)

    # ---- frozen replay: the behaviour audit MUST be applied here too -------------------
    # A first version passed an empty audit dict, so `valid` defaulted to True for all 30
    # agents and the frozen cells were the only ones in the table NOT audit-filtered. That
    # put audit-filtered simulator cells beside unfiltered recorded-book cells and presented
    # them identically -- an inconsistency no caption wording can repair. The frozen audit
    # criterion is the same one the simulators use, recorded per run as `dl_flag`: an agent
    # is invalid when it relies on the forced deadline purchase (deadline-residual frequency
    # above 10%).
    L2 = SCRATCH_L2 / "l2_test_results"
    frozen_audit = {}
    for split, rec in (("val", "val_recheck.json"), ("test", "test_runs_10s.json")):
        doc = json.loads((L2 / rec).read_text())
        frozen_audit[split] = {r["run"]: not bool(r.get("dl_flag"))
                               for ds in doc["datasets"] for r in ds["runs"]}
        n_bad = sum(1 for v in frozen_audit[split].values() if not v)
        print(f"frozen/{split}: behaviour audit applied, {n_bad} of "
              f"{len(frozen_audit[split])} agents invalid (deadline-leaning)")

    frozen_dir = S / "per_episode_frozen"
    for split in ("val", "test"):
        p = frozen_dir / f"{split}.npz"
        if p.exists():
            cells += analyse_block("frozen", split, p, frozen_audit[split], False)
        else:
            print(f"SKIP (missing) frozen/{split}: {p}")

    xb = cross_block(cells)

    W = 108
    print("=" * W)
    print("MINIMUM DETECTABLE EFFECT BY ENVIRONMENT AND BLOCK "
          "-- one-sided across-seed t-test, alpha = 0.05")
    print("=" * W)
    print(f"{'env':<10}{'block':<13}{'cell':<18}{'n':>3}{'pooled':>9}{'SD(ep)':>9}"
          f"{'SD(seed)':>10}{'MDE80':>9}{'[95% interval]':>20}{'pwr@.05':>9}")
    for c in cells:
        if c.get("n_valid", 0) < 2:
            print(f"{c['env']:<10}{c['block']:<13}{c['algo']+'/'+c['group']:<18}"
                  f"{c['n_valid']:>3}   {c['note']}")
            continue
        print(f"{c['env']:<10}{c['block']:<13}{c['algo']+'/'+c['group']:<18}{c['n_valid']:>3}"
              f"{c['observed_pooled_bps']:>9.4f}{c['sd_episode_mean_bps']:>9.4f}"
              f"{c['sd_across_seed_bps']:>10.4f}{c['mde_80_bps']:>9.4f}"
              f"   [{c['mde_80_lo_bps']:.4f}-{c['mde_80_hi_bps']:.4f}]"
              f"{c['power_at_materiality_0p05']:>9.3f}"
              + ("" if c["quotable"] else "   NOT QUOTABLE"))

    print("\n" + "=" * W)
    print("DOES DISPERSION TRANSFER ACROSS BLOCKS?  "
          "(the assumption the whole MDE claim rested on)")
    print("=" * W)
    for r in xb:
        blocks = " | ".join(f"{b}={v:.4f}" for b, v in sorted(r["sd_episode_by_block"].items()))
        print(f"  {r['env']:<9}{r['algo']+'/'+r['group']:<18} sigma_episode: {blocks}")
        print(f"  {'':<9}{'':<18} spread {r['sd_spread_pct']:+.1f}%  "
              f"ratio {r['sd_ratio_max_over_min']:.3f}")
    if xb:
        worst = max(xb, key=lambda r: r["sd_ratio_max_over_min"])
        print(f"\n  WORST CASE: {worst['env']} {worst['algo']}/{worst['group']} "
              f"ratio {worst['sd_ratio_max_over_min']:.3f} "
              f"({worst['sd_spread_pct']:+.1f}%)")

    print("\n" + "=" * W)
    print("READING")
    print("=" * W)
    ok = [c for c in cells if c.get("quotable")]
    for env in sorted({c["env"] for c in ok}):
        sub = [c for c in ok if c["env"] == env]
        print(f"  {env:<10} MDE(80%) {min(c['mde_80_bps'] for c in sub):.4f}"
              f" to {max(c['mde_80_bps'] for c in sub):.4f} bps"
              f"  | upper 95% bound {max(c['mde_80_hi_bps'] for c in sub):.4f}"
              f"  | {len(sub)} quotable cells")
    thin = [c for c in cells if c.get("n_valid", 0) >= 2 and not c.get("quotable")]
    for c in thin:
        print(f"  NOT QUOTABLE: {c['env']}/{c['block']} {c['algo']}/{c['group']} "
              f"-- {c['n_valid']} surviving seeds, SD on 1 d.o.f.")
    print(f"  Registered materiality threshold: {MATERIALITY} bps.")

    OUT.write_text(json.dumps({"materiality_bps": MATERIALITY, "alpha_one_sided": ALPHA,
                               "ceiling_bps": CEILING, "cells": cells,
                               "cross_block": xb}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
