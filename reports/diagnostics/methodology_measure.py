"""Measure everything the Methodology chapter's exhibits need. No number is typed in a builder.

Writes scratch_hyperliquid/oxford_l4/methodology_measurements.json, which
reports/tables/make_methodology_tables.py and reports/figures/methodology/ then read.
Same contract as data_chapter_measure.py -> make_data_tables.py: if a value is not in the
JSON, it was never measured, and the builder must fail rather than invent it.

WHAT IS MEASURED HERE

  agent census      per track, from run directories, with the logging/duplicate copies
                    EXCLUDED by name. Counting directories overstates: logging re-runs and
                    duplicate copies exist alongside the agents of record.
  training budget   per build, from each run's own meta.json. It is NOT uniform on the
                    recorded-book builds, and the JSON therefore carries the
                    distinct values per build rather than one number.
  order ladders     from meta.json, not from the configs: the configs declared a ladder
                    that never ran and omitted one that did; meta.json is authoritative.
  observation width from the environment code's own arithmetic and the recorded-book
                    config's feature list -- six cells, three environments x two algorithms.
  design constants  decision cadence, episode length, horizon, from configs and meta.

WHAT IS NOT MEASURED HERE, AND WHY

  Block governance (development / reserve / confirmation / spent) is a human designation
  recorded in the partition audit, not derivable from any file. Same declared exception
  the Data chapter's Panel B makes. Those cells are literals in the builder, marked there.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SCRATCH = Path("/Users/fardeenidrus/Desktop/MSc Dissertation/scratch_hyperliquid")
OX = SCRATCH / "oxford_l4"
REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs"
OUT = OX / "methodology_measurements.json"

# Directories that hold reproduction-gated logging copies or byte-identical duplicates of
# agents already counted. Excluded BY NAME so the exclusion is auditable, not inferred.
# The 20 earlier logging re-runs, 10 duplicate copies, and 30 later logging re-runs.
NOT_AGENTS_OF_RECORD = {
    "runs_signal_logged",              # logging copy of runs_signal_phaseD's PPO arm
    "runs_signal_logged_dqn",          # logging copy of runs_signal_phaseD's DQN arm
    "runs_signal_obsfix_logged",       # (Y24) reproduction-gated logging copy
    "runs_primary_v3_obsfix_logged",   # (Y24) reproduction-gated logging copy
    "runs_primary_v3_logged",          # (Y24) reproduction-gated logging copy
    # Byte-identical copies of runs_signal_phaseD's base PPO agents, kept under a second
    # name for the sealed confirmation. NOT ten more agents. Re-verified independently on
    # 2026-08-11 rather than taken from the record: 10 of 10 model files sha-256 identical
    # to their Phase D partner, 0 differing, 0 missing. The check runs below.
    "runs_signal_sealed_confirm",
}

# Pairs asserted to be byte-identical duplicates. The census is only right if they are, so
# the claim is TESTED at measurement time and the run aborts if it stops holding.
DUPLICATE_OF = {"runs_signal_sealed_confirm": "runs_signal_phaseD"}


def verify_duplicates() -> dict:
    """Re-check every 'byte-identical copy' claim the census depends on."""
    import hashlib
    out = {}
    for dup, src in DUPLICATE_OF.items():
        same = diff = missing = 0
        for run in _meta_dirs(OX / dup):
            partner = OX / src / run.name
            models = sorted(run.glob("*.zip"))
            if not partner.is_dir() or not models:
                missing += 1
                continue
            for m in models:
                q = partner / m.name
                if not q.exists():
                    missing += 1
                elif (hashlib.sha256(m.read_bytes()).hexdigest()
                      == hashlib.sha256(q.read_bytes()).hexdigest()):
                    same += 1
                else:
                    diff += 1
        if diff or missing:
            raise SystemExit(
                f"{dup} is excluded from the census as a byte-identical copy of {src}, and it "
                f"is NOT: {same} identical, {diff} differing, {missing} missing. The census is "
                f"wrong until this is resolved -- do not build exhibits from it.")
        out[dup] = {"duplicate_of": src, "identical": same, "differing": diff}
    return out


def _meta_dirs(root: Path) -> list[Path]:
    """Every immediate child of root holding a meta.json."""
    if not root.is_dir():
        return []
    return sorted(p.parent for p in root.glob("*/meta.json"))


def _load(p: Path) -> dict:
    return json.loads((p / "meta.json").read_text())


def census() -> dict:
    """Agents of record per track, plus the directory count that overstates it."""
    tracks: dict[str, dict] = {}

    # --- recorded books: three builds, each its own top-level directory -----------------
    rb_dirs = {"1-minute": SCRATCH / "runs",
               "10-second/30-minute": SCRATCH / "runs_10s",
               "10-second/10-minute": SCRATCH / "runs_10s_10min"}
    rb_builds, rb_total, rb_folders = {}, 0, 0
    for label, d in rb_dirs.items():
        runs = _meta_dirs(d)
        rb_folders += len(runs)
        metas = [_load(r) for r in runs]
        # A stub is a run directory whose meta.json records no completed training.
        real = [m for m in metas if m.get("total_timesteps")]
        budgets = sorted({int(m["total_timesteps"]) for m in real})
        sizes = sorted({float(m["size_btc"]) for m in real if "size_btc" in m})
        rb_builds[label] = {
            "agents": len(real),
            "directories": len(runs),
            "stubs": len(metas) - len(real),
            "training_budget_steps": budgets,
            "order_sizes_btc": sizes,
            "algos": sorted({m["algo"] for m in real}),
            "seeds": sorted({int(m["seed"]) for m in real}),
        }
        rb_total += len(real)
    tracks["recorded_books"] = {
        "agents": rb_total, "directories": rb_folders, "builds": rb_builds,
        "budget_is_uniform": len({b for v in rb_builds.values()
                                  for b in v["training_budget_steps"]}) == 1,
    }

    # --- the two simulator tracks: many campaign directories under oxford_l4 ------------
    sim_dirs = sorted(d for d in OX.glob("runs*") if d.is_dir())
    reacting, injected, excluded = [], [], []
    folders = 0
    for d in sim_dirs:
        runs = _meta_dirs(d)
        folders += len(runs)
        if d.name in NOT_AGENTS_OF_RECORD:
            excluded.append({"campaign": d.name, "agents": len(runs)})
            continue
        for r in runs:
            m = _load(r)
            rec = {"campaign": d.name, "run": r.name, "algo": m.get("algo"),
                   "steps": m.get("steps"), "order_btc": m.get("order_btc"),
                   "env_steps": m.get("env_steps"), "regime": m.get("regime"),
                   "injected": bool(m.get("injected", False))}
            (injected if rec["injected"] else reacting).append(rec)

    def summarise(rows: list[dict]) -> dict:
        return {
            "agents": len(rows),
            "campaigns": len({r["campaign"] for r in rows}),
            "training_budget_steps": sorted({r["steps"] for r in rows if r["steps"]}),
            "order_sizes_btc": sorted({r["order_btc"] for r in rows if r["order_btc"]}),
            "decisions_per_episode": sorted({r["env_steps"] for r in rows if r["env_steps"]}),
            "algos": sorted({r["algo"] for r in rows if r["algo"]}),
            "regimes": sorted({r["regime"] for r in rows if r["regime"]}),
        }

    # The PRIMARY design, distinguished from the robustness grid. Both tracks run at a 1 s
    # cadence, so decisions-per-episode and horizon-in-seconds coincide; the grid campaigns
    # vary the horizon, so the raw set spans them all and min()/max() over it is meaningless
    # as "the design". The primary is the cell the injected track was run at, which is also
    # the modal reacting cell (criteria section 1: "Cadence 1 s; horizon 300 s").
    def modal(rows, key):
        from collections import Counter
        c = Counter(r[key] for r in rows if r[key])
        return c.most_common(1)[0][0] if c else None

    tracks["primary_design"] = {
        "cadence_s": 1,
        "decisions_per_episode": modal(injected, "env_steps") or modal(reacting, "env_steps"),
        "order_btc": modal(injected, "order_btc") or modal(reacting, "order_btc"),
        "horizon_variants_decisions": sorted({r["env_steps"] for r in reacting if r["env_steps"]}),
        "source": "meta.json env_steps/order_btc; cadence from criteria section 1",
    }
    tracks["reacting_simulator"] = summarise(reacting)
    tracks["injected_simulator"] = summarise(injected)
    tracks["_simulator_directories"] = folders
    tracks["_excluded_logging_copies"] = excluded

    total = rb_total + len(reacting) + len(injected)
    all_folders = rb_folders + folders
    tracks["TOTAL_agents_of_record"] = total
    tracks["TOTAL_directories"] = all_folders
    tracks["directories_overstate_by"] = all_folders - total
    return tracks


def observation_widths() -> dict:
    """Six cells. Recorded books from its config's feature list; simulators from the code."""
    cfg = (CONFIGS / "experiment_10s.yaml").read_text()
    feats = re.search(r"^\s*features:\s*\[(.+?)\]", cfg, re.M).group(1)
    feats = [f.strip() for f in feats.split(",")]

    # The simulator's width is 2 + 2K + 1 + 2 + 2, and K is NOT a literal in the code --
    # it comes from the calibrated bundle (reactive_env.py:109, self.K = self.bundle.K).
    # Read it from the bundles themselves, both regimes, and refuse to proceed if they
    # disagree. This is the direct-from-the-bundle confirmation the plan's §7.4 item 2 asks
    # for; inferring K from the observation width would be circular.
    import numpy as np
    bundles = {r: SCRATCH / "oxford_l4" / "step3g" / f"qrm_bundle_{r}_b.npz"
               for r in ("calm", "volatile")}
    Ks, Qs = {}, {}
    for r, p in bundles.items():
        with np.load(p, allow_pickle=True) as b:
            Ks[r], Qs[r] = int(b["K"]), int(b["Q"])
    if len(set(Ks.values())) != 1:
        raise SystemExit(f"the two bundles disagree on K: {Ks}. Resolve before building.")
    K = Ks["calm"]

    env = (REPO / "src" / "execution" / "qrm" / "reactive_env.py").read_text()
    if not re.search(r"self\.obs_dim\s*=\s*2\s*\+\s*2\s*\*\s*self\.K\s*\+\s*1\s*\+\s*2\s*\+\s*2", env):
        raise SystemExit("reactive_env.py's obs_dim arithmetic changed; re-derive before building.")
    reacting = 2 + 2 * K + 1 + 2 + 2

    return {
        "recorded_books": {"base": 2 + len(feats), "with_arrival_price": None,
                           "features": feats,
                           "note": "never re-trained under the amendment; disclosed"},
        "reacting_simulator": {"base": reacting, "with_arrival_price": reacting + 1},
        "injected_simulator": {"base": reacting + 1, "with_arrival_price": reacting + 2},
        "K_per_side": K,
        "Q_by_regime": Qs,
        "groups": {"inventory": 1, "time remaining": 1, "queue sizes": 2 * K,
                   "spread": 1, "own fills": 2, "trailing market flow": 2},
        "source": ("real_data_env.py:104 (2 + len(obs_features)); reactive_env.py:134-146; "
                   "K and Q read from step3g/qrm_bundle_{calm,volatile}_b.npz"),
    }


def action_grid() -> dict:
    cfg = (CONFIGS / "experiment.yaml").read_text()
    acts = re.search(r"^\s*actions:\s*\[(.+?)\]", cfg, re.M).group(1)
    return {"multiples": [float(a) for a in acts.split(",")],
            "twap_action": 1.0,
            "source": "configs/experiment.yaml"}


def main() -> None:
    dup = verify_duplicates()
    out = {
        "_provenance": "reports/diagnostics/methodology_measure.py",
        "duplicate_verification": dup,
        "census": census(),
        "observation_widths": observation_widths(),
        "action_grid": action_grid(),
    }
    OUT.write_text(json.dumps(out, indent=1, default=str))
    c = out["census"]
    print(f"census: recorded books {c['recorded_books']['agents']} | "
          f"reacting {c['reacting_simulator']['agents']} | "
          f"injected {c['injected_simulator']['agents']} | "
          f"TOTAL {c['TOTAL_agents_of_record']}")
    print(f"directories {c['TOTAL_directories']}, overstating by {c['directories_overstate_by']}")
    print(f"recorded-book budget uniform: {c['recorded_books']['budget_is_uniform']}")
    for k, v in c["recorded_books"]["builds"].items():
        print(f"  {k}: {v['agents']} agents, budgets {v['training_budget_steps']}, "
              f"sizes {v['order_sizes_btc']}, stubs {v['stubs']}")
    print(f"duplicate check: {dup}")
    ow = out["observation_widths"]
    print(f"observation widths: recorded books {ow['recorded_books']['base']} | "
          f"reacting {ow['reacting_simulator']['base']}/{ow['reacting_simulator']['with_arrival_price']} | "
          f"injected {ow['injected_simulator']['base']}/{ow['injected_simulator']['with_arrival_price']} "
          f"| K={ow['K_per_side']} Q={ow['Q_by_regime']}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
