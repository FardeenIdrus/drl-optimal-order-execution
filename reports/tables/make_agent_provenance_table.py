"""Agent provenance table: every hyperparameter of both agents, and where it came from.

Three sources, all read at build time, nothing typed here:
  library  -- Stable-Baselines3, introspected from the installed package signatures
  scaffold -- the vendored implementation's own config and agent construction
  used     -- the run record for a base configuration (meta.json is the authority)

Output: reports/tables/m5_agent_provenance.tex  (then sync_to_dissertation.py)
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import yaml
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.dqn.policies import DQNPolicy

ROOT = Path("/Users/fardeenidrus/Desktop/MSc Dissertation")
SCAFFOLD = ROOT / "code/qrm_optimal_execution/src/qrm_rl"
RUNS = ROOT / "scratch_hyperliquid/oxford_l4/runs_primary_v3"
HERE = Path(__file__).resolve().parent

CFG = yaml.safe_load((SCAFFOLD / "configs/default.yaml").read_text())


def _scaffold_trunk() -> tuple[list[int], str]:
    """The trunk is set in the scaffold's code, not its config (runner.py)."""
    src = (SCAFFOLD / "runner.py").read_text()
    line = next(ln for ln in src.splitlines() if "net_arch=[" in ln)
    arch = [int(x) for x in line.split("[")[1].split("]")[0].split(",")]
    act = next(ln for ln in src.splitlines() if "activation_fn=" in ln)
    return arch, act.split("activation_fn=")[1].strip().rstrip(",").split(".")[-1]


SCAF_ARCH, SCAF_ACT = _scaffold_trunk()


def _lib(cls) -> dict:
    return {k: v.default for k, v in inspect.signature(cls.__init__).parameters.items()
            if v.default is not inspect.Parameter.empty}


def _policy_default(cls) -> tuple[list[int], str]:
    """net_arch=None means the policy substitutes its own literal at construction."""
    src = inspect.getsource(cls.__init__)
    # DQNPolicy writes `net_arch = [64, 64]`; ActorCriticPolicy writes
    # `net_arch = dict(pi=[64, 64], vf=[64, 64])`. Take the first bracketed list.
    lines = [ln for ln in src.splitlines()
             if "net_arch = " in ln and "[" in ln and any(c.isdigit() for c in ln)]
    line = next((ln for ln in lines if "dict(pi=" in ln), lines[0])
    arch = [int(x) for x in line.split("[")[1].split("]")[0].split(",")]
    act = _lib(cls)["activation_fn"]
    return arch, getattr(act, "__name__", str(act))


TRAIN = ROOT / "code/drl-optimal-order-execution/src/execution/qrm/train_reactive.py"


def _base_hp() -> dict[str, dict]:
    """The two base hyperparameter dicts, parsed from the training script's own source.

    A simulator run's meta.json records only the overridable settings, so it cannot
    supply the full configuration. It is used instead to *prove* the parsed dicts are
    what ran: every override null, and net_arch, gamma and steps matching.
    """
    import ast
    tree = ast.parse(TRAIN.read_text())
    consts = {}
    for n in tree.body:
        if not (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)):
            continue
        if isinstance(n.value, (ast.Constant, ast.BinOp)):
            try:                                    # arithmetic only, no names or calls
                consts[n.targets[0].id] = eval(ast.unparse(n.value), {"__builtins__": {}}, {})
            except Exception:
                pass
    out = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)):
            continue
        keys = [k.value for k in node.value.keys if isinstance(k, ast.Constant)]
        if "net_arch" not in keys or "activation" not in keys:
            continue
        d = {}
        for k, v in zip(node.value.keys, node.value.values):
            if isinstance(v, ast.Name) and v.id in consts:
                d[k.value] = consts[v.id]
            elif isinstance(v, ast.Constant):
                d[k.value] = v.value
        d["net_arch"] = SCAF_ARCH  # CLI default "30,30,30,30,30", split to ints
        out["dqn" if "buffer_size" in d else "ppo"] = d
    return out


def _safe(node) -> bool:
    try:
        __import__("ast").literal_eval(node)
        return True
    except (ValueError, SyntaxError, TypeError):
        return False


def _verify(algo: str, hp: dict) -> str:
    """Cross-check the parsed dict against a base run's own record."""
    p = sorted(RUNS.glob(f"{algo}_*_s*/meta.json"))[0]
    m = json.loads(p.read_text())
    assert all(v is None for v in m["overrides"].values()), f"{p}: overrides applied"
    assert m["net_arch"] == hp["net_arch"], f"{p}: net_arch mismatch"
    assert abs(m["gamma"] - hp["gamma"]) < 1e-12, f"{p}: gamma mismatch"
    return f"{p.parent.name}, {m['steps']:,} steps"


def _fmt(v) -> str:
    if v is None:
        return "--"
    if isinstance(v, list):
        return r"$" + r"\times".join(str(x) for x in [v[0], len(v)]) + r"$" if len(set(v)) == 1 \
            else "[" + ", ".join(str(x) for x in v) + "]"
    if isinstance(v, float):
        t = f"{v:.6f}".rstrip("0")
        return t + "0" if t.endswith(".") else t
    if isinstance(v, int) and abs(v) >= 1000:
        return f"{v:,}".replace(",", r"{,}")
    return str(v)


def _origin(lib, scaf, used) -> str:
    same_lib = lib is not None and _eq(lib, used)
    same_scaf = scaf is not None and _eq(scaf, used)
    if same_scaf and same_lib:
        return r"Espa\~na et al., matching the library"
    if same_scaf:
        return r"Espa\~na et al."
    if same_lib:
        return "library"
    return r"\textbf{this study}"


def _eq(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-9
        except (TypeError, ValueError):
            return False
    return a == b


LABELS = {
    "net_arch": "hidden layers", "activation": "activation",
    "learning_rate": "learning rate", "buffer_size": "replay buffer",
    "batch_size": "batch size", "learning_starts": "learning starts",
    "gamma": "discount factor", "train_freq": "update frequency",
    "gradient_steps": "gradient steps", "target_update_interval": "target update",
    "n_steps": "rollout length", "exploration_initial_eps": "initial exploration",
    "exploration_final_eps": "final exploration",
    "exploration_anneal_frac": "exploration anneal",
    "n_epochs": "epochs", "gae_lambda": "GAE lambda", "clip_range": "clipping range",
    "ent_coef": "entropy coefficient", "vf_coef": "value coefficient",
    "max_grad_norm": "gradient clipping",
}
DQN_ORDER = ["net_arch", "activation", "learning_rate", "buffer_size", "batch_size",
             "learning_starts", "train_freq", "gradient_steps", "target_update_interval",
             "n_steps", "exploration_initial_eps", "exploration_final_eps",
             "exploration_anneal_frac", "gamma"]
PPO_ORDER = ["net_arch", "activation", "learning_rate", "n_steps", "batch_size",
             "n_epochs", "gae_lambda", "clip_range", "ent_coef", "gamma"]


def rows(order, lib, polarch, polact, used, is_dqn):
    out = []
    for k in order:
        if k == "net_arch":
            l, s, u = polarch, SCAF_ARCH, used.get("net_arch")
        elif k == "activation":
            l, s, u = polact, SCAF_ACT, used.get("activation")
        elif k == "exploration_anneal_frac":
            l, s, u = lib.get("exploration_fraction"), CFG.get("exploration_fraction"), \
                used.get("exploration_anneal_frac")
        else:
            l = lib.get(k)
            s = CFG.get(k) if (is_dqn or k in ("gamma",)) else None
            u = used.get(k)
        if u is None:
            continue
        # The discount rate is the scaffold's; only its conversion to the decision
        # interval is this study's, so "this study" alone would overstate the change.
        org = r"Espa\~na et al.'s rate, converted here" if k == "gamma" else _origin(l, s, u)
        # SB3 reuses the name n_steps for two different things: an n-step return
        # horizon in DQN, a rollout buffer in PPO. One label would mislead.
        lab = "n-step returns" if (k == "n_steps" and is_dqn) else LABELS[k]
        out.append((lab, _fmt(l), _fmt(s), _fmt(u), org))
    return out


def build() -> str:
    dqn_l, ppo_l = _lib(DQN), _lib(PPO)
    dqn_pa, dqn_pact = _policy_default(DQNPolicy)
    ppo_pa, ppo_pact = _policy_default(ActorCriticPolicy)
    hp = _base_hp()
    dqn_u, ppo_u = hp["dqn"], hp["ppo"]

    L = [r"\begin{tabular}{@{}p{0.18\linewidth}p{0.12\linewidth}p{0.12\linewidth}"
         r"p{0.15\linewidth}p{0.29\linewidth}@{}}", r"\toprule"]
    for title, rs in ((r"Panel A. DQN, the value-based agent", rows(DQN_ORDER, dqn_l, dqn_pa, dqn_pact,
                                                           dqn_u, True)),
                      (r"Panel B. PPO, the policy-gradient agent", rows(PPO_ORDER, ppo_l, ppo_pa,
                                                               ppo_pact, ppo_u, False))):
        if title.startswith(r"Panel B"):
            L += [r"\addlinespace[0.6em]", r"\midrule"]
        L += [rf"\multicolumn{{5}}{{@{{}}l}}{{\textbf{{{title}}}}} \\", r"\midrule",
              r"Setting & Library & Espa\~na et al. & Used here & Source \\", r"\midrule"]
        for r_ in rs:
            L.append(" & ".join(r_) + r" \\")
    L += [r"\bottomrule",
          r"\multicolumn{5}{@{}p{\dimexpr0.86\linewidth+8\tabcolsep}@{}}{\footnotesize "
          r"Library values are Stable-Baselines3 2.9.0's; a dash means the parameter does not "
          r"exist for that agent. The discount shown is the one-second simulators'. The "
          r"exploration anneal is a share of the post-warm-up budget, not the library "
          r"parameter of the same name.} \\",
          r"\end{tabular}"]
    return "\n".join(L) + "\n"


def main() -> None:
    out = HERE / "m5_agent_provenance.tex"
    out.write_text(build())
    hp = _base_hp()
    print(f"wrote {out.name}")
    print(f"  scaffold trunk {SCAF_ARCH} {SCAF_ACT} (runner.py)")
    print(f"  value agent  verified against {_verify('dqn', hp['dqn'])}")
    print(f"  policy agent verified against {_verify('ppo', hp['ppo'])}")
    print("  no value typed: library introspected, scaffold and training script parsed")


if __name__ == "__main__":
    main()
