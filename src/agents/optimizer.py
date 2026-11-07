"""Self-improving workflow search (AFlow-style), G1 + core, Phase 2.

This is the optimizer your run script calls. The public API is exactly:

    optimise(...)        run the AFlow loop, return the search tree (list of
                         OptimisationRecord, root = seed at index 0)
    best_record(history) return the highest-scoring record from a search tree
    evaluate_spec(...)   score one WorkflowSpec on a task set, return an
                         OptimisationRecord
    AgentWorkflow        re-exported from workflow.py (so `from
                         src.agents.optimizer import AgentWorkflow` works)

It uses your code, not a parallel implementation:
  * starts from BLANK_SEED / PROCESS_SEED in src/agents/seeds.py
  * scores via run_tasks + summarise in src/eval/evaluate.py
  * the cost-aware score comes from summarise() in src/eval/evaluate.py, which
    uses G3's objective (src/agents/objective.py) when it is present
  * every candidate workflow is run by AgentWorkflow in src/agents/workflow.py
  * trajectories are logged via src/agents/trajectory.py (one JSONL file per
    search, written when a traj_logger is passed)

The loop (one search):
  1. SELECT   a parent workflow with AFlow's soft mixed-probability selection
              (mostly the best, sometimes explore).
  2. PROPOSE  an LLM proposer picks ONE modification from a 10-action library,
              seeing what earlier modifications did (the experience pool).
              Invalid outputs fall back to a random valid action.
  3. EXPAND   apply_action produces a new WorkflowSpec.
  4. SCORE    evaluate_spec runs the spec on the validation set via run_tasks
              and summarise; the score is summarise()'s cost-aware objective.
  5. KEEP     append to the search tree; track the best; early-stop on
              `patience` rounds with no improvement.

Where the three research gaps live:
  Gap 1   business-process tasks (next-activity + anomaly) instead of math/code/QA
  Gap 2   seed = "process" starts from the process-mined workflow (Gap 2 ON)
  Gap 3   cost_weight > 0 puts token cost into the score (Gap 3 ON)

CLI:
    python -m src.agents.optimizer --provider mock --seed process --rounds 5
    python -m src.agents.optimizer --seed blank --rounds 8 --cost-weight 0.05
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from pathlib import Path

import pandas as pd

from src.agents.state import (CostRecord, OptimisationRecord, Step, WorkflowSpec)
from src.agents.trajectory import TrajectoryLogger
from src.agents.workflow import AgentWorkflow  # re-exported (see module docstring)
from src.baselines.process_mining import ProcessMiningBaseline
from src.config import load_config
from src.data.build_tasks import build_anomaly, build_next_activity
from src.data.preprocess import temporal_split, to_traces
from src.eval.evaluate import run_tasks, summarise
from src.llm.client import make_llm

# The action library: the discrete moves the optimiser can make. Wider than
# Phase 1 (7 actions) so the proposer can pick cheap vs expensive variants of
# each operator — needed for the cost-aware Pareto front (Gap 3).
ACTIONS = [
    "add process_context (top_k=1)",
    "add process_context (top_k=3)",
    "add process_context (top_k=5)",
    "remove process_context",
    "add verify (mode=revise)",
    "add verify (mode=flag)",
    "remove verify",
    "add ensemble (n=3)",
    "add ensemble (n=5)",
    "remove ensemble",
]


# ============================================================ public API ====

def evaluate_spec(spec: WorkflowSpec, llm, vocab: list[str], pm: ProcessMiningBaseline,
                  na_tasks: list[dict], an_tasks: list[dict],
                  cost_weight: float = 0.0, token_scale: float = 1000.0,
                  name: str | None = None, round_num: int = 0, parent: int | None = None,
                  modification: str = "", proposer_reasoning: str = "",
                  traj_logger: TrajectoryLogger | None = None) -> OptimisationRecord:
    """Score one WorkflowSpec on the validation set.

    Uses run_tasks + summarise from src/eval/evaluate.py, so the candidate
    workflow is scored by exactly the same code as the baselines and the
    final test evaluation. Returns an OptimisationRecord.
    """
    agent = AgentWorkflow(spec, llm, vocab, pm, traj_logger=traj_logger)
    method_name = name or spec.name
    rows = run_tasks(agent, method_name, na_tasks, an_tasks, progress=None, keep_steps=True)
    summary = summarise(rows, cost_weight=cost_weight, token_scale=token_scale)

    na_acc = summary.get("next_activity", {}).get("accuracy")
    an_acc = summary.get("anomaly", {}).get("accuracy")
    overall = summary.get("overall", {})
    tokens_per_task = overall.get("tokens_per_task", 0.0)
    score = overall.get("objective", 0.0)     # accuracy - cost_weight * tokens / token_scale
    acc = overall.get("task_score", 0.0)      # mean of the two task accuracies
    failures = [_compact_failure(r) for r in rows if not r["correct"]][:6]

    cost = CostRecord(
        tokens_per_task=tokens_per_task,
        latency_s_per_task=overall.get("latency_s_per_task", 0.0),
        llm_calls_per_task=overall.get("llm_calls_per_task", 0.0),
        tool_calls_per_task=overall.get("tool_calls_per_task", 0.0),
        cache_hit_rate=round(sum(1 for r in rows if r.get("cached")) / max(len(rows), 1), 4),
        format_retry_rate=overall.get("format_retry_rate", 0.0),
        fallback_rate=overall.get("fallback_rate", 0.0),
    )
    return OptimisationRecord(
        round=round_num, parent=parent, modification=modification,
        proposer_reasoning=proposer_reasoning,
        spec=spec, accuracy=acc,
        accuracy_next_activity=na_acc or 0.0,
        accuracy_anomaly=an_acc or 0.0,
        tokens_per_task=tokens_per_task, score=score,
        improved=False,  # set by the caller, who knows the parent's score
        cost=cost, children=[], summary=summary, failure_examples=failures,
    )


def best_record(history: list[OptimisationRecord]) -> OptimisationRecord:
    """Return the highest-scoring record from a search tree."""
    return max(history, key=lambda r: r.score)


def optimise(seed_spec: WorkflowSpec, llm, vocab: list[str], pm: ProcessMiningBaseline,
             na_tasks: list[dict], an_tasks: list[dict],
             optimizer_llm=None, rounds: int = 5,
             cost_weight: float = 0.0, token_scale: float = 1000.0,
             patience: int = 3, top_k: int = 3, mix_lambda: float = 0.2,
             alpha: float = 0.4, seed: int = 42,
             traj_logger: TrajectoryLogger | None = None,
             experience_path: str | Path | None = None,
             log=print) -> list[OptimisationRecord]:
    """Run the AFlow loop. Returns the full search tree (root = seed at index 0).

    Parameters
    ----------
    seed_spec        starting WorkflowSpec (use seeds.get_seed("blank"|"process"))
    llm              executor LLM (passed to AgentWorkflow)
    vocab            activity vocabulary
    pm               fitted ProcessMiningBaseline (the DFG)
    na_tasks, an_tasks  validation tasks (NEVER the test set)
    optimizer_llm    proposer LLM (defaults to `llm` if None)
    rounds           max number of modifications to try (excluding the seed)
    cost_weight      Gap 3 switch: 0 = accuracy-only, >0 = cost-aware
    token_scale      normaliser for the cost penalty (default 1000)
    patience         early-stop after this many rounds with no improvement
    top_k, mix_lambda, alpha  soft-mixed-probability selection params (AFlow Eq. 1)
    seed             RNG seed for reproducibility
    traj_logger      if set, every candidate's trajectories are logged
    experience_path  if set, one record per round is appended here (the
                     experience pool, used as in-context examples next round)
    log              callable for human-readable progress (default print)
    """
    rng = random.Random(seed)
    proposer = optimizer_llm or llm

    # Round 0: evaluate the seed.
    seed_rec = evaluate_spec(seed_spec, llm, vocab, pm, na_tasks, an_tasks,
                             cost_weight, token_scale, name=seed_spec.name,
                             round_num=0, parent=None, modification=f"seed={seed_spec.name}",
                             proposer_reasoning="initial seed", traj_logger=None)
    seed_rec.improved = True
    history = [seed_rec]
    best = seed_rec
    no_improve = 0
    log(f"round 0 (seed={seed_spec.name} spec={seed_spec.signature()}): "
        f"acc_na={seed_rec.accuracy_next_activity:.3f} acc_an={seed_rec.accuracy_anomaly:.3f} "
        f"acc={seed_rec.accuracy:.3f} tokens/task={seed_rec.cost.tokens_per_task:.0f} "
        f"score={seed_rec.score:.3f}")

    exp_file = open(experience_path, "a", encoding="utf-8") if experience_path else None
    if exp_file:
        exp_file.write(json.dumps(_exp_row(seed_rec)) + "\n")

    try:
        seen = {seed_rec.spec.signature()}   # every workflow already evaluated
        for rnd in range(1, rounds + 1):
            parent = soft_mixed_select(history, rng, k=top_k, lam=mix_lambda, alpha=alpha)
            spec, action, reasoning = _next_new_spec(proposer, parent, history, seen, rng, rnd)
            if spec is None:   # every action from this parent was a no-op or a repeat
                log(f"round {rnd}: parent={parent.round} has no untried change left; skipped")
                no_improve += 1
                if no_improve >= patience:
                    log(f"early stop: no improvement for {no_improve} rounds")
                    break
                continue
            seen.add(spec.signature())

            rec = evaluate_spec(spec, llm, vocab, pm, na_tasks, an_tasks,
                                cost_weight, token_scale, name=spec.name,
                                round_num=rnd, parent=parent.round,
                                modification=action, proposer_reasoning=reasoning,
                                traj_logger=traj_logger)
            rec.improved = rec.score > parent.score
            parent.children.append(rnd)
            history.append(rec)
            if exp_file:
                exp_file.write(json.dumps(_exp_row(rec)) + "\n")
                exp_file.flush()

            log(f"round {rnd}: parent={parent.round} action='{action}' -> "
                f"acc_na={rec.accuracy_next_activity:.3f} acc_an={rec.accuracy_anomaly:.3f} "
                f"acc={rec.accuracy:.3f} tokens/task={rec.cost.tokens_per_task:.0f} "
                f"score={rec.score:.3f} {'IMPROVED' if rec.improved else 'no improve'}")
            log(f"   reason: {reasoning[:140]}")

            if rec.improved and rec.score > best.score:
                best = rec
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                log(f"early stop: no improvement for {no_improve} rounds")
                break
    finally:
        if exp_file:
            exp_file.close()

    log(f"\nBest: round {best.round} spec={best.spec.signature()} "
        f"score={best.score:.3f} (acc={best.accuracy:.3f} "
        f"tokens/task={best.cost.tokens_per_task:.0f})")
    return history


# ============================================================ internals ====

def soft_mixed_select(records: list[OptimisationRecord], rng: random.Random,
                      k: int = 3, lam: float = 0.2, alpha: float = 0.4) -> OptimisationRecord:
    """AFlow's soft mixed-probability selection.

    Picks one of the top-k nodes: with probability `lam` uniformly (explore),
    otherwise proportional to exp(alpha * (score - max_score)) (exploit).
    """
    top = sorted(records, key=lambda r: r.score, reverse=True)[:k]
    s_max = top[0].score
    w = [math.exp(alpha * (r.score - s_max)) for r in top]
    probs = [lam / len(top) + (1 - lam) * wi / sum(w) for wi in w]
    return rng.choices(top, weights=probs, k=1)[0]


def propose(llm, parent: OptimisationRecord, history: list[OptimisationRecord],
            rng: random.Random, rejected: list[str] | None = None) -> tuple[str, str]:
    """Ask the optimiser LLM to pick the next action.

    The prompt shows the parent workflow, what was already tried on it, a few
    validation cases it got wrong, and any proposals just rejected as no-ops or
    repeats. Returns (action, reasoning). On any parse failure or invalid output
    we fall back to a random valid action and record that we did so.
    """
    tried = "\n".join(f"- round {h.round}: {h.modification} -> score {h.score:.3f} "
                      f"({'better' if h.improved else 'worse'})"
                      for h in history if h.parent == parent.round) or "- nothing yet"
    fails = parent.failure_examples
    shown = rng.sample(fails, min(4, len(fails))) if fails else []
    fail_txt = "\n".join(_fmt_failure(f) for f in shown) or "- none (all validation cases correct)"
    reject_txt = ("\nThese choices were rejected because they change nothing or repeat a "
                  "workflow already tried; pick a different one:\n"
                  + "\n".join(f"- {r}" for r in rejected) + "\n") if rejected else ""
    options = "\n".join(f"[{i + 1}] {a}" for i, a in enumerate(ACTIONS))
    prompt = (
        "You improve an AI workflow for business-process tasks (predict the next\n"
        "step of a travel-expense declaration, and detect anomalous executions).\n"
        f"Current workflow: {parent.spec.signature()} (score {parent.score:.3f},\n"
        f"accuracy {parent.accuracy:.3f}, tokens/task {parent.cost.tokens_per_task:.0f}).\n\n"
        f"Modifications already tried on this workflow:\n{tried}\n\n"
        f"Validation cases this workflow got wrong:\n{fail_txt}\n"
        f"{reject_txt}\n"
        "Choose ONE modification that is most likely to fix these mistakes and raise the\n"
        "score. The score rewards accuracy and penalises token cost, so do not add\n"
        "expensive operators unless they pay off.\n"
        f"{options}\n\n"
        "Reply with the option number on the first line, then one short sentence\n"
        "explaining why."
    )
    r = llm.chat([{"role": "user", "content": prompt}])
    text = r.text.strip()
    m = re.match(r"\s*\[?(\d+)", text)
    if m and 1 <= int(m.group(1)) <= len(ACTIONS):
        action = ACTIONS[int(m.group(1)) - 1]
        reasoning = text.split("\n", 1)[1].strip() if "\n" in text else text
        return action, reasoning
    # Fall back: invalid proposer output -> random valid action.
    return rng.choice(ACTIONS), f"[fallback: proposer returned '{text[:60]}', chose random action]"


def _next_new_spec(proposer, parent: OptimisationRecord, history: list[OptimisationRecord],
                   seen: set[str], rng: random.Random, rnd: int):
    """Get a modification that produces a NEW workflow.

    Up to 3 proposals; a proposal that changes nothing (e.g. 'remove verify'
    when there is no verify) or recreates an already-evaluated workflow is
    rejected and the proposer is told why. If all 3 fail, pick a random action
    that does give a new workflow. Returns (spec, action, reasoning) or
    (None, None, None) when no new workflow is reachable from this parent.
    """
    rejected: list[str] = []
    for _ in range(3):
        action, reasoning = propose(proposer, parent, history, rng, rejected)
        spec = apply_action(parent.spec, action, f"r{rnd}")
        if spec.signature() == parent.spec.signature():
            rejected.append(f"{action} (changes nothing)")
        elif spec.signature() in seen:
            rejected.append(f"{action} (already tried)")
        else:
            return spec, action, reasoning
    fresh = [a for a in ACTIONS
             if (sig := apply_action(parent.spec, a, "tmp").signature()) != parent.spec.signature()
             and sig not in seen]
    if not fresh:
        return None, None, None
    action = rng.choice(fresh)
    return (apply_action(parent.spec, action, f"r{rnd}"), action,
            f"[fallback: proposals {rejected} were no-ops or repeats; chose a random new change]")


def _compact_failure(r: dict) -> dict:
    """A small, prompt-safe summary of one wrong validation answer."""
    inp = r.get("input") or []
    return {"task": r["task"], "input": inp[-5:] if r["task"] == "next_activity" else inp[:12],
            "prediction": r.get("prediction"), "target": r.get("target"),
            "anomaly_type": r.get("anomaly_type")}


def _fmt_failure(f: dict) -> str:
    if f["task"] == "next_activity":
        return (f"- next step after ...{' -> '.join(f['input'])}: answered '{f['prediction']}', "
                f"correct '{f['target']}'")
    lab = {1: "ANOMALOUS", 0: "NORMAL", None: "no answer"}
    kind = f" (perturbation: {f['anomaly_type']})" if f.get("anomaly_type") else ""
    return (f"- case {' -> '.join(f['input'])}: answered {lab.get(f['prediction'])}, "
            f"correct {lab.get(f['target'])}{kind}")


def apply_action(spec: WorkflowSpec, action: str, name: str) -> WorkflowSpec:
    """Return a new WorkflowSpec with the action applied. Always keeps a `generate`.
    Idempotent on no-ops (e.g. `remove ensemble` on a spec with no ensemble)."""
    steps = [s.model_copy() for s in spec.steps]
    drop = lambda op: [s for s in steps if s.operator != op]

    if action.startswith("add process_context"):
        k = int(re.search(r"top_k=(\d+)", action).group(1))
        steps = [Step(operator="process_context", params={"top_k": k})] + drop("process_context")
    elif action == "remove process_context":
        steps = drop("process_context")
    elif action.startswith("add verify"):
        mode = re.search(r"mode=(\w+)", action).group(1)
        steps = drop("verify") + [Step(operator="verify", params={"mode": mode})]
    elif action == "remove verify":
        steps = drop("verify")
    elif action.startswith("add ensemble"):
        n = int(re.search(r"n=(\d+)", action).group(1))
        steps = drop("ensemble")
        # Place ensemble after process_context (if any) and before generate.
        insert_at = sum(1 for s in steps if s.operator == "process_context")
        steps.insert(insert_at, Step(operator="ensemble", params={"n": n}))
    elif action == "remove ensemble":
        steps = drop("ensemble")

    if not any(s.operator == "generate" for s in steps):
        steps.append(Step(operator="generate"))
    return WorkflowSpec(name=name, steps=steps)


def _exp_row(rec: OptimisationRecord) -> dict:
    """One row of the experience pool: small, safe to feed back to the proposer."""
    return {
        "round": rec.round, "parent": rec.parent, "modification": rec.modification,
        "spec": rec.spec.signature(), "score": round(rec.score, 4),
        "accuracy": round(rec.accuracy, 4), "tokens_per_task": round(rec.cost.tokens_per_task, 1),
        "improved": rec.improved, "reasoning": rec.proposer_reasoning[:200],
    }


# ============================================================ CLI ===========

def _setup_file_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger(f"optim_{log_path.stem}")
    log.setLevel(logging.INFO)
    log.handlers = []
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    log.addHandler(fh)
    return log


def _log_to(loggers):
    """Compose a print-like callable that writes to every logger."""
    def _emit(msg):
        for lg in loggers:
            lg.info(msg) if isinstance(lg, logging.Logger) else lg(msg)
    return _emit


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--provider")
    ap.add_argument("--seed", choices=["blank", "process"], default="blank")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--cost-weight", type=float, default=0.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rng_seed = cfg["seed"]
    d = Path(cfg["data"]["processed_dir"])
    train = pd.read_csv(d / "train.csv", parse_dates=["timestamp"])
    vocab = json.loads((d / "vocab.json").read_text())

    # Validation = the latest 20% of TRAINING cases; the test set stays untouched.
    fit_df, val_df = temporal_split(train, 0.8)
    pm = ProcessMiningBaseline().fit(list(to_traces(fit_df).values()))
    val_traces = to_traces(val_df)
    variants = {tuple(t) for t in list(to_traces(fit_df).values()) + list(val_traces.values())}
    n_na = cfg["optimizer"]["n_val_next_activity"]
    n_an = cfg["optimizer"]["n_val_anomaly"]
    na_tasks = build_next_activity(val_traces, n_na, rng_seed + 100)
    an_tasks = build_anomaly(val_traces, variants, vocab, n_an,
                             cfg["tasks"]["anomaly"]["anomaly_ratio"], rng_seed + 101)
    for r in na_tasks + an_tasks:
        r["task_id"] = "val_" + r["task_id"]

    llm = make_llm(cfg, args.provider)

    # Use the canonical seeds from seeds.py (Gap 2 switch).
    from src.agents.seeds import get_seed
    seed_spec = get_seed(args.seed)

    res = Path(cfg["eval"]["results_dir"])
    logs = Path(cfg["eval"]["logs_dir"])
    tag = f"{args.seed}_cw{int(args.cost_weight * 1000):03d}"
    traj_path = logs / f"trajectories_optim_{tag}.jsonl"
    traj_logger = TrajectoryLogger(traj_path)
    exp_path = res / f"optimiser_experience_{tag}.jsonl"
    exp_path.write_text("")  # truncate at the start of a run
    file_log = _setup_file_logger(logs / f"optimiser_run_{tag}.log")
    log = _log_to([file_log, print])

    history = optimise(seed_spec, llm, vocab, pm, na_tasks, an_tasks,
                       optimizer_llm=None, rounds=args.rounds,
                       cost_weight=args.cost_weight,
                       token_scale=cfg["optimizer"]["token_scale"],
                       patience=cfg["optimizer"]["patience"],
                       top_k=cfg["optimizer"]["top_k"],
                       mix_lambda=cfg["optimizer"]["mix_lambda"],
                       alpha=cfg["optimizer"]["alpha"],
                       seed=rng_seed, traj_logger=traj_logger,
                       experience_path=exp_path, log=log)

    best = best_record(history)
    out = res / f"optimisation_{tag}.json"
    out.write_text(json.dumps([h.model_dump() for h in history], indent=2))
    log(f"Search tree ({len(history)} nodes) saved to {out}")
    log(f"Experience pool saved to {exp_path}")
    log(f"Trajectories saved to {traj_path}")


if __name__ == "__main__":
    main()
