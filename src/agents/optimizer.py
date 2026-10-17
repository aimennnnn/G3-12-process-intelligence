"""Self-improving workflow search (AFlow-style, simplified). PHASE 2 SKELETON.

Each round:
  1. select a parent workflow from the best ones so far
     (AFlow's soft mixed-probability selection: mostly the best, sometimes explore)
  2. an LLM optimizer picks ONE modification from a fixed action list,
     seeing what earlier modifications did (tree-structured experience)
  3. the new workflow is scored on a VALIDATION set built from the training
     split (never the test set)
  4. the result is stored, and the best workflow is kept

Where the three research gaps live:
  Gap 1  business-process tasks instead of math/code/QA benchmarks
  Gap 2  --seed process   starts from the process-mined workflow, not a blank one
  Gap 3  --cost-weight    score = accuracy - cost_weight * (tokens per task / 1000)

    python -m src.agents.optimizer --provider mock --rounds 5
    python -m src.agents.optimizer --seed process --rounds 10 --cost-weight 0.05
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import pandas as pd

from src.agents.state import (BLANK_SEED, PROCESS_SEED, OptimisationRecord, Step,
                              WorkflowSpec)
from src.agents.workflow import AgentWorkflow
from src.baselines.process_mining import ProcessMiningBaseline
from src.config import load_config
from src.data.build_tasks import build_next_activity
from src.data.preprocess import temporal_split, to_traces
from src.llm.client import make_llm

ACTIONS = [
    "add process_context (top_k=3)",
    "add process_context (top_k=1)",
    "remove process_context",
    "add verify",
    "remove verify",
    "add ensemble (n=3)",
    "remove ensemble",
]


def apply_action(spec: WorkflowSpec, action: str, name: str) -> WorkflowSpec:
    steps = [s.model_copy() for s in spec.steps]
    drop = lambda op: [s for s in steps if s.operator != op]
    if action.startswith("add process_context"):
        k = int(re.search(r"top_k=(\d+)", action).group(1))
        steps = [Step(operator="process_context", params={"top_k": k})] + drop("process_context")
    elif action == "remove process_context":
        steps = drop("process_context")
    elif action == "add verify":
        steps = drop("verify") + [Step(operator="verify")]
    elif action == "remove verify":
        steps = drop("verify")
    elif action.startswith("add ensemble"):
        steps = drop("ensemble")
        steps.insert(len([s for s in steps if s.operator == "process_context"]),
                     Step(operator="ensemble", params={"n": 3}))
    elif action == "remove ensemble":
        steps = drop("ensemble")
    if not any(s.operator == "generate" for s in steps):
        steps.append(Step(operator="generate"))
    return WorkflowSpec(name=name, steps=steps)


def soft_mixed_select(records: list[OptimisationRecord], rng: random.Random,
                      k: int = 3, lam: float = 0.2, alpha: float = 0.4) -> OptimisationRecord:
    top = sorted(records, key=lambda r: r.score, reverse=True)[:k]
    s_max = top[0].score
    w = [math.exp(alpha * (r.score - s_max)) for r in top]
    probs = [lam / len(top) + (1 - lam) * wi / sum(w) for wi in w]
    return rng.choices(top, weights=probs, k=1)[0]


def propose(llm, parent: OptimisationRecord, history: list[OptimisationRecord],
            rng: random.Random) -> str:
    tried = "\n".join(f"- round {h.round}: {h.modification} -> score {h.score:.3f}"
                      f" ({'better' if h.improved else 'worse'})"
                      for h in history if h.parent == parent.round) or "- nothing yet"
    options = "\n".join(f"[{i + 1}] {a}" for i, a in enumerate(ACTIONS))
    prompt = (
        "You improve an AI workflow for predicting the next step of a business process.\n"
        f"Current workflow steps: {[s.operator for s in parent.spec.steps]} "
        f"(score {parent.score:.3f}).\n"
        f"Modifications already tried on this workflow:\n{tried}\n\n"
        f"Choose ONE modification that is most likely to raise the score:\n{options}\n\n"
        "Reply with only the option number."
    )
    r = llm.chat([{"role": "user", "content": prompt}])
    m = re.search(r"\d+", r.text)
    if m and 1 <= int(m.group()) <= len(ACTIONS):
        return ACTIONS[int(m.group()) - 1]
    return rng.choice(ACTIONS)  # invalid output: fall back to a random valid action


def score_workflow(spec, llm, vocab, pm, tasks, cost_weight: float):
    agent = AgentWorkflow(spec, llm, vocab, pm)
    correct = tokens = 0
    for t in tasks:
        pred, meta = agent.predict_next(t["prefix"])
        correct += int(pred == t["target"])
        tokens += meta["prompt_tokens"] + meta["completion_tokens"]
    acc = correct / max(len(tasks), 1)
    tpt = tokens / max(len(tasks), 1)
    return acc, tpt, acc - cost_weight * tpt / 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--provider")
    ap.add_argument("--seed", choices=["blank", "process"], default="blank")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--n-val", type=int, default=30)
    ap.add_argument("--cost-weight", type=float, default=0.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rng = random.Random(cfg["seed"])
    d = Path(cfg["data"]["processed_dir"])
    train = pd.read_csv(d / "train.csv", parse_dates=["timestamp"])
    vocab = json.loads((d / "vocab.json").read_text())

    # Validation = the latest 20% of TRAINING cases; the test set stays untouched.
    fit_df, val_df = temporal_split(train, 0.8)
    pm = ProcessMiningBaseline().fit(list(to_traces(fit_df).values()))
    val_tasks = build_next_activity(to_traces(val_df), args.n_val, cfg["seed"])
    llm = make_llm(cfg, args.provider)

    seed_spec = PROCESS_SEED if args.seed == "process" else BLANK_SEED
    acc, tpt, score = score_workflow(seed_spec, llm, vocab, pm, val_tasks, args.cost_weight)
    history = [OptimisationRecord(round=0, parent=None, modification="seed", spec=seed_spec,
                                  accuracy=acc, tokens_per_task=tpt, score=score, improved=True)]
    print(f"round 0 (seed={args.seed}): acc={acc:.3f} tokens/task={tpt:.0f} score={score:.3f}")

    for rnd in range(1, args.rounds + 1):
        parent = soft_mixed_select(history, rng)
        action = propose(llm, parent, history, rng)
        spec = apply_action(parent.spec, action, f"r{rnd}")
        acc, tpt, score = score_workflow(spec, llm, vocab, pm, val_tasks, args.cost_weight)
        rec = OptimisationRecord(round=rnd, parent=parent.round, modification=action, spec=spec,
                                 accuracy=acc, tokens_per_task=tpt, score=score,
                                 improved=score > parent.score)
        history.append(rec)
        print(f"round {rnd}: parent={parent.round} '{action}' -> acc={acc:.3f} "
              f"tokens/task={tpt:.0f} score={score:.3f} {'↑' if rec.improved else '↓'}")

    best = max(history, key=lambda r: r.score)
    out = Path(cfg["eval"]["results_dir"]) / f"optimisation_{args.seed}.json"
    out.write_text(json.dumps([h.model_dump() for h in history], indent=2))
    print(f"\nBest: round {best.round} {[s.operator for s in best.spec.steps]} "
          f"score={best.score:.3f}. Full search tree saved to {out}")


if __name__ == "__main__":
    main()
