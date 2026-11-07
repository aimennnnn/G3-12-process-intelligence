"""E5: Ablation study. Remove each gap-specific component one at a time and
measure its contribution on the TEST set.

The full ProcFlow system is the process-seeded workflow (Gap 1+2+3 ON).
The ablation removes one component at a time:

  full             process_context + generate + verify   (Gap 1, 2, 3 ON)
  no_context       remove process_context                 -> tests Gap 2 (process knowledge)
  no_verify        remove verify                          -> tests the verification operator
  no_ensemble      remove ensemble (if present)
  ensemble         add ensemble (n=3) self-consistency
  blank_seed       start from blank seed, no process tool -> tests Gap 2 cold-start
  cost_unaware     score with cost_weight=0 in the optimiser -> tests Gap 3 (cost-aware search)
  fixed_llm_only   no agent at all                        -> baseline floor

For each variant we report:
  * next-activity accuracy & macro-F1
  * anomaly precision / recall / F1
  * tokens/task, latency/task, LLM calls/task  (E6 metrics, captured here too)
  * the cost-aware score: objective(na, an, tokens, 0.05, 1000)

This script uses AgentWorkflow + run_tasks + summarise (the same scoring
code as the baselines and the optimiser), so the numbers are directly
comparable across E1, E3, E5.

    python -m experiments.e5_ablation
    python -m experiments.e5_ablation --provider groq --limit 100

Outputs (results/tables/):
  ablation.json     the full table
  ablation.md       markdown for the paper
Outputs (logs/):
  trajectories_e5_<variant>.jsonl   per-task trajectories for each variant
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from src.agents.state import Step, WorkflowSpec
from src.agents.trajectory import TrajectoryLogger
from src.agents.workflow import AgentWorkflow
from src.baselines.fixed_llm import FixedLLMWorkflow
from src.baselines.process_mining import ProcessMiningBaseline
from src.config import load_config
from src.data.preprocess import to_traces
from src.eval import metrics
from src.eval.evaluate import run_tasks, summarise
from src.llm.client import make_llm


# The variants under test. Each is a (label, spec_builder) pair so the
# ablation is self-documenting: you can see exactly what is removed.
def _full_spec() -> WorkflowSpec:
    return WorkflowSpec(name="full", steps=[
        Step(operator="process_context", params={"top_k": 3}),
        Step(operator="generate"),
        Step(operator="verify", params={"mode": "revise"}),
    ])


def _no_context_spec() -> WorkflowSpec:
    return WorkflowSpec(name="no_context", steps=[
        Step(operator="generate"),
        Step(operator="verify", params={"mode": "revise"}),
    ])


def _no_verify_spec() -> WorkflowSpec:
    return WorkflowSpec(name="no_verify", steps=[
        Step(operator="process_context", params={"top_k": 3}),
        Step(operator="generate"),
    ])


def _ensemble_spec() -> WorkflowSpec:
    return WorkflowSpec(name="ensemble", steps=[
        Step(operator="process_context", params={"top_k": 3}),
        Step(operator="ensemble", params={"n": 3}),
        Step(operator="generate"),
        Step(operator="verify", params={"mode": "revise"}),
    ])


def _blank_seed_spec() -> WorkflowSpec:
    return WorkflowSpec(name="blank_seed", steps=[Step(operator="generate")])


VARIANTS = {
    "full":          ("Full ProcFlow (G1+G2+G3)", _full_spec),
    "no_context":    ("Gap 2 OFF: no process_context", _no_context_spec),
    "no_verify":     ("verify operator removed", _no_verify_spec),
    "ensemble":      ("+ ensemble (n=3) self-consistency", _ensemble_spec),
    "blank_seed":    ("Gap 2 OFF: blank seed only", _blank_seed_spec),
}


def evaluate_variant(name: str, spec: WorkflowSpec, llm, vocab, pm,
                     na_tasks, an_tasks, logs: Path, cost_weight: float = 0.05,
                     token_scale: float = 1000.0) -> dict:
    """Run one variant on the test set and return a result dict."""
    traj_logger = TrajectoryLogger(logs / f"trajectories_e5_{name}.jsonl")
    agent = AgentWorkflow(spec, llm, vocab, pm, traj_logger=traj_logger)
    rows = run_tasks(agent, name, na_tasks, an_tasks, progress=name, keep_steps=True)
    summary = summarise(rows, cost_weight=cost_weight, token_scale=token_scale)
    return {
        "spec": spec.signature(),
        "next_activity": summary.get("next_activity", {}),
        "anomaly": summary.get("anomaly", {}),
        "overall": summary.get("overall", {}),
        "cost_aware_score": summary.get("overall", {}).get("objective"),
    }


def run_fixed_llm(llm, vocab, na_tasks, an_tasks, cost_weight: float = 0.05,
                  token_scale: float = 1000.0) -> dict:
    """The non-agentic baseline: one LLM call per task, no operators."""
    f = FixedLLMWorkflow(llm, vocab)
    rows = run_tasks(f, "fixed_llm", na_tasks, an_tasks, progress="fixed_llm", keep_steps=True)
    summary = summarise(rows, cost_weight=cost_weight, token_scale=token_scale)
    return {
        "spec": "fixed_llm (one LLM call, no agent)",
        "next_activity": summary.get("next_activity", {}),
        "anomaly": summary.get("anomaly", {}),
        "overall": summary.get("overall", {}),
        "cost_aware_score": summary.get("overall", {}).get("objective"),
    }


def to_markdown(results: dict, meta: dict) -> str:
    lines = [f"# E5 Ablation study ({meta['dataset']})", "",
             f"Provider: {meta['provider']} / {meta['model']} | seed {meta['seed']} | {meta['when']}", "",
             "## Next-activity prediction", "",
             "| Variant | Spec | Accuracy | Macro-F1 | Tokens/task | Score |",
             "|---|---|---|---|---|---|"]
    for name, r in results.items():
        if r is None or "next_activity" not in r or not r["next_activity"]:
            continue
        x = r["next_activity"]; c = r["overall"]
        lines.append(f"| {name} | `{r['spec']}` | {x['accuracy']:.3f} | {x['macro_f1']:.3f} | "
                     f"{c['tokens_per_task']:.0f} | {r['cost_aware_score']:.3f} |")
    lines += ["", "## Anomaly detection", "",
              "| Variant | Spec | Accuracy | Precision | Recall | F1 | Tokens/task | Score |",
              "|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        if r is None or "anomaly" not in r or not r["anomaly"]:
            continue
        x = r["anomaly"]; c = r["overall"]
        lines.append(f"| {name} | `{r['spec']}` | {x['accuracy']:.3f} | {x['precision']:.3f} | "
                     f"{x['recall']:.3f} | {x['f1']:.3f} | {c['tokens_per_task']:.0f} | "
                     f"{r['cost_aware_score']:.3f} |")
    lines += ["", "## Variants", ""]
    for name, (desc, _) in VARIANTS.items():
        lines.append(f"- **{name}**: {desc}")
    lines.append("- **fixed_llm**: non-agentic baseline (no agent, no operators)")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--provider")
    ap.add_argument("--limit", type=int, default=60)
    args = ap.parse_args()
    cfg = load_config(args.config)

    d = Path(cfg["data"]["processed_dir"])
    train = pd.read_csv(d / "train.csv")
    vocab = json.loads((d / "vocab.json").read_text())
    na = [json.loads(l) for l in (d / "next_activity.jsonl").read_text().splitlines() if l.strip()]
    an = [json.loads(l) for l in (d / "anomaly.jsonl").read_text().splitlines() if l.strip()]
    if args.limit:
        na, an = na[:args.limit], an[:args.limit]
    pm = ProcessMiningBaseline().fit(list(to_traces(train).values()))
    llm = make_llm(cfg, args.provider)
    logs = Path(cfg["eval"]["logs_dir"])
    res = Path(cfg["eval"]["results_dir"]) / "tables"
    res.mkdir(parents=True, exist_ok=True)

    cw = cfg.get("ablation", {}).get("cost_weight", 0.05)
    ts = cfg.get("ablation", {}).get("token_scale", 1000.0)

    results = {}
    for name, (desc, builder) in VARIANTS.items():
        spec = builder()
        print(f"\nE5: {name} ({desc}) ...")
        t0 = time.time()
        results[name] = evaluate_variant(name, spec, llm, vocab, pm, na, an, logs, cw, ts)
        na_acc = results[name]["next_activity"].get("accuracy", 0)
        print(f"   done in {time.time() - t0:.0f}s: na_acc={na_acc:.3f} "
              f"score={results[name]['cost_aware_score']:.3f}")
    print("\nE5: fixed_llm ...")
    results["fixed_llm"] = run_fixed_llm(llm, vocab, na, an, cw, ts)

    stats_path = Path(cfg["eval"]["results_dir"]) / "data_stats.json"
    dataset = json.loads(stats_path.read_text())["source"] if stats_path.exists() else "?"
    meta = {"dataset": dataset, "provider": args.provider or cfg["llm"]["provider"],
            "model": cfg["llm"]["model"], "seed": cfg["seed"],
            "when": time.strftime("%Y-%m-%d %H:%M"), "n_na": len(na), "n_an": len(an)}
    out = {"meta": meta, "results": results}
    (res / "ablation.json").write_text(json.dumps(out, indent=2, default=str))
    md = to_markdown(results, meta)
    (res / "ablation.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
