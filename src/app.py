"""Command-line interface: use a trained ProcFlow workflow on new cases (owner: Aimen).

    # what happens next in this running case?
    python -m src.app next "Declaration SUBMITTED by EMPLOYEE" "Declaration APPROVED by ADMINISTRATION"

    # is this finished case normal?
    python -m src.app check "Declaration SUBMITTED by EMPLOYEE" "Request Payment" "Payment Handled"

    # options
    --run e3_procflow_s1     which saved run's best workflow to use (default)
    --workflow seed          use the process-seeded workflow instead of a searched one
    --provider mock          no API key needed (answers are meaningless)

Prints the answer and the agent's trajectory (every tool and LLM step).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.agents.seeds import get_seed
from src.agents.state import WorkflowSpec
from src.agents.tools import ProcessModelTool
from src.agents.workflow import AgentWorkflow
from src.config import load_config
from src.data.preprocess import to_traces
from src.llm.client import make_llm


def load_workflow(cfg: dict, run: str, workflow: str | None) -> WorkflowSpec:
    if workflow:
        return get_seed(workflow)
    p = Path(cfg["eval"]["results_dir"]) / "runs" / f"{run}.json"
    if not p.exists():
        print(f"(no saved run {p}; using the process-seeded workflow)")
        return get_seed("process")
    return WorkflowSpec(**json.loads(p.read_text(encoding="utf-8"))["workflow"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("mode", choices=["next", "check"])
    ap.add_argument("activities", nargs="+", help="activity names in order")
    ap.add_argument("--run", default="e3_procflow_s1")
    ap.add_argument("--workflow", choices=["blank", "process"])
    ap.add_argument("--provider")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d = Path(cfg["data"]["processed_dir"])
    vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
    unknown = [a for a in args.activities if a not in vocab]
    if unknown:
        raise SystemExit(f"Unknown activity name(s): {unknown}\nKnown: {vocab}")
    tool = ProcessModelTool.from_traces(list(to_traces(pd.read_csv(d / "train.csv")).values()),
                                        vocab)
    spec = load_workflow(cfg, args.run, args.workflow)
    agent = AgentWorkflow(spec, make_llm(cfg, args.provider), vocab, tool.pm)
    desc = spec.describe() if hasattr(spec, "describe") else spec.signature()
    print(f"Workflow: {desc}\n")

    if args.mode == "next":
        ans, meta = agent.predict_next(args.activities)
        print(f"Predicted next activity: {ans}")
    else:
        ans, meta = agent.check_trace(args.activities)
        print(f"Verdict: {'ANOMALOUS' if ans == 1 else 'NORMAL' if ans == 0 else 'no answer'}")

    print("\nTrajectory:")
    for i, s in enumerate(meta["steps"], 1):
        if isinstance(s, dict):
            out = json.dumps(s.get("output")) if isinstance(s.get("output"), dict) else s.get("output")
            print(f"  {i}. {s.get('op')} ({s.get('type')}): {str(out)[:300]}")
        else:
            print(f"  {i}. {s}")
    print(f"\n{meta['llm_calls']} LLM call(s), "
          f"{meta['prompt_tokens'] + meta['completion_tokens']} tokens, "
          f"{meta['tool_calls']} tool call(s)")


if __name__ == "__main__":
    main()
