"""Failure analysis: cluster failures into categories, count them, and pull
representative trajectories (Section 10 of the Phase 2 paper, rubric B4).

Failure categories (designed from looking at real failure modes on
next-activity and anomaly tasks; see docs/devlog.md):

  next_activity
    wrong_at_branch      the LLM picked the wrong option at a known DFG fork
                         (the right answer was also seen after the same prefix)
    unseen_transition    the LLM picked an activity the DFG says never follows
                         (the verify operator should catch this; if it didn't,
                          verify was off or the model check failed too)
    length_bias          the LLM picked the most frequent activity globally
                         (a length-1 prefix problem: too little context)
    invalid_output       the LLM returned an unparseable answer

  anomaly
    missed_skip          a skip perturbation was labelled NORMAL
    missed_swap          a swap perturbation was labelled NORMAL
    missed_insert        an insert perturbation was labelled NORMAL
    missed_rework        a rework perturbation was labelled NORMAL
    false_alarm          a normal trace was labelled ANOMALOUS

Each category is computed from the prediction records in
results/predictions_<method>.jsonl (or wherever run_tasks wrote them) plus the
trajectory file logs/trajectories_<method>.jsonl (for the operator trace).
For each category we keep up to 2 representative trajectories.

This file is independent of G1's optimizer — it only reads the prediction
rows that run_tasks (evaluate.py) writes, so it works for any method,
including the baselines and the ablation variants.

    python -m src.eval.failure_analysis --method agent_seeded
    python -m src.eval.failure_analysis --all
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from src.agents.trajectory import read_trajectories
from src.baselines.process_mining import START, ProcessMiningBaseline
from src.config import load_config
from src.data.preprocess import to_traces


def _categorise_next_activity(rec: dict, pm: ProcessMiningBaseline) -> str:
    pred, target = rec.get("prediction"), rec.get("target")
    prefix = rec.get("input") or []
    if pred is None:
        return "invalid_output"
    if pred == target:
        return ""  # not a failure
    last = prefix[-1] if prefix else START
    seen_after = set(a for a, _ in pm.dfg.get(last, Counter()).most_common() if a != "<END>")
    if pred not in seen_after:
        return "unseen_transition"
    if target in seen_after:
        return "wrong_at_branch"
    if len(prefix) <= 1:
        return "length_bias"
    return "wrong_at_branch"


def _categorise_anomaly(rec: dict) -> str:
    pred, target, kind = rec.get("prediction"), rec.get("target"), rec.get("anomaly_type")
    if pred == target:
        return ""
    if target == 1:  # we missed a real anomaly
        return f"missed_{kind}" if kind else "missed_anomaly"
    return "false_alarm"


def categorise(rec: dict, pm: ProcessMiningBaseline) -> str:
    if rec.get("task") == "next_activity":
        return _categorise_next_activity(rec, pm)
    return _categorise_anomaly(rec)


def analyse_method(method: str, cfg: dict) -> dict:
    """Read predictions_<method>.jsonl and trajectories_<method>.jsonl, return
    a failure-analysis report."""
    res = Path(cfg["eval"]["results_dir"])
    logs = Path(cfg["eval"]["logs_dir"])
    pred_path = res / f"predictions_{method}.jsonl"
    traj_path = logs / f"trajectories_{method}.jsonl"
    if not pred_path.exists():
        return {"method": method, "error": f"missing {pred_path}"}

    # Reload the train.csv so we can build the DFG (for next-activity categorisation).
    d = Path(cfg["data"]["processed_dir"])
    train = pd.read_csv(d / "train.csv")
    pm = ProcessMiningBaseline().fit(list(to_traces(train).values()))

    preds = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    trajs = {t["task_id"]: t for t in read_trajectories(traj_path)} if traj_path.exists() else {}

    n_total = len(preds)
    n_fail = sum(1 for p in preds if not p.get("correct", False))
    cats = Counter()
    reps: dict[str, list[dict]] = defaultdict(list)
    for p in preds:
        if p.get("correct", False):
            continue
        cat = categorise(p, pm)
        if not cat:
            continue
        cats[cat] += 1
        if len(reps[cat]) < 2:
            reps[cat].append({
                "task_id": p["task_id"], "task": p["task"],
                "prefix": p.get("input") if p["task"] == "next_activity" else None,
                "trace": p.get("input") if p["task"] == "anomaly" else None,
                "prediction": p["prediction"], "target": p["target"],
                "anomaly_type": p.get("anomaly_type"),
                "explanation": p.get("explanation", ""),
                "steps": p.get("steps"),
                "trajectory": trajs.get(p["task_id"]),
            })

    # Two representative SUCCESS cases for the paper.
    successes = []
    for p in preds:
        if p.get("correct", False) and len(successes) < 2:
            successes.append({
                "task_id": p["task_id"], "task": p["task"],
                "prefix": p.get("input") if p["task"] == "next_activity" else None,
                "trace": p.get("input") if p["task"] == "anomaly" else None,
                "prediction": p["prediction"], "target": p["target"],
                "anomaly_type": p.get("anomaly_type"),
                "trajectory": trajs.get(p["task_id"]),
            })

    return {
        "method": method, "n_total": n_total, "n_failures": n_fail,
        "failure_rate": round(n_fail / max(n_total, 1), 4),
        "categories": dict(cats),
        "category_rates": {k: round(v / max(n_total, 1), 4) for k, v in cats.items()},
        "representative_failures": {k: v for k, v in reps.items()},
        "representative_successes": successes,
    }


def to_markdown(report: dict) -> str:
    if "error" in report:
        return f"## {report['method']}\n\n_{report['error']}_\n"
    lines = [f"## Failure analysis: {report['method']}", "",
             f"Total tasks: {report['n_total']} | Failures: {report['n_failures']} "
             f"({report['failure_rate']:.1%})", "",
             "| Category | Count | Rate |", "|---|---|---|"]
    for cat, n in sorted(report["categories"].items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {n} | {report['category_rates'][cat]:.1%} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--method")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    methods = cfg["eval"]["methods"] if args.all else [args.method]
    res = Path(cfg["eval"]["results_dir"])
    out_path = res / "failure_analysis.json"
    out = {}
    md = []
    for m in methods:
        if not m:
            continue
        r = analyse_method(m, cfg)
        out[m] = r
        md.append(to_markdown(r))
    out_path.write_text(json.dumps(out, indent=2, default=str))
    (res / "failure_analysis.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
