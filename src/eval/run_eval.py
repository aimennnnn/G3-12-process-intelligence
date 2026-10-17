"""Run every method on both tasks and write results.

    python -m src.eval.run_eval                       # methods from config.yaml
    python -m src.eval.run_eval --limit 20            # quick check
    python -m src.eval.run_eval --provider mock       # offline, no API key
    python -m src.eval.run_eval --methods process_mining fixed_llm agent_seeded

Outputs (results/):
  predictions_<method>.jsonl   one record per task: answer, correctness, cost, trace
  summary.json / summary.md    the results table for the proposal
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.agents.state import BLANK_SEED, PROCESS_SEED, Prediction
from src.agents.workflow import AgentWorkflow
from src.baselines.fixed_llm import FixedLLMWorkflow
from src.baselines.process_mining import ProcessMiningBaseline
from src.config import load_config
from src.data.preprocess import to_traces
from src.eval import metrics

import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_processed(cfg: dict):
    d = Path(cfg["data"]["processed_dir"])
    if not (d / "train.csv").exists():
        raise SystemExit("No processed data. Run: python -m src.data.prepare  (or --sample)")
    train = pd.read_csv(d / "train.csv")
    vocab = json.loads((d / "vocab.json").read_text())
    return to_traces(train), vocab, read_jsonl(d / "next_activity.jsonl"), read_jsonl(d / "anomaly.jsonl")


def build_method(name: str, cfg: dict, pm: ProcessMiningBaseline, vocab, provider):
    if name == "process_mining":
        return pm
    from src.llm.client import make_llm  # only needed for LLM methods
    llm = make_llm(cfg, provider)
    if name == "fixed_llm":
        return FixedLLMWorkflow(llm, vocab)
    if name == "agent_blank":
        return AgentWorkflow(BLANK_SEED, llm, vocab, pm)
    if name == "agent_seeded":
        return AgentWorkflow(PROCESS_SEED, llm, vocab, pm)
    raise SystemExit(f"Unknown method '{name}'")


def _call(method, fn: str, arg):
    out = getattr(method, fn)(arg)
    if isinstance(out[1], str):          # process-mining baseline: (answer, explanation)
        return out[0], {"explanation": out[1]}
    return out                            # LLM methods: (answer, meta dict)


def _explain(meta: dict) -> str:
    """Human-readable trace of how the answer was produced."""
    if meta.get("explanation"):
        return meta["explanation"]
    parts = []
    raw = meta.get("raw")
    if raw:
        parts.append("llm: " + (" / ".join(raw) if isinstance(raw, list) else str(raw)))
    parts += meta.get("steps", [])
    return " | ".join(parts)


def evaluate(method, name: str, na_tasks, an_tasks) -> list[dict]:
    rows = []
    jobs = [("next_activity", t, "predict_next", t["prefix"], t["target"]) for t in na_tasks] + \
           [("anomaly", t, "check_trace", t["trace"], t["label"]) for t in an_tasks]
    t0 = time.time()
    for i, (task, t, fn, arg, target) in enumerate(jobs, 1):
        if fn == "predict_next" and name == "process_mining":
            pred, meta = method.predict_next(arg), {"explanation": "most frequent successor"}
        else:
            pred, meta = _call(method, fn, arg)
        expl = _explain(meta)
        rec = Prediction(
            task_id=t["task_id"], task=task, method=name, prediction=pred, target=target,
            correct=(pred == target), explanation=str(expl)[:500],
            llm_calls=meta.get("llm_calls", 0), prompt_tokens=meta.get("prompt_tokens", 0),
            completion_tokens=meta.get("completion_tokens", 0),
            latency_s=round(meta.get("latency_s", 0.0), 3), cached=meta.get("cached", False),
        ).model_dump()
        rec["anomaly_type"] = t.get("anomaly_type")
        rows.append(rec)
        if i % 25 == 0 or i == len(jobs):
            print(f"  {name}: {i}/{len(jobs)} tasks ({time.time() - t0:.0f}s)")
    return rows


def summarise(rows: list[dict]) -> dict:
    na = [r for r in rows if r["task"] == "next_activity"]
    an = [r for r in rows if r["task"] == "anomaly"]
    out = {}
    if na:
        p, t = [r["prediction"] for r in na], [r["target"] for r in na]
        out["next_activity"] = {"n": len(na), "accuracy": round(metrics.accuracy(p, t), 4),
                                "macro_f1": round(metrics.macro_f1(p, t), 4),
                                **metrics.cost_summary(na)}
    if an:
        p, t = [r["prediction"] for r in an], [r["target"] for r in an]
        prf = metrics.binary_prf(p, t)
        out["anomaly"] = {"n": len(an), "accuracy": round(metrics.accuracy(p, t), 4),
                          **{k: round(v, 4) for k, v in prf.items()},
                          "recall_by_type": metrics.recall_by_type(an),
                          **metrics.cost_summary(an)}
    return out


def to_markdown(summary: dict, meta: dict) -> str:
    lines = [f"# Baseline results ({meta['dataset']})", "",
             f"LLM: {meta['provider']} / {meta['model']} | seed {meta['seed']} | {meta['when']}", "",
             "## Next-activity prediction", "",
             "| Method | N | Accuracy | Macro-F1 | LLM calls/task | Tokens/task | Latency (s)/task | Invalid |",
             "|---|---|---|---|---|---|---|---|"]
    for m, s in summary.items():
        if "next_activity" in s:
            x = s["next_activity"]
            lines.append(f"| {m} | {x['n']} | {x['accuracy']:.3f} | {x['macro_f1']:.3f} | "
                         f"{x['llm_calls_per_task']} | {x['tokens_per_task']} | "
                         f"{x['latency_s_per_task']} | {x['invalid_output_rate']:.1%} |")
    lines += ["", "## Anomaly detection", "",
              "| Method | N | Accuracy | Precision | Recall | F1 | LLM calls/task | Tokens/task | Invalid |",
              "|---|---|---|---|---|---|---|---|---|"]
    for m, s in summary.items():
        if "anomaly" in s:
            x = s["anomaly"]
            lines.append(f"| {m} | {x['n']} | {x['accuracy']:.3f} | {x['precision']:.3f} | "
                         f"{x['recall']:.3f} | {x['f1']:.3f} | {x['llm_calls_per_task']} | "
                         f"{x['tokens_per_task']} | {x['invalid_output_rate']:.1%} |")
    lines += ["", "## Anomaly recall by perturbation type", "",
              "| Method | " + " | ".join(["skip", "swap", "insert", "rework"]) + " |",
              "|---|---|---|---|---|"]
    for m, s in summary.items():
        if "anomaly" in s:
            r = s["anomaly"]["recall_by_type"]
            lines.append(f"| {m} | " + " | ".join(str(r.get(k, "-")) for k in
                                                   ["skip", "swap", "insert", "rework"]) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--provider", help="override llm.provider (groq/gemini/ollama/mock)")
    ap.add_argument("--limit", type=int, help="only the first N tasks of each type")
    args = ap.parse_args()

    cfg = load_config(args.config)
    train_traces, vocab, na, an = load_processed(cfg)
    if args.limit:
        na, an = na[:args.limit], an[:args.limit]
    pm = ProcessMiningBaseline().fit(list(train_traces.values()))

    res = Path(cfg["eval"]["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    stats_path = res / "data_stats.json"
    dataset = json.loads(stats_path.read_text())["source"] if stats_path.exists() else "?"

    summary = {}
    for name in args.methods or cfg["eval"]["methods"]:
        print(f"Running {name} on {len(na)} + {len(an)} tasks ...")
        method = build_method(name, cfg, pm, vocab, args.provider)
        rows = evaluate(method, name, na, an)
        with open(res / f"predictions_{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        summary[name] = summarise(rows)

    meta = {"dataset": dataset, "provider": args.provider or cfg["llm"]["provider"],
            "model": cfg["llm"]["model"], "seed": cfg["seed"],
            "when": time.strftime("%Y-%m-%d %H:%M")}
    (res / "summary.json").write_text(json.dumps({"meta": meta, "results": summary}, indent=2))
    md = to_markdown(summary, meta)
    (res / "summary.md").write_text(md)
    print("\n" + md)


if __name__ == "__main__":
    main()
