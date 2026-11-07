"""Run one method over a task set and score it (owner: Aimen).

Used for the baselines, for every candidate workflow during the search
(validation tasks) and for the final test evaluation, so all methods are
scored by exactly the same code.
"""
from __future__ import annotations

import time

from src.eval import metrics

try:   # G3's cost-aware objective
    from src.agents.objective import objective, task_score
except ImportError:   # before G3 is pushed: plain accuracy, cost ignored
    def task_score(na, an):
        parts = [x for x in (na, an) if x is not None]
        return sum(parts) / len(parts) if parts else 0.0

    def objective(na, an, tokens_per_task, cost_weight, token_scale=1000.0):
        return task_score(na, an) - cost_weight * tokens_per_task / token_scale


def _call(method, fn: str, arg, name: str):
    if name == "process_mining":
        if fn == "predict_next":
            return method.predict_next(arg), {"explanation": "most frequent successor"}
        flag, why = method.check_trace(arg)
        return flag, {"explanation": why}
    return getattr(method, fn)(arg)


def run_tasks(method, name: str, na_tasks: list[dict], an_tasks: list[dict],
              progress: str | None = None, keep_steps: bool = True) -> list[dict]:
    rows = []
    jobs = [("next_activity", t, "predict_next", t["prefix"], t["target"]) for t in na_tasks] + \
           [("anomaly", t, "check_trace", t["trace"], t["label"]) for t in an_tasks]
    t0 = time.time()
    for i, (task, t, fn, arg, target) in enumerate(jobs, 1):
        pred, meta = _call(method, fn, arg, name)
        row = {
            "task_id": t["task_id"], "task": task, "method": name,
            "input": arg, "prediction": pred, "target": target, "correct": pred == target,
            "anomaly_type": t.get("anomaly_type"),
            "llm_calls": meta.get("llm_calls", 0), "tool_calls": meta.get("tool_calls", 0),
            "prompt_tokens": meta.get("prompt_tokens", 0),
            "completion_tokens": meta.get("completion_tokens", 0),
            "latency_s": round(meta.get("latency_s", 0.0), 3),
            "cached": meta.get("cached", False),
            "format_retries": meta.get("format_retries", 0),
            "fallback": meta.get("fallback", False),
        }
        if meta.get("explanation"):
            row["explanation"] = meta["explanation"]
        if keep_steps:
            row["steps"] = meta.get("steps", [])
            if isinstance(meta.get("raw"), str):
                row["steps"] = [{"op": "generate", "type": "llm", "output": meta["raw"][:300]}]
        rows.append(row)
        if progress and (i % 25 == 0 or i == len(jobs)):
            print(f"    {progress}: {i}/{len(jobs)} tasks ({time.time() - t0:.0f}s)")
    return rows


def summarise(rows: list[dict], cost_weight: float = 0.0, token_scale: float = 1000.0) -> dict:
    na = [r for r in rows if r["task"] == "next_activity"]
    an = [r for r in rows if r["task"] == "anomaly"]
    out: dict = {}
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
    allc = metrics.cost_summary(rows)
    na_acc = out.get("next_activity", {}).get("accuracy")
    an_acc = out.get("anomaly", {}).get("accuracy")
    out["overall"] = {
        "task_score": round(task_score(na_acc, an_acc), 4),
        "objective": round(objective(na_acc, an_acc, allc["tokens_per_task"], cost_weight,
                                     token_scale), 4),
        "cost_weight": cost_weight,
        **allc,
        "tool_calls_per_task": round(sum(r.get("tool_calls", 0) for r in rows)
                                     / max(len(rows), 1), 3),
        "format_retry_rate": round(sum(r.get("format_retries", 0) > 0 for r in rows)
                                   / max(len(rows), 1), 4),
        "fallback_rate": round(sum(bool(r.get("fallback")) for r in rows)
                               / max(len(rows), 1), 4),
    }
    return out
