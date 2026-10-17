"""Metrics. Kept dependency-free so they are easy to unit-test."""
from __future__ import annotations

from collections import defaultdict


def accuracy(preds: list, targets: list) -> float:
    if not targets:
        return 0.0
    return sum(p == t for p, t in zip(preds, targets)) / len(targets)


def macro_f1(preds: list, targets: list) -> float:
    """Macro-averaged F1 over the classes that appear in the targets.
    A missing prediction (None) counts as wrong."""
    classes = sorted(set(targets), key=str)
    if not classes:
        return 0.0
    f1s = []
    for c in classes:
        tp = sum(p == c and t == c for p, t in zip(preds, targets))
        fp = sum(p == c and t != c for p, t in zip(preds, targets))
        fn = sum(p != c and t == c for p, t in zip(preds, targets))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s)


def binary_prf(preds: list, targets: list, positive=1) -> dict:
    """Precision / recall / F1 for the 'anomalous' class. None counts as negative."""
    tp = sum(p == positive and t == positive for p, t in zip(preds, targets))
    fp = sum(p == positive and t != positive for p, t in zip(preds, targets))
    fn = sum(p != positive and t == positive for p, t in zip(preds, targets))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def recall_by_type(records: list[dict]) -> dict:
    """Anomaly recall per perturbation type (skip/swap/insert/rework): failure analysis."""
    hit, tot = defaultdict(int), defaultdict(int)
    for r in records:
        kind = r.get("anomaly_type")
        if kind:
            tot[kind] += 1
            hit[kind] += int(r["prediction"] == 1)
    return {k: round(hit[k] / tot[k], 3) for k in sorted(tot)}


def cost_summary(records: list[dict]) -> dict:
    n = max(len(records), 1)
    calls = sum(r.get("llm_calls", 0) for r in records)
    tokens = sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in records)
    latency = sum(r.get("latency_s", 0.0) for r in records)
    invalid = sum(r.get("prediction") is None for r in records)
    return {"llm_calls_per_task": round(calls / n, 3),
            "tokens_per_task": round(tokens / n, 1),
            "latency_s_per_task": round(latency / n, 3),
            "invalid_output_rate": round(invalid / n, 3)}
