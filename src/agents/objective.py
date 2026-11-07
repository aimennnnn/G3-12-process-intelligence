"""Cost-aware optimization objective and efficiency metrics (G3, Phase 2).

Implements the cost-aware objective defined in Section VIII-E of the proposal:
    Objective Score = Predictive Performance - λ * Normalized Token Cost

Where:
    - Predictive Performance is the unweighted average of task accuracies:
      task_score(na_accuracy, anomaly_accuracy).
    - Normalized Token Cost = tokens_per_task / token_scale (default scale: 1000 tokens).
    - λ (lambda / cost_weight) is the penalty factor (0.0 = accuracy-only, 0.05, 0.1, 0.2).
    - When λ = 0, the objective matches accuracy-only workflow search (AFlow baseline).
    - When λ > 0, the optimizer penalizes expensive operator combinations (such as
      redundant self-consistency or oversized context retrieval), guiding the search
      toward the Pareto-optimal frontier of high accuracy and low token cost.
"""
from __future__ import annotations

from typing import Any, Sequence


def task_score(na_accuracy: float | None, anomaly_accuracy: float | None) -> float:
    """Compute overall predictive performance as the mean of valid task accuracies.

    Parameters
    ----------
    na_accuracy : float | None
        Next-activity top-1 accuracy (or None if task omitted).
    anomaly_accuracy : float | None
        Anomaly detection accuracy or F1 (or None if task omitted).

    Returns
    -------
    float
        Average score in [0.0, 1.0], or 0.0 if both are None.
    """
    valid = [x for x in (na_accuracy, anomaly_accuracy) if x is not None]
    if not valid:
        return 0.0
    return float(sum(valid) / len(valid))


def normalize_token_cost(tokens_per_task: float, token_scale: float = 1000.0) -> float:
    """Scale measured token consumption to a normalized reference unit.

    Parameters
    ----------
    tokens_per_task : float
        Average tokens consumed per task execution.
    token_scale : float, default=1000.0
        Reference token scale factor (1000 tokens = 1.0 unit).

    Returns
    -------
    float
        Normalized token consumption.
    """
    if token_scale <= 0.0:
        raise ValueError(f"token_scale must be positive, got {token_scale}")
    return max(0.0, float(tokens_per_task)) / float(token_scale)


def cost_penalty(tokens_per_task: float, cost_weight: float = 0.0,
                 token_scale: float = 1000.0) -> float:
    """Compute the penalty deducted from task accuracy."""
    if cost_weight <= 0.0:
        return 0.0
    norm_tokens = normalize_token_cost(tokens_per_task, token_scale)
    return float(cost_weight * norm_tokens)


def objective(na_accuracy: float | None, anomaly_accuracy: float | None,
              tokens_per_task: float, cost_weight: float = 0.0,
              token_scale: float = 1000.0) -> float:
    """Compute the cost-aware objective score for workflow selection.

    Parameters
    ----------
    na_accuracy : float | None
        Validation accuracy on next-activity prediction.
    anomaly_accuracy : float | None
        Validation accuracy or F1 on anomaly detection.
    tokens_per_task : float
        Average tokens consumed per task by the candidate workflow.
    cost_weight : float, default=0.0
        Weight λ controlling cost penalty. 0.0 disables cost penalty (AFlow baseline).
    token_scale : float, default=1000.0
        Normalizing scale for token counts.

    Returns
    -------
    float
        Objective score: task_score - (cost_weight * tokens_per_task / token_scale).
    """
    base_acc = task_score(na_accuracy, anomaly_accuracy)
    if cost_weight <= 0.0:
        return float(base_acc)
    penalty = cost_penalty(tokens_per_task, cost_weight, token_scale)
    return float(base_acc - penalty)


def pareto_frontier(candidates: Sequence[dict[str, Any]],
                    score_key: str = "task_score",
                    cost_key: str = "tokens_per_task") -> list[dict[str, Any]]:
    """Identify Pareto-optimal candidates (maximizing score while minimizing cost).

    A candidate A dominates candidate B if:
        score(A) >= score(B) AND cost(A) <= cost(B),
        with at least one strict inequality.

    Returns
    -------
    list[dict[str, Any]]
        Subset of non-dominated candidates, sorted by cost ascending.
    """
    valid = [c for c in candidates if score_key in c and cost_key in c]
    if not valid:
        return []

    frontier = []
    for cand in valid:
        c_score = float(cand[score_key])
        c_cost = float(cand[cost_key])
        dominated = False
        for other in valid:
            if other is cand:
                continue
            o_score = float(other[score_key])
            o_cost = float(other[cost_key])
            if (o_score >= c_score and o_cost <= c_cost) and (o_score > c_score or o_cost < c_cost):
                dominated = True
                break
        if not dominated and cand not in frontier:
            frontier.append(cand)

    frontier.sort(key=lambda x: float(x[cost_key]))
    return frontier


def efficiency_gain(baseline_tokens: float, target_tokens: float,
                    baseline_score: float, target_score: float) -> dict[str, float]:
    """Calculate token savings percentage and score retention delta."""
    b_tok = max(1e-9, float(baseline_tokens))
    savings_pct = max(-100.0, ((b_tok - float(target_tokens)) / b_tok) * 100.0)
    score_delta = float(target_score) - float(baseline_score)
    return {
        "token_reduction_pct": round(savings_pct, 2),
        "score_delta": round(score_delta, 4),
        "tokens_saved_per_task": round(float(baseline_tokens) - float(target_tokens), 2),
    }

