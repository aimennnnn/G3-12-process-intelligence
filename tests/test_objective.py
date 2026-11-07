"""Tests for G3 cost-aware objective and Pareto frontier."""
import pytest
from src.agents.objective import (
    cost_penalty,
    efficiency_gain,
    normalize_token_cost,
    objective,
    pareto_frontier,
    task_score,
)


def test_task_score_both_present():
    assert task_score(0.8, 0.6) == pytest.approx(0.7)


def test_task_score_one_missing():
    assert task_score(0.85, None) == pytest.approx(0.85)
    assert task_score(None, 0.75) == pytest.approx(0.75)


def test_task_score_both_none():
    assert task_score(None, None) == 0.0


def test_normalize_token_cost():
    assert normalize_token_cost(1000.0, 1000.0) == pytest.approx(1.0)
    assert normalize_token_cost(500.0, 1000.0) == pytest.approx(0.5)
    assert normalize_token_cost(0.0) == 0.0


def test_cost_penalty():
    # with cost_weight 0, penalty is always 0
    assert cost_penalty(1500, cost_weight=0.0) == 0.0
    # with cost_weight 0.1 and 1000 tokens / scale 1000 => 0.1 * 1.0 = 0.1
    assert cost_penalty(1000, cost_weight=0.1, token_scale=1000.0) == pytest.approx(0.1)


def test_objective_zero_cost_weight():
    # λ=0 should match task_score exactly
    score = objective(0.9, 0.7, tokens_per_task=5000, cost_weight=0.0)
    assert score == pytest.approx(0.8)


def test_objective_with_cost_penalty():
    # task_score = (0.9 + 0.7)/2 = 0.8
    # cost penalty: 0.1 * (1000 / 1000) = 0.1
    # expected objective = 0.8 - 0.1 = 0.7
    score = objective(0.9, 0.7, tokens_per_task=1000, cost_weight=0.1, token_scale=1000.0)
    assert score == pytest.approx(0.7)


def test_pareto_frontier():
    candidates = [
        {"name": "heavy_accurate", "task_score": 0.95, "tokens_per_task": 1200},
        {"name": "light_good", "task_score": 0.90, "tokens_per_task": 400},
        {"name": "dominated", "task_score": 0.85, "tokens_per_task": 600},  # dominated by light_good
        {"name": "very_cheap", "task_score": 0.80, "tokens_per_task": 150},
    ]
    frontier = pareto_frontier(candidates)
    names = [c["name"] for c in frontier]
    assert "heavy_accurate" in names
    assert "light_good" in names
    assert "very_cheap" in names
    assert "dominated" not in names


def test_efficiency_gain():
    gain = efficiency_gain(baseline_tokens=1000, target_tokens=700,
                           baseline_score=0.80, target_score=0.82)
    assert gain["token_reduction_pct"] == pytest.approx(30.0)
    assert gain["score_delta"] == pytest.approx(0.02)
    assert gain["tokens_saved_per_task"] == pytest.approx(300.0)

