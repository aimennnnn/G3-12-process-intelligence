"""Structured contracts shared by every component.

Each prediction, workflow and optimisation step is a validated record, so
traces are consistent across methods and can be analysed later (observability).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Operator = Literal["process_context", "generate", "ensemble", "verify"]


class NextActivityTask(BaseModel):
    task_id: str
    case_id: str
    prefix: list[str]
    target: str


class AnomalyTask(BaseModel):
    task_id: str
    case_id: str
    trace: list[str]
    label: int = Field(ge=0, le=1)
    anomaly_type: Optional[str] = None


class Prediction(BaseModel):
    """One method's answer to one task, with its cost."""
    task_id: str
    task: Literal["next_activity", "anomaly"]
    method: str
    prediction: Optional[str | int]
    target: str | int
    correct: bool
    explanation: str = ""
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    cached: bool = False


class Step(BaseModel):
    operator: Operator
    params: dict = Field(default_factory=dict)


class WorkflowSpec(BaseModel):
    """A workflow is an ordered list of operator steps (AFlow-style, simplified)."""
    name: str
    steps: list[Step]

    def has(self, op: str) -> bool:
        return any(s.operator == op for s in self.steps)

    def param(self, op: str, key: str, default=None):
        for s in self.steps:
            if s.operator == op:
                return s.params.get(key, default)
        return default


class OptimisationRecord(BaseModel):
    """One node of the search tree: what changed, and did it help."""
    round: int
    parent: Optional[int]
    modification: str
    spec: WorkflowSpec
    accuracy: float
    tokens_per_task: float
    score: float
    improved: bool


# Starting points for the search (Gap 2: cold start vs process-mined seed).
BLANK_SEED = WorkflowSpec(name="blank", steps=[Step(operator="generate")])
PROCESS_SEED = WorkflowSpec(name="process_seeded", steps=[
    Step(operator="process_context", params={"top_k": 3}),
    Step(operator="generate"),
    Step(operator="verify"),
])
