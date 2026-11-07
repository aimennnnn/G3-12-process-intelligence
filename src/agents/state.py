"""Structured contracts shared by every component (G1 + core, Phase 2).

Each prediction, workflow and optimisation step is a validated record, so
traces are consistent across methods and can be analysed later (observability).

Phase 2 / G1 additions (all backward-compatible — every new field has a
default, so Phase 1 code that constructs these models keeps working):
  * WorkflowSpec.signature()       short comparable spec string, e.g.
                                   'process_context(top_k=3)+generate+verify(mode=revise)'
  * OptimisationRecord             now also carries per-task accuracy, the
                                   proposer's reasoning, the cost breakdown,
                                   and the list of child rounds (search tree)
  * CostRecord                     auditable per-evaluation cost breakdown
  * FailureCase                    one row of the failure-analysis table

The BLANK_SEED / PROCESS_SEED constants are kept here for backward
compatibility with Phase 1 imports; seeds.py is the canonical source.
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
    """A workflow is an ordered list of operator steps (AFlow-style)."""
    name: str
    steps: list[Step]

    def has(self, op: str) -> bool:
        return any(s.operator == op for s in self.steps)

    def param(self, op: str, key: str, default=None):
        for s in self.steps:
            if s.operator == op:
                return s.params.get(key, default)
        return default

    def signature(self) -> str:
        """Short comparable description, e.g. 'process_context(top_k=3)+generate+verify(mode=revise)'."""
        parts = []
        for s in self.steps:
            if s.params:
                parts.append(f"{s.operator}({','.join(f'{k}={v}' for k, v in s.params.items())})")
            else:
                parts.append(s.operator)
        return "+".join(parts)


class CostRecord(BaseModel):
    """Auditable cost breakdown for one workflow evaluation."""
    tokens_per_task: float = 0.0
    latency_s_per_task: float = 0.0
    llm_calls_per_task: float = 0.0
    tool_calls_per_task: float = 0.0
    cache_hit_rate: float = 0.0
    format_retry_rate: float = 0.0
    fallback_rate: float = 0.0


class OptimisationRecord(BaseModel):
    """One node of the AFlow search tree.

    Phase 1 fields (kept exactly): round, parent, modification, spec, accuracy,
    tokens_per_task, score, improved.
    Phase 2 fields (all optional, default to 0/empty): per-task accuracy,
    proposer reasoning, cost breakdown, children list, failure examples.
    """
    round: int
    parent: Optional[int]
    modification: str
    spec: WorkflowSpec
    accuracy: float                       # = task_score(na_acc, an_acc)
    tokens_per_task: float
    score: float                          # = objective(...)
    improved: bool
    # ---- Phase 2 additions (backward-compatible) ----
    proposer_reasoning: str = ""
    accuracy_next_activity: float = 0.0
    accuracy_anomaly: float = 0.0
    cost: CostRecord = Field(default_factory=CostRecord)
    children: list[int] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)   # full `summarise()` output, for the paper
    failure_examples: list[dict] = Field(default_factory=list)  # up to 6 wrong validation answers,
                                                                 # shown to the proposer next round


class FailureCase(BaseModel):
    """One failure case for the failure-analysis table."""
    task_id: str
    task: Literal["next_activity", "anomaly"]
    category: str
    prefix_or_trace: list[str]
    prediction: Optional[str | int]
    target: str | int
    anomaly_type: Optional[str] = None
    explanation: str = ""


# Starting points (kept here for backward compat with Phase 1 imports;
# seeds.py is the canonical source and adds the `mode=revise` param).
BLANK_SEED = WorkflowSpec(name="blank", steps=[Step(operator="generate")])
PROCESS_SEED = WorkflowSpec(name="process_seeded", steps=[
    Step(operator="process_context", params={"top_k": 3}),
    Step(operator="generate"),
    Step(operator="verify", params={"mode": "revise"}),
])
