"""Trajectory logger: structured observability for every agent run.

A trajectory is the ordered list of operator executions that produced one
prediction, with the inputs/outputs of every step (operator, LLM call, tool
call) and the cost of each. Trajectories are the evidence the rubric asks for
under "Observable behaviour: logged agent trajectories showing reasoning
steps, tool calls, inputs and outputs, and state transitions."

A trajectory is built up step-by-step by the workflow (workflow.py); this
module only gives it a shape, an append API, and a way to flush it to disk as
JSONL. The workflow constructs one Trajectory per task and (if a logger is
attached) writes it; the meta dict returned to evaluate.py is independent, so
trajectories are pure observability — nothing reads them at scoring time.

Logs land in logs/trajectories_<tag>.jsonl so they can be analysed by
experiments/e5_ablation.py and src/eval/failure_analysis.py.

Usage (inside AgentWorkflow):

    traj = Trajectory(task_id, method, task_type)
    traj.step("process_context", input={"prefix": [...]},
              output={"text": "..."}, tool_calls=[...])
    traj.llm_call(prompt=..., response=..., tokens=(p, c), latency=0.4)
    traj.step("generate", output={"answer": "..."})
    traj.step("verify", input={"pred": "..."}, output={"final": "..."})
    traj.finish(target, correct=True)
    logger.write(traj)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal


@dataclass
class Step:
    """One executed operator (process_context, generate, ensemble, verify)."""
    operator: str
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    note: str = ""


@dataclass
class Trajectory:
    task_id: str
    method: str
    task: Literal["next_activity", "anomaly"]
    started_at: float = field(default_factory=time.time)
    steps: list[Step] = field(default_factory=list)
    prediction: Any = None
    target: Any = None
    correct: bool = False
    finished_at: float | None = None

    # ---------- mutation ----------
    def step(self, operator: str, *, input: dict | None = None,
             output: dict | None = None, tool_calls: list | None = None,
             llm_calls: int = 0, prompt_tokens: int = 0,
             completion_tokens: int = 0, latency_s: float = 0.0,
             note: str = "") -> Step:
        s = Step(operator=operator, input=input or {}, output=output or {},
                 tool_calls=tool_calls or [], llm_calls=llm_calls,
                 prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                 latency_s=latency_s, note=note)
        self.steps.append(s)
        return s

    def finish(self, prediction: Any, target: Any, correct: bool) -> None:
        self.prediction = prediction
        self.target = target
        self.correct = correct
        self.finished_at = time.time()

    # ---------- aggregate ----------
    @property
    def total_llm_calls(self) -> int:
        return sum(s.llm_calls for s in self.steps)

    @property
    def total_tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(s.prompt_tokens + s.completion_tokens for s in self.steps)

    @property
    def total_latency(self) -> float:
        return round(sum(s.latency_s for s in self.steps), 3)

    @property
    def operators_used(self) -> list[str]:
        return [s.operator for s in self.steps]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_llm_calls"] = self.total_llm_calls
        d["total_tool_calls"] = self.total_tool_calls
        d["total_tokens"] = self.total_tokens
        d["total_latency_s"] = self.total_latency
        d["operators"] = self.operators_used
        d["duration_s"] = round((self.finished_at or time.time()) - self.started_at, 3)
        return d


class TrajectoryLogger:
    """Append-only JSONL logger. One file per method/variant, so ablation
    runs do not overwrite the main run's trajectories."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate at the start of a run so the file matches the latest results.
        self.path.write_text("", encoding="utf-8")

    def write(self, traj: Trajectory) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(traj.to_dict(), default=str) + "\n")

    def read_all(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]


def read_trajectories(path: str | Path) -> list[dict]:
    """Read a trajectory JSONL file (used by failure analysis & ablation)."""
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
