"""Starting workflows for the search: the switch for Gap 2 (owner: Aimen).

AFlow starts every search from a blank workflow (one LLM call that knows
nothing about the domain), so the optimiser must discover any domain
knowledge by trial and error (the "cold start" problem).

ProcFlow can instead start from a workflow built from the mined process
model: the process-model tool supplies likely next steps / transition
frequencies, the LLM answers, and the answer is checked against the model.

    --seed blank     Gap 2 component OFF (AFlow behaviour, used in E2 and ablation)
    --seed process   Gap 2 component ON  (ProcFlow)

Both searches use the same operator library, so a blank search can still
discover the process tool on its own; the only difference is where it starts.
"""
from __future__ import annotations

from src.agents.state import Step, WorkflowSpec

BLANK_SEED = WorkflowSpec(name="blank", steps=[Step(operator="generate")])

PROCESS_SEED = WorkflowSpec(name="process_seeded", steps=[
    Step(operator="process_context", params={"top_k": 3}),
    Step(operator="generate"),
    Step(operator="verify", params={"mode": "revise"}),
])

SEEDS = {"blank": BLANK_SEED, "process": PROCESS_SEED}


def get_seed(name: str) -> WorkflowSpec:
    if name not in SEEDS:
        raise ValueError(f"Unknown seed '{name}'. Use one of {list(SEEDS)}.")
    return SEEDS[name].model_copy(deep=True)
