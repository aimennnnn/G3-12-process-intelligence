"""Tests for the G1 optimiser and agent workflow (no network, stub LLMs)."""
import random

from src.agents.optimizer import ACTIONS, _next_new_spec, apply_action
from src.agents.seeds import get_seed
from src.agents.state import OptimisationRecord
from src.agents.workflow import AgentWorkflow
from src.baselines.process_mining import ProcessMiningBaseline
from src.llm.client import LLMResponse

TRAIN = [["submit", "approve", "pay"]] * 8 + [["submit", "reject", "submit", "approve", "pay"]] * 2
VOCAB = ["approve", "pay", "reject", "submit"]


class StubLLM:
    """Always gives the same reply, and counts calls."""
    def __init__(self, text):
        self.text, self.calls = text, 0

    def chat(self, messages, *args, **kwargs):
        self.calls += 1
        return LLMResponse(self.text, 10, 1)


def node(spec):
    return OptimisationRecord(round=0, parent=None, modification="seed", spec=spec,
                              accuracy=0.5, tokens_per_task=100, score=0.5, improved=True)


def test_no_op_action_keeps_signature():
    blank = get_seed("blank")
    assert apply_action(blank, "remove verify", "x").signature() == blank.signature()


def test_no_op_proposals_are_skipped():
    parent = node(get_seed("blank"))
    proposer = StubLLM(str(ACTIONS.index("remove verify") + 1))   # always a no-op here
    spec, action, why = _next_new_spec(proposer, parent, [parent], {parent.spec.signature()},
                                       random.Random(0), 1)
    assert proposer.calls == 3                                     # asked 3 times, then fallback
    assert spec.signature() != parent.spec.signature()
    assert why.startswith("[fallback")


def test_repeated_workflows_are_skipped():
    parent = node(get_seed("blank"))
    choice = "add process_context (top_k=3)"
    seen = {parent.spec.signature(), apply_action(parent.spec, choice, "r").signature()}
    spec, action, _ = _next_new_spec(StubLLM(str(ACTIONS.index(choice) + 1)), parent, [parent],
                                     seen, random.Random(0), 2)
    assert spec.signature() not in seen


def test_anomaly_verify_revise_asks_llm_again():
    pm = ProcessMiningBaseline().fit(TRAIN)
    llm = StubLLM("NORMAL")
    pred, meta = AgentWorkflow(get_seed("process"), llm, VOCAB, pm).check_trace(
        ["submit", "pay", "approve"])                              # 'submit -> pay' never seen
    assert llm.calls == 2                                          # answer + reconsideration
    assert pred == 0 and not meta["fallback"]                      # the LLM decides, not the tool
