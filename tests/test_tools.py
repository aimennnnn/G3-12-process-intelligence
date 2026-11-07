"""Tests for the process-model tool and the seed workflows (Gap 2 component)."""
from src.agents.seeds import get_seed
from src.agents.tools import ProcessModelTool

TRAIN = [["submit", "approve", "pay"]] * 8 + [["submit", "reject", "submit", "approve", "pay"]] * 2
VOCAB = ["approve", "pay", "reject", "submit"]


def tool():
    return ProcessModelTool.from_traces(TRAIN, VOCAB)


def test_next_candidates_ranked_with_option_numbers():
    c = tool().next_candidates(["submit"], k=2)
    assert [x["activity"] for x in c["candidates"]] == ["approve", "reject"]
    assert c["candidates"][0]["option"] == 1          # 'approve' is option [1] in VOCAB
    assert "(seen in" in c["text"]


def test_next_candidates_uses_two_step_context():
    c = tool().next_candidates(["submit", "reject"], k=1)
    assert c["context"] == ["submit", "reject"]
    assert c["candidates"][0]["activity"] == "submit"


def test_check_next():
    t = tool()
    assert t.check_next(["submit"], "approve")["supported"]
    assert not t.check_next(["submit"], "pay")["supported"]
    assert not t.check_next(["submit"], None)["supported"]


def test_transition_report_and_check_trace():
    t = tool()
    ok = t.check_trace(["submit", "approve", "pay"])
    bad = t.check_trace(["submit", "pay", "approve"])
    assert ok["flag"] == 0 and bad["flag"] == 1 and bad["n_unseen"] >= 1
    rep = t.transition_report(["submit", "pay"])
    assert "NEVER seen" in rep["text"]


def test_seeds():
    assert [s.operator for s in get_seed("blank").steps] == ["generate"]
    assert [s.operator for s in get_seed("process").steps] == ["process_context", "generate", "verify"]
    s = get_seed("process")
    s.steps.pop()                                     # copies must not change the original
    assert len(get_seed("process").steps) == 3
