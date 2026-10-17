from src.agents.state import PROCESS_SEED
from src.agents.optimizer import apply_action
from src.baselines.fixed_llm import parse_label, parse_option
from src.baselines.process_mining import ProcessMiningBaseline
from src.eval import metrics

TRAIN = [["submit", "approve", "pay"]] * 8 + [["submit", "reject", "submit", "approve", "pay"]] * 2


def test_pm_predicts_most_frequent_successor():
    pm = ProcessMiningBaseline().fit(TRAIN)
    assert pm.predict_next(["submit"]) == "approve"
    assert pm.predict_next(["submit", "reject"]) == "submit"


def test_pm_flags_unseen_transition():
    pm = ProcessMiningBaseline().fit(TRAIN)
    assert pm.check_trace(["submit", "approve", "pay"])[0] == 0
    assert pm.check_trace(["submit", "pay", "approve"])[0] == 1


def test_output_parsers():
    vocab = ["a", "b", "c"]
    assert parse_option("2", vocab) == "b"
    assert parse_option("Option [3]", vocab) == "c"
    assert parse_option("9", vocab) is None
    assert parse_label("ANOMALOUS.") == 1
    assert parse_label("normal") == 0
    assert parse_label("unsure") is None


def test_metrics():
    assert metrics.accuracy(["a", "b"], ["a", "c"]) == 0.5
    prf = metrics.binary_prf([1, 0, 1, None], [1, 1, 0, 0])
    assert prf["precision"] == 0.5 and prf["recall"] == 0.5


def test_apply_action_keeps_generate():
    spec = apply_action(PROCESS_SEED, "remove verify", "x")
    assert [s.operator for s in spec.steps] == ["process_context", "generate"]
    spec = apply_action(spec, "add ensemble (n=3)", "y")
    assert spec.has("ensemble") and spec.has("generate")
