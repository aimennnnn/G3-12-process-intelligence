import random
from pathlib import Path

import pytest

from scripts.make_sample_log import write_xes
from src.data.build_tasks import build_anomaly, build_next_activity, perturb
from src.data.preprocess import clean, temporal_split, to_traces
from src.data.xes_loader import load_xes


@pytest.fixture(scope="module")
def log(tmp_path_factory):
    path = tmp_path_factory.mktemp("d") / "s.xes.gz"
    write_xes(path, n_cases=150, seed=1)
    return clean(load_xes(path))


def test_loader_columns(log):
    assert {"case_id", "activity", "timestamp"} <= set(log.columns)
    assert log["timestamp"].notna().all()
    assert log["case_id"].nunique() == 150


def test_events_sorted_within_case(log):
    for _, g in log.groupby("case_id"):
        assert g["timestamp"].is_monotonic_increasing


def test_split_has_no_leakage(log):
    train, test = temporal_split(log, 0.8)
    assert not set(train["case_id"]) & set(test["case_id"])
    assert train.groupby("case_id")["timestamp"].min().max() <= \
        test.groupby("case_id")["timestamp"].min().min()


def test_next_activity_targets_follow_prefix(log):
    traces = to_traces(log)
    for t in build_next_activity(traces, 50, seed=0):
        full = traces[t["case_id"]]
        assert full[:len(t["prefix"])] == t["prefix"]
        assert full[len(t["prefix"])] == t["target"]


def test_anomaly_set_is_balanced_and_new(log):
    traces = to_traces(log)
    vocab = sorted(log["activity"].unique())
    variants = {tuple(t) for t in traces.values()}
    tasks = build_anomaly(traces, variants, vocab, 60, 0.5, seed=0)
    assert sum(t["label"] for t in tasks) == 30
    for t in tasks:
        if t["label"] == 1:
            assert tuple(t["trace"]) not in variants


def test_perturbations_change_trace():
    rng = random.Random(0)
    base = ["a", "b", "c", "d"]
    for kind in ("skip", "swap", "insert", "rework"):
        assert perturb(base, kind, ["x"], rng) != base
