"""Turn the test split into labelled evaluation tasks.

Task 1 - next activity: given the first k steps of a case, predict step k+1.
Task 2 - anomaly detection: is this case normal, or was it perturbed?
          Perturbations follow common practice in process-mining anomaly
          detection: skip, swap, insert, rework.
Ground truth comes from the log itself, so scoring is fully automatic.
"""
from __future__ import annotations

import random

PERTURBATIONS = ("skip", "swap", "insert", "rework")


def build_next_activity(test_traces: dict[str, list[str]], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    candidates = [(cid, t) for cid, t in test_traces.items() if len(t) >= 2]
    rng.shuffle(candidates)
    tasks = []
    for cid, trace in candidates[:n]:
        k = rng.randint(1, len(trace) - 1)
        tasks.append({
            "task_id": f"na_{len(tasks):04d}",
            "case_id": cid,
            "prefix": trace[:k],
            "target": trace[k],
        })
    return tasks


def perturb(trace: list[str], kind: str, vocab: list[str], rng: random.Random) -> list[str]:
    t = list(trace)
    if kind == "skip":
        del t[rng.randrange(len(t))]
    elif kind == "swap":
        options = [i for i in range(len(t) - 1) if t[i] != t[i + 1]]
        if not options:
            return t
        i = rng.choice(options)
        t[i], t[i + 1] = t[i + 1], t[i]
    elif kind == "insert":
        t.insert(rng.randrange(len(t) + 1), rng.choice(vocab))
    elif kind == "rework":
        i = rng.randrange(len(t))
        t.insert(i + 1, t[i])
    return t


def build_anomaly(test_traces: dict[str, list[str]], known_variants: set[tuple],
                  vocab: list[str], n: int, anomaly_ratio: float, seed: int) -> list[dict]:
    """known_variants = every variant seen in train+test, used to avoid
    labelling a perturbed trace 'anomalous' when it actually occurs in the log."""
    rng = random.Random(seed + 1)
    candidates = [(cid, t) for cid, t in test_traces.items() if len(t) >= 3]
    rng.shuffle(candidates)
    target = min(n, len(candidates))
    n_anom = round(target * anomaly_ratio)
    tasks: list[dict] = []
    n_made_anom = n_made_norm = 0
    for cid, trace in candidates:
        if len(tasks) >= target:
            break
        make_anomaly = n_made_anom < n_anom and (
            n_made_norm >= target - n_anom or rng.random() < anomaly_ratio)
        if make_anomaly:
            for _ in range(10):  # retry until the perturbation is genuinely new
                kind = rng.choice(PERTURBATIONS)
                new = perturb(trace, kind, vocab, rng)
                if tuple(new) not in known_variants:
                    break
            else:
                continue
            tasks.append({"task_id": f"an_{len(tasks):04d}", "case_id": cid,
                          "trace": new, "label": 1, "anomaly_type": kind})
            n_made_anom += 1
        else:
            tasks.append({"task_id": f"an_{len(tasks):04d}", "case_id": cid,
                          "trace": trace, "label": 0, "anomaly_type": None})
            n_made_norm += 1
    rng.shuffle(tasks)
    return tasks
