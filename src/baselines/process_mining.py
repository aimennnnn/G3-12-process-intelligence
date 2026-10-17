"""Process-mining baseline (no LLM).

Learns a directly-follows graph (DFG) from the training log:
  * next activity  -> most frequent successor, using the last two steps when
                      that context was seen in training, otherwise the last step
                      (a standard frequency / Markov baseline)
  * anomaly        -> flag a trace if it uses a transition the training log
                      never (or almost never) shows, a lightweight form of
                      conformance checking against the discovered model
"""
from __future__ import annotations

from collections import Counter, defaultdict

START, END = "<START>", "<END>"


class ProcessMiningBaseline:
    name = "process_mining"

    def __init__(self, min_edge_prob: float = 0.0):
        # 0.0 => only transitions never seen in training count as anomalous
        self.min_edge_prob = min_edge_prob
        self.dfg: dict[str, Counter] = defaultdict(Counter)
        self.ctx2: dict[tuple, Counter] = defaultdict(Counter)
        self.global_freq: Counter = Counter()

    # ---------- training ----------
    def fit(self, traces: list[list[str]]) -> "ProcessMiningBaseline":
        for t in traces:
            seq = [START, *t, END]
            for a, b in zip(seq, seq[1:]):
                self.dfg[a][b] += 1
            for i in range(1, len(seq) - 1):
                self.ctx2[(seq[i - 1], seq[i])][seq[i + 1]] += 1
            self.global_freq.update(t)
        return self

    def edge_prob(self, a: str, b: str) -> float:
        total = sum(self.dfg[a].values())
        return self.dfg[a][b] / total if total else 0.0

    # ---------- task 1 ----------
    def predict_next(self, prefix: list[str]) -> str:
        seq = [START, *prefix]
        candidates = self.ctx2.get((seq[-2], seq[-1])) if len(seq) >= 2 else None
        if not candidates:
            candidates = self.dfg.get(seq[-1])
        # Ignore END: the task always asks for a real next activity.
        if candidates:
            ranked = [a for a, _ in candidates.most_common() if a != END]
            if ranked:
                return ranked[0]
        return self.global_freq.most_common(1)[0][0]

    # ---------- task 2 ----------
    def check_trace(self, trace: list[str]) -> tuple[int, str]:
        """Return (1 if anomalous else 0, short explanation)."""
        seq = [START, *trace, END]
        for a, b in zip(seq, seq[1:]):
            p = self.edge_prob(a, b)
            if p <= self.min_edge_prob:
                return 1, f"rare/unseen transition: '{a}' -> '{b}' (p={p:.3f})"
        return 0, "all transitions seen in training"

    def to_text(self, top_k: int = 3) -> str:
        """Readable summary of the discovered model (used to seed LLM agents)."""
        lines = []
        for a in sorted(self.dfg):
            succ = [f"{b} ({self.edge_prob(a, b):.0%})"
                    for b, _ in self.dfg[a].most_common(top_k)]
            lines.append(f"- after '{a}': " + ", ".join(succ))
        return "\n".join(lines)
