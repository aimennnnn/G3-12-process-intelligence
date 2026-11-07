"""Process-model tool: the Gap 2 component (owner: Aimen).

The agent does not read the event log directly. It calls this tool, which
answers questions from a directly-follows graph (DFG) mined from the training
cases. It is deterministic and costs no LLM tokens.

Tool calls
  next_candidates(prefix, k)   most common next activities after this prefix,
                               with how often each followed it in past cases
  transition_report(trace)     every transition in a case with its frequency;
                               never-seen transitions are marked
  check_next(prefix, answer)   is "last step -> answer" a transition the log shows?
  check_trace(trace)           does the case contain a never-seen transition?

Every call returns a dict (logged in the agent trajectory) plus a short text
block that is placed in the LLM prompt.
"""
from __future__ import annotations

from src.baselines.process_mining import END, START, ProcessMiningBaseline


class ProcessModelTool:
    def __init__(self, pm: ProcessMiningBaseline, vocab: list[str]):
        self.pm, self.vocab = pm, vocab
        self.index = {a: i + 1 for i, a in enumerate(vocab)}   # option numbers in prompts

    @classmethod
    def from_traces(cls, traces: list[list[str]], vocab: list[str]) -> "ProcessModelTool":
        return cls(ProcessMiningBaseline().fit(traces), vocab)

    # ------------------------------------------------------------ next activity
    def next_candidates(self, prefix: list[str], k: int = 3) -> dict:
        seq = [START, *prefix]
        counts, context = None, None
        if len(seq) >= 2 and self.pm.ctx2.get((seq[-2], seq[-1])):
            counts, context = self.pm.ctx2[(seq[-2], seq[-1])], [seq[-2], seq[-1]]
        elif self.pm.dfg.get(seq[-1]):
            counts, context = self.pm.dfg[seq[-1]], [seq[-1]]
        if not counts:
            return {"tool": "next_candidates", "context": [], "candidates": [],
                    "text": "Process-model tool: this situation never occurred in past cases."}
        total = sum(v for a, v in counts.items() if a != END) or 1
        ranked = [(a, v) for a, v in counts.most_common() if a != END][:k]
        cands = [{"activity": a, "option": self.index.get(a), "share": round(v / total, 3),
                  "count": v} for a, v in ranked]
        ctx_txt = " then ".join(f"'{c}'" for c in context if c != START) or "the start"
        lines = [f"[{c['option']}] {c['activity']} (seen in {c['share']:.0%} of past cases)"
                 for c in cands]
        text = (f"Process-model tool (learned from past cases): after {ctx_txt}, the most "
                f"common next activities were:\n" + "\n".join(lines))
        return {"tool": "next_candidates", "context": context, "candidates": cands, "text": text}

    def check_next(self, prefix: list[str], answer: str | None) -> dict:
        last = prefix[-1] if prefix else START
        p = self.pm.edge_prob(last, answer) if answer else 0.0
        supported = p > 0.0
        text = (f"Process-model check: '{last}' -> '{answer}' occurred in {p:.0%} of past cases."
                if supported else
                f"Process-model check: '{last}' -> '{answer}' NEVER occurred in past cases.")
        return {"tool": "check_next", "last": last, "answer": answer,
                "share": round(p, 4), "supported": supported, "text": text}

    # ------------------------------------------------------------ anomaly
    def transition_report(self, trace: list[str]) -> dict:
        seq = [START, *trace, END]
        rows = []
        for a, b in zip(seq, seq[1:]):
            rows.append({"from": a, "to": b, "count": int(self.pm.dfg[a][b]),
                         "share": round(self.pm.edge_prob(a, b), 4)})
        unseen = [r for r in rows if r["count"] == 0]
        lines = []
        for r in rows:
            seen = (f"seen {r['count']} times" if r["count"] else "NEVER seen")
            lines.append(f"- {_name(r['from'])} -> {_name(r['to'])}: {seen}")
        text = ("Process-model tool: each transition in this case and how often it occurred "
                "in past cases:\n" + "\n".join(lines))
        return {"tool": "transition_report", "transitions": rows,
                "n_unseen": len(unseen), "text": text}

    def check_trace(self, trace: list[str]) -> dict:
        rep = self.transition_report(trace)
        unseen = [r for r in rep["transitions"] if r["count"] == 0]
        if unseen:
            u = unseen[0]
            text = (f"Process-model check: the transition {_name(u['from'])} -> {_name(u['to'])} "
                    f"NEVER occurred in past cases ({len(unseen)} unseen transition(s) in total).")
        else:
            rarest = min(rep["transitions"], key=lambda r: r["count"])
            text = (f"Process-model check: every transition in this case occurred in past cases "
                    f"(the rarest, {_name(rarest['from'])} -> {_name(rarest['to'])}, "
                    f"{rarest['count']} times).")
        return {"tool": "check_trace", "flag": int(bool(unseen)), "n_unseen": len(unseen),
                "text": text}


def _name(a: str) -> str:
    return {START: "(case start)", END: "(case end)"}.get(a, f"'{a}'")
