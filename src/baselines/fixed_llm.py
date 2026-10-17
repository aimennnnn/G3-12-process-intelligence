"""Fixed LLM workflow baseline (the non-adaptive, non-agentic comparison).

One LLM call per task with a hand-written prompt that never changes. It gets
the list of possible activities but NO process knowledge from the event log,
and it never learns from its mistakes. This mirrors AFlow's simplest "IO"
baseline, applied to business-process tasks.
"""
from __future__ import annotations

import re

SYSTEM = "You are an assistant that analyses business process executions. Answer tersely."


def format_trace(steps: list[str]) -> str:
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))


def next_activity_prompt(prefix: list[str], vocab: list[str]) -> str:
    options = "\n".join(f"[{i + 1}] {a}" for i, a in enumerate(vocab))
    return (
        "A travel-expense declaration process has executed these steps so far:\n"
        f"{format_trace(prefix)}\n\n"
        "Which activity happens next? Options:\n"
        f"{options}\n\n"
        "Reply with only the option number."
    )


def anomaly_prompt(trace: list[str]) -> str:
    return (
        "Here is one complete execution of a travel-expense declaration process:\n"
        f"{format_trace(trace)}\n\n"
        "Does this execution follow a normal, valid order of steps, or does it "
        "contain a step that is skipped, repeated, out of order, or does not belong?\n"
        "Reply with exactly one word: NORMAL or ANOMALOUS."
    )


def parse_option(text: str, vocab: list[str]) -> str | None:
    m = re.search(r"\d+", text)
    if m and 1 <= int(m.group()) <= len(vocab):
        return vocab[int(m.group()) - 1]
    for a in vocab:  # fallback: model wrote the activity name
        if a.lower() in text.lower():
            return a
    return None


def parse_label(text: str) -> int | None:
    t = text.upper()
    if "ANOMAL" in t:
        return 1
    if "NORMAL" in t:
        return 0
    return None


class FixedLLMWorkflow:
    name = "fixed_llm"

    def __init__(self, llm, vocab: list[str]):
        self.llm, self.vocab = llm, vocab

    def _ask(self, prompt: str):
        return self.llm.chat([{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt}])

    def predict_next(self, prefix: list[str]) -> tuple[str | None, dict]:
        resp = self._ask(next_activity_prompt(prefix, self.vocab))
        return parse_option(resp.text, self.vocab), _meta(resp)

    def check_trace(self, trace: list[str]) -> tuple[int | None, dict]:
        resp = self._ask(anomaly_prompt(trace))
        return parse_label(resp.text), _meta(resp)


def _meta(resp) -> dict:
    return {"raw": resp.text, "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "latency_s": resp.latency_s, "cached": resp.cached, "llm_calls": 1}
