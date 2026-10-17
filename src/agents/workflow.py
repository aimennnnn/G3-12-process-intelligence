"""Executable agent workflow built from operators (Phase 2 system, early version).

Operators
  process_context : retrieve the discovered process model (DFG) as text and
                    give it to the LLM as grounding knowledge
  generate        : one LLM reasoning call
  ensemble        : run generate n times and take the majority answer
  verify          : check the answer against the process model; if it uses a
                    transition the log never shows, fall back / flag it

The fixed LLM baseline is just WorkflowSpec([generate]). The self-improving
system changes the spec round by round (see optimizer.py).
"""
from __future__ import annotations

from collections import Counter

from src.agents.state import WorkflowSpec
from src.baselines.fixed_llm import (SYSTEM, anomaly_prompt, next_activity_prompt,
                                     parse_label, parse_option)
from src.baselines.process_mining import START, ProcessMiningBaseline


class AgentWorkflow:
    def __init__(self, spec: WorkflowSpec, llm, vocab: list[str], pm: ProcessMiningBaseline):
        self.spec, self.llm, self.vocab, self.pm = spec, llm, vocab, pm
        self.name = f"agent[{spec.name}]"

    # ---------- shared ----------
    def _context(self) -> str:
        if not self.spec.has("process_context"):
            return ""
        k = self.spec.param("process_context", "top_k", 3)
        return ("Known process behaviour learned from past executions "
                "(most common next steps):\n" + self.pm.to_text(top_k=k) + "\n\n")

    def _generate(self, prompt: str, parse, meta: dict):
        n = self.spec.param("ensemble", "n", 1) if self.spec.has("ensemble") else 1
        answers = []
        for _ in range(n):
            r = self.llm.chat([{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": prompt}])
            meta["llm_calls"] += 1
            meta["prompt_tokens"] += r.prompt_tokens
            meta["completion_tokens"] += r.completion_tokens
            meta["latency_s"] += r.latency_s
            meta["cached"] = meta["cached"] or r.cached
            meta["raw"].append(r.text)
            answers.append(parse(r.text))
        valid = [a for a in answers if a is not None]
        return Counter(valid).most_common(1)[0][0] if valid else None

    @staticmethod
    def _meta() -> dict:
        return {"raw": [], "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                "latency_s": 0.0, "cached": False, "steps": []}

    # ---------- task 1 ----------
    def predict_next(self, prefix: list[str]):
        meta = self._meta()
        prompt = self._context() + next_activity_prompt(prefix, self.vocab)
        pred = self._generate(prompt, lambda t: parse_option(t, self.vocab), meta)
        meta["steps"].append(f"generate -> {pred}")
        if self.spec.has("verify"):
            last = prefix[-1] if prefix else START
            if pred is None or self.pm.edge_prob(last, pred) == 0.0:
                fallback = self.pm.predict_next(prefix)
                meta["steps"].append(f"verify: '{last}'->'{pred}' unseen, use {fallback}")
                pred = fallback
            else:
                meta["steps"].append("verify: transition seen in log, keep")
        return pred, meta

    # ---------- task 2 ----------
    def check_trace(self, trace: list[str]):
        meta = self._meta()
        prompt = self._context() + anomaly_prompt(trace)
        pred = self._generate(prompt, parse_label, meta)
        meta["steps"].append(f"generate -> {pred}")
        if self.spec.has("verify"):
            flag, why = self.pm.check_trace(trace)
            if flag == 1 and pred != 1:
                meta["steps"].append(f"verify: override to ANOMALOUS ({why})")
                pred = 1
            elif pred is None:
                pred = flag
                meta["steps"].append("verify: no valid LLM answer, use model check")
        return pred, meta
