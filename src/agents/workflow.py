"""Executable agent workflow built from operators (G1 + core, Phase 2).

Operators (AFlow-style atomic units):
  process_context : retrieve context from the process-model tool (tools.py)
                    and prepend it to the prompt as grounding knowledge.
                    Uses ProcessModelTool.next_candidates / transition_report,
                    so the agent talks to the tool, not to the log directly.
  generate        : one LLM reasoning call. With ensemble on, run n times at
                    ensemble_temperature and take the majority answer.
  ensemble        : meta-operator that turns generate into a self-consistency
                    vote (n samples). Costs n × generate tokens.
  verify          : check the answer against the process model.
                    next activity: if the transition is unseen, fall back to the
                    model's prediction (mode=revise) or just flag it (mode=flag).
                    anomaly: if the model finds a never-seen transition and the
                    LLM said NORMAL, the LLM reconsiders with the tool's evidence
                    (mode=revise) or its answer is kept and flagged (mode=flag).

The fixed LLM baseline is WorkflowSpec([generate]). The self-improving system
changes the spec round by round (see optimizer.py).

This file connects to the rest of the repo:
  * uses ProcessModelTool from src/agents/tools.py        (your process-map tool)
  * starts from BLANK_SEED / PROCESS_SEED from seeds.py   (your seeds)
  * returns (pred, meta) where meta has every field that
    src/eval/evaluate.py:run_tasks reads                  (your scoring code)
  * emits a Trajectory per task (trajectory.py) when a logger is attached,
    so the rubric's "logged agent trajectories" requirement is satisfied
    without changing the scoring contract.

AgentWorkflow constructor signature is the same as Phase 1:
    AgentWorkflow(spec, llm, vocab, pm)
with an optional `traj_logger` keyword arg added (defaults to None — backward
compatible with any code that constructs it without the logger).
"""
from __future__ import annotations

import time
from collections import Counter

from src.agents.state import WorkflowSpec
from src.agents.trajectory import Trajectory, TrajectoryLogger
from src.agents.tools import ProcessModelTool
from src.baselines.fixed_llm import (SYSTEM, anomaly_prompt, next_activity_prompt,
                                     parse_label, parse_option)
from src.baselines.process_mining import START, ProcessMiningBaseline

# How many times to retry an unparseable LLM answer before falling back.
MAX_FORMAT_RETRIES = 2


class AgentWorkflow:
    """Run a WorkflowSpec on a task and emit a Trajectory.

    Backward-compatible with Phase 1: same constructor signature (plus an
    optional traj_logger kwarg), same predict_next / check_trace contract.
    """

    def __init__(self, spec: WorkflowSpec, llm, vocab: list[str],
                 pm: ProcessMiningBaseline, traj_logger: TrajectoryLogger | None = None):
        self.spec, self.llm, self.vocab, self.pm = spec, llm, vocab, pm
        # Use the process-model tool from tools.py — the Gap 2 component.
        self.tool = ProcessModelTool(pm, vocab)
        self.traj_logger = traj_logger
        self.name = f"agent[{spec.name}]"

    # ---------- shared ----------
    def _context_for_next(self, prefix: list[str]) -> tuple[str, dict]:
        """Return (prompt_block, tool_call_record) for process_context on the
        next-activity task. Returns ("", {}) when process_context is off."""
        if not self.spec.has("process_context"):
            return "", {}
        k = self.spec.param("process_context", "top_k", 3)
        rec = self.tool.next_candidates(prefix, k=k)
        return rec["text"] + "\n\n", rec

    def _context_for_anomaly(self, trace: list[str]) -> tuple[str, dict]:
        """Return (prompt_block, tool_call_record) for process_context on the
        anomaly task. Uses transition_report so the LLM sees every transition's
        frequency, not just the most common successors."""
        if not self.spec.has("process_context"):
            return "", {}
        rec = self.tool.transition_report(trace)
        return rec["text"] + "\n\n", rec

    def _generate(self, prompt: str, parse) -> tuple[object, list[str], dict]:
        """Run the generate operator (with optional ensemble). Returns
        (pred, raw_answers, cost_dict). Retries on unparseable output."""
        n = self.spec.param("ensemble", "n", 1) if self.spec.has("ensemble") else 1
        answers, raws = [], []
        llm_calls = prompt_tokens = completion_tokens = 0
        latency = 0.0
        cached = False
        format_retries = 0
        for _ in range(n):
            # Retry loop: if the LLM output is unparseable, ask again with a
            # short reminder (counts as a format_retry).
            attempts = 0
            parsed = None
            raw = ""
            while attempts <= MAX_FORMAT_RETRIES:
                r = self.llm.chat([{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": prompt}])
                llm_calls += 1
                prompt_tokens += r.prompt_tokens
                completion_tokens += r.completion_tokens
                latency += r.latency_s
                cached = cached or r.cached
                raw = r.text
                parsed = parse(raw)
                if parsed is not None or attempts == MAX_FORMAT_RETRIES:
                    break
                # Remind the model of the expected format and retry.
                prompt = prompt + "\n\nReply with ONLY the option number (or ONLY the word NORMAL / ANOMALOUS)."
                format_retries += 1
                attempts += 1
            answers.append(parsed)
            raws.append(raw)
        valid = [a for a in answers if a is not None]
        pred = Counter(valid).most_common(1)[0][0] if valid else None
        cost = {"llm_calls": llm_calls, "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens, "latency_s": round(latency, 3),
                "cached": cached, "format_retries": format_retries,
                "raw_answers": raws, "ensemble_n": n}
        return pred, raws, cost

    @staticmethod
    def _new_traj(task_id: str, method: str, task: str) -> Trajectory:
        return Trajectory(task_id=task_id, method=method, task=task)

    @staticmethod
    def _meta(template: dict | None = None) -> dict:
        """Build the meta dict that evaluate.py:run_tasks reads. Every key
        run_tasks accesses must be present."""
        m = {"raw": [], "llm_calls": 0, "tool_calls": 0, "prompt_tokens": 0,
             "completion_tokens": 0, "latency_s": 0.0, "cached": False,
             "format_retries": 0, "fallback": False, "steps": []}
        if template:
            m.update(template)
        return m

    # ---------- task 1: next activity ----------
    def predict_next(self, prefix: list[str], task_id: str = "?") -> tuple[object, dict]:
        traj = self._new_traj(task_id, self.name, "next_activity")
        meta = self._meta()
        prompt = next_activity_prompt(prefix, self.vocab)

        # 1. process_context (optional)
        ctx_block, ctx_rec = self._context_for_next(prefix)
        if ctx_block:
            traj.step("process_context",
                      input={"prefix": prefix, "top_k": self.spec.param("process_context", "top_k", 3)},
                      output={"text_chars": len(ctx_block)},
                      tool_calls=[ctx_rec],
                      note="inject mined DFG candidates as grounding knowledge")
            meta["tool_calls"] += 1
            meta["steps"].append(f"process_context: top-{self.spec.param('process_context', 'top_k', 3)} candidates from DFG")

        # 2. generate (with optional ensemble)
        gen_pred, raws, cost = self._generate(ctx_block + prompt,
                                              lambda t: parse_option(t, self.vocab))
        meta["llm_calls"] += cost["llm_calls"]
        meta["tool_calls"] += 0
        meta["prompt_tokens"] += cost["prompt_tokens"]
        meta["completion_tokens"] += cost["completion_tokens"]
        meta["latency_s"] += cost["latency_s"]
        meta["cached"] = meta["cached"] or cost["cached"]
        meta["format_retries"] += cost["format_retries"]
        meta["raw"] = raws
        traj.step("generate",
                  input={"prompt_chars": len(ctx_block + prompt), "ensemble_n": cost["ensemble_n"]},
                  output={"answer": gen_pred, "raw_answers": raws, "format_retries": cost["format_retries"]},
                  llm_calls=cost["llm_calls"], prompt_tokens=cost["prompt_tokens"],
                  completion_tokens=cost["completion_tokens"], latency_s=cost["latency_s"],
                  note=f"generate -> {gen_pred}" + (f" ({cost['format_retries']} retries)" if cost["format_retries"] else ""))
        meta["steps"].append(f"generate -> {gen_pred}")
        pred = gen_pred

        # 3. verify (optional)
        if self.spec.has("verify"):
            mode = self.spec.param("verify", "mode", "revise")
            check = self.tool.check_next(prefix, pred)
            meta["tool_calls"] += 1
            if not check["supported"]:  # unseen transition
                fallback = self.pm.predict_next(prefix)
                if mode == "revise":
                    note = f"verify (revise): '{check['last']}'->'{pred}' unseen in DFG; fall back to '{fallback}'"
                    final = fallback
                    meta["fallback"] = True
                else:  # mode == "flag"
                    note = f"verify (flag): '{check['last']}'->'{pred}' unseen in DFG; keep but flag"
                    final = pred
                traj.step("verify",
                          input={"candidate": pred, "edge_prob": check["share"]},
                          output={"final": final, "fallback_used": mode == "revise"},
                          tool_calls=[check], note=note)
                meta["steps"].append(note)
                pred = final
            else:
                note = f"verify ({mode}): transition '{check['last']}'->'{pred}' seen in DFG (p={check['share']:.2f}); keep"
                traj.step("verify",
                          input={"candidate": pred, "edge_prob": check["share"]},
                          output={"final": pred, "fallback_used": False},
                          tool_calls=[check], note=note)
                meta["steps"].append(note)

        traj.finish(pred, target=None, correct=False)  # target/correct filled by the evaluator
        if self.traj_logger:
            self.traj_logger.write(traj)
        return pred, meta

    # ---------- task 2: anomaly ----------
    def check_trace(self, trace: list[str], task_id: str = "?") -> tuple[object, dict]:
        traj = self._new_traj(task_id, self.name, "anomaly")
        meta = self._meta()
        prompt = anomaly_prompt(trace)

        # 1. process_context (optional)
        ctx_block, ctx_rec = self._context_for_anomaly(trace)
        if ctx_block:
            traj.step("process_context",
                      input={"trace_len": len(trace), "top_k": self.spec.param("process_context", "top_k", 3)},
                      output={"text_chars": len(ctx_block), "n_unseen": ctx_rec.get("n_unseen", 0)},
                      tool_calls=[ctx_rec],
                      note="inject transition report from DFG as grounding knowledge")
            meta["tool_calls"] += 1
            meta["steps"].append(f"process_context: transition report ({ctx_rec.get('n_unseen', 0)} unseen)")

        # 2. generate (with optional ensemble)
        gen_pred, raws, cost = self._generate(ctx_block + prompt, parse_label)
        meta["llm_calls"] += cost["llm_calls"]
        meta["prompt_tokens"] += cost["prompt_tokens"]
        meta["completion_tokens"] += cost["completion_tokens"]
        meta["latency_s"] += cost["latency_s"]
        meta["cached"] = meta["cached"] or cost["cached"]
        meta["format_retries"] += cost["format_retries"]
        meta["raw"] = raws
        traj.step("generate",
                  input={"prompt_chars": len(ctx_block + prompt), "ensemble_n": cost["ensemble_n"]},
                  output={"answer": gen_pred, "raw_answers": raws, "format_retries": cost["format_retries"]},
                  llm_calls=cost["llm_calls"], prompt_tokens=cost["prompt_tokens"],
                  completion_tokens=cost["completion_tokens"], latency_s=cost["latency_s"],
                  note=f"generate -> {gen_pred}" + (f" ({cost['format_retries']} retries)" if cost["format_retries"] else ""))
        meta["steps"].append(f"generate -> {gen_pred}")
        pred = gen_pred

        # 3. verify (optional)
        if self.spec.has("verify"):
            mode = self.spec.param("verify", "mode", "revise")
            check = self.tool.check_trace(trace)
            meta["tool_calls"] += 1
            conflict = check["flag"] == 1 and pred != 1   # model saw a never-seen transition
            if mode == "revise" and conflict:
                # The LLM reconsiders, now seeing the tool's evidence (Review & Revise).
                said = {0: "NORMAL", None: "no clear answer"}.get(pred, str(pred))
                fix_prompt = (ctx_block + prompt + f"\n\nYour previous answer was {said}. "
                              f"{check['text']}\nReconsider. Reply with exactly one word: "
                              f"NORMAL or ANOMALOUS.")
                r = self.llm.chat([{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": fix_prompt}])
                meta["llm_calls"] += 1
                meta["prompt_tokens"] += r.prompt_tokens
                meta["completion_tokens"] += r.completion_tokens
                meta["latency_s"] += r.latency_s
                meta["cached"] = meta["cached"] or r.cached
                revised = parse_label(r.text)
                if revised is None:
                    final = check["flag"]
                    meta["fallback"] = True
                    note = (f"verify (revise): DFG reports unseen transition; LLM asked to "
                            f"reconsider gave no valid answer -> use model check ({final})")
                else:
                    final = revised
                    note = (f"verify (revise): DFG reports unseen transition; LLM reconsidered "
                            f"-> {'ANOMALOUS' if final else 'NORMAL'}")
                traj.step("verify",
                          input={"candidate": pred, "model_flag": check["flag"]},
                          output={"final": final, "revised_answer": r.text[:100],
                                  "fallback_used": revised is None},
                          tool_calls=[check], llm_calls=1, prompt_tokens=r.prompt_tokens,
                          completion_tokens=r.completion_tokens, latency_s=r.latency_s,
                          note=note)
            else:
                if pred is None:
                    final = check["flag"]
                    meta["fallback"] = True
                    note = f"verify ({mode}): no valid LLM answer; use model check ({final})"
                elif conflict:   # mode == "flag": keep the LLM's answer, record the warning
                    final = pred
                    note = f"verify (flag): DFG reports unseen transition; LLM answer kept (flagged)"
                else:
                    final = pred
                    note = (f"verify ({mode}): LLM answer stands ({'ANOM' if pred else 'NORM'}); "
                            f"model says {check['flag']}")
                traj.step("verify",
                          input={"candidate": pred, "model_flag": check["flag"]},
                          output={"final": final, "fallback_used": final != pred},
                          tool_calls=[check], note=note)
            meta["steps"].append(note)
            pred = final

        traj.finish(pred, target=None, correct=False)
        if self.traj_logger:
            self.traj_logger.write(traj)
        return pred, meta
