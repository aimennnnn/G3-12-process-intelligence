# Development log

Each entry: date, who, what failed or was limited, what we changed, and the measured effect.
Numbers come from files in `results/` unless marked otherwise.

| # | Date | Who | Observed failure / limitation | Change | Measured effect |
|---|---|---|---|---|---|
| 1 | 2026-10-17 | Umar | Groq returned `404 model_not_found` for `llama-3.1-8b-instant` (model retired) | Switched the executor to `openai/gpt-oss-20b` (non-Chinese, open-weight) and raised `max_tokens` to 1024 because it is a reasoning model | Baseline run completed: 400 tasks, 0.0% invalid next-activity answers (`results/summary.md`, Phase 1) |
| 2 | 2026-10-17 | Umar | Phase 1 baseline run hit many HTTP 429 responses on the free tier | Retries with back-off + `retry-after`, and a disk cache of every answer | Run finished in 297 s despite the 429s; re-runs cost 0 API calls |
| 3 | 2026-10-17 | Aimen | Anomaly task builder made 100 of 120 sample tasks perturbed instead of half | Capped the target at the number of candidate cases and counted perturbed/normal separately | Balanced sets: 100 perturbed / 100 normal on the real log (`results/data_stats.json`) |
| 4 | 2026-10-17 | Aimen | Fixed LLM baseline missed **every** skipped-activity anomaly (skip recall 0.0) and reached only 59.5% next-activity accuracy vs 92.5% for process mining | Built the process-model tool (`tools.py`) and the process-seeded workflow (`seeds.py`) so the agent can see transition frequencies and check its answers | Seed workflow vs fixed LLM on the test set: see E5 `e5_seed_only` in `results/tables/main_results.md` |
| 5 | 2026-11-06 | Aimen | No validation set existed, so a search would have had to pick workflows on the test set | `prepare.py` builds validation tasks from the latest 20% of training cases; the search's process model is mined only from the earlier 80% | Test tasks unchanged from Phase 1 (same seed); test set untouched by the search |
| 6 | 2026-11-07 | Aimen | The G1 optimiser's interface did not match the experiment runner (it takes the mined process model rather than the tool, and `evaluate_spec` returns a search record rather than per-task rows), so every search run crashed | Runner now calls `optimise()` with keyword arguments and the fit-split process model, runs the final test evaluation itself with `AgentWorkflow` + `run_tasks`, logs the search to `logs/<run>/`, and falls back to the executor model when no separate optimiser model is available | Offline (mock) smoke test: 17/17 runs complete and the analysis tables build |
