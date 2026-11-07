# Self-Improving Agentic Workflows for Business Process Intelligence

**AI4015 Agentic AI · Fall 2026 · Project G3-12** (Process Intelligence with Agentic AI: Designing Self-Improving Business Workflows)
Supervisor: Naeem Tareen · Instructor: Muhammad Owais Idrees
Team: Aimen Ahmad (22I-0922, group head) · Umar Javed (21I-0412) · Abdul Rafay (22I-0954)

Base paper: J. Zhang *et al.*, "AFlow: Automating Agentic Workflow Generation," ICLR 2025.

---

## What this does

LLM agents run two business-process tasks on a real event log, and are compared
against simpler baselines:

| Task | Input | Output | Correct answer comes from |
|---|---|---|---|
| **Next activity** | first *k* steps of a case | the step that comes next | the real log |
| **Anomaly detection** | a complete case | NORMAL / ANOMALOUS | the log + controlled perturbations (skip, swap, insert, rework) |

Methods:

| Method | What it is |
|---|---|
| `process_mining` | Non-LLM baseline: directly-follows graph learned from training cases |
| `fixed_llm` | Non-adaptive baseline: one fixed prompt, no process knowledge, never changes |
| `agent_seeded` | Early agentic workflow: process-model context + LLM + verification against the model |
| `src/agents/optimizer.py` | Phase 2: AFlow-style search that improves the workflow round by round |

Scoring is automatic and objective: no LLM grades another LLM.

## Dataset

**BPI Challenge 2020: Domestic Declarations** (travel-expense claims at Eindhoven University of Technology)
B. van Dongen, 4TU.ResearchData, 2020. DOI: [10.4121/uuid:3f422315-ed9d-4882-891f-e180b5b4feb5](https://doi.org/10.4121/uuid:3f422315-ed9d-4882-891f-e180b5b4feb5)

### Get the dataset
1. Open https://data.4tu.nl/articles/dataset/BPI_Challenge_2020_Domestic_Declarations/12692543/1
2. Download `DomesticDeclarations.xes.gz`
3. Put it in `data/raw/DomesticDeclarations.xes.gz` (do not unzip)

The real log is not committed to git (it is public, and each person downloads it).

## Setup (5 minutes)

```bash
git clone <repo-url> && cd g3-12-process-intelligence
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
```

Get a **free** Groq key at https://console.groq.com/keys or a Google Gemini API key at https://aistudio.google.com/apikey, and paste into `.env`:
```
GROQ_API_KEY=gsk_...
GROQ_API_KEY_2=gsk_...
GROQ_API_KEY_3=gsk_...

GEMINI_API_KEY=AIzaSy...
GEMINI_API_KEY_2=AIzaSy...
GEMINI_API_KEY_3=AIzaSy...
```
The client features a **3-key rotation ring** that automatically rotates keys upon HTTP 429 (rate-limit) or quota thresholds.
Supported providers: `groq` (default, `openai/gpt-oss-20b`), `gemini` (`gemini-1.5-flash`), `ollama`, or `mock`.

## Run

```bash
# 1. Data pipeline: load, clean, split, build tasks
python -m src.data.prepare

# 2. Quick check with 10 tasks of each type (about 1 minute)
python -m src.eval.run_eval --limit 10

# 3. Full baseline run (400 tasks, ~15-20 minutes on the free tier)
python -m src.eval.run_eval

# 4. Optional: run with Google Gemini API
python -m src.eval.run_eval --provider gemini

# 5. Run efficiency analysis (E6)
python -m experiments.e6_efficiency
```

Results land in `results/summary.md` (table for the report), `results/summary.json`,
`results/data_stats.json` (dataset numbers), and `results/predictions_<method>.jsonl`
(one record per task with the answer, cost, and a trace of how it was produced).

**No key or no internet?** Everything runs offline with a mock model, using a synthetic log:
```bash
python scripts/make_sample_log.py
python -m src.data.prepare --sample
python -m src.eval.run_eval --provider mock
python -m src.agents.optimizer --provider mock --seed process --rounds 5
```
Mock and sample results are only for testing the code. Never report them.

### Tests
```bash
python -m pytest -q
```

## Reproducibility

* Fixed seed `42` for the split, task sampling, and perturbations
* Temperature `0`, and every LLM answer is cached in `results/llm_cache.jsonl`, so re-runs are identical and free
* Temporal split: the earliest 80% of cases train, the latest 20% test. The optimizer validates on part of the *training* split only
* All settings live in `config.yaml`

## Repository structure

```
config.yaml                 all settings
src/
  data/                     xes_loader, preprocess, build_tasks, prepare (entry point)
  baselines/                process_mining (DFG), fixed_llm (one fixed prompt)
  llm/client.py             Groq / Gemini (3-key rotation ring), rate limit, retries, stats tracking, cache
  agents/
    state.py                structured contracts (tasks, predictions, workflow spec)
    workflow.py             operators: process_context, generate, ensemble, verify
    seeds.py                seed workflows (blank vs process-seeded)
    objective.py            G3: cost-aware objective & Pareto-frontier ranking (Gap 3)
    optimizer.py            Phase 2 self-improvement loop (AFlow-style)
  eval/                     metrics, evaluate, failure_analysis, run_eval
experiments/
  runs.yaml                 specification of all benchmark runs (E1-E5)
  run_all.py                experiment execution runner with key rotation
  analyze.py                results analysis and table generation
  e5_ablation.py            systematic single-operator ablation study
  e6_efficiency.py          G3: inference and search efficiency analysis (E6)
scripts/make_sample_log.py  synthetic test log
tests/                      unit tests
docs/summaries/             literature review notes per member
results/                    outputs
```

## Research gaps (from the Phase 1 literature review)

| Gap | Where it is in the code | Measured by |
|---|---|---|
| 1. Workflow optimisers are only tested on QA/code/math, not business processes | the tasks in `src/data/build_tasks.py` | accuracy / F1 vs both baselines |
| 2. Searches start from a blank workflow and ignore process knowledge | `--seed process` vs `--seed blank` in `optimizer.py` | rounds needed to reach a target score; final score |
| 3. Search optimises accuracy only; cost is reported afterwards | `--cost-weight` in `optimizer.py` | tokens per task at equal accuracy; cost–accuracy curve |

## Team and ownership

| Member | Owns |
|---|---|
| Aimen Ahmad (22I-0922) | Data pipeline: `src/data/`, `scripts/`, `tests/test_data.py` |
| Umar Javed (21I-0412) | Baselines and LLM client: `src/baselines/`, `src/llm/` |
| Abdul Rafay (22I-0954) | Evaluation, agents, docs: `src/eval/`, `src/agents/`, `tests/test_baselines.py`, README |

Every member must be able to explain the whole system in the viva.
