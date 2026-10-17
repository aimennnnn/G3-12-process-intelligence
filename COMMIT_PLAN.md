# Commit plan (Phase 1)

Each person commits **their own folder from their own GitHub account**, in this order
(each step uses the one before it). Run your part once before pushing, and read
through it: in the viva anyone can be asked about any file.

| Role | Member | Files |
|---|---|---|
| A — Data pipeline | Aimen Ahmad (22I-0922) | `src/data/`, `scripts/`, `tests/test_data.py`, `config.yaml`, `src/config.py` |
| B — Baselines + LLM client | Umar Javed (21I-0412) | `src/baselines/`, `src/llm/`, `.env.example` |
| C — Evaluation, agents, docs | Abdul Rafay (22I-0954) | `src/eval/`, `src/agents/`, `tests/test_baselines.py`, `README.md`, `results/summary.md`, `results/data_stats.json` |

## Step 1 — Aimen (creates the repo)

1. Create an empty GitHub repo `G3-12-process-intelligence` (private).
2. Settings → Collaborators: add both teammates, **Owais-Idrees**, and the supervisor (Naeem Tareen).
3. From the `A_data_pipeline` folder:

```bash
git init && git branch -M main
git add .gitignore requirements.txt config.yaml src/__init__.py src/config.py data/raw/.gitkeep data/processed/.gitkeep data/sample/.gitkeep results/.gitkeep
git commit -m "Set up project structure and configuration"
git add src/data/ scripts/
git commit -m "Add event log loader, cleaning, temporal split and task builder"
git add tests/__init__.py tests/test_data.py
git commit -m "Add data pipeline tests"
git remote add origin https://github.com/aimennnnn/G3-12-process-intelligence.git
git push -u origin main
```

## Step 2 — Member B

```bash
git clone https://github.com/aimennnnn/G3-12-process-intelligence.git && cd G3-12-process-intelligence
# copy in the files from the B_baselines folder, then:
git add src/llm/ .env.example
git commit -m "Add LLM client with rate limiting, retries and caching"
git add src/baselines/
git commit -m "Add process-mining and fixed LLM baselines"
git push
```

## Step 3 — Member C

```bash
git pull   # (or clone)
# copy in the files from the C_evaluation folder, then:
git add src/agents/
git commit -m "Add structured state contracts, agent workflow and optimiser skeleton"
git add src/eval/ tests/test_baselines.py
git commit -m "Add metrics and evaluation runner"
git add README.md COMMIT_PLAN.md
git commit -m "Add README with setup and run instructions"
git push
```

## Step 4 — Everyone runs the real baseline (needs the dataset + a Groq key)

```bash
python -m src.data.prepare
python -m src.eval.run_eval
python -m pytest -q
```

Then **Member C** commits the real results:

```bash
git add results/summary.md results/summary.json results/data_stats.json
git commit -m "Add initial baseline results on BPI Challenge 2020"
git push
```

## Step 5 — Everyone

Add your paper notes as `docs/summaries/<your-name>.md` and commit them
("Add literature notes for papers X–Y").
