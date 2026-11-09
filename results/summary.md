# Baseline results (data/raw/DomesticDeclarations.xes.gz)

LLM: groq / openai/gpt-oss-20b | seed 42 | 2026-10-17 21:30

## Next-activity prediction

| Method | N | Accuracy | Macro-F1 | LLM calls/task | Tokens/task | Latency (s)/task | Invalid |
|---|---|---|---|---|---|---|---|
| process_mining | 200 | 0.925 | 0.651 | 0.0 | 0.0 | 0.0 | 0.0% |
| fixed_llm | 200 | 0.595 | 0.323 | 1.0 | 462.8 | 0.875 | 0.0% |

## Anomaly detection

| Method | N | Accuracy | Precision | Recall | F1 | LLM calls/task | Tokens/task | Invalid |
|---|---|---|---|---|---|---|---|---|
| process_mining | 200 | 0.990 | 1.000 | 0.980 | 0.990 | 0.0 | 0.0 | 0.0% |
| fixed_llm | 200 | 0.820 | 0.874 | 0.760 | 0.813 | 1.0 | 433.5 | 1.0% |

## Anomaly recall by perturbation type

| Method | skip | swap | insert | rework |
|---|---|---|---|---|
| process_mining | 0.933 | 1.0 | 0.958 | 1.0 |
| fixed_llm | 0.0 | 0.773 | 0.833 | 1.0 |
