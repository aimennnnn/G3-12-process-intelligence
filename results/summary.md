# Baseline results (data/sample/sample_declarations.xes.gz)

LLM: gemini / openai/gpt-oss-20b | seed 42 | 2026-11-07 20:46

## Next-activity prediction

| Method | N | Accuracy | Macro-F1 | LLM calls/task | Tokens/task | Latency (s)/task | Invalid |
|---|---|---|---|---|---|---|---|
| process_mining | 5 | 0.800 | 0.667 | 0.0 | 0.0 | 0.0 | 0.0% |
| fixed_llm | 5 | 0.200 | 0.250 | 1.0 | 162.6 | 2.153 | 0.0% |

## Anomaly detection

| Method | N | Accuracy | Precision | Recall | F1 | LLM calls/task | Tokens/task | Invalid |
|---|---|---|---|---|---|---|---|---|
| process_mining | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.0 | 0.0 | 0.0% |
| fixed_llm | 5 | 0.400 | 0.500 | 0.667 | 0.571 | 1.0 | 124.0 | 0.0% |

## Anomaly recall by perturbation type

| Method | skip | swap | insert | rework |
|---|---|---|---|---|
| process_mining | - | 1.0 | 1.0 | 1.0 |
| fixed_llm | - | 0.0 | 1.0 | 1.0 |
