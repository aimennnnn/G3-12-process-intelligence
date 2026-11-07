# E5 Ablation study (data/sample/sample_declarations.xes.gz)

Provider: gemini / openai/gpt-oss-20b | seed 42 | 2026-11-07 21:04

## Next-activity prediction

| Variant | Spec | Accuracy | Macro-F1 | Tokens/task | Score |
|---|---|---|---|---|---|
| full | `process_context(top_k=3)+generate+verify(mode=revise)` | 0.800 | 0.667 | 265 | 0.887 |
| no_context | `generate+verify(mode=revise)` | 0.400 | 0.417 | 162 | 0.492 |
| no_verify | `process_context(top_k=3)+generate` | 0.800 | 0.667 | 265 | 0.887 |
| ensemble | `process_context(top_k=3)+ensemble(n=3)+generate+verify(mode=revise)` | 0.800 | 0.667 | 796 | 0.860 |
| blank_seed | `generate` | 0.200 | 0.250 | 143 | 0.293 |
| fixed_llm | `fixed_llm (one LLM call, no agent)` | 0.200 | 0.250 | 143 | 0.293 |

## Anomaly detection

| Variant | Spec | Accuracy | Precision | Recall | F1 | Tokens/task | Score |
|---|---|---|---|---|---|---|---|
| full | `process_context(top_k=3)+generate+verify(mode=revise)` | 1.000 | 1.000 | 1.000 | 1.000 | 265 | 0.887 |
| no_context | `generate+verify(mode=revise)` | 0.600 | 0.600 | 1.000 | 0.750 | 162 | 0.492 |
| no_verify | `process_context(top_k=3)+generate` | 1.000 | 1.000 | 1.000 | 1.000 | 265 | 0.887 |
| ensemble | `process_context(top_k=3)+ensemble(n=3)+generate+verify(mode=revise)` | 1.000 | 1.000 | 1.000 | 1.000 | 796 | 0.860 |
| blank_seed | `generate` | 0.400 | 0.500 | 0.667 | 0.571 | 143 | 0.293 |
| fixed_llm | `fixed_llm (one LLM call, no agent)` | 0.400 | 0.500 | 0.667 | 0.571 | 143 | 0.293 |

## Variants

- **full**: Full ProcFlow (G1+G2+G3)
- **no_context**: Gap 2 OFF: no process_context
- **no_verify**: verify operator removed
- **ensemble**: + ensemble (n=3) self-consistency
- **blank_seed**: Gap 2 OFF: blank seed only
- **fixed_llm**: non-agentic baseline (no agent, no operators)
