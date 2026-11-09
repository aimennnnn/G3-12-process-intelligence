### Main results on the test set (mean ± std over runs)

| Exp | Method | Runs | NA acc (%) | NA macro-F1 | Anom acc (%) | Anom F1 | Task score | Tokens/task | LLM calls/task |
|---|---|---|---|---|---|---|---|---|---|
| E1 | Process mining (DFG) | 1 | 95.0 | 0.700 | 98.0 | 0.979 | 0.965 | 0 | 0.00 |
| E1 | Fixed LLM | 1 | 60.0 | 0.413 | 84.0 | 0.818 | 0.720 | 449 | 1.00 |
| E2 | AFlow (blank, accuracy-only) | 3 | 95.0 ± 0.0 | 0.700 ± 0.000 | 95.0 ± 0.0 | 0.945 ± 0.000 | 0.950 ± 0.000 | 552 ± 9 | 1.00 ± 0.00 |
| E3 | ProcFlow (full) | 3 | 95.0 ± 0.0 | 0.700 ± 0.000 | 98.0 ± 0.0 | 0.979 ± 0.000 | 0.965 ± 0.000 | 554 ± 0 | 1.02 ± 0.00 |
| E5 | ProcFlow w/o process seed | 3 | 95.0 ± 0.0 | 0.700 ± 0.000 | 96.0 ± 1.7 | 0.956 ± 0.019 | 0.955 ± 0.009 | 546 ± 7 | 1.01 ± 0.01 |
| E5 | ProcFlow w/o cost objective | 3 | 95.0 ± 0.0 | 0.700 ± 0.000 | 98.0 ± 0.0 | 0.979 ± 0.000 | 0.965 ± 0.000 | 569 ± 0 | 1.02 ± 0.00 |
| E5 | ProcFlow w/o search (seed workflow) | 1 | 95.0 | 0.700 | 98.0 | 0.979 | 0.965 | 553 | 1.01 |
