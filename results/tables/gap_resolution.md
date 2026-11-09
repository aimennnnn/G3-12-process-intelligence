### Gap-resolution table (draft; check wording)

| Gap | Method / component | Metric | Base paper result | Our result | Addressed? |
|---|---|---|---|---|---|
| G1: business-process evaluation | AFlow-style search applied to next-activity + anomaly tasks on BPI 2020 | NA accuracy / anomaly F1 (test) | 0.950 / 0.945 | 0.950 / 0.979 | Partially |
| G2: cold start | Process-seeded initial workflow (seeds.py, --seed process) | rounds to reach blank's best val score; test task score | rounds [1, 1, 1, 1, 4, 1]; score 0.950 | rounds [0, 0, 0, 2, 2, 2]; score 0.965 | Partially |
| G3: cost-blind objective | Cost-aware objective (objective.py, cost_weight 0.1) | tokens/task; task score (test) | 569; 0.965 | 554 (-3%); 0.965 | Partially |
