# Literature Review Summaries
**Name:** Abdul Rafay (22I-0954)  
**Role:** G3 — Cost-Aware Optimization, LLM Client Upgrade & Efficiency Analysis  

---

## Paper 1: AFlow: Automating Agentic Workflow Generation
* **Authors:** Jiayi Zhang et al.
* **Venue:** ICLR 2025 (A*)
* **Summary:** AFlow formulates agentic workflow optimization as an automated code search problem over modular building blocks (planning, tools, self-consistency ensembles, verification). Using an experience pool and Monte Carlo tree search, it iteratively modifies workflows to maximize validation accuracy on reasoning, math, and code generation benchmarks (e.g., HumanEval, GSM8K).
* **Limitations:**
  1. AFlow evaluates purely on static synthetic datasets and does not address business process monitoring tasks.
  2. The search begins from a blank prompt (`blank_seed`), incurring substantial cold-start search penalties.
  3. The objective function is **entirely cost-blind**: the optimizer selects workflows purely by raw predictive accuracy, producing bloated architectures with redundant verification and high-degree ensembles.
* **Relevance to G3:** Motivates our cost-aware scoring objective ($\text{Score} = \text{Accuracy} - \lambda \times \frac{\text{Tokens}}{\tau}$) to penalize token inflation and discover the Pareto frontier.

---

## Paper 2: Multi-Agent Architecture Search via Agentic Supernet (MaAS)
* **Authors:** Guobin Zhang et al.
* **Venue:** ICML 2025 (A*)
* **Summary:** MaAS proposes training an agentic supernet that learns a probability distribution over multi-agent topological structures, dynamically sampling an architecture conditioned on the query complexity. Notably, it includes inference cost (token counts and latency) directly into the optimization objective, achieving 6–45% inference cost reductions while improving task performance by 0.54–11.82%.
* **Limitations:** Training a full agentic supernet requires expensive pre-training and hundreds of thousands of evaluation samples, making it impractical for lightweight or budget-constrained process mining scenarios.
* **Relevance to G3:** Directly provides academic justification that token cost and predictive accuracy can and should be optimized jointly. Inspires ProcFlow's lightweight, non-parametric cost penalty integrated into AFlow-style iterative search.

---

## Paper 3: AgentSquare: Automatic LLM Agent Search in Modular Design Space
* **Authors:** Yu Shang et al.
* **Venue:** ICLR 2025 (A*)
* **Summary:** AgentSquare abstracts agent design into four distinct modular sub-spaces: planning, reasoning, tool use, and memory. It explores this space using evolutionary recombination and a performance predictor to skip unpromising designs, reporting average performance gains of 17.2% over human-engineered baselines across six benchmarks.
* **Limitations:** The search space combinatorial explosion leads to heavy API consumption during exploration, and no explicit cost constraint is enforced during module selection.
* **Relevance to G3:** Highlights the importance of discrete action spaces for workflow mutation, reinforcing G3's need to penalize operator overhead (e.g., distinguishing cheap `top_k=1` vs expensive `top_k=5` context retrieval).

