"""E6: Efficiency and Cost-Awareness Analysis (G3, Phase 2).

Computes efficiency metrics from experiment runs:
  1. Inference Efficiency: tokens per task (prompt/completion), latency (s/task),
     LLM calls per task, and tool calls per task.
  2. Search Efficiency: rounds to converge to best validation score, cumulative
     search tokens consumed, and cold-start (AFlow blank) vs warm-start (ProcFlow seeded).
  3. Gap 3 Pareto Analysis: trade-off between task score and token consumption across
     cost weights λ in [0.0, 0.05, 0.1, 0.2], measuring token reduction percentage
     and accuracy retention.

Usage:
    python -m experiments.e6_efficiency                 # reads results/runs/
    python -m experiments.e6_efficiency --mock          # reads results/mock/runs/

Outputs (saved to results/tables/ and results/figures/):
    e6_efficiency.csv / .md
    e6_search_efficiency.csv / .md
    fig_e6_pareto.png
    fig_e6_latency_tokens.png
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.agents.objective import pareto_frontier


def load_all_runs(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load all run JSON files from results/runs or results/mock/runs."""
    runs_dir = results_dir / "runs"
    if not runs_dir.exists():
        return {}
    runs = {}
    for p in sorted(runs_dir.glob("*.json")):
        try:
            runs[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
    return runs


def compute_inference_efficiency(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract and aggregate inference cost metrics per method."""
    by_config: dict[str, list[dict[str, Any]]] = {}
    for run_id, r in runs.items():
        # Strip random seed suffix like _s1, _s2, _s3
        base_id = "_".join(run_id.split("_")[:-1]) if run_id.endswith(("_s1", "_s2", "_s3")) else run_id
        by_config.setdefault(base_id, []).append(r)

    rows = []
    for base_id, r_list in sorted(by_config.items()):
        scores, tokens, prompt_toks, comp_toks, latencies, calls, tools = [], [], [], [], [], [], []
        for r in r_list:
            t = r.get("test", {}).get("overall", {})
            sc = t.get("task_score")
            tok = t.get("tokens_per_task")
            if sc is not None:
                scores.append(float(sc))
            if tok is not None:
                tokens.append(float(tok))
            p_tok = t.get("prompt_tokens_per_task", tok)
            if p_tok is not None:
                prompt_toks.append(float(p_tok))
            c_tok = t.get("completion_tokens_per_task", 0.0)
            comp_toks.append(float(c_tok))
            lat = t.get("latency_s_per_task")
            if lat is not None:
                latencies.append(float(lat))
            call = t.get("llm_calls_per_task")
            if call is not None:
                calls.append(float(call))
            tool = t.get("tool_calls_per_task")
            if tool is not None:
                tools.append(float(tool))

        if not tokens:
            continue

        mean_tok = st.mean(tokens)
        mean_score = st.mean(scores) if scores else 0.0
        # Score per 1k tokens
        score_per_1k = (mean_score / (mean_tok / 1000.0)) if mean_tok > 0 else 0.0

        rows.append({
            "Method": base_id,
            "Runs": len(r_list),
            "Task Score": f"{mean_score:.3f}" if scores else "-",
            "Tokens / Task": f"{mean_tok:.0f}",
            "Latency (s/task)": f"{st.mean(latencies):.2f}" if latencies else "-",
            "LLM Calls / Task": f"{st.mean(calls):.2f}" if calls else "-",
            "Tool Calls / Task": f"{st.mean(tools):.2f}" if tools else "-",
            "Score / 1k Tokens": f"{score_per_1k:.2f}",
        })
    return rows


def compute_search_efficiency(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract search convergence rounds, search tokens, and warm vs cold start."""
    rows = []
    for run_id, r in sorted(runs.items()):
        if r.get("kind") != "search" or "search" not in r:
            continue
        history = r.get("search", [])
        if not history:
            continue

        best_score = max(h["score"] for h in history)
        # Round where best score was first achieved
        rounds_to_best = next((h["round"] for h in history if abs(h["score"] - best_score) < 1e-6), len(history) - 1)
        total_rounds = len(history)

        # Estimate search tokens: round count * validation tasks * tokens per validation task
        c_stats = r.get("client_stats", {})
        search_tokens = c_stats.get("total_tokens", 0)

        seed_type = r.get("seed", "unknown")
        cost_weight = r.get("cost_weight", 0.0)

        rows.append({
            "Run ID": run_id,
            "Seed Type": seed_type,
            "Cost Weight (λ)": str(cost_weight),
            "Rounds to Best": rounds_to_best,
            "Total Rounds": total_rounds,
            "Best Val Score": f"{best_score:.3f}",
            "Search Tokens": str(search_tokens) if search_tokens > 0 else "-",
        })
    return rows


def compute_cost_sweep_pareto(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Analyze the Pareto trade-off across cost weights: 0.0, 0.05, 0.1, 0.2."""
    sweep_keys = [
        ("0.0", "e5_no_cost_s1"),
        ("0.05", "e4_cost_0.05_s1"),
        ("0.1", "e3_procflow_s1"),
        ("0.2", "e4_cost_0.2_s1"),
    ]
    points = []
    baseline_tokens = None
    baseline_score = None

    for cw, rid in sweep_keys:
        r = runs.get(rid)
        if not r:
            continue
        overall = r.get("test", {}).get("overall", {})
        sc = overall.get("task_score")
        tok = overall.get("tokens_per_task")
        if sc is None or tok is None:
            continue

        if cw == "0.0":
            baseline_tokens = float(tok)
            baseline_score = float(sc)

        tok_reduction = 0.0
        if baseline_tokens and baseline_tokens > 0:
            tok_reduction = ((baseline_tokens - float(tok)) / baseline_tokens) * 100.0

        score_delta = 0.0
        if baseline_score is not None:
            score_delta = float(sc) - baseline_score

        points.append({
            "Cost Weight (λ)": cw,
            "Run ID": rid,
            "Tokens / Task": round(float(tok), 1),
            "Task Score": round(float(sc), 4),
            "Token Reduction (%)": round(tok_reduction, 1),
            "Score Delta": round(score_delta, 4),
            "Best Workflow": r.get("workflow_text", ""),
        })

    return points


def save_table(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    """Save rows as CSV and Markdown."""
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())

    # CSV
    csv_file = out_path.with_suffix(".csv")
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f'"{r.get(c, "")}"' for c in cols) + "\n")

    # Markdown
    md_file = out_path.with_suffix(".md")
    lines = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    md_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_pareto(pareto_points: list[dict[str, Any]], out_path: Path) -> None:
    """Plot task score vs tokens per task (Pareto trade-off)."""
    if not pareto_points:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    x = [p["Tokens / Task"] for p in pareto_points]
    y = [p["Task Score"] for p in pareto_points]
    labels = [f"λ={p['Cost Weight (λ)']}" for p in pareto_points]

    ax.plot(x, y, "o-", color="#1f77b4", linewidth=2, markersize=7)
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), fontsize=8, xytext=(6, 5), textcoords="offset points")

    ax.set_xlabel("Tokens per Task (Test)")
    ax.set_ylabel("Task Score (Accuracy)")
    ax.set_title("E6: Cost-Aware Pareto Frontier (Gap 3)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_latency_vs_tokens(inference_rows: list[dict[str, Any]], out_path: Path) -> None:
    """Plot latency vs token usage across configurations."""
    valid_rows = [r for r in inference_rows if r["Latency (s/task)"] != "-" and r["Tokens / Task"] != "-"]
    if not valid_rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))

    x = [float(r["Tokens / Task"]) for r in valid_rows]
    y = [float(r["Latency (s/task)"]) for r in valid_rows]
    names = [r["Method"].replace("e1_", "").replace("e2_", "").replace("e3_", "").replace("e5_", "") for r in valid_rows]

    ax.scatter(x, y, color="#2ca02c", s=60, edgecolors="black", alpha=0.85)
    for xi, yi, name in zip(x, y, names):
        ax.annotate(name, (xi, yi), fontsize=7.5, xytext=(5, 4), textcoords="offset points")

    ax.set_xlabel("Tokens per Task")
    ax.set_ylabel("Latency (s / task)")
    ax.set_title("E6: Inference Latency vs Token Footprint")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_e6_analysis(results_dir: Path) -> None:
    runs = load_all_runs(results_dir)
    if not runs:
        print(f"No completed runs found in {results_dir / 'runs'}. Run experiments first.")
        return

    tables_dir = results_dir / "tables"
    figs_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Inference efficiency
    inf_rows = compute_inference_efficiency(runs)
    save_table(inf_rows, tables_dir / "e6_efficiency", "E6: Inference Efficiency and Token Footprint")

    # 2. Search efficiency
    search_rows = compute_search_efficiency(runs)
    if search_rows:
        save_table(search_rows, tables_dir / "e6_search_efficiency", "E6: Workflow Search Convergence Efficiency")

    # 3. Cost sweep & Pareto
    pareto_rows = compute_cost_sweep_pareto(runs)
    if pareto_rows:
        save_table(pareto_rows, tables_dir / "e6_cost_pareto", "E6: Cost-Aware Objective Sweep (Pareto Frontier)")
        plot_pareto(pareto_rows, figs_dir / "fig_e6_pareto.png")

    plot_latency_vs_tokens(inf_rows, figs_dir / "fig_e6_latency_tokens.png")

    print(f"\n[E6 Efficiency Analysis Complete]")
    if (tables_dir / "e6_efficiency.md").exists():
        print((tables_dir / "e6_efficiency.md").read_text(encoding="utf-8"))
    if pareto_rows and (tables_dir / "e6_cost_pareto.md").exists():
        print((tables_dir / "e6_cost_pareto.md").read_text(encoding="utf-8"))
    print(f"Saved tables to {tables_dir}/ and figures to {figs_dir}/")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    res_dir = Path(args.results) / ("mock" if args.mock else "")
    run_e6_analysis(res_dir)


if __name__ == "__main__":
    main()

