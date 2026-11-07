"""Turn saved runs into the paper's tables and figures (owner: Aimen).

    python -m experiments.analyze            # reads results/runs/*.json
    python -m experiments.analyze --mock     # reads results/mock/ (testing only)

Writes results/tables/*.csv + *.md and results/figures/*.png. Every number in
the paper comes from these files.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# config label -> (display name, experiment)
CONFIGS = {
    "e1_process_mining": ("Process mining (DFG)", "E1"),
    "e1_fixed_llm": ("Fixed LLM", "E1"),
    "e2_aflow_blank": ("AFlow (blank, accuracy-only)", "E2"),
    "e3_procflow": ("ProcFlow (full)", "E3"),
    "e5_no_seed": ("ProcFlow w/o process seed", "E5"),
    "e5_no_cost": ("ProcFlow w/o cost objective", "E5"),
    "e5_seed_only": ("ProcFlow w/o search (seed workflow)", "E5"),
}


def label_of(run_id: str) -> str:
    return re.sub(r"_s\d+$", "", run_id)


def load_runs(res: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((res / "runs").glob("*.json"))}


def ms(vals: list[float], pct: bool = False, digits: int = 3) -> str:
    vals = [v for v in vals if v is not None]
    if not vals:
        return "-"
    f = (lambda v: f"{100 * v:.1f}") if pct else (lambda v: f"{v:.{digits}f}")
    if len(vals) == 1:
        return f(vals[0])
    return f"{f(st.mean(vals))} ± {f(st.stdev(vals))}"


def metric(run: dict, path: str):
    cur = run["test"]
    for k in path.split("."):
        cur = cur.get(k) if isinstance(cur, dict) else None
        if cur is None:
            return None
    return cur


def write_table(rows: list[dict], name: str, out: Path, title: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else []
    with open(out / f"{name}.csv", "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(f'"{r[c]}"' for c in cols) + "\n")
    md = [f"### {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    (out / f"{name}.md").write_text(encoding="utf-8", data="\n".join(md) + "\n")


def group(runs: dict) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = {}
    for rid, r in runs.items():
        g.setdefault(label_of(rid), []).append(r)
    return g


# ------------------------------------------------------------------ tables
def main_table(groups: dict) -> list[dict]:
    rows = []
    for lab, (name, exp) in CONFIGS.items():
        rs = groups.get(lab)
        if not rs:
            continue
        g = lambda p, pct=False: ms([metric(r, p) for r in rs], pct)
        rows.append({
            "Exp": exp, "Method": name, "Runs": len(rs),
            "NA acc (%)": g("next_activity.accuracy", True),
            "NA macro-F1": g("next_activity.macro_f1"),
            "Anom acc (%)": g("anomaly.accuracy", True),
            "Anom F1": g("anomaly.f1"),
            "Task score": g("overall.task_score"),
            "Tokens/task": ms([metric(r, "overall.tokens_per_task") for r in rs], digits=0),
            "LLM calls/task": ms([metric(r, "overall.llm_calls_per_task") for r in rs], digits=2),
        })
    return rows


def recall_table(groups: dict) -> list[dict]:
    rows = []
    for lab, (name, _) in CONFIGS.items():
        rs = groups.get(lab)
        if not rs:
            continue
        row = {"Method": name}
        for k in ["skip", "swap", "insert", "rework"]:
            row[k] = ms([metric(r, f"anomaly.recall_by_type.{k}") for r in rs])
        rows.append(row)
    return rows


def best_so_far(search: list[dict]) -> list[float]:
    """Best validation objective after each round (search may stop early: carry forward)."""
    out, best = [], -1e9
    for h in sorted(search, key=lambda h: h["round"]):
        best = max(best, h["score"])
        out.append(best)
    return out


def rounds_to_reach(search: list[dict], target: float) -> int | None:
    for i, v in enumerate(best_so_far(search)):
        if v >= target - 1e-9:
            return i
    return None


def g2_table(runs: dict) -> list[dict]:
    """Seeded vs blank at equal cost weight, paired by run seed."""
    rows = []
    pairs = [("e2_aflow_blank", "e5_no_cost", "0"), ("e5_no_seed", "e3_procflow", "0.1")]
    for blank, proc, cw in pairs:
        for s in (1, 2, 3):
            b, p = runs.get(f"{blank}_s{s}"), runs.get(f"{proc}_s{s}")
            if not (b and p and b.get("search") and p.get("search")):
                continue
            b_best = max(h["score"] for h in b["search"])
            rows.append({
                "Cost weight": cw, "Run seed": s,
                "Blank: round-0 val": f"{b['search'][0]['score']:.3f}",
                "Blank: best val": f"{b_best:.3f}",
                "Blank: rounds to best": rounds_to_reach(b["search"], b_best),
                "Seeded: round-0 val": f"{p['search'][0]['score']:.3f}",
                "Seeded: best val": f"{max(h['score'] for h in p['search']):.3f}",
                "Seeded: rounds to blank's best": rounds_to_reach(p["search"], b_best),
                "Test task score (blank / seeded)":
                    f"{metric(b, 'overall.task_score'):.3f} / {metric(p, 'overall.task_score'):.3f}",
            })
    return rows


def g3_table(runs: dict) -> list[dict]:
    pts = [("0.0", "e5_no_cost_s1"), ("0.05", "e4_cost_0.05_s1"), ("0.1", "e3_procflow_s1"),
           ("0.2", "e4_cost_0.2_s1")]
    rows = []
    for cw, rid in pts:
        r = runs.get(rid)
        if not r:
            continue
        rows.append({"Cost weight": cw, "Run": rid, "Best workflow": r.get("workflow_text", ""),
                     "Test tokens/task": f"{metric(r, 'overall.tokens_per_task'):.0f}",
                     "LLM calls/task": f"{metric(r, 'overall.llm_calls_per_task'):.2f}",
                     "Test task score": f"{metric(r, 'overall.task_score'):.3f}",
                     "NA acc": f"{metric(r, 'next_activity.accuracy'):.3f}",
                     "Anom F1": f"{metric(r, 'anomaly.f1'):.3f}"})
    return rows


def mean_of(groups, lab, path):
    vals = [metric(r, path) for r in groups.get(lab, []) if metric(r, path) is not None]
    return st.mean(vals) if vals else None


def gap_table(groups: dict, runs: dict, g2rows: list[dict]) -> list[dict]:
    """Rule-based first draft of the gap-resolution table; check the wording by hand."""
    rows = []
    e1f, e1p = groups.get("e1_fixed_llm"), groups.get("e1_process_mining")
    m = lambda lab, p: mean_of(groups, lab, p)

    # Gap 1: business-process evaluation (H1)
    na3, f13 = m("e3_procflow", "next_activity.accuracy"), m("e3_procflow", "anomaly.f1")
    na2, f12 = m("e2_aflow_blank", "next_activity.accuracy"), m("e2_aflow_blank", "anomaly.f1")
    naf, f1f = m("e1_fixed_llm", "next_activity.accuracy"), m("e1_fixed_llm", "anomaly.f1")
    f1p = m("e1_process_mining", "anomaly.f1")
    if None not in (na3, f13, naf, f1f):
        beats_fixed = na3 - naf >= 0.10 and f13 - f1f >= 0.10
        beats_pm = f1p is not None and f13 > f1p
        verdict = "Fully" if beats_fixed and beats_pm else ("Partially" if beats_fixed else "Not")
        rows.append({"Gap": "G1: business-process evaluation",
                     "Method / component": "AFlow-style search applied to next-activity + anomaly tasks on BPI 2020",
                     "Metric": "NA accuracy / anomaly F1 (test)",
                     "Base paper result": f"{na2:.3f} / {f12:.3f}" if na2 is not None else "-",
                     "Our result": f"{na3:.3f} / {f13:.3f}",
                     "Addressed?": verdict})
    # Gap 2: process-seeded initialisation (H2)
    if g2rows:
        halves, better = [], []
        for r in g2rows:
            bt, sd = r["Blank: rounds to best"], r["Seeded: rounds to blank's best"]
            halves.append(sd is not None and (bt == 0 or sd <= bt / 2))
            b, s = (float(x) for x in r["Test task score (blank / seeded)"].split(" / "))
            better.append(s - b >= 0.03)
        verdict = ("Fully" if all(halves) and all(better) else
                   "Partially" if any(halves) or any(better) else "Not")
        sd_round = [r["Seeded: rounds to blank's best"] for r in g2rows]
        bl_round = [r["Blank: rounds to best"] for r in g2rows]
        rows.append({"Gap": "G2: cold start",
                     "Method / component": "Process-seeded initial workflow (seeds.py, --seed process)",
                     "Metric": "rounds to reach blank's best val score; test task score",
                     "Base paper result": f"rounds {bl_round}; score {m('e2_aflow_blank', 'overall.task_score'):.3f}",
                     "Our result": f"rounds {sd_round}; score {m('e5_no_cost', 'overall.task_score'):.3f}",
                     "Addressed?": verdict})
    # Gap 3: cost-aware objective (H3)
    t0, t1 = m("e5_no_cost", "overall.tokens_per_task"), m("e3_procflow", "overall.tokens_per_task")
    a0, a1 = m("e5_no_cost", "overall.task_score"), m("e3_procflow", "overall.task_score")
    if None not in (t0, t1, a0, a1):
        cut = (t0 - t1) / t0 if t0 else 0.0
        drop = a0 - a1
        verdict = ("Fully" if cut >= 0.30 and drop <= 0.02 else
                   "Partially" if cut > 0 and drop <= 0.05 else "Not")
        rows.append({"Gap": "G3: cost-blind objective",
                     "Method / component": "Cost-aware objective (objective.py, cost_weight 0.1)",
                     "Metric": "tokens/task; task score (test)",
                     "Base paper result": f"{t0:.0f}; {a0:.3f}",
                     "Our result": f"{t1:.0f} ({-100 * cut:+.0f}%); {a1:.3f}",
                     "Addressed?": verdict})
    return rows


# ------------------------------------------------------------------ figures
def fig_main(groups, out):
    labs = [l for l in CONFIGS if l in groups]
    names = [CONFIGS[l][0].replace(" (", "\n(") for l in labs]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    w = 0.38
    for j, (path, lab) in enumerate([("next_activity.accuracy", "Next-activity accuracy"),
                                     ("anomaly.f1", "Anomaly F1")]):
        means, sds = [], []
        for l in labs:
            v = [metric(r, path) for r in groups[l]]
            means.append(st.mean(v))
            sds.append(st.stdev(v) if len(v) > 1 else 0)
        ax.bar([i + (j - 0.5) * w for i in range(len(labs))], means, w, yerr=sds, capsize=3,
               label=lab, color=["#4C72B0", "#DD8452"][j])
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(names, fontsize=6.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (test)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_main_results.png", dpi=300)
    plt.close(fig)


def fig_convergence(runs, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    styles = {"e2_aflow_blank": ("Blank, cost 0 (AFlow)", "#C44E52", "--"),
              "e5_no_cost": ("Seeded, cost 0", "#4C72B0", "-"),
              "e5_no_seed": ("Blank, cost 0.1", "#DD8452", "--"),
              "e3_procflow": ("Seeded, cost 0.1 (ProcFlow)", "#55A868", "-")}
    for lab, (name, c, ls) in styles.items():
        curves = [best_so_far(r["search"]) for rid, r in runs.items()
                  if label_of(rid) == lab and r.get("search")]
        if not curves:
            continue
        n = max(len(c_) for c_ in curves)
        curves = [c_ + [c_[-1]] * (n - len(c_)) for c_ in curves]
        mean = [st.mean(col) for col in zip(*curves)]
        ax.plot(range(n), mean, ls, color=c, marker="o", ms=3, label=name)
    ax.set_xlabel("Search round")
    ax.set_ylabel("Best validation objective")
    ax.legend(fontsize=6)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_convergence.png", dpi=300)
    plt.close(fig)


def fig_cost(g3rows, out):
    if not g3rows:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = [float(r["Test tokens/task"]) for r in g3rows]
    y = [float(r["Test task score"]) for r in g3rows]
    ax.plot(x, y, "o-", color="#4C72B0")
    for xi, yi, r in zip(x, y, g3rows):
        ax.annotate(f"w={r['Cost weight']}", (xi, yi), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("Tokens per task (test)")
    ax.set_ylabel("Task score (test)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_cost_tradeoff.png", dpi=300)
    plt.close(fig)


def fig_recall(groups, out):
    labs = [l for l in ["e1_process_mining", "e1_fixed_llm", "e2_aflow_blank", "e3_procflow"]
            if l in groups]
    kinds = ["skip", "swap", "insert", "rework"]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    w = 0.8 / max(len(labs), 1)
    for j, l in enumerate(labs):
        vals = [st.mean([metric(r, f"anomaly.recall_by_type.{k}") or 0 for r in groups[l]])
                for k in kinds]
        ax.bar([i + j * w for i in range(len(kinds))], vals, w, label=CONFIGS[l][0])
    ax.set_xticks([i + w * (len(labs) - 1) / 2 for i in range(len(kinds))])
    ax.set_xticklabels(kinds)
    ax.set_ylabel("Recall (test)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=5.5, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_recall_by_type.png", dpi=300)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="results")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    res = Path(args.results) / ("mock" if args.mock else "")
    runs = load_runs(res)
    if not runs:
        raise SystemExit(f"No runs in {res}/runs. Run: python -m experiments.run_all")
    groups = group(runs)
    tables, figs = res / "tables", res / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    main_rows = main_table(groups)
    write_table(main_rows, "main_results", tables, "Main results on the test set (mean ± std over runs)")
    write_table(recall_table(groups), "recall_by_type", tables, "Anomaly recall by perturbation type")
    g2 = g2_table(runs)
    if g2:
        write_table(g2, "g2_seeded_vs_blank", tables, "Gap 2: process-seeded vs blank start")
    g3 = g3_table(runs)
    if g3:
        write_table(g3, "g3_cost_sweep", tables, "Gap 3: cost-weight sweep (run seed 1)")
    gaps = gap_table(groups, runs, g2)
    if gaps:
        write_table(gaps, "gap_resolution", tables, "Gap-resolution table (draft; check wording)")

    fig_main(groups, figs)
    fig_convergence(runs, figs)
    fig_cost(g3, figs)
    fig_recall(groups, figs)

    # G3: E6 Efficiency Analysis
    try:
        from experiments.e6_efficiency import run_e6_analysis
        run_e6_analysis(res)
    except Exception as e:
        print(f"Note: E6 efficiency analysis skipped or partial: {e}")

    print((tables / "main_results.md").read_text(encoding="utf-8"))
    if gaps:
        print((tables / "gap_resolution.md").read_text(encoding="utf-8"))
    print(f"Tables in {tables}/, figures in {figs}/")


if __name__ == "__main__":
    main()
