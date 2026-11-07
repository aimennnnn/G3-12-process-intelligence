"""Run every experiment (E1-E5 runs; E6 is computed from their logs) (owner: Aimen).

    python -m experiments.run_all                 # everything in experiments/runs.yaml
    python -m experiments.run_all --only e3 e2    # runs whose id starts with these
    python -m experiments.run_all --list          # show runs and which are done
    python -m experiments.run_all --provider mock # offline smoke test (never report)

Resumable: a run whose results/runs/<id>.json exists is skipped, and every LLM
answer is cached, so if the free quota runs out just run the same command
again later and it continues where it stopped.

Outputs per run
  results/runs/<id>.json             config, search tree, best workflow, val + test metrics
  logs/<id>/search.jsonl             one line per search round (change, score, proposer
                                     reasoning): the optimisation trajectory
  logs/<id>/search_log.txt           the search's progress messages
  logs/<id>/search_trajectories.jsonl  agent steps of every candidate during the search
  logs/<id>/test_predictions.jsonl   every test task with its agent trajectory
                                     (tool calls, LLM outputs, verification steps)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from src.agents.seeds import get_seed
from src.agents.tools import ProcessModelTool
from src.baselines.fixed_llm import FixedLLMWorkflow
from src.baselines.process_mining import ProcessMiningBaseline
from src.config import load_config
from src.data.preprocess import temporal_split, to_traces

# The search parts (G1: optimizer, evaluate) and the key-rotating client (G3) are
# imported when needed, so the baselines (E1) and --list work on their own.
try:
    from src.llm.client import QuotaExhausted
except ImportError:          # G3's client not pushed yet
    class QuotaExhausted(Exception):
        pass


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")


class Data:
    """Everything loaded once: splits, tools and task sets."""

    def __init__(self, cfg: dict):
        d = Path(cfg["data"]["processed_dir"])
        if not (d / "val_next_activity.jsonl").exists():
            sys.exit("Processed data is missing or from Phase 1. Run: python -m src.data.prepare")
        train = pd.read_csv(d / "train.csv")
        self.vocab = json.loads((d / "vocab.json").read_text(encoding="utf-8"))
        fit, _ = temporal_split(train.assign(timestamp=pd.to_datetime(train["timestamp"],
                                                                        utc=True, format="mixed")),
                                cfg["data"]["train_ratio"])
        # The search only sees a model mined from the FIT part of training (no validation leak);
        # the final test evaluation uses the model mined from ALL training cases, like the baseline.
        self.pm_full = ProcessMiningBaseline().fit(list(to_traces(train).values()))
        self.tool_full = ProcessModelTool(self.pm_full, self.vocab)
        self.tool_fit = ProcessModelTool.from_traces(list(to_traces(fit).values()), self.vocab)
        lim = cfg["eval"]["test_limit"]
        o = cfg["optimizer"]
        self.test_na = read_jsonl(d / "next_activity.jsonl")[:lim]
        self.test_an = read_jsonl(d / "anomaly.jsonl")[:lim]
        self.val_na = read_jsonl(d / "val_next_activity.jsonl")[:o["n_val_next_activity"]]
        self.val_an = read_jsonl(d / "val_anomaly.jsonl")[:o["n_val_anomaly"]]


def _stats(client) -> dict:
    """API-usage counters, if the LLM client keeps them (G3's client does)."""
    st = getattr(client, "stats", None)
    return dict(st.as_dict()) if st is not None and hasattr(st, "as_dict") else {}


def _describe(spec) -> str:
    return spec.describe() if hasattr(spec, "describe") else spec.signature()


def run_one(run: dict, cfg: dict, data: Data, llm, opt_llm, res: Path, logs: Path) -> dict:
    from src.eval.evaluate import run_tasks, summarise
    if run["kind"] != "baseline":
        from src.agents.optimizer import best_record, optimise   # G1
        from src.agents.workflow import AgentWorkflow           # G1
    o = cfg["optimizer"]
    cw = float(run.get("cost_weight", 0.0))
    out = {"id": run["id"], "experiment": run["exp"], "kind": run["kind"], "run": run,
           "models": {"executor": cfg["llm"]["model"],
                      "optimizer": (cfg.get("optimizer_llm", {}).get("model")
                                    if opt_llm is not llm else cfg["llm"]["model"]),
                      "provider": getattr(llm, "provider", "?"),
                      "reasoning_effort": cfg["llm"].get("reasoning_effort"),
                      "temperature": cfg["llm"].get("temperature")},
           "n_test": {"next_activity": len(data.test_na), "anomaly": len(data.test_an)},
           "n_val": {"next_activity": len(data.val_na), "anomaly": len(data.val_an)},
           "started": time.strftime("%Y-%m-%d %H:%M")}
    before = _stats(llm)
    t0 = time.time()

    if run["kind"] == "baseline":
        method = data.pm_full if run["method"] == "process_mining" else FixedLLMWorkflow(llm, data.vocab)
        rows = run_tasks(method, run["method"], data.test_na, data.test_an, progress=run["id"])
    else:
        seed_spec = get_seed(run["seed"])
        if run["kind"] == "search":
            run_logs = logs / run["id"]
            run_logs.mkdir(parents=True, exist_ok=True)
            log_path = run_logs / "search_log.txt"
            log_path.write_text(encoding="utf-8", data="")

            def log(msg):
                print(f"    {msg}")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(str(msg) + "\n")

            extra = {}
            try:   # G1's trajectory logger: every candidate's agent steps during the search
                from src.agents.trajectory import TrajectoryLogger
                extra["traj_logger"] = TrajectoryLogger(run_logs / "search_trajectories.jsonl")
            except ImportError:
                pass
            exp_path = run_logs / "search.jsonl"        # one line per round (the experience pool)
            exp_path.write_text(encoding="utf-8", data="")

            # The search sees only the process model mined from the FIT part of training.
            history = optimise(seed_spec, llm, data.vocab, data.tool_fit.pm,
                               data.val_na, data.val_an, optimizer_llm=opt_llm,
                               rounds=o["rounds"], cost_weight=cw,
                               token_scale=o["token_scale"], patience=o["patience"],
                               top_k=o["top_k"], mix_lambda=o["mix_lambda"], alpha=o["alpha"],
                               seed=int(run["run_seed"]), experience_path=exp_path,
                               log=log, **extra)
            best = best_record(history)
            spec = best.spec
            out["search"] = [h.model_dump() for h in history]
            out["best_round"] = best.round
        else:
            spec = seed_spec
        out["workflow"] = spec.model_dump()
        out["workflow_text"] = _describe(spec)
        print(f"    test evaluation of [{_describe(spec)}]")
        # Final test: the process model mined from ALL training cases, like the baseline.
        agent = AgentWorkflow(spec, llm, data.vocab, data.pm_full)
        rows = run_tasks(agent, run["id"], data.test_na, data.test_an, progress="test")

    write_jsonl(logs / run["id"] / "test_predictions.jsonl", rows)
    out["test"] = summarise(rows, cw, o["token_scale"])
    after = _stats(llm)
    if after:
        out["client_stats"] = {k: after[k] - before.get(k, 0) for k in after}
    if opt_llm is not None and opt_llm is not llm and run["kind"] == "search" and _stats(opt_llm):
        out["optimizer_client_stats"] = _stats(opt_llm)
    out["wall_time_s"] = round(time.time() - t0, 1)
    (res / "runs").mkdir(parents=True, exist_ok=True)
    (res / "runs" / f"{run['id']}.json").write_text(encoding="utf-8", data=json.dumps(out, indent=2, default=str))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--runs", default="experiments/runs.yaml")
    ap.add_argument("--only", nargs="+", help="run ids (or prefixes) to run")
    ap.add_argument("--provider", help="override llm.provider (e.g. mock)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even if results exist")
    args = ap.parse_args()

    cfg = load_config(args.config)
    runs = yaml.safe_load(Path(args.runs).read_text(encoding="utf-8"))["runs"]
    if args.only:
        runs = [r for r in runs if any(r["id"].startswith(p) for p in args.only)]
    res, logs = Path(cfg["eval"]["results_dir"]), Path(cfg["eval"]["logs_dir"])
    if args.provider == "mock":   # keep mock output away from real results
        res, logs = res / "mock", logs / "mock"

    if args.list:
        for r in runs:
            done = (res / "runs" / f"{r['id']}.json").exists()
            print(f"  [{'done' if done else '    '}] {r['id']:<22} {r['exp']}  {r['kind']}")
        return

    data = Data(cfg)
    from src.llm.client import make_llm
    needs_llm = any(r["kind"] != "baseline" or r.get("method") != "process_mining" for r in runs)
    llm = make_llm(cfg, args.provider) if needs_llm else None
    opt_llm = None
    if needs_llm:
        try:      # G3's client can build a separate optimiser model (config: optimizer_llm)
            opt_llm = make_llm(cfg, args.provider, role="optimizer")
        except TypeError:   # older client: the optimiser uses the executor model
            opt_llm = llm

    todo = [r for r in runs if args.force or not (res / "runs" / f"{r['id']}.json").exists()]
    print(f"{len(runs) - len(todo)} runs already done, {len(todo)} to go.")
    for i, run in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {run['id']} ({run['exp']}, {run['kind']})")
        try:
            out = run_one(run, cfg, data, llm, opt_llm, res, logs)
        except QuotaExhausted as e:
            print(f"\nSTOPPED: {e}")
            sys.exit(2)
        t = out["test"]
        na, an = t.get("next_activity", {}), t.get("anomaly", {})
        print(f"  -> test next-activity acc {na.get('accuracy', 0):.3f} | anomaly F1 "
              f"{an.get('f1', 0):.3f} | tokens/task {t['overall']['tokens_per_task']:.0f} "
              f"| {out['wall_time_s']:.0f}s")
    print("\nAll runs finished. Next: python -m experiments.analyze")


if __name__ == "__main__":
    main()
