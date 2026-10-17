"""Data pipeline entry point.

    python -m src.data.prepare               # real BPI 2020 log
    python -m src.data.prepare --sample      # small synthetic log for testing

Writes to data/processed/: train.csv, test.csv, vocab.json,
next_activity.jsonl, anomaly.jsonl, and results/data_stats.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.data.build_tasks import build_anomaly, build_next_activity
from src.data.preprocess import clean, log_stats, temporal_split, to_traces
from src.data.xes_loader import load_log


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", help="override the event log path")
    ap.add_argument("--sample", action="store_true", help="use the synthetic sample log")
    args = ap.parse_args()

    cfg = load_config(args.config)
    d, t, seed = cfg["data"], cfg["tasks"], cfg["seed"]
    src = args.input or (d["sample_path"] if args.sample else d["raw_path"])
    out = Path(d["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {src} ...")
    df = clean(load_log(src), d["min_trace_len"])
    train, test = temporal_split(df, d["train_ratio"])

    train_traces, test_traces = to_traces(train), to_traces(test)
    vocab = sorted(train["activity"].unique().tolist())
    variants = {tuple(x) for x in list(train_traces.values()) + list(test_traces.values())}

    na = build_next_activity(test_traces, t["next_activity"]["n_samples"], seed)
    an = build_anomaly(test_traces, variants, vocab, t["anomaly"]["n_samples"],
                       t["anomaly"]["anomaly_ratio"], seed)

    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    (out / "vocab.json").write_text(json.dumps(vocab, indent=2))
    write_jsonl(out / "next_activity.jsonl", na)
    write_jsonl(out / "anomaly.jsonl", an)

    stats = {"source": str(src), "full": log_stats(df), "train": log_stats(train),
             "test": log_stats(test), "tasks": {"next_activity": len(na),
             "anomaly": len(an), "anomalies": sum(x["label"] for x in an)}}
    res = Path(cfg["eval"]["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    (res / "data_stats.json").write_text(json.dumps(stats, indent=2))

    f = stats["full"]
    print(f"Cases {f['n_cases']} | events {f['n_events']} | activities {f['n_activities']}"
          f" | variants {f['n_variants']}")
    print(f"Train cases {stats['train']['n_cases']} | test cases {stats['test']['n_cases']}")
    print(f"Tasks: {len(na)} next-activity, {len(an)} anomaly "
          f"({stats['tasks']['anomalies']} perturbed). Saved to {out}/")


if __name__ == "__main__":
    main()
