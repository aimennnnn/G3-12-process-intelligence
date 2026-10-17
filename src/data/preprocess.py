"""Clean an event log and split it into train/test by time (no leakage)."""
from __future__ import annotations

import pandas as pd


def clean(df: pd.DataFrame, min_trace_len: int = 2) -> pd.DataFrame:
    """Drop broken rows, sort events, and remove very short cases."""
    df = df.dropna(subset=["case_id", "activity", "timestamp"]).copy()
    df["activity"] = df["activity"].astype(str).str.strip()
    df = df[df["activity"] != ""]
    df = df.drop_duplicates()
    # Stable sort keeps the log's original order for events with equal timestamps.
    df = df.sort_values(["case_id", "timestamp"], kind="stable")
    sizes = df.groupby("case_id")["activity"].transform("size")
    return df[sizes >= min_trace_len].reset_index(drop=True)


def temporal_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """Split by case start time: earliest cases train, latest cases test.

    This mimics real use (learn from the past, predict the future) and
    guarantees no case appears in both splits.
    """
    starts = df.groupby("case_id")["timestamp"].min().sort_values(kind="stable")
    cut = int(len(starts) * train_ratio)
    train_ids = set(starts.index[:cut])
    train = df[df["case_id"].isin(train_ids)].reset_index(drop=True)
    test = df[~df["case_id"].isin(train_ids)].reset_index(drop=True)
    return train, test


def to_traces(df: pd.DataFrame) -> dict[str, list[str]]:
    """case_id -> ordered list of activity names."""
    return {cid: g["activity"].tolist()
            for cid, g in df.groupby("case_id", sort=False)}


def log_stats(df: pd.DataFrame) -> dict:
    """Numbers to quote in the proposal's dataset section."""
    traces = to_traces(df)
    lengths = [len(t) for t in traces.values()]
    return {
        "n_cases": len(traces),
        "n_events": int(len(df)),
        "n_activities": int(df["activity"].nunique()),
        "n_variants": len({tuple(t) for t in traces.values()}),
        "avg_trace_len": round(sum(lengths) / max(len(lengths), 1), 2),
        "max_trace_len": max(lengths, default=0),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
    }
