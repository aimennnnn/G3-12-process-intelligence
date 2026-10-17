"""Load XES (or CSV) event logs into a flat pandas DataFrame.

Output columns: case_id, activity, timestamp, plus resource/role when present.
Uses a streaming parser, so large logs do not need to fit in memory as XML.
"""
from __future__ import annotations

import gzip
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ATTR_TAGS = {"string", "date", "int", "float", "boolean", "id"}
RENAME = {
    "concept:name": "activity",
    "time:timestamp": "timestamp",
    "org:resource": "resource",
    "org:role": "role",
}


def _local(tag: str) -> str:
    """Strip an XML namespace: '{ns}event' -> 'event'."""
    return tag.rsplit("}", 1)[-1]


def _open(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")


def load_xes(path: str | Path) -> pd.DataFrame:
    """Parse an .xes or .xes.gz file into one row per event."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Event log not found: {path}. See README section 'Get the dataset'."
        )

    rows: list[dict] = []
    stack: list[str] = []
    trace_attrs: dict = {}
    trace_events: list[dict] = []
    event: dict = {}
    n_traces = 0

    with _open(path) as fh:
        for kind, elem in ET.iterparse(fh, events=("start", "end")):
            tag = _local(elem.tag)
            if kind == "start":
                stack.append(tag)
                if tag == "trace":
                    trace_attrs, trace_events = {}, []
                elif tag == "event":
                    event = {}
                continue

            stack.pop()
            parent = stack[-1] if stack else None

            if tag in ATTR_TAGS:
                key, value = elem.get("key"), elem.get("value")
                if parent == "event":
                    event[key] = value
                elif parent == "trace":
                    trace_attrs[key] = value
            elif tag == "event" and parent == "trace":
                trace_events.append(event)
            elif tag == "trace":
                n_traces += 1
                case_id = trace_attrs.get("concept:name", f"case_{n_traces}")
                for ev in trace_events:
                    rows.append({"case_id": case_id, **ev})
                elem.clear()  # free memory as we go

    return _normalise(pd.DataFrame(rows))


def load_csv(path: str | Path, case_col="case_id", activity_col="activity",
             timestamp_col="timestamp") -> pd.DataFrame:
    """Load a CSV event log with configurable column names."""
    df = pd.read_csv(path)
    df = df.rename(columns={case_col: "case_id", activity_col: "activity",
                            timestamp_col: "timestamp"})
    return _normalise(df)


def load_log(path: str | Path) -> pd.DataFrame:
    """Pick the right loader from the file extension."""
    name = str(path).lower()
    if name.endswith(".xes") or name.endswith(".xes.gz"):
        return load_xes(path)
    if name.endswith(".csv"):
        return load_csv(path)
    raise ValueError(f"Unsupported event log format: {path}")


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RENAME)
    missing = {"case_id", "activity", "timestamp"} - set(df.columns)
    if missing:
        raise ValueError(f"Event log is missing required fields: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601",
                                     errors="coerce")
    df["case_id"] = df["case_id"].astype(str)
    keep = [c for c in ["case_id", "activity", "timestamp", "resource", "role"]
            if c in df.columns]
    return df[keep]
