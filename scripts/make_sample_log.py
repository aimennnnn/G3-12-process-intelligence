"""Generate a SMALL SYNTHETIC event log shaped like BPI Challenge 2020
(Domestic Declarations), so the pipeline can be tested without the real data.

    python scripts/make_sample_log.py            # 600 cases -> data/sample/

This is NOT real data. Never report results on it in the paper.
"""
from __future__ import annotations

import argparse
import gzip
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.sax.saxutils import quoteattr

SUB = ("Declaration SUBMITTED by EMPLOYEE", "EMPLOYEE")
ADM_OK = ("Declaration APPROVED by ADMINISTRATION", "ADMINISTRATION")
ADM_NO = ("Declaration REJECTED by ADMINISTRATION", "ADMINISTRATION")
PRE_OK = ("Declaration APPROVED by PRE_APPROVER", "PRE_APPROVER")
BUD_OK = ("Declaration APPROVED by BUDGET OWNER", "BUDGET OWNER")
SUP_OK = ("Declaration FINAL_APPROVED by SUPERVISOR", "SUPERVISOR")
SUP_NO = ("Declaration REJECTED by SUPERVISOR", "SUPERVISOR")
EMP_NO = ("Declaration REJECTED by EMPLOYEE", "EMPLOYEE")
REQ = ("Request Payment", "UNDEFINED")
PAY = ("Payment Handled", "UNDEFINED")


def sample_trace(rng: random.Random) -> list[tuple[str, str]]:
    trace = [SUB]
    for _ in range(3):  # at most 3 rejection loops
        r = rng.random()
        if r < 0.12:
            trace += [ADM_NO, EMP_NO, SUB]
            continue
        if r < 0.16:
            trace += [ADM_OK, SUP_NO, EMP_NO, SUB]
            continue
        break
    if rng.random() < 0.10:
        trace.append(PRE_OK)
    else:
        trace.append(ADM_OK)
        if rng.random() < 0.22:
            trace.append(BUD_OK)
    trace += [SUP_OK, REQ, PAY]
    return trace


def write_xes(path: Path, n_cases: int, seed: int) -> None:
    rng = random.Random(seed)
    start = datetime(2017, 1, 1, tzinfo=timezone.utc)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<log xes.version="1.0" xmlns="http://www.xes-standard.org/">',
             '<string key="concept:name" value="SYNTHETIC sample (not real data)"/>']
    for i in range(n_cases):
        t = start + timedelta(hours=rng.uniform(0, 24 * 700))
        lines.append("<trace>")
        lines.append(f'<string key="concept:name" value="declaration {i:05d}"/>')
        for act, role in sample_trace(rng):
            t += timedelta(hours=rng.expovariate(1 / 30))
            lines += ["<event>",
                      f'<string key="concept:name" value={quoteattr(act)}/>',
                      f'<string key="org:role" value={quoteattr(role)}/>',
                      f'<date key="time:timestamp" value="{t.isoformat()}"/>',
                      "</event>"]
        lines.append("</trace>")
    lines.append("</log>")
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/sample/sample_declarations.xes.gz")
    ap.add_argument("--cases", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    write_xes(Path(a.out), a.cases, a.seed)
    print(f"Wrote {a.cases} synthetic cases to {a.out}")
