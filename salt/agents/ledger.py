# -*- coding: utf-8 -*-
"""The record of what a session handed to its workers.

One line per delegation in `<session>/delegations.jsonl`, written after
the answer has already been shown, so the file is a history rather than
a thing the answer waits on. It holds what the delegation was and what
it cost, never the worker's prose: the text was printed, and a session
that wants it in memory ingests it as a turn instead.

Reading is deliberately forgiving. A line the reader cannot make sense
of is skipped with a warning and the rest of the file still loads, which
is what makes the last line safe to lose to a crash mid-write.
"""

import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "salt-delegation/1"
LEDGER_NAME = "delegations.jsonl"
FIELDS = ("schema", "id", "target", "task", "context_stats", "status",
          "usage", "t_start", "t_end", "ingest")
CONTEXT_FIELDS = ("n_selected", "words_used")


@dataclass(frozen=True)
class Ledger:
    """One session's delegation history as it was found on disk."""

    records: tuple = ()
    warnings: tuple = ()
    last_id: int = 0

    def __len__(self):
        return len(self.records)


def ledger_path(session_dir):
    return Path(session_dir) / LEDGER_NAME


def record(result, ingest=False):
    """The line one delegation leaves behind."""
    ctx = result.context
    return {"schema": SCHEMA,
            "id": int(result.id),
            "target": result.target,
            "task": result.task,
            "context_stats": {"n_selected": ctx.n_selected if ctx else 0,
                              "words_used": ctx.words_used if ctx else 0},
            "status": result.status,
            "usage": dict(result.usage or {}),
            "t_start": round(float(result.t_start), 3),
            "t_end": round(float(result.t_end), 3),
            "ingest": bool(ingest)}


def append(session_dir, result, ingest=False):
    """File one delegation. The line is built whole and written once, so
    a concurrent reader sees either all of it or none of it."""
    rec = record(result, ingest)
    with open(ledger_path(session_dir), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
    return rec


def _scan(path):
    """(line number, record or None) for every line with anything on it."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for n, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            yield n, None
            continue
        yield n, rec if isinstance(rec, dict) else None


def read(session_dir):
    """Everything this session has delegated, and what could not be read.

    `last_id` counts every line carrying one, including lines skipped as
    unreadable by schema, so a newer salt's ids are never handed out a
    second time by an older one. A line torn in half by a crash takes its
    id with it and that one number can come round again.
    """
    path = ledger_path(session_dir)
    records, warnings, last = [], [], 0
    for n, rec in _scan(path):
        if rec is None:
            warnings.append(f"{path} line {n} is unreadable and was skipped "
                            f"(a delegation that did not finish writing).")
            continue
        try:
            last = max(last, int(rec.get("id", 0)))
        except (TypeError, ValueError):
            pass
        if rec.get("schema") != SCHEMA:
            warnings.append(f"{path} line {n} was written as "
                            f"{rec.get('schema')!r}, which this version of "
                            f"salt does not read, and was skipped.")
            continue
        records.append(rec)
    return Ledger(records=tuple(records), warnings=tuple(warnings),
                  last_id=last)
