# -*- coding: utf-8 -*-
"""The record of the turns a session planned out instead of answering.

One line per agent turn in `<session>/agent_trace.jsonl`, written after
the reply is already on screen, so the file is a history rather than
something the reply waits on.

It holds what the round decided and what each piece of it cost, never
the prose. The reply was kept as the session's own turn and each helper
answer already has a line in the delegation ledger, so repeating either
here would be a second copy that can disagree with the first.

Reading is forgiving in the same way the ledger is: a line that cannot
be made sense of is skipped with a warning and the rest still loads,
which is what makes the last line safe to lose to a crash mid-write.
"""

import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "salt-agent-trace/1"
TRACE_NAME = "agent_trace.jsonl"
FIELDS = ("schema", "ask", "action", "subtasks", "pieces", "planning",
          "synthesis", "protocol_failures", "fell_back", "answered_directly",
          "rounds", "reply_words", "t_start", "t_end", "seconds", "route")
PIECE_FIELDS = ("id", "target", "status", "ran", "usage", "seconds")
CALL_FIELDS = ("calls", "prompt_tokens", "cached_tokens")


@dataclass(frozen=True)
class Trace:
    """One session's agent turns as they were found on disk."""

    rounds: tuple = ()
    warnings: tuple = ()

    def __len__(self):
        return len(self.rounds)


def trace_path(session_dir):
    return Path(session_dir) / TRACE_NAME


def piece(result):
    """One subtask as the trace keeps it. `ran` is written down rather
    than derived from the status, so a later salt reading this file can
    tell a delegation from a piece nobody was asked without having to
    know which statuses this one had."""
    return {"id": int(result.id),
            "target": result.target,
            "status": result.status,
            "ran": bool(result.ran),
            "usage": dict(result.usage or {}),
            "seconds": round(result.seconds, 3)}


def call_numbers(stats):
    """One call's engine numbers, whichever way they were written down.

    A worker handle writes what a call cost into a dict the caller
    holds; a runner leaves the same two numbers on itself under the
    names its engine gave them. Both are read here so a round planned by
    a roster endpoint and one planned by the session's own model are
    added up the same way.
    """
    stats = stats or {}
    prompt = stats.get("prompt_tokens")
    if prompt is None:
        prompt = stats.get("apc_prompt_tokens")
    cached = stats.get("cached_tokens")
    if cached is None:
        cached = stats.get("apc_cached_tokens")
    return prompt, cached


def call_cost(calls):
    """What a round's own calls to one model cost, added up.

    Empty when nothing was measured, so a call that never happened and a
    backend that reports nothing read the same way. A zero would be a
    call that cost no prompt at all, which no call does, and an in
    process model has no request body for a server to count.
    """
    rows = [call_numbers(c) for c in calls or ()]
    rows = [(p, c) for p, c in rows if p is not None or c is not None]
    if not rows:
        return {}
    prompt = [int(p) for p, _ in rows if p is not None]
    cached = [int(c) for _, c in rows if c is not None]
    return {"calls": len(rows),
            "prompt_tokens": sum(prompt) if prompt else None,
            "cached_tokens": sum(cached) if cached else None}


def record(round_, route=None, planning=None):
    """The line one agent turn leaves behind.

    `route` is what decided this turn was worth planning, empty when
    nobody decided. Empty rather than absent: a round that nobody routed
    and a round whose route was lost are different things, and a reader
    counting how often a policy actually acted has to be able to tell
    them apart.

    `planning` is what the round's own questions to its planner cost,
    the way `synthesis` is what writing it up cost. Both are the numbers
    the model that answered reported, so a turn's whole prompt bill is
    this line plus the pieces under it.
    """
    directive = round_.directive
    subtasks = getattr(directive, "subtasks", ()) or ()
    return {"schema": SCHEMA,
            "ask": round_.ask,
            "action": getattr(directive, "action", ""),
            "subtasks": [{"id": s.id, "task": s.task, "target": s.target}
                         for s in subtasks],
            "pieces": [piece(r) for r in round_.results],
            "planning": dict(planning or {}),
            "synthesis": dict(round_.synthesis or {}),
            "protocol_failures": int(round_.protocol_failures),
            "fell_back": bool(round_.fell_back),
            "answered_directly": bool(round_.answered_directly),
            "rounds": int(round_.rounds),
            "reply_words": len(round_.text.split()),
            "t_start": round(float(round_.t_start), 3),
            "t_end": round(float(round_.t_end), 3),
            "seconds": round(round_.seconds, 3),
            "route": dict(route or {})}


def append(session_dir, rec):
    """File one round. The line is built whole and written once, so a
    concurrent reader sees either all of it or none of it."""
    with open(trace_path(session_dir), "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
    return rec


def blank_summary():
    return {"turns": 0, "pieces": 0, "delegated": 0, "failed": 0,
            "protocol_failures": 0, "direct": 0, "seconds": 0.0}


def tally(summary, rec):
    """Fold one round into a running summary. Takes a trace record
    rather than a Round, so a resumed session and a live one are counted
    by the same arithmetic."""
    pieces = rec.get("pieces") or []
    ran = [p for p in pieces if p.get("ran")]
    summary["turns"] += 1
    summary["pieces"] += len(pieces)
    summary["delegated"] += len(ran)
    summary["failed"] += sum(1 for p in ran if p.get("status") != "ok")
    summary["protocol_failures"] += int(rec.get("protocol_failures") or 0)
    summary["direct"] += bool(rec.get("answered_directly"))
    summary["seconds"] += max(0.0, float(rec.get("seconds") or 0.0))
    return summary


def summarize(rounds):
    summary = blank_summary()
    for rec in rounds:
        tally(summary, rec)
    return summary


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
    """Every agent turn this session has taken, and what could not be
    read."""
    path = trace_path(session_dir)
    rounds, warnings = [], []
    for n, rec in _scan(path):
        if rec is None:
            warnings.append(f"{path} line {n} is unreadable and was skipped "
                            f"(a round that did not finish writing).")
            continue
        if rec.get("schema") != SCHEMA:
            warnings.append(f"{path} line {n} was written as "
                            f"{rec.get('schema')!r}, which this version of "
                            f"salt does not read, and was skipped.")
            continue
        rounds.append(rec)
    return Trace(rounds=tuple(rounds), warnings=tuple(warnings))
