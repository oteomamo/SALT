# -*- coding: utf-8 -*-
"""What a decision about planning a turn is allowed to read.

The switch seam decides HOW a turn selects its memory. This one decides
WHETHER a turn is planned out over helper models at all, and what that
plan may spend. They are siblings rather than one thing: a planned turn
is three or more model calls where a plain turn is one, so routing every
line is the wrong default and routing none wastes the layer.

A route signal set is the snapshot plus sixteen names of its own, in one
flat namespace, so a rule reads `n_turns` and `ask_words` the same way
and nobody writes either down twice. It deliberately does not grow
`snapshot.KEYS`: that set is a shipped contract the MCP server answers
with, and it is asserted equal to what a SWITCH rule may read, so adding
route names there would quietly widen a different seam.

SIX OF THESE SIGNALS ARE DOWNSTREAM OF ROUTING'S OWN ACTION, and they
are marked. A rule that reads one without also constraining
`turns_since_round` does not merely stop firing, it freezes: "the last
round took too long, so do not plan" is true once, and then no round
runs, so the number never updates and the rule is answered forever by a
number from an hour ago. `turns_since_round` exists to qualify such a
rule, and the live firing census exists to notice when one has stopped
saying anything.
"""

from salt.agents.snapshot import KEYS as SNAPSHOT_KEYS
from salt.agents.snapshot import snapshot

SCHEMA = "salt-route-signals/1"

# what the person actually asked, which is the one family that is not
# downstream of anything this decides
ASK_KEYS = ("ask_words", "ask_questions", "ask_sentences", "ask_lines",
            "ask_list_items", "ask_names_worker")
# who is available to help, read off the roster the session already holds
ROSTER_KEYS = ("n_workers", "n_workers_ready", "n_workers_busy",
               "worker_kinds")
# what the last round did. Everything here is a consequence of a routing
# decision, so a rule reading one of them is reading its own output
FEEDBACK_KEYS = ("rounds_taken", "turns_since_round", "last_round_s",
                 "last_round_pieces", "last_round_answered",
                 "last_round_direct")
# `turns_since_round` is the exception that makes the rest usable: it
# counts turns rather than describing a round, so it keeps moving when
# nothing is being planned, which is exactly when a frozen rule needs
# something that still changes
CLOSED_LOOP = tuple(k for k in FEEDBACK_KEYS if k != "turns_since_round")

ROUTE_KEYS = ASK_KEYS + ROSTER_KEYS + FEEDBACK_KEYS
SIGNALS = SNAPSHOT_KEYS + ROUTE_KEYS

# a line that offers a helper a piece of the work by name, which is a
# person routing the turn themselves
_AT_NAME = "@"
# what reads as one item of a list somebody wrote out
_BULLETS = ("-", "*", "+", "•")


def _list_item(line):
    line = line.strip()
    if not line:
        return False
    if line[0] in _BULLETS:
        return True
    head = line.split(".", 1)[0].split(")", 1)[0]
    return head.isdigit() and len(head) <= 2 and len(head) < len(line)


def ask_signals(ask):
    """The question itself, measured. Somebody who wants three things
    tends to write three lines, which is a more honest reading of "this
    is several pieces of work" than the length of the sentence."""
    text = ask or ""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    sentences = [s for s in text.replace("!", ".").replace("?", ".").split(".")
                 if s.strip()]
    return {"ask_words": len(text.split()),
            "ask_questions": text.count("?"),
            "ask_sentences": len(sentences),
            "ask_lines": len(lines),
            "ask_list_items": sum(1 for ln in lines if _list_item(ln)),
            "ask_names_worker": _AT_NAME in text}


def _kind(entry):
    """What a worker is FOR, as the planner is shown it. The notes are
    the whole lever: an orchestrator concentrates on one helper unless
    the roster describes disjoint jobs, so a roster of clones never fans
    out however the ask is phrased."""
    return " ".join((entry.notes or entry.alias or entry.name).lower().split())


def roster_signals(state):
    """Who is there to help, without opening anything that is not open.

    A handle is asked what it already knows. Opening a client to answer
    a signal would make building the signals cost a connection, and a
    policy that has not decided anything yet has not earned one.
    """
    from salt.agents.worker import BUSY
    roster = getattr(state, "roster", None)
    workers = list(getattr(roster, "workers", ()) or ())
    ready, busy, kinds = 0, 0, set()
    for entry in workers:
        handle = state.worker(entry.name)
        if handle.opened() is not None:
            ready += 1
            kinds.add(_kind(entry))
        if handle.state == BUSY:
            busy += 1
    return {"n_workers": len(workers),
            "n_workers_ready": ready,
            "n_workers_busy": busy,
            "worker_kinds": len(kinds)}


def round_signals(state):
    """What the last round did, or None everywhere when there has been
    none. None means "this session cannot say", never zero, so a rule
    about a slow round does not fire on a session that has never had
    one."""
    summary = getattr(state, "agent_stats", None) or {}
    last = getattr(state, "last_round", None)
    at = getattr(state, "last_round_turn", None)
    turns = getattr(getattr(state, "trie", None), "n_turns", None)
    since = None if (at is None or turns is None) else max(0, turns - at)
    if last is None:
        return {"rounds_taken": int(summary.get("turns") or 0),
                "turns_since_round": since,
                "last_round_s": None, "last_round_pieces": None,
                "last_round_answered": None, "last_round_direct": None}
    return {"rounds_taken": int(summary.get("turns") or 0),
            "turns_since_round": since,
            "last_round_s": last.seconds,
            "last_round_pieces": len(last.delegated),
            "last_round_answered": len(last.answered),
            "last_round_direct": bool(last.answered_directly)}


def route_signals(state, ask, stats=None):
    """This session and this question, as the closed set a route
    decision may read. The snapshot first, in its own order, so the two
    schemas can be told apart by reading rather than by trusting."""
    out = dict(snapshot(state, stats))
    out.update(ask_signals(ask))
    out.update(roster_signals(state))
    out.update(round_signals(state))
    assert tuple(out) == SIGNALS, "the route signals grew a key nothing pinned"
    return out
