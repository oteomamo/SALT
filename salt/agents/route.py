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

from dataclasses import dataclass, field, replace

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


class RouteError(Exception):
    """A route decision asked for something a turn cannot give it."""


# what a decision may set, and what each one has to be. Closed, so a
# decision naming anything else is refused by name rather than ignored
FIELDS = {"plan": bool, "max_pieces": int, "max_wall_s": (int, float),
          "rounds": int, "targets": list, "why": str}


@dataclass(frozen=True)
class RouteDecision:
    """Whether to plan this turn out, and what the plan may spend.

    Every field is None where the decision has no opinion, which is the
    normal answer. None means "leave the session's own flag alone", and
    it is told apart from a zero that would mean something quite else.
    """

    plan: bool = None
    max_pieces: int = None
    max_wall_s: float = None
    rounds: int = None
    targets: tuple = None
    why: str = ""

    @property
    def quiet(self):
        """Whether this decision changes anything at all."""
        return all(getattr(self, name) is None for name in FIELDS
                   if name != "why")


@dataclass(frozen=True)
class Ceiling:
    """What the session's own flags already allow, which is the most a
    decision can be permitted to reach. A route policy exists to spend
    less than a person allowed, never more: whatever a model or a rules
    file proposes, the flags stay the ceiling."""

    max_pieces: int = 4
    max_wall_s: float = 600.0
    rounds: int = 1
    targets: tuple = field(default_factory=tuple)
    ready: tuple = field(default_factory=tuple)


def ceiling(state):
    """The session's own flags and roster as a ceiling."""
    roster = getattr(state, "roster", None)
    workers = tuple(e.name for e in getattr(roster, "workers", ()) or ())
    ready = tuple(name for name in workers
                  if state.worker(name).opened() is not None)
    return Ceiling(max_pieces=getattr(state, "agent_max_delegations", 4),
                   max_wall_s=getattr(state, "agent_max_wall", 600.0),
                   rounds=getattr(state, "agent_rounds", 1),
                   targets=workers, ready=ready)


def check(proposal):
    """A proposal as a RouteDecision, or a refusal naming what is wrong.

    Names and types only. Whether a number is reachable is the guard's
    question, and it is a different question: a decision can be
    perfectly well formed and still ask for more than the session
    allows, and that one is clamped rather than refused.
    """
    if isinstance(proposal, RouteDecision):
        return proposal
    if not isinstance(proposal, dict):
        raise RouteError(f"a route policy answers with a dict, and this one "
                         f"answered with {type(proposal).__name__}")
    unknown = [name for name in proposal if name not in FIELDS]
    if unknown:
        raise RouteError(f"a route decision named {unknown}, which is not "
                         f"something it can set. Allowed: "
                         f"{', '.join(FIELDS)}")
    out = {}
    for name, value in proposal.items():
        if value is None:
            continue
        want = FIELDS[name]
        # bool is an int in Python, so a number written where a yes or no
        # belongs would be honoured as one
        if want is bool and not isinstance(value, bool):
            raise RouteError(f"{name} is a yes or no, and this decision "
                             f"gave {value!r}")
        if want is not bool and isinstance(value, bool):
            raise RouteError(f"{name} is not a yes or no, and this decision "
                             f"gave {value!r}")
        if want is list:
            if not isinstance(value, (list, tuple)) or not all(
                    isinstance(v, str) and v for v in value):
                raise RouteError(f"{name} is a list of worker names, and "
                                 f"this decision gave {value!r}")
            out[name] = tuple(value)
            continue
        if not isinstance(value, want):
            raise RouteError(f"{name} must be "
                             f"{getattr(want, '__name__', 'a number')}, and "
                             f"this decision gave {value!r}")
        out[name] = value
    return RouteDecision(**out)


def guard(decision, ceiling_, why=None):
    """A checked decision made safe to act on, and the reasons it moved.

    Clamps DOWN and never up. Refuses to plan when nobody is ready or
    when the pieces have been clamped to none, drops a target the roster
    does not carry, and never raises: a route policy having a bad turn
    costs the turn nothing, exactly as a round having one does.
    """
    notes = [] if why is None else why
    out = decision
    for name in ("max_pieces", "max_wall_s", "rounds"):
        asked = getattr(out, name)
        allowed = getattr(ceiling_, name)
        if asked is None:
            continue
        if asked > allowed:
            notes.append(f"{name} {asked} is more than this session allows, "
                         f"so {allowed} stands")
            out = replace(out, **{name: allowed})
        elif asked < 0:
            notes.append(f"{name} cannot be {asked}, so {allowed} stands")
            out = replace(out, **{name: allowed})
    if out.targets is not None:
        keep = tuple(t for t in out.targets if t in ceiling_.ready)
        dropped = [t for t in out.targets if t not in ceiling_.ready]
        if dropped:
            notes.append(f"this session has no ready worker called "
                         f"{dropped}, so it was left out")
        out = replace(out, targets=keep)
        if not keep:
            notes.append("the decision named no worker this session has, "
                         "so the turn is not planned")
            out = replace(out, plan=False)
    if out.plan and not ceiling_.ready:
        notes.append("no worker is ready, so there is nothing to plan with")
        out = replace(out, plan=False)
    if out.plan and out.max_pieces == 0:
        notes.append("a plan with no pieces in it is a plain turn, and "
                     "costs a call to say so")
        out = replace(out, plan=False)
    if out.rounds is not None and out.rounds < 1:
        out = replace(out, rounds=1)
    return out


class RoutePolicy:
    """The seam. One question per turn, answered with a decision.

    `decide(signals)` returns what this turn should do, and returning
    nothing at all is the normal answer. The session's own flags are the
    starting point every turn, so a decision cannot accumulate.
    """

    name = "policy"
    # whether asking is worth the signals it costs. A policy that never
    # decides anything is not asked, and building the signals is what
    # asking costs
    decides = True

    def decide(self, signals):
        return RouteDecision()

    def bind(self, state):
        return self

    def explain(self):
        """Why the last decision came out as it did, one entry per
        reason. Empty from a policy with no reasons to give."""
        return ()


class NullRoute(RoutePolicy):
    """Nobody decides. The session's own flags, every turn."""

    name = "none"
    decides = False
