# -*- coding: utf-8 -*-
"""Deciding what a turn needs, before anything is done about it.

Planning is one question put to one model: given what this session
remembers and what was just asked, answer it outright or name the pieces
of it and the helper each piece goes to. The reply is read as a
directive, and a reply that is not one costs a repair and then stops
costing anything, because a model that would not plan still said
something and what it said is an answer.

Nothing here touches the session. The memory block arrives already
built, no coverage is committed, no worker is called, no file is
written. A round that planned badly leaves exactly as much behind as one
that never planned, which is what lets the caller decide what a
directive is worth after seeing it rather than before.

Version 1 plans with the session's own chat model. A schema cannot be
demanded through the chat seam, which carries generation settings and
nothing else, so that model is planned for as one that will not be held
to a schema however capable the server behind it is: it is shown a
worked directive instead of being handed one to fill.
"""

import time
from dataclasses import dataclass

from salt.agents import protocol
from salt.agents.delegate import (DelegationRequest, DelegationResult,
                                  close_quietly, delegate)
from salt.agents.roster import GUIDED_CAPABLE, GUIDED_PLAIN, RosterError

ASK_HEADER = "ASK: "
# a plan is a decision rather than prose: the same session asked the same
# thing twice should decide the same way
PLANNING_GEN = {"temperature": 0.0}
MAIN_LABEL = "the chat model"
REFUSED = "refused"
STOPPED = "stopped"


class OrchestratorError(Exception):
    """A round could not be planned at all, for want of a model."""


@dataclass(frozen=True)
class AgentLimits:
    """What one round of delegating is allowed to cost.

    Three ceilings, each of which stops the round rather than the
    subtask that crossed it: what has been answered already is worth
    keeping, and the pieces that never ran say so in their own results.

    Tokens are counted as what the helpers generated. A prompt is mostly
    context this session would have selected anyway, and counting it
    would make one large conversation spend the whole round's allowance
    on its first subtask.
    """

    max_delegations_per_turn: int = 4
    max_wall_s: float = 600.0
    max_total_delegated_tokens: int = 8192
    # one round of delegation and no deeper: a worker cannot start a
    # round of its own. Version 1 refuses anything else rather than
    # pretending to enforce a depth it does not implement
    depth: int = 1


@dataclass(frozen=True)
class Endpoint:
    """The model a round plans with, and what it can be held to."""

    label: str
    send: object
    capability: str = GUIDED_PLAIN
    model_id: str = None

    @property
    def guided(self):
        return self.capability == GUIDED_CAPABLE


def main_runner_send(state):
    """One prompt put to the session's chat model, waited out in full.

    ``guided`` is accepted and ignored. Nothing carries a schema through
    the chat seam, and a capability that cannot be exercised is one the
    round must not plan around.
    """

    def send(messages, guided=False):
        pieces = []
        # held in a name rather than left to the loop, so an interrupt
        # closes it here: closing the generator is what stops a model
        # that is still generating
        stream = state.runner.stream_chat(messages, **PLANNING_GEN)
        try:
            for piece in stream:
                pieces.append(piece)
        finally:
            close_quietly(stream)
        return "".join(pieces)

    return send


def orchestrator_endpoint(state):
    """Which model plans this turn, or None when the session has none.

    One answer for now, the session's own chat model. The shape is here
    rather than inline because a roster orchestrator is the next thing
    to arrive, and when it does this function is what changes.
    """
    runner = getattr(state, "runner", None)
    if runner is None:
        return None
    cfg = getattr(runner, "cfg", None) or {}
    return Endpoint(label=getattr(runner, "alias", None) or MAIN_LABEL,
                    send=main_runner_send(state),
                    capability=GUIDED_PLAIN,
                    model_id=cfg.get("hf_id"))


def targets_for(state):
    """The helpers a plan may name, spelled the way it has to spell them.

    Every worker the roster carries, running or not. A plan is a decision
    about the work, and a worker that turns out to be down fails its own
    subtask with a reason the round can report, which is worth more than
    quietly narrowing what the session was willing to consider.
    """
    roster = getattr(state, "roster", None)
    if roster is None:
        return ()
    return tuple((e.name, e.notes or e.alias) for e in roster.workers)


def planning_messages(capability, ask, memory_block="", targets=()):
    """The two messages a planning call is: the standing instructions,
    then this session's memory with the question under it.

    The shape a worker is sent, for the same reason. The head is
    byte-stable so a prefix cache can hold it, and everything that
    changes from turn to turn sits below it.
    """
    body = f"{ASK_HEADER}{ask}"
    if memory_block:
        body = f"{memory_block}\n\n{body}"
    return [{"role": "system",
             "content": protocol.orchestrator_instructions(capability,
                                                           targets)},
            {"role": "user", "content": body}]


def plan(state, ask, memory_block="", endpoint=None):
    """What this turn should do, decided once.

    Returns the protocol's outcome rather than the bare directive: a
    round that has to say how it went needs to know whether the model
    was repaired into a directive or whether what came back is simply
    what it would have said anyway.
    """
    endpoint = orchestrator_endpoint(state) if endpoint is None else endpoint
    if endpoint is None:
        raise OrchestratorError(
            "this session has no chat model, so there is nothing to plan "
            "the turn with")
    messages = planning_messages(endpoint.capability, ask, memory_block,
                                 targets_for(state))
    return protocol.ask_directive(endpoint.send, messages,
                                  guided=endpoint.guided)


def not_run(subtask, status, why):
    """A subtask that never reached a worker, as a result all the same.

    The round has to be able to say what it did not do. A missing entry
    would be indistinguishable from a subtask the plan never had, and
    the synthesis that follows must be able to tell the model which
    pieces of its own plan came back empty. No id: ids are handed out by
    delegations that happened.
    """
    now = time.time()
    return DelegationResult(id=0, target=subtask.target, task=subtask.task,
                            status=status, error=why, t_start=now, t_end=now)


def delegated_tokens(result):
    return int((result.usage or {}).get("output_tokens") or 0)


def stop_reason(limits, results, spent, started):
    """Why this round should stop before the next subtask, or None.

    Asked before each one rather than after, so a cap that has been
    reached costs nothing further, and the subtask that would have
    crossed it is reported as stopped rather than half done.
    """
    ran = [r for r in results if r.ran]
    if any(r.status == "aborted" for r in ran):
        return "the round was interrupted"
    if len(ran) >= limits.max_delegations_per_turn:
        return (f"this turn's limit of {limits.max_delegations_per_turn} "
                f"delegations is used up")
    if spent >= limits.max_total_delegated_tokens:
        return (f"the helpers have generated {spent} tokens for this turn, "
                f"which is its budget of {limits.max_total_delegated_tokens}")
    waited = time.time() - started
    if waited >= limits.max_wall_s:
        return (f"this turn has spent {waited:.0f}s delegating, which is "
                f"its limit of {limits.max_wall_s:.0f}s")
    return None


def request_timeout(state, entry):
    """How long a subtask's worker may go quiet. The roster's number is a
    fact about that model, so the session's own timeout stands in only
    for workers nothing was said about."""
    if entry.timeout_s is not None:
        return None
    return getattr(state, "offload_timeout", None)


def subtask_request(state, subtask, entry):
    """One subtask as a delegation this session can send."""
    return DelegationRequest(task=subtask.task, target=subtask.target,
                             context_query=subtask.query,
                             budget_pct=subtask.budget_pct,
                             max_tokens=subtask.max_tokens,
                             ingest=bool(getattr(state, "offload_ingest",
                                                 False)),
                             timeout_s=request_timeout(state, entry))


def worker_entry(state, name):
    """The roster entry a subtask names, refused by name when the plan
    invented one or reached for the model doing the planning."""
    entry = state.worker(name).entry
    if entry.role != "worker":
        raise RosterError(f"{name!r} is this roster's {entry.role}, not a "
                          f"worker a task can be handed to.")
    return entry


def execute(state, directive, limits=None, on_result=None):
    """Run a directive's subtasks in the order it put them.

    One at a time, on this thread, because the trie is read on it: every
    subtask's context is selected, sent and answered before the next one
    is looked at. What comes back is a result per subtask in plan order,
    always the same length as the plan, so a round can account for every
    piece of what it decided to do.

    Nothing here ends the round early by raising. A worker that does not
    exist, one that is down, a cap that has been reached and an
    interrupted call are each a result with a status on it, and the
    synthesis that follows is told about all of them.
    """
    limits = AgentLimits() if limits is None else limits
    if limits.depth != 1:
        raise OrchestratorError(
            f"this salt delegates one round deep, and these limits ask for "
            f"{limits.depth}")
    results, spent, started = [], 0, time.time()
    for subtask in directive.subtasks:
        reason = stop_reason(limits, results, spent, started)
        if reason:
            result = not_run(subtask, STOPPED, reason)
        else:
            try:
                entry = worker_entry(state, subtask.target)
            except RosterError as exc:
                result = not_run(subtask, REFUSED, str(exc))
            else:
                result = delegate(state, subtask_request(state, subtask,
                                                         entry))
                spent += delegated_tokens(result)
        results.append(result)
        if on_result is not None:
            on_result(result)
    return results
