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

from dataclasses import dataclass

from salt.agents import protocol
from salt.agents.delegate import close_quietly
from salt.agents.roster import GUIDED_CAPABLE, GUIDED_PLAIN

ASK_HEADER = "ASK: "
# a plan is a decision rather than prose: the same session asked the same
# thing twice should decide the same way
PLANNING_GEN = {"temperature": 0.0}
MAIN_LABEL = "the chat model"


class OrchestratorError(Exception):
    """A round could not be planned at all, for want of a model."""


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
