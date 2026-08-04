# -*- coding: utf-8 -*-
"""Handing one task to a roster worker.

A delegation is a task plus the memory the session would have selected
for it, sent to a model beside the chat model. The context comes from the
same compressed selection a chat turn builds, so a worker reads the
conversation the way the chat model does, and building it changes
nothing: coverage, the verbatim tail and the trie are exactly as they
were. A worker is shown a snapshot, and the session's own memory never
learns that it was.

Tail exclusion is deliberately off here. The chat model already sees the
recent turns verbatim, so selecting them again would spend its budget
twice; a worker sees only what it is handed, which makes a tail-resident
sentence ordinary context for it.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from salt.agents.worker import DEAD, WorkerError, is_read_timeout

INSTRUCTIONS_PATH = Path(__file__).resolve().parent / "worker_instructions.md"
FALLBACK_INSTRUCTIONS = (
    "Answer the task on the line beginning 'TASK:' using the context above "
    "it, which is a partial selection of excerpts rather than a full "
    "transcript. Say which part is missing when the context does not cover "
    "the task, do not invent material, and return the answer itself with no "
    "preamble.")
TASK_HEADER = "TASK: "
# ok: the worker answered. timeout: it went quiet mid-reply and is still
# usable. dead: it is not answering at all. error: everything else,
# including a server that rejected the request
STATUSES = ("ok", "timeout", "dead", "error")


class DelegationError(Exception):
    """A delegation could not be sent (no target, no such worker)."""


@dataclass(frozen=True)
class DelegationRequest:
    """One task to hand over, and how to build the context for it."""

    task: str
    target: str = None
    context_query: str = None
    budget_pct: float = None
    max_tokens: int = None
    ingest: bool = False
    timeout_s: float = None

    @property
    def query(self):
        """What the context is selected for: the task itself, unless the
        caller knows a better line to search the conversation with."""
        return self.context_query or self.task


@dataclass(frozen=True)
class DelegationContext:
    """The memory a worker is handed, and what it cost to select."""

    text: str = ""
    selected_idx: tuple = ()
    stats: dict = field(default_factory=dict)

    @property
    def n_selected(self):
        return len(self.selected_idx)

    @property
    def words_used(self):
        return self.stats.get("words_used", 0)

    @property
    def empty(self):
        return not self.text


@dataclass(frozen=True)
class DelegationResult:
    """What came back from one delegation, answered or not."""

    id: int
    target: str
    task: str
    status: str
    text: str = ""
    error: str = ""
    usage: dict = field(default_factory=dict)
    context: DelegationContext = None
    t_start: float = 0.0
    t_end: float = 0.0

    @property
    def ok(self):
        return self.status == "ok"

    @property
    def seconds(self):
        return max(0.0, self.t_end - self.t_start)


def worker_instructions():
    """The worker's system prompt. Re-read per delegation like the chat
    model's own instructions, so the wording can be tuned live, and
    tolerant of a broken file: a delegation must not die over it."""
    try:
        text = INSTRUCTIONS_PATH.read_text(encoding="utf-8").strip()
        return text or FALLBACK_INSTRUCTIONS
    except (OSError, ValueError):
        return FALLBACK_INSTRUCTIONS


def build_context(state, req):
    """Select this session's memory for `req` without committing any of it.

    Returns a DelegationContext, empty when the session has no memory yet:
    a task that needs none is still a legal delegation.
    """
    # imported here, not at module load: the chat layer carries the encoder
    # stack, and importing the agent layer has to stay free
    from salt.chat.cli import (format_memory_block, memory_word_cap,
                               report_ingest_failures)
    # the ingest worker owns add_turn on its own thread, and in async mode a
    # turn submits its user line before generation, so a delegation raised
    # mid-turn would read the trie while that write is still in flight
    report_ingest_failures(state.ingest.drain())
    if state.trie.n_sentences == 0:
        return DelegationContext()
    query = req.query
    budget = state.budget if req.budget_pct is None else req.budget_pct
    comp = state.trie.compress(query=query, budget_pct=budget,
                               tokenizer=state.bge_tok,
                               model=state.bge_model,
                               device=state.bge_device,
                               coverage_half_life=state.coverage_half_life,
                               coverage_decay_docs=state.coverage_decay_docs,
                               shift_damping=state.shift_damping,
                               shift_margin=state.shift_margin,
                               shift_query_boost=state.shift_query_boost,
                               per_source_themes=state.per_source_themes,
                               max_words=memory_word_cap(state, query),
                               stable_keys=state.stable_coverage_keys,
                               coverage_gc=state.coverage_gc,
                               coverage_max_keys=state.coverage_max_keys,
                               defer_commit=True,
                               exclude_sent_idx=None)
    # the commit is deliberately dropped on the floor: dropping it is what
    # keeps a delegation invisible to the session's own memory
    selected = comp["selected_sent_idx"]
    text = format_memory_block(state.trie, selected, state.turn_labels,
                               state.conversation_map)
    return DelegationContext(text=text, selected_idx=tuple(selected),
                             stats=dict(comp["stats"]))


def build_messages(context, req):
    """The two messages a worker is sent: its standing instructions, then
    the context with the task under it. A task with no context is still a
    legal delegation, so the header carries the whole message then."""
    body = f"{TASK_HEADER}{req.task}"
    if context.text:
        body = f"{context.text}\n\n{body}"
    return [{"role": "system", "content": worker_instructions()},
            {"role": "user", "content": body}]


def call_overrides(entry, req):
    """Generation settings for this call: the roster entry's, with the
    request's max_tokens winning where it names one."""
    over = {}
    max_new = entry.max_tokens if req.max_tokens is None else req.max_tokens
    if max_new is not None:
        over["max_new_tokens"] = int(max_new)
    if entry.temperature is not None:
        over["temperature"] = float(entry.temperature)
    return over


def output_tokens(runner, text):
    """The reply measured with the worker's own tokenizer. Never raises:
    a usage number is worth less than the answer it describes."""
    tokenizer = getattr(runner, "tokenizer", None)
    if not text or tokenizer is None:
        return 0
    try:
        return len(tokenizer(text, add_special_tokens=False).input_ids)
    except Exception:
        return None


def call_usage(handle, text):
    """What this call cost, read off the worker's client afterwards."""
    runner = handle.runner
    stats = getattr(runner, "last_engine_stats", None) or {}
    prompt = stats.get("apc_prompt_tokens")
    if prompt is None:
        prompt = getattr(runner, "last_prompt_tokens", None)
    return {"prompt_tokens": prompt,
            "cached_tokens": stats.get("apc_cached_tokens"),
            "output_tokens": output_tokens(runner, text)}


def failure_status(handle, exc):
    """Which kind of failure this was, from the state the worker is in
    rather than from the wording of the message."""
    if is_read_timeout(getattr(exc, "__cause__", None)):
        return "timeout"
    return "dead" if handle.state == DEAD else "error"


def delegate(state, req, context=None):
    """Send `req` to its worker and wait for the whole reply.

    Blocking: the stream is consumed to completion before this returns,
    so the caller gets an answer rather than a generator. Failures come
    back as a DelegationResult with a status, never as an exception, so
    one worker having a bad day cannot end the turn that asked it.
    """
    if not req.target:
        raise DelegationError(
            "a delegation needs a worker to send to, and this request "
            "names none")
    handle = state.worker(req.target)
    ctx = build_context(state, req) if context is None else context
    messages = build_messages(ctx, req)
    overrides = call_overrides(handle.entry, req)
    state.delegation_seq += 1
    t_start = time.time()
    pieces, status, error = [], "ok", ""
    prior = _apply_request_timeout(handle, req)
    try:
        for piece in handle.call(messages, **overrides):
            pieces.append(piece)
    except WorkerError as exc:
        status, error = failure_status(handle, exc), str(exc)
    except Exception as exc:
        status = failure_status(handle, exc)
        error = f"{type(exc).__name__}: {exc}"
    finally:
        _restore_timeout(handle, prior)
    text = "".join(pieces)
    return DelegationResult(id=state.delegation_seq, target=handle.name,
                            task=req.task, status=status, text=text,
                            error=error, usage=call_usage(handle, text),
                            context=ctx, t_start=t_start, t_end=time.time())


def _apply_request_timeout(handle, req):
    """Let one request go quiet for longer or shorter than the roster
    allows this worker. Returns what to put back."""
    if req.timeout_s is None:
        return None
    try:
        runner = handle.ready()
    except WorkerError:
        # opening failed, so the call below will fail too and say why
        return None
    prior = getattr(runner, "read_timeout", None)
    runner.read_timeout = req.timeout_s
    return runner, prior


def _restore_timeout(handle, prior):
    if prior is not None:
        runner, value = prior
        runner.read_timeout = value
