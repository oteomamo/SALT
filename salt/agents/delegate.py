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
# only used to size the FIRST cut when a context has to be trimmed; the
# loop that follows measures, so a wrong guess costs a pass, not accuracy
TOKENS_PER_WORD = 1.6
# ok: the worker answered. timeout: it went quiet mid-reply and is still
# usable. dead: it is not answering at all. aborted: the person asking
# changed their mind. error: everything else, including a server that
# rejected the request
STATUSES = ("ok", "timeout", "dead", "aborted", "error")
# how many times a severing close is retried against arriving interrupts
CLOSE_ATTEMPTS = 3


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
    a task that needs none is still a legal delegation. Two ceilings can
    bound it, the session's own memory cap and the offload cap, and
    whichever bites first wins.
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
    caps = [c for c in (memory_word_cap(state, query),
                        getattr(state, "offload_context_cap", None)) if c]
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
                               max_words=min(caps) if caps else None,
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


def window_max_tokens(runner):
    """A reply cap taken from the worker's own window, for a roster entry
    that names none. input_budget_for never lets reply headroom exceed
    half the window, so half is the most the window can be asked for
    without squeezing the prompt out of it."""
    limit = getattr(runner, "max_input_len", None)
    return int(limit) // 2 if limit else None


def call_overrides(entry, req, runner=None):
    """Generation settings for this call: the roster entry's, with the
    request's max_tokens winning where it names one, and the worker's
    window as the backstop when neither does."""
    over = {}
    max_new = entry.max_tokens if req.max_tokens is None else req.max_tokens
    if max_new is None and runner is not None:
        max_new = window_max_tokens(runner)
    if max_new is not None:
        over["max_new_tokens"] = int(max_new)
    if entry.temperature is not None:
        over["temperature"] = float(entry.temperature)
    return over


def count_tokens(runner, text):
    """`text` measured with the worker's own tokenizer, None when it
    cannot be measured. A budget nobody can compute is not a budget, so
    every caller treats None as 'no trimming'."""
    tokenizer = getattr(runner, "tokenizer", None)
    if tokenizer is None:
        return None
    try:
        return len(tokenizer(text, add_special_tokens=False).input_ids)
    except Exception:
        return None


def fit_messages(runner, messages, req, overrides=None):
    """Trim the head of the context until the whole task fits the worker.

    The serve client tail-truncates tokens when a prompt overflows, which
    keeps the task but can cut the instructions off and leave the context
    starting mid-word. Trimming here instead hands the worker something
    coherent: whole words leave the front of the context, and the task
    and the instructions are never touched. A delegation that lost its
    task would come back answering the wrong question with confidence.

    Returns the messages to send and a one-line note, empty when nothing
    was trimmed.
    """
    budget = None
    if hasattr(runner, "input_budget"):
        budget = runner.input_budget((overrides or {}).get("max_new_tokens"))
    if not budget:
        return messages, ""
    body = messages[-1]["content"]
    tail = f"{TASK_HEADER}{req.task}"
    total = count_tokens(runner, messages[0]["content"] + body)
    if total is None or total <= budget or not body.endswith(tail):
        return messages, ""
    words = body[:-len(tail)].split()
    # words are cheaper to count than to tokenize, so cut an estimated
    # slice first and close the remaining gap by measurement
    drop = min(len(words), int((total - budget) / TOKENS_PER_WORD) + 1)
    while True:
        head = " ".join(words[drop:])
        candidate = f"{head}\n\n{tail}" if head else tail
        size = count_tokens(runner, messages[0]["content"] + candidate)
        if size is None or size <= budget or drop >= len(words):
            break
        drop = min(len(words), drop + max(1, (len(words) - drop) // 4))
    messages = messages[:-1] + [{"role": "user", "content": candidate}]
    note = (f"context trimmed to fit the worker's window: "
            f"{len(words) - drop} of {len(words)} words kept, "
            f"the task in full")
    return messages, note


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


def close_quietly(stream, attempts=CLOSE_ATTEMPTS):
    """Sever the response even while Ctrl-C is still arriving.

    Closing the stream is what aborts the request on the worker, so a
    second interrupt landing during the cleanup must not be what leaves
    a worker generating into a connection nobody is reading. Bounded on
    purpose: after a few tries the interrupt is what the caller wants
    more than the tidy close.
    """
    for _ in range(attempts):
        try:
            stream.close()
            return True
        except KeyboardInterrupt:
            continue
    return False


def failure_status(handle, exc):
    """Which kind of failure this was, from the state the worker is in
    rather than from the wording of the message."""
    if is_read_timeout(getattr(exc, "__cause__", None)):
        return "timeout"
    return "dead" if handle.state == DEAD else "error"


def delegate(state, req, context=None):
    """Send `req` to its worker and wait for the whole reply.

    Blocking: the stream is consumed to completion before this returns,
    so the caller gets an answer rather than a generator. Every way this
    can end short comes back as a DelegationResult with a status, never
    as an exception, so one worker having a bad day cannot end the turn
    that asked it, and an interrupted delegation is still recorded as
    one that happened rather than one the session never knew about.
    """
    if not req.target:
        raise DelegationError(
            "a delegation needs a worker to send to, and this request "
            "names none")
    handle = state.worker(req.target)
    ctx = build_context(state, req) if context is None else context
    messages = build_messages(ctx, req)
    # opened here rather than left to handle.call, because the window and
    # the tokenizer that size this call live on the runner. A worker that
    # will not open is left to call() below, which reports why
    runner = handle.opened()
    overrides = call_overrides(handle.entry, req, runner)
    if runner is not None:
        messages, note = fit_messages(runner, messages, req, overrides)
        if note:
            print(f"  {note}")
    state.delegation_seq += 1
    t_start = time.time()
    pieces, status, error = [], "ok", ""
    prior = _apply_request_timeout(handle, req)
    # held in a name rather than left to the for loop, so Ctrl-C closes it
    # here and now: closing the generator is what severs the response and
    # aborts the request on the worker
    stream = handle.call(messages, **overrides)
    try:
        for piece in stream:
            pieces.append(piece)
    except WorkerError as exc:
        status, error = failure_status(handle, exc), str(exc)
    except KeyboardInterrupt:
        # not re-raised: the interrupt has already done what it was for.
        # Closing the stream below severs the response and aborts the
        # request on the worker, so the caller is back at the prompt with
        # the worker free, and the delegation is recorded as abandoned
        # rather than vanishing from the session's history.
        status = "aborted"
        error = "the delegation was interrupted"
    except Exception as exc:
        status = failure_status(handle, exc)
        error = f"{type(exc).__name__}: {exc}"
    finally:
        close_quietly(stream)
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
