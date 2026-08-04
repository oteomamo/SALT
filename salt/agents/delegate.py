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

from dataclasses import dataclass, field
from pathlib import Path

INSTRUCTIONS_PATH = Path(__file__).resolve().parent / "worker_instructions.md"
FALLBACK_INSTRUCTIONS = (
    "Answer the task on the line beginning 'TASK:' using the context above "
    "it, which is a partial selection of excerpts rather than a full "
    "transcript. Say which part is missing when the context does not cover "
    "the task, do not invent material, and return the answer itself with no "
    "preamble.")


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
