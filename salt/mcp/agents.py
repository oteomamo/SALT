# -*- coding: utf-8 -*-
"""Roster workers, reached over MCP.

A delegation hands one task and a conversation's memory to a smaller
model beside the one the client is using. The selection behind it is the
one saltChat builds for the same task: the same compression, the same
labels, and nothing committed, so a conversation is exactly as it was
after a delegation as before it.

The delegation path reads a saltChat session object. Rather than fork
it, an open MCP session is dressed as one here, which is what keeps a
delegation asked for over the protocol and one typed at the prompt the
same delegation. Worker handles live on the runtime instead of the
call, because a handle holds an open client and rebuilding it per call
would reconnect every time.

Nothing here runs off the main loop: a worker call is HTTP and the trie
is only ever read by the thread the tool call arrives on.
"""

import contextlib
import sys

from salt.agents.delegate import (DelegationContext, DelegationError,
                                  DelegationRequest, build_context, delegate)
from salt.agents.roster import RosterError

NO_ROSTER = ("this server reaches no workers - start salt-mcp with "
             "--roster FILE, e.g. salt/agents/roster_sample.json")
_DEFAULTS = None


def chat_defaults():
    """The launch defaults a saltChat session would have.

    Read off the chat parser rather than restated here, so a delegation
    over MCP selects memory exactly as a default session does and cannot
    drift from it as those defaults move.
    """
    global _DEFAULTS
    if _DEFAULTS is None:
        from salt.chat.cli import build_parser
        _DEFAULTS = build_parser().parse_args([])
    return _DEFAULTS


class DelegationState:
    """An open conversation dressed as the session a delegation reads.

    Without a session this is still legal: a task that needs no memory
    is a context-free delegation, and only the worker lookup and the id
    counter are read then.
    """

    def __init__(self, runtime, session=None):
        args = chat_defaults()
        engine = runtime.engine
        self.runtime = runtime
        self.session = session
        self.trie = session.trie if session is not None else None
        self.ingest = session.ingest if session is not None else None
        self.bge_tok = engine.tokenizer
        self.bge_model = engine.model
        self.bge_device = engine.device
        # there is no chat model in this process, so there is no window to
        # fit the block to: the budget is the only thing bounding it
        self.runner = None
        self.memory_cap = args.memory_cap
        self.tokens_per_word = 1.0
        self.budget = (self.trie.config.get("budget_pct_default")
                       if self.trie is not None else args.budget_pct)
        self.offload_context_cap = args.offload_context_cap
        self.coverage_half_life = args.coverage_half_life
        self.coverage_decay_docs = args.coverage_decay_docs
        self.shift_damping = args.shift_damping
        self.shift_margin = args.shift_margin
        self.shift_query_boost = args.shift_query_boost
        self.per_source_themes = args.per_source_themes
        self.stable_coverage_keys = args.stable_coverage_keys
        self.coverage_gc = args.coverage_gc
        self.coverage_max_keys = args.coverage_max_keys
        self.dedup_cos = args.dedup_cos
        self.max_sentences = args.max_sentences
        self.turn_labels = not args.no_turn_labels
        self.conversation_map = args.conversation_map

    @property
    def conversation_id(self):
        return self.session.conversation_id if self.session else ""

    def worker(self, name):
        return self.runtime.worker(name)

    @property
    def delegation_seq(self):
        return self.runtime.seq(self.conversation_id)

    @delegation_seq.setter
    def delegation_seq(self, value):
        # written through rather than held here: the state is per call and
        # the numbering belongs to the conversation
        self.runtime.set_seq(self.conversation_id, value)


class AgentRuntime:
    """The roster this server can reach, and the handles it has opened."""

    def __init__(self, engine, pool=None, roster=None):
        self.engine = engine
        self.pool = pool
        self.roster = roster
        self.workers = {}
        self._seq = {}

    @property
    def read_only(self):
        return bool(self.pool is not None and self.pool.read_only)

    def worker(self, name):
        from salt.agents.worker import WorkerHandle
        if self.roster is None:
            raise RosterError(NO_ROSTER)
        if name not in self.workers:
            self.workers[name] = WorkerHandle(self.roster.get(name))
        return self.workers[name]

    def handles(self):
        if self.roster is None:
            return []
        return [self.worker(e.name) for e in self.roster.entries]

    def target(self, name=None):
        """Which worker a task goes to. The prompt's rule: with one
        worker it needs no naming, with several the caller says which."""
        if self.roster is None:
            raise RosterError(NO_ROSTER)
        if name:
            return self.worker(name)
        workers = self.roster.workers
        if len(workers) == 1:
            return self.worker(workers[0].name)
        if not workers:
            raise RosterError(f"{self.roster.path} names no worker to "
                              f"delegate to")
        known = ", ".join(e.name for e in workers)
        raise RosterError(f"this roster has {len(workers)} workers, so name "
                          f"one in target (known: {known})")

    def seq(self, conversation_id):
        """The last delegation id this conversation handed out, resumed
        from its ledger the first time it is asked for, so a restarted
        server never issues a number twice."""
        if conversation_id not in self._seq:
            last = 0
            if conversation_id and self.pool is not None:
                from salt.agents import ledger
                last = ledger.read(
                    self.pool.cache_dir / conversation_id).last_id
            self._seq[conversation_id] = last
        return self._seq[conversation_id]

    def set_seq(self, conversation_id, value):
        self._seq[conversation_id] = int(value)

    def close(self):
        for handle in self.workers.values():
            handle.close()


def roster_payload(runtime, probe=False):
    """The roster as data: one row per entry, in file order.

    With `probe` each endpoint is contacted for the model it is actually
    serving, which is the only way to learn that a declared worker is
    not there.
    """
    if runtime.roster is None:
        return {"roster": "", "workers": [], "n": 0, "note": NO_ROSTER}
    from salt.agents.roster import probe_roster
    probes = probe_roster(runtime.roster) if probe else {}
    rows = []
    for entry in runtime.roster.entries:
        handle = runtime.worker(entry.name)
        row = {"name": entry.name,
               "role": entry.role,
               "alias": entry.alias,
               "mode": "attach" if entry.attach else "spawn",
               "endpoint": handle.endpoint,
               "state": handle.state,
               "calls": handle.calls,
               "capabilities": list(entry.capabilities),
               "notes": entry.notes}
        result = probes.get(entry.name)
        if result is not None:
            row["probe"] = result.state
            row["served_model"] = result.served_model or ""
            row["max_model_len"] = result.max_model_len
            row["detail"] = result.detail
        rows.append(row)
    return {"roster": str(runtime.roster.path), "workers": rows,
            "n": len(rows), "probed": bool(probe)}


def remember_answer(state, result, target):
    """Keep a worker's answer as a turn of its own.

    Headed with the worker it came from rather than shown as something
    the client said, and never part of any verbatim tail: a delegated
    answer is memory the next read can select, not something the
    conversation claims to have said.
    """
    session = state.session
    engine = state.runtime.engine
    session.ingest.submit(
        lambda: session.trie.add_turn(
            result.text, role="worker", origin=target,
            tokenizer=engine.tokenizer, model=engine.model,
            device=engine.device, dedup_cos=state.dedup_cos,
            max_sentences=state.max_sentences, save=False),
        label="worker-message ingest", payload=result.text)
    session.ingest.submit(
        lambda: session.trie.save() if session.trie.dirty else None,
        label="session save")
    return True


def file_record(state, result, remembered):
    """File one delegation in the conversation's ledger. Best effort:
    the answer is already on its way back, and failing to write the
    history of it must not take the answer down with it."""
    from salt.agents import ledger
    try:
        ledger.append(state.trie.cache_dir, ledger.record(result, remembered))
        return True
    except OSError:
        return False


def run_delegation(runtime, task, conversation_id="", target=None,
                   context_query=None, budget_pct=None, ingest=False,
                   max_chars=None):
    """Hand one task to a worker and wait for the whole answer.

    With a conversation the task is sent under that conversation's
    memory, selected for the task and committed to nothing. Without one
    it is a context-free delegation: the worker gets the task alone.
    """
    from salt.mcp.errors import DEFAULT_MAX_CHARS, ToolError, need_budget, \
        need_text
    from salt.mcp.server import known, refuse_write
    task = need_text("salt_delegate's task", task,
                     DEFAULT_MAX_CHARS if max_chars is None else max_chars)
    budget_pct = need_budget(budget_pct)
    if ingest and not conversation_id:
        raise ToolError("invalid_argument",
                        "ingest asks for the answer to be remembered, "
                        "which needs a conversation_id to remember it in")
    if ingest and runtime.read_only:
        refuse_write("salt_delegate", "remember a worker's answer")
    session = None
    if conversation_id:
        if runtime.pool is None:
            raise ToolError("invalid_argument",
                            "this server keeps no conversations")
        session = runtime.pool.get(known(runtime.pool, conversation_id))
    handle = runtime.target(target)
    state = DelegationState(runtime, session)
    req = DelegationRequest(task=task, target=handle.name,
                            context_query=context_query or None,
                            budget_pct=budget_pct, ingest=bool(ingest))
    # the chat layer talks to a person on stdout, and here stdout is the
    # protocol itself: a trim note printed onto it would be read as a
    # message and end the session
    with contextlib.redirect_stdout(sys.stderr):
        runtime.engine.ready()
        context = (build_context(state, req) if session is not None
                   else DelegationContext())
        result = delegate(state, req, context=context)
        remembered = bool(session is not None and req.ingest and result.ok
                          and result.text.strip()
                          and remember_answer(state, result, handle.name))
        recorded = (file_record(state, result, remembered)
                    if session is not None and not runtime.read_only
                    else False)
    return {"conversation_id": state.conversation_id,
            "id": result.id,
            "target": result.target,
            "task": task,
            "status": result.status,
            "answer": result.text,
            "error": result.error,
            "seconds": round(result.seconds, 3),
            "usage": dict(result.usage or {}),
            "context": {"n_selected": context.n_selected,
                        "words_used": context.words_used},
            "remembered": remembered,
            "recorded": recorded}


def delegation_error(exc):
    """A roster or delegation problem restated as a tool error. These
    are questions about the roster rather than about the task, and a
    client should read them without a traceback attached."""
    return isinstance(exc, (RosterError, DelegationError))
