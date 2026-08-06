# -*- coding: utf-8 -*-
"""Open conversations for the MCP server, and the cap on how many.

One SessionTrie and one ingest worker per open conversation, built on
first use and kept warm after, because a client that adds a turn is
about to ask for memory. Past the cap the least recently used session is
closed, and closing means: drain its ingest, save it if it is dirty,
then drop it. That ordering is the same one saltChat follows when it
switches conversations, generalized to a pool that holds several.

ONE CLIENT PER SERVER. Nothing here locks a session directory, and two
servers holding the same conversation would each save over the other.
A double open is detected on a best-effort basis and reported as a
warning on the session, not refused: a stale sentinel from a crashed
server must not be what stops the next one from working.
"""

import os
import time
from pathlib import Path

# how many conversations stay open at once. Each one costs its trie and
# its embeddings in memory, so this is a memory bound, not a policy
DEFAULT_CAPACITY = 8
SENTINEL = "mcp_open.json"
# a sentinel older than this belongs to a server that is gone
SENTINEL_STALE_S = 3600


class SessionError(Exception):
    """A session request that cannot be honoured (bad id, unknown id)."""


class OpenSession:
    """One open conversation: its trie, its ingest worker, its warnings."""

    def __init__(self, trie, ingest, warnings=(), read_only=False):
        self.trie = trie
        self.ingest = ingest
        # everything the client should be told about this conversation
        # that is not a refusal: another server holding it, a repair the
        # open had to make. Reported, never raised
        self.warnings = list(warnings)
        self.read_only = read_only
        self.touched = time.monotonic()
        # what the last read of this conversation measured, kept so the
        # signals a compression produces can be reported without running
        # one to find them
        self.last_stats = None

    @property
    def conversation_id(self):
        return self.trie.conversation_id

    def drain(self):
        """Every submitted job finished, so a read sees every write."""
        return self.ingest.drain()

    def close(self):
        """Drain, save what changed, then let go. A read-only server
        saves nothing at all: its whole promise is that opening a
        conversation leaves it exactly as it was found.

        The drain comes first and its failure cannot skip the save: a
        session being closed under a queue still holding jobs is exactly
        when the rows are worth the most, and dropping them because the
        last one failed would lose the ones that did not.
        """
        try:
            return self.ingest.close()
        finally:
            if self.trie.dirty and not self.read_only:
                self.trie.save()


def repair_note(trie):
    """What opening this conversation had to put right. A session whose
    files disagreed after a crash is rolled back to its last complete
    state as it loads, and the client is told rather than left to notice
    that a sentence is gone."""
    repair = getattr(trie, "load_repair", None)
    if not repair:
        return ""
    return (f"this conversation was repaired as it opened: "
            f"{repair.get('dropped_sentences', 0)} sentence(s) and "
            f"{repair.get('orphan_rows', 0)} embedding row(s) were dropped "
            f"as incomplete, leaving {repair.get('kept', 0)}. The full "
            f"record is in load_repairs.jsonl beside the conversation.")


def sentinel_path(cache_dir, conversation_id):
    return Path(cache_dir) / conversation_id / SENTINEL


def claim(cache_dir, conversation_id):
    """Mark this session as open here, and say so when somebody else
    already had. Best effort by design: this is a warning, not a lock."""
    path = sentinel_path(cache_dir, conversation_id)
    warning = ""
    try:
        if path.is_file():
            age = time.time() - path.stat().st_mtime
            if age < SENTINEL_STALE_S:
                warning = (f"session {conversation_id!r} looks open in "
                           f"another process (opened {int(age)}s ago). Two "
                           f"servers holding one session overwrite each "
                           f"other's saves.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'{{"pid": {os.getpid()}}}', encoding="utf-8")
    except OSError:
        pass
    return warning


def release(cache_dir, conversation_id):
    try:
        sentinel_path(cache_dir, conversation_id).unlink()
    except OSError:
        pass


class SessionPool:
    """The open sessions, newest use first, bounded by `capacity`."""

    def __init__(self, cache_dir, capacity=DEFAULT_CAPACITY,
                 budget_pct=0.20, synchronous=False, read_only=False):
        self.cache_dir = Path(cache_dir)
        self.capacity = max(1, int(capacity))
        self.budget_pct = budget_pct
        self.synchronous = synchronous
        self.read_only = read_only
        self.open = {}
        self.evictions = 0

    # ── opening ──────────────────────────────────────────────────────
    def _build(self, conversation_id):
        from salt.chat.cli import BGE_MODEL
        from salt.chat.ingest import IngestWorker
        from salt.engine.session_trie import SessionTrie
        from salt.mcp.errors import ToolError
        # both halves are built before anything is published, so a failed
        # constructor leaves the pool exactly as it was
        try:
            trie = SessionTrie(conversation_id, cache_dir=self.cache_dir,
                               model_name=BGE_MODEL,
                               budget_pct_default=self.budget_pct)
        except Exception as exc:
            # a session whose files cannot be read at all. The reason and
            # the folder are what a person needs; the traceback is not
            raise ToolError(
                "failed",
                f"conversation {conversation_id!r} could not be opened "
                f"({type(exc).__name__}: {exc}). Its files are under "
                f"{self.cache_dir / conversation_id}") from exc
        ingest = IngestWorker(
            journal_path=trie.cache_dir / "ingest_failures.jsonl",
            synchronous=self.synchronous)
        warnings = []
        # no sentinel on a read-only server: marking the directory is
        # itself a write, and this server promises not to make any
        if not self.read_only:
            warnings.append(claim(self.cache_dir, conversation_id))
        warnings.append(repair_note(trie))
        return OpenSession(trie, ingest, warnings=[w for w in warnings if w],
                           read_only=self.read_only)

    def get(self, conversation_id):
        """The open session for this id, opening it if needed."""
        validate_id(conversation_id)
        session = self.open.get(conversation_id)
        if session is None:
            session = self._build(conversation_id)
            self.open[conversation_id] = session
            self._evict_down()
        session.touched = time.monotonic()
        return session

    def exists(self, conversation_id):
        """Whether this conversation is on disk, saved by anyone."""
        validate_id(conversation_id)
        return (self.cache_dir / conversation_id / "config.json").is_file()

    # ── closing ──────────────────────────────────────────────────────
    def _evict_down(self):
        while len(self.open) > self.capacity:
            oldest = min(self.open.values(), key=lambda s: s.touched)
            self.close(oldest.conversation_id)
            self.evictions += 1

    def close(self, conversation_id):
        session = self.open.pop(conversation_id, None)
        if session is None:
            return False
        session.close()
        if not self.read_only:
            release(self.cache_dir, conversation_id)
        return True

    def close_all(self):
        for cid in list(self.open):
            self.close(cid)


def validate_id(conversation_id):
    """The REPL's rule, so a conversation named at the prompt and one
    named over the protocol are the same conversation."""
    from salt.chat.cli import valid_session_id
    if not isinstance(conversation_id, str) or not valid_session_id(
            conversation_id):
        raise SessionError(
            f"session ids may only contain letters, digits, '.', '_' and "
            f"'-', and {conversation_id!r} does not")
    return conversation_id
