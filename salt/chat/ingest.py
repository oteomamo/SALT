# -*- coding: utf-8 -*-
"""Background ingest worker: SALT's per-turn encode passes off the REPL's
critical path.

Every chat turn ends with two `SessionTrie.add_turn` calls (the user line
and the assistant reply), each running the dense-attention keyword pass and
the BGE embedding pass before the next prompt can appear — a long pasted
message costs seconds of encoder work between the reply's last token and
the returned `you>`. IngestWorker moves those calls onto ONE background
thread: the REPL enqueues them and returns the prompt immediately, and the
encode work overlaps the two idle spans a turn already has (the model's own
generation, the user's typing time).

The concurrency contract is deliberately blunt:

  * ONE worker thread, FIFO. The trie keeps a single mutator, so sentence
    order, turn indices, dedupe-hash evolution and the near-dup gate's
    keep-first semantics are byte-identical to the synchronous order.
  * `drain()` is the barrier. Any code that reads or mutates the shared
    state jobs touch must drain first; after a drain the state is exactly
    what inline execution would have produced.
  * The worker never prints. A job failure is captured, appended to the
    failure journal (one JSON line, WITH its payload text — a user message
    whose ingest raised must stay recoverable), and reported by the next
    `drain()`/`close()` return value, so a background error can never
    corrupt the prompt line it would have printed over.

Jobs are opaque callables: the worker knows nothing about tries or models,
which keeps it unit-testable without an encoder and reusable for attachment
ingest later. A worker built with `synchronous=True` runs each job inline:
a failing job is journaled, then raises at the submit site exactly like the
direct call it replaces — the escape hatch back to today's timing AND
today's failure control-flow, sharing one code path instead of forking
every call site.

Threads, not processes: the encode passes are `@torch.no_grad()` functions
on weights pinned to the GPU, torch releases the GIL inside forwards, and
the growing embedding matrix would be expensive to ship across a process
boundary. The REPL already runs one background thread per generation
(`TextIteratorStreamer`), so this adds a second, longer-lived one.
"""

import atexit
import json
import queue
import threading
import time
import traceback
from datetime import datetime


class IngestWorker:
    """One FIFO background thread for trie-ingest jobs, with a drain barrier.

    `stats` (lifetime, for /stats): `jobs` completed, `failures`, `busy_s`
    seconds spent inside jobs — the encoder time taken off the REPL's
    critical path.
    """

    def __init__(self, journal_path=None, synchronous=False):
        self.journal_path = journal_path
        self.synchronous = synchronous
        self.stats = {"jobs": 0, "failures": 0, "busy_s": 0.0}
        self._failures = []             # records since the last drain
        self._lock = threading.Lock()
        self._pending = 0
        self._closed = False
        self._q = None
        self._thread = None
        if not synchronous:
            self._q = queue.Queue()
            self._thread = threading.Thread(target=self._run,
                                            name="salt-ingest", daemon=True)
            self._thread.start()
            # The thread is daemon (a wedged encode must never block a hard
            # exit), so an exit path that bypasses close() — an exception
            # propagating out of the REPL, sys.exit from a handler — would
            # hard-kill it mid-job. The atexit hook drains those soft-exit
            # paths cleanly; close() unregisters it.
            atexit.register(self.close)

    @property
    def pending(self):
        """Jobs submitted but not yet finished (always 0 in sync mode)."""
        with self._lock:
            return self._pending

    def submit(self, fn, label="ingest", payload=None):
        """Queue `fn` (no args) for the worker; inline in sync mode, where
        a failing job also raises here, like the direct call it replaces.

        `label` names the job in failure reports; `payload` is the text the
        job would ingest, preserved verbatim in the failure journal so an
        error can never silently lose a user's words. Raises RuntimeError
        after `close()` — a job aimed at a torn-down session must fail
        loudly at the submit site, not vanish.
        """
        if self._closed:
            raise RuntimeError("IngestWorker is closed")
        if self.synchronous:
            self._execute(fn, label, payload)
            return
        with self._lock:
            self._pending += 1
        try:
            self._q.put((fn, label, payload))
        except BaseException:           # Ctrl-C on the put: job not queued
            with self._lock:
                self._pending -= 1
            raise

    def drain(self):
        """Block until every submitted job has finished, then return the
        failure records accumulated since the last drain (already
        journaled; the caller only needs to report them).

        This is THE barrier: main-thread code may touch the state jobs
        mutate only after a drain. `Queue.join()` is task_done-exact and
        interruptible by Ctrl-C on the main thread — an interrupted drain
        leaves the worker running, and the next drain resumes waiting.
        """
        if self._q is not None:
            self._q.join()
        with self._lock:
            failures, self._failures = self._failures, []
            # Submits come from the thread draining (single-submitter
            # contract), so an empty queue means truly nothing in flight;
            # resetting heals any counter drift a Ctrl-C mid-submit left.
            self._pending = 0
        return failures

    def close(self):
        """Finish the queue, stop the thread, reject later submits.
        Idempotent. Returns undelivered failure records, like drain()."""
        with self._lock:
            was_closed = self._closed
            self._closed = True
        if not was_closed and self._q is not None:
            self._q.put(None)           # after all queued jobs: FIFO
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        atexit.unregister(self.close)   # idempotent; sync mode never registered
        with self._lock:
            failures, self._failures = self._failures, []
        return failures

    # ── worker side ───────────────────────────────────────────────────────
    def _run(self):
        while True:
            item = self._q.get()
            if item is None:            # close() sentinel
                self._q.task_done()
                return
            fn, label, payload = item
            try:
                self._execute(fn, label, payload)
            finally:
                with self._lock:
                    self._pending -= 1
                self._q.task_done()

    def _execute(self, fn, label, payload):
        t0 = time.monotonic()
        try:
            fn()
            with self._lock:
                self.stats["jobs"] += 1
        except BaseException as exc:
            # Sync mode preserves the inline call's control flow exactly:
            # the exception propagates to the submit site (a Ctrl-C reaches
            # the REPL's own handler untouched), and a real failure is
            # journaled + counted on the way out so the durability story
            # holds in both modes. On the worker thread nothing may escape
            # — a dead worker would hang every future drain — so there
            # everything is recorded instead.
            if self.synchronous:
                if isinstance(exc, Exception):
                    self._record_failure(label, payload, exc, deliver=False)
                raise
            try:
                self._record_failure(label, payload, exc)
            except BaseException:
                with self._lock:        # record failed; keep the report honest
                    self.stats["failures"] += 1
                    self._failures.append(
                        {"ts": "", "label": label, "payload": payload,
                         "error": "failure recording itself failed",
                         "traceback": ""})
        finally:
            with self._lock:
                self.stats["busy_s"] += time.monotonic() - t0

    def _record_failure(self, label, payload, exc, deliver=True):
        try:
            err = f"{type(exc).__name__}: {exc}"
        except BaseException:           # a __str__ that raises
            err = type(exc).__name__
        try:
            tb = traceback.format_exc()
        except BaseException:
            tb = ""
        rec = {"ts": datetime.now().isoformat(timespec="seconds"),
               "label": label, "error": err, "traceback": tb,
               "payload": payload}
        # Deliver BEFORE the fallible journal write: a journal error must
        # never cost the drain() report too. `deliver=False` (sync mode)
        # journals and counts only — the exception itself propagates, so
        # queuing it for drain would report the same failure twice.
        with self._lock:
            self.stats["failures"] += 1
            if deliver:
                self._failures.append(rec)
        self._journal(rec)

    def _journal(self, rec):
        # Best-effort, like SessionTrie's near-dup log: journaling must
        # never take down the worker — Exception also covers a payload
        # json can't serialize and MemoryError/RecursionError on a
        # pathological one.
        if self.journal_path is None:
            return
        try:
            with open(self.journal_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass
