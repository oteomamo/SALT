# -*- coding: utf-8 -*-
"""Background ingest worker: saltChat's per-turn keyword/embedding passes
run on one FIFO thread so the prompt returns immediately. drain() is the
barrier before any main-thread trie access. The worker never prints:
failures are journaled with their message text and reported at the next
drain(). synchronous=True runs each job inline, re-raising at the submit
site."""

import atexit
import json
import queue
import threading
import time
import traceback
from datetime import datetime


class IngestWorker:
    """One FIFO background thread for trie-ingest jobs, with a drain
    barrier. `stats` counts lifetime jobs, failures, and busy_s seconds
    spent inside jobs."""

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
            # daemon thread: atexit drains exit paths that bypass close()
            atexit.register(self.close)

    @property
    def pending(self):
        """Jobs submitted but not yet finished (always 0 in sync mode)."""
        with self._lock:
            return self._pending

    def submit(self, fn, label="ingest", payload=None):
        """Queue `fn` (inline in sync mode, where a failure raises here).
        `payload` is the text the failure journal preserves. Raises
        RuntimeError after close()."""
        if self._closed:
            raise RuntimeError("IngestWorker is closed")
        if self.synchronous:
            self._execute(fn, label, payload)
            return
        with self._lock:
            self._pending += 1
        try:
            self._q.put((fn, label, payload))
        except BaseException:
            with self._lock:
                self._pending -= 1
            raise

    def drain(self):
        """Block until every submitted job has finished, then return the
        failure records accumulated since the last drain."""
        if self._q is not None:
            self._q.join()
        with self._lock:
            failures, self._failures = self._failures, []
            self._pending = 0           # heal any Ctrl-C mid-submit drift
        return failures

    def close(self):
        """Finish the queue, stop the thread, reject later submits.
        Idempotent. Returns undelivered failure records, like drain()."""
        with self._lock:
            was_closed = self._closed
            self._closed = True
        if not was_closed and self._q is not None:
            self._q.put(None)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        atexit.unregister(self.close)
        with self._lock:
            failures, self._failures = self._failures, []
        return failures

    # ── worker side ───────────────────────────────────────────────────────
    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
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
            if self.synchronous:
                if isinstance(exc, Exception):
                    self._record_failure(label, payload, exc, deliver=False)
                raise
            # nothing may escape on the worker thread: a dead worker
            # would hang every future drain
            try:
                self._record_failure(label, payload, exc)
            except BaseException:
                with self._lock:
                    self.stats["failures"] += 1
                    self._failures.append(
                        {"ts": "", "label": label, "payload": payload,
                         "error": "failure recording itself failed",
                         "traceback": "", "journaled": False})
        finally:
            with self._lock:
                self.stats["busy_s"] += time.monotonic() - t0

    def _record_failure(self, label, payload, exc, deliver=True):
        try:
            err = f"{type(exc).__name__}: {exc}"
        except BaseException:
            err = type(exc).__name__
        try:
            tb = traceback.format_exc()
        except BaseException:
            tb = ""
        rec = {"ts": datetime.now().isoformat(timespec="seconds"),
               "label": label, "error": err, "traceback": tb,
               "payload": payload}
        # deliver before the fallible journal write
        with self._lock:
            self.stats["failures"] += 1
            if deliver:
                self._failures.append(rec)
        rec["journaled"] = self._journal(rec)

    def _journal(self, rec):
        # best-effort: journaling must never take down the worker
        if self.journal_path is None:
            return False
        try:
            with open(self.journal_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return True
        except Exception:
            return False
