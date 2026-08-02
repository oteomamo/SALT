# -*- coding: utf-8 -*-
"""Worker handles for the saltChat agent layer.

A ``WorkerHandle`` is one roster entry's live connection to a model
serving beside the chat model. An attach entry points at a ``saltServe``
that is already running: the handle opens and closes its own client and
never starts or stops that server. Building a handle costs nothing, so a
roster may name workers a session never uses - the HTTP session and the
tokenizer load on the first call.

States: DECLARED (nothing contacted yet), PROBED (the endpoint answered
and serves the right model), READY (a client exists), BUSY (a call is in
flight), DEAD (the endpoint failed). PROBED and DEAD are spelled the same
as the roster's probe states on purpose, so one word describes a worker
whether it was reached by a probe or by a call.

One call at a time per handle. The serve client keeps this turn's token
counts on itself and shares a single HTTP session between calls, so two
concurrent streams would read each other's numbers.
"""

import threading
import time

from salt.agents.roster import UNPROBED, probe as probe_endpoint

DECLARED = "DECLARED"
PROBED = "PROBED"
READY = "READY"
BUSY = "BUSY"
DEAD = "DEAD"
STATES = (DECLARED, PROBED, READY, BUSY, DEAD)


class WorkerError(Exception):
    """A worker could not be reached, opened, or used."""


class WorkerHandle:
    """One roster entry, its connection state, and its call counters."""

    def __init__(self, entry, cfg=None):
        self.entry = entry
        self.cfg = entry.model if cfg is None else cfg
        self.state = DECLARED
        self.probe_result = UNPROBED
        self.runner = None
        self.calls = 0
        self.busy_s = 0.0
        self.last_error = ""
        self._lock = threading.Lock()

    @property
    def name(self):
        return self.entry.name

    @property
    def role(self):
        return self.entry.role

    @property
    def endpoint(self):
        if self.entry.attach:
            return self.entry.server_url
        return f"port {self.entry.spawn['port']}"

    @property
    def mean_latency(self):
        return self.busy_s / self.calls if self.calls else 0.0

    @property
    def note(self):
        return self.last_error if self.state == DEAD else self.probe_result.note

    def probe(self, timeout=5, url=None):
        """Contact the endpoint and record what it answered. Never
        raises: the ProbeResult carries the reason."""
        result = probe_endpoint(self.entry, url=url, timeout=timeout)
        self.probe_result = result
        if result.state == PROBED:
            # an answering endpoint clears an earlier failure, and an open
            # client is worth more than a probe: it needs no reopening
            if self.state != BUSY:
                self.state = READY if self.runner is not None else PROBED
            self.last_error = ""
        elif result.state == DEAD:
            self.state = DEAD
            self.last_error = result.detail
        return result

    def ready(self):
        """Open the client if it is not open yet and return the runner."""
        with self._lock:
            return self._open()

    def close(self):
        """Drop this session's client. The server keeps running and keeps
        its prefix cache warm: attach mode never stops what it did not
        start. Idempotent."""
        with self._lock:
            runner, self.runner = self.runner, None
            if runner is not None:
                runner.unload()
            if self.state in (READY, BUSY):
                self.state = (PROBED if self.probe_result.state == PROBED
                              else DECLARED)

    def call(self, messages, **overrides):
        """Stream one reply from the worker, yielding text pieces.

        A generator that holds the single-flight lock for its whole
        iteration, so a second caller waits for the first to finish
        instead of interleaving on one HTTP session. Abandoning it
        (``close()`` on the generator, or Ctrl-C) severs the response,
        which aborts the request server-side and frees the handle."""
        with self._lock:
            runner = self._open()
            self.state = BUSY
            t0 = time.monotonic()
            stream = runner.stream_chat(messages, **overrides)
            try:
                for piece in stream:
                    yield piece
            except Exception as exc:
                self.state = DEAD
                self.last_error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                # closed here rather than left to collection: severing the
                # response is what aborts the request on the worker, and it
                # has to happen the moment the caller walks away
                stream.close()
                self.calls += 1
                self.busy_s += time.monotonic() - t0
                if self.state == BUSY:
                    self.state = READY

    def _open(self):
        if self.runner is not None:
            return self.runner
        if not self.entry.attach:
            raise WorkerError(
                f"worker {self.name!r} is a spawn entry and nothing is "
                f"running for it yet")
        if not self.cfg:
            raise WorkerError(
                f"worker {self.name!r} has no resolved model, so its "
                f"tokenizer cannot be loaded")
        # imported here, not at module load: the serve client pulls in
        # transformers and torch, and a roster that names workers must
        # stay free for a session that never calls one
        from salt.chat.runner_serve import VLLMServeChatRunner
        try:
            self.runner = VLLMServeChatRunner(
                self.cfg, server_url=self.entry.server_url)
        except Exception as exc:
            self.state = DEAD
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise WorkerError(f"worker {self.name!r}: {exc}") from exc
        self.state = READY
        self.last_error = ""
        return self.runner
