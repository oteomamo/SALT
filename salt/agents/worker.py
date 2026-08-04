# -*- coding: utf-8 -*-
"""Worker handles for the saltChat agent layer.

A ``WorkerHandle`` is one roster entry's live connection to a model
serving beside the chat model. An attach entry points at a ``saltServe``
that is already running: the handle opens and closes its own client and
never starts or stops that server. A spawn entry describes a server this
session launches itself, as a child process on a port of its own, and
that one it does stop, at the latest when the session exits.
Building a handle costs nothing, so a roster may name workers a session
never uses - the HTTP session and the tokenizer load on the first call.

States: DECLARED (nothing contacted yet), STARTING (a spawned server is
coming up), PROBED (the endpoint answered and serves the right model),
READY (a client exists), BUSY (a call is in flight), DEAD (the endpoint
failed). PROBED and DEAD are spelled the same as the roster's probe
states on purpose, so one word describes a worker whether it was reached
by a probe or by a call.

One call at a time per handle. The serve client keeps this turn's token
counts on itself and shares a single HTTP session between calls, so two
concurrent streams would read each other's numbers.
"""

import atexit
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from salt.agents.roster import UNPROBED, probe as probe_endpoint

DECLARED = "DECLARED"
STARTING = "STARTING"
PROBED = "PROBED"
READY = "READY"
BUSY = "BUSY"
DEAD = "DEAD"
STATES = (DECLARED, STARTING, PROBED, READY, BUSY, DEAD)

HOST = "127.0.0.1"
# a cold vLLM start loads weights and builds a graph, so minutes is normal
READY_TIMEOUT = 180
READY_POLL = 0.5
# what SIGTERM gets before SIGKILL, and how much log a failure shows
STOP_GRACE = 10
LOG_TAIL_LINES = 20
# how long a worker may go silent mid-reply before the call is given up on
CALL_TIMEOUT = 300
# a refused connection is worth one more try, because a worker restarting
# beside a live session is the ordinary case. Only before the first byte:
# once text has streamed, a second attempt would duplicate a half-written
# reply, and only for connection errors, never for a stall.
CALL_RETRIES = 1
# a moment before the retry, because the case worth retrying is a server
# that is coming back up, and an instant second attempt would just be
# refused again
RETRY_DELAY = 0.5
# one bad call can be anything; two in a row is a worker to stop using
# until a probe says otherwise
FAILURES_TO_DEAD = 2


class WorkerError(Exception):
    """A worker could not be reached, opened, or used."""


def free_port(host=HOST):
    """A port nothing is listening on right now, taken and released."""
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host, 0))
        return s.getsockname()[1]


def port_available(port, host=HOST):
    """Whether a server could bind this port. REUSEADDR matches what the
    server itself binds with, so a just-stopped worker's socket waiting
    out TIME_WAIT does not read as occupied."""
    try:
        with socket.socket() as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
        return True
    except OSError:
        return False


def serve_executable():
    """How to run saltServe from here. The one next to this interpreter
    comes first: an unactivated env still finds its own, the way
    saltServe itself picks its vllm."""
    sibling = os.path.join(os.path.dirname(sys.executable), "saltServe")
    if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
        return [sibling]
    found = shutil.which("saltServe")
    if found:
        return [found]
    return [sys.executable, "-m", "salt.chat.serve"]


def is_connection_error(exc):
    """Whether the worker could not be reached at all, as opposed to
    answering and then stalling. This is the only failure worth retrying:
    nothing was said, so nothing can be duplicated by asking again."""
    import requests
    if is_read_timeout(exc):
        return False
    return isinstance(exc, requests.exceptions.ConnectionError)


def is_read_timeout(exc):
    """Whether this is the server going quiet mid-reply rather than the
    connection failing. requests reports it as ReadTimeout, or wraps
    urllib3's ReadTimeoutError in a ConnectionError once the stream is
    already open, so both shapes count."""
    import requests
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return False
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return True
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "ReadTimeoutError" in repr(exc.args)
    return False


def spawn_argv(entry, port):
    """The command line that serves this entry on this port."""
    spawn = entry.spawn or {}
    argv = list(spawn.get("command") or serve_executable() + [entry.alias])
    argv += ["--port", str(port)]
    if spawn.get("gpu") is not None:
        argv += ["--gpu", str(spawn["gpu"])]
    if spawn.get("gpu_mem_util") is not None:
        argv += ["--gpu-mem-util", str(spawn["gpu_mem_util"])]
    if spawn.get("max_model_len"):
        argv += ["--max-model-len", str(spawn["max_model_len"])]
    return argv


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
        self.failures = 0
        self.retries = 0
        self.process = None
        self.port = None
        self.url = entry.server_url
        self.log_path = None
        self.record_path = None
        self._lock = threading.Lock()

    @property
    def name(self):
        return self.entry.name

    @property
    def role(self):
        return self.entry.role

    @property
    def endpoint(self):
        if self.url:
            return self.url
        return f"port {self.entry.spawn['port']}"

    @property
    def mean_latency(self):
        return self.busy_s / self.calls if self.calls else 0.0

    @property
    def note(self):
        return self.last_error if self.state == DEAD else self.probe_result.note

    def start(self, workers_dir):
        """Launch this entry's own saltServe as a child process and
        return it. The server is not up yet when this returns: it holds
        the port and writes its startup to the log, and S012's readiness
        poll is what turns that into a usable worker. Calling it again
        while the child lives is a no-op."""
        if self.entry.attach:
            raise WorkerError(
                f"worker {self.name!r} attaches to {self.entry.server_url}, "
                f"so there is nothing for this session to start")
        with self._lock:
            if self.process is not None and self.process.poll() is None:
                return self.process
            port = self.entry.spawn.get("port", "auto")
            if port == "auto":
                port = free_port()
            elif not port_available(port):
                raise WorkerError(
                    f"worker {self.name!r}: port {port} is already taken. "
                    f"Point the entry at it with server_url to attach "
                    f"instead, or give spawn a free port.")
            argv = spawn_argv(self.entry, port)
            d = Path(workers_dir)
            d.mkdir(parents=True, exist_ok=True)
            log_path = d / f"{self.name}.log"
            try:
                # appended, never truncated: a restart's log has to keep
                # the previous crash that explains why it was restarted
                with open(log_path, "ab") as log:
                    self.process = subprocess.Popen(
                        argv, stdout=log, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, cwd=str(d))
            except OSError as exc:
                self.state = DEAD
                self.last_error = f"{type(exc).__name__}: {exc}"
                raise WorkerError(
                    f"worker {self.name!r}: cannot run {argv[0]!r}: "
                    f"{exc}") from exc
            self.port = port
            self.url = f"http://{HOST}:{port}"
            self.log_path = log_path
            self.record_path = self._write_record(d, argv)
            self.state = STARTING
            self.last_error = ""
            # a spawned server outlives the REPL unless something stops it,
            # and an exit path that skips /worker stop is still an exit
            atexit.register(self.stop)
            return self.process

    @property
    def ready_timeout(self):
        spawn = self.entry.spawn or {}
        return spawn.get("ready_timeout", READY_TIMEOUT)

    @property
    def timeout_s(self):
        """How long this worker may go silent mid-reply. Unlike the chat
        model, a worker that stalls is worth giving up on: the session
        has its own model to fall back to."""
        return (CALL_TIMEOUT if self.entry.timeout_s is None
                else self.entry.timeout_s)

    def log_tail(self, lines=LOG_TAIL_LINES):
        """The end of this worker's log, which is where a server that
        refused to start says why."""
        if self.log_path is None:
            return ""
        try:
            text = Path(self.log_path).read_text(errors="replace")
        except OSError:
            return ""
        return "\n".join(text.splitlines()[-lines:])

    def _explain(self, msg):
        tail = self.log_tail()
        if not tail:
            return msg
        return f"{msg}\nlast lines of {self.log_path}:\n{tail}"

    def wait_ready(self, timeout=None, poll=READY_POLL, on_wait=None):
        """Wait for a started worker to answer, and return what it is
        serving. Raises WorkerError naming the reason: the server died
        during startup (its log tail comes with the error) or it never
        answered in time. A failing poll does NOT mark the handle DEAD -
        not answering yet is what starting up looks like. ``on_wait`` is
        called with the seconds waited after each poll that found
        nothing, which is how a caller shows progress through a start
        that takes minutes."""
        if not self.entry.attach and self.process is None:
            raise WorkerError(
                f"worker {self.name!r} has not been started, so there is "
                f"nothing to wait for")
        limit = self.ready_timeout if timeout is None else timeout
        started = time.monotonic()
        deadline = started + limit
        probe_timeout = max(1.0, min(5.0, limit))
        while True:
            result = probe_endpoint(self.entry, url=self.url,
                                    timeout=probe_timeout)
            if result.state == PROBED:
                self.probe_result = result
                self.last_error = ""
                if self.state != BUSY:
                    self.state = READY if self.runner is not None else PROBED
                return result
            code = None if self.process is None else self.process.poll()
            if code is not None:
                self.state = DEAD
                self.last_error = self._explain(
                    f"the server for worker {self.name!r} exited with code "
                    f"{code} before it was ready")
                raise WorkerError(self.last_error)
            if time.monotonic() >= deadline:
                self.state = DEAD
                self.last_error = self._explain(
                    f"worker {self.name!r} did not answer at {self.url} "
                    f"within {limit:g}s. Raise spawn.ready_timeout if this "
                    f"model is simply slow to load.")
                raise WorkerError(self.last_error)
            if on_wait is not None:
                on_wait(time.monotonic() - started)
            time.sleep(poll)

    def healthy(self, timeout=5):
        """Whether this worker can take a call right now. A spawned
        server that has exited is DEAD without a network round trip:
        checking the child costs nothing and is never wrong."""
        code = None if self.process is None else self.process.poll()
        if code is not None:
            self.state = DEAD
            self.last_error = self._explain(
                f"the server for worker {self.name!r} exited with code "
                f"{code}")
            return False
        return self.probe(timeout=timeout).state == PROBED

    def stop(self, grace=STOP_GRACE):
        """Stop a server this session spawned and return its exit code.
        SIGTERM first so vLLM can shut its engine down, SIGKILL only if
        it will not go. Idempotent, and never raises at exit."""
        if self.entry.attach:
            raise WorkerError(
                f"worker {self.name!r} attaches to {self.entry.server_url}, "
                f"so this session must not stop it")
        atexit.unregister(self.stop)
        proc, self.process = self.process, None
        if proc is None:
            return None
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(grace)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(grace)
        # the client outlived its server, and a call in flight has already
        # broken on the closed connection - a bounded wait, because this
        # also runs at interpreter exit
        if self._lock.acquire(timeout=grace):
            try:
                runner, self.runner = self.runner, None
                if runner is not None:
                    runner.unload()
            finally:
                self._lock.release()
        if self.record_path is not None:
            # a record left on disk means a worker still running, which is
            # what tells a later session there is something to clean up
            try:
                os.remove(self.record_path)
            except OSError:
                pass
            self.record_path = None
        self.url, self.port = None, None
        self.state = DECLARED
        self.probe_result = UNPROBED
        self.last_error = ""
        return proc.returncode

    def _write_record(self, d, argv):
        """Leave the pid and the port on disk, so a later session can
        tell a live worker from one this machine has forgotten."""
        path = d / f"{self.name}.json"
        record = {"name": self.name, "alias": self.entry.alias,
                  "pid": self.process.pid, "port": self.port, "url": self.url,
                  "started_at": time.time(), "argv": list(argv),
                  "log": str(self.log_path)}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(record, indent=2))
        os.replace(tmp, path)
        return path

    def probe(self, timeout=5, url=None):
        """Contact the endpoint and record what it answered. Never
        raises: the ProbeResult carries the reason."""
        result = probe_endpoint(self.entry, url=url or self.url,
                                timeout=timeout)
        self.probe_result = result
        if result.state == PROBED:
            # an answering endpoint clears an earlier failure, and an open
            # client is worth more than a probe: it needs no reopening
            if self.state != BUSY:
                self.state = READY if self.runner is not None else PROBED
            self.last_error = ""
            self.failures = 0
        elif result.state == DEAD:
            self.state = DEAD
            self.last_error = result.detail
        return result

    def ready(self):
        """Open the client if it is not open yet and return the runner."""
        with self._lock:
            return self._open()

    def opened(self):
        """The runner if this worker will open, None if it will not.

        For callers that want what the runner knows - its window, its
        tokenizer - rather than to send on it. Reporting a worker that
        cannot be reached is the sending call's job, and it says why.
        """
        try:
            return self.ready()
        except WorkerError:
            return None

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

    def _note_failure(self, reason):
        """Record a failed call. Two in a row and the worker is DEAD:
        one can be anything, a second says stop sending work there until
        a probe proves otherwise."""
        self.failures += 1
        self.last_error = reason
        if self.failures >= FAILURES_TO_DEAD:
            self.state = DEAD
        return reason

    def _open_stream(self, runner, messages, overrides):
        """Open the response and return (stream, first piece), retrying a
        refused connection. Retries happen HERE and nowhere else, because
        this is the only point at which nothing has been said yet."""
        attempt = 0
        while True:
            stream = runner.stream_chat(messages, **overrides)
            try:
                return stream, next(stream)
            except StopIteration:
                return stream, None
            except Exception as exc:
                stream.close()
                if is_connection_error(exc) and attempt < CALL_RETRIES:
                    attempt += 1
                    self.retries += 1
                    time.sleep(RETRY_DELAY)
                    continue
                raise

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
            stream = None
            try:
                stream, first = self._open_stream(runner, messages, overrides)
                if first is not None:
                    yield first
                    for piece in stream:
                        yield piece
                self.failures = 0
                self.last_error = ""
            except Exception as exc:
                if is_read_timeout(exc):
                    # the server is alive, this reply is not coming. The
                    # finally below severs it, which frees the worker to
                    # take the next call, so this is not a DEAD worker and
                    # a stall never counts toward one.
                    self.last_error = (
                        f"worker {self.name!r} sent nothing for "
                        f"{self.timeout_s:g}s, so the call was given up on")
                    raise WorkerError(self.last_error) from exc
                self._note_failure(f"{type(exc).__name__}: {exc}")
                raise
            finally:
                # closed here rather than left to collection: severing the
                # response is what aborts the request on the worker, and it
                # has to happen the moment the caller walks away
                if stream is not None:
                    stream.close()
                self.calls += 1
                self.busy_s += time.monotonic() - t0
                if self.state == BUSY:
                    self.state = READY

    def _open(self):
        if self.runner is not None:
            return self.runner
        if self.url is None:
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
            self.runner = VLLMServeChatRunner(self.cfg, server_url=self.url,
                                              read_timeout=self.timeout_s)
        except Exception as exc:
            self._note_failure(f"{type(exc).__name__}: {exc}")
            raise WorkerError(f"worker {self.name!r}: {exc}") from exc
        self.state = READY
        self.last_error = ""
        return self.runner
