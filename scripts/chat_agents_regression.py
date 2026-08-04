# -*- coding: utf-8 -*-
"""Regression harness for the agent layer (--roster, /roster, /worker).

The fixture is a small scripted conversation plus a stub HTTP endpoint
that speaks the two routes a saltServe worker answers on, so every check
runs on CPU with no vLLM, no GPU and no second process.

  1. Roster validation: the shape errors a bad file has to produce.
  2. Roster loading: what a good file yields, and what the shipped
     sample says.
  3. Probe: a healthy endpoint, one holding another model, a closed
     port, a hung server, and a body that is not JSON.
  4. Worker handles: unopened until asked, and refusals with reasons.
  5. Worker calls: one call at a time, and closing without stopping
     the server.
  6. Abandoning a call: the response is severed so the worker aborts.
  7. Spawning: port picking, the command line the child gets, and the
     starts that are refused instead.
  8. What a spawned worker leaves on disk: the pid record's fields and
     the log the child writes.
  9. Readiness: a slow start polled through, one that dies with its log
     tail, and a deadline that is honoured.
 10. Stopping: SIGTERM then SIGKILL, and a session that exits without
     stopping its worker.
 11. Placement: which GPU a worker may spawn onto beside the chat model
     and the workers already running.
 12. Call timeouts: a worker that goes quiet mid-reply is given up on
     and the next call still works.
 13. Retry policy: what is retried, what never is, and when a worker is
     given up on for good.
 14. The /worker commands: start, stop, start --all, and the flag that
     starts the roster with the session.
 15. Delegation context: the memory a worker is handed is the same
     selection the chat model would get, and building it commits
     nothing.
 16. Executing a delegation: what the worker answered, what it cost,
     and how each way of failing is named.
 17. The /offload command: who a task goes to, what is printed, and
     what interrupting one leaves behind.
 18. The delegation ledger: what one delegation leaves on disk, where a
     resumed session picks its ids up, and what a damaged line costs.
 19. Worker rows: a remembered answer's role, origin and gating, and
     the flag that decides whether one is remembered at all.
 20. Identity: a scripted conversation runs byte-identically with and
     without a roster loaded, prompts and coverage included.
 21. Import purity: importing the agent layer costs nothing, and no
     entry point reaches the serve client or an MCP server on import.
 22. Frozen core: the agent work has not touched the eval files.
 23. Command surfaces: HELP, TAB completion and the docs
     command table agree on what the REPL accepts.

Groups 7 to 14 spawn stub servers instead of saltServe, through the
roster's undocumented spawn.command, so the whole worker lifecycle runs
on CPU with no vLLM, no GPU and no downloaded chat model.

Needs only the salt install and the BGE encoder (downloaded to the HF
cache on first use). Assert-based: refuses to run under python -O.

Usage:
    python scripts/chat_agents_regression.py
"""

import _thread
import argparse
import io
import json
import os
import pickle
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

if not __debug__:
    sys.exit("this harness is assert-based - run it without python -O")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from salt.agents import delegate as D                            # noqa: E402
from salt.agents import ledger as L                              # noqa: E402
from salt.agents import roster as R                              # noqa: E402
from salt.agents.roster import (BGE_CARD_MB, PLACEMENT_CEILING,   # noqa: E402
                                check_placement)
from salt.agents.worker import (BUSY, CALL_RETRIES, CALL_TIMEOUT,  # noqa: E402
                                DECLARED, DEAD, FAILURES_TO_DEAD, HOST,
                                PROBED, READY, READY_TIMEOUT, RETRY_DELAY,
                                STARTING, STATES, WorkerError, WorkerHandle,
                                free_port, is_connection_error,
                                is_read_timeout, port_available,
                                serve_executable, spawn_argv)
from salt.chat import cli                                        # noqa: E402
from salt.chat.registry import RegistryError, resolve_model      # noqa: E402
from salt.engine.compressor import load_bge                      # noqa: E402
from salt.engine.session_trie import (CONVERSATION_ROLES,        # noqa: E402
                                      VALID_ROLES, SessionTrie)

BGE_MODEL = "BAAI/bge-small-en-v1.5"
SAMPLE = REPO / "salt" / "agents" / "roster_sample.json"
SAMPLE_ALIAS = "qwen05"
CARDS = [{"id": "some/model", "max_model_len": 4096}]

# the eval core, frozen by CONTRIBUTING: the agent work is designed to
# need no edit here, and group 9 is what proves it kept to that
FROZEN = ("salt/compress.py", "salt/engine/celf.py",
          "salt/engine/trie_core.py", "salt/engine/embedder.py",
          "salt/engine/dataset_modes.py", "salt/engine/sentence_filter.py",
          "salt/engine/retrieval.py")
# the commit before the agent ladder's first, so the guard spans it whole
LADDER_BASE = "v2.10.0^"

TRANSCRIPT = [
    "We are sizing a home battery for a house with 9 kW of panels.",
    "The inverter is rated at 5 kW continuous.",
    "Winter evenings are the worst case, roughly 4 hours of draw.",
    "What size battery does that argue for?",
    "Assume the panels produce almost nothing in December.",
    "Summarize the sizing argument for the installer.",
]
REPLIES = [f"noted point {i}." for i in range(len(TRANSCRIPT))]


class _StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.server.raw is not None:
            body = self.server.raw
        else:
            body = json.dumps({"data": list(self.server.cards)}).encode()
        try:
            if self.server.delay:
                time.sleep(self.server.delay)
            self.send_response(self.server.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.server.aborted.set()

    def _frame(self, text):
        self.wfile.write(b"data: " + json.dumps(
            {"choices": [{"text": text}]}).encode() + b"\n\n")
        self.wfile.flush()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.server.last_payload = json.loads(self.rfile.read(n) or b"{}")
        with self.server.gauge:
            self.server.inflight += 1
            self.server.peak = max(self.server.peak, self.server.inflight)
            self.server.posts += 1
        try:
            if self.server.post_status != 200:
                body = b"no model is loaded on this server"
                self.send_response(self.server.post_status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in self.server.pieces:
                self._frame(piece)
                if self.server.delay:
                    time.sleep(self.server.delay)
            if self.server.drop:
                self.close_connection = True
                self.connection.close()
                return
            if self.server.stall:
                # quiet mid-reply without hanging up, then talking again:
                # the second write is where a client that walked away
                # surfaces as a broken pipe
                self.server.stalled.set()
                time.sleep(self.server.stall)
                for i in range(400):
                    self._frame(f"late{i} ")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            self.server.aborted.set()
        finally:
            with self.server.gauge:
                self.server.inflight -= 1


class Stub:
    """One /v1/models plus /v1/completions endpoint with a scripted answer.

    Port 0 by default so a real server on the sample roster's port never
    collides with, or silently satisfies, one of these checks. A fixed
    port is for the checks that stop the server and bring it back with a
    session's client still pointing at it.
    """

    def __init__(self, cards=(), pieces=("he", "llo"), delay=0.0,
                 status=200, raw=None, port=0, stall=0.0, drop=False,
                 serving=True, post_status=200):
        self.cfg = dict(cards=list(cards), pieces=list(pieces), delay=delay,
                        status=status, raw=raw, stall=stall, drop=drop,
                        post_status=post_status)
        self.port = port
        self.httpd = None
        self.aborted = threading.Event()
        self.stalled = threading.Event()
        self.url = f"http://127.0.0.1:{port}"
        if serving:
            self.start()

    def start(self):
        self.httpd = ThreadingHTTPServer((HOST, self.port), _StubHandler)
        for key, value in self.cfg.items():
            setattr(self.httpd, key, value)
        self.httpd.last_payload = None
        self.httpd.aborted, self.httpd.stalled = self.aborted, self.stalled
        self.httpd.inflight, self.httpd.peak, self.httpd.posts = 0, 0, 0
        self.httpd.gauge = threading.Lock()
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self

    def stop(self):
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
            self.httpd = None

    @property
    def posts(self):
        return 0 if self.httpd is None else self.httpd.posts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()


class _FakeRunner:
    """A runner that answers from a script and keeps every prompt it was
    given. The turn path needs a tokenizer, two settable fields and a
    stream, so this keeps the identity arm on CPU with no chat model."""

    kind = "fake"

    def __init__(self, tokenizer, replies):
        self.tokenizer = tokenizer
        self.alias = "fake"
        self.cfg = {"alias": "fake", "hf_id": "test/fake", "path": "-"}
        self.max_input_len = 4096
        self.last_prompt_tokens = None
        self.last_engine_stats = None
        self.replies = list(replies)
        self.prompts = []

    def input_budget(self, max_new_tokens=None):
        return self.max_input_len

    def stream_chat(self, messages, **overrides):
        self.prompts.append(json.loads(json.dumps(messages)))
        yield self.replies[(len(self.prompts) - 1) % len(self.replies)]

    def unload(self):
        pass


def closed_port():
    """A port nothing is listening on, taken and released."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


STUB_SERVER = '''
import argparse, json, sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int)
ap.add_argument("--gpu")
ap.add_argument("--gpu-mem-util")
ap.add_argument("--max-model-len")
ap.add_argument("--delay", type=float, default=0.0)
ap.add_argument("--die", type=float, default=None)
ap.add_argument("--ignore-term", action="store_true")
a, rest = ap.parse_known_args()
if a.ignore_term:
    import signal
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("loading weights, gpu", a.gpu, "extra", rest, flush=True)
if a.die is not None:
    time.sleep(a.die)
    print("CUDA out of memory: tried to allocate 24.00 GiB", flush=True)
    print("engine failed to start", flush=True)
    sys.exit(7)
time.sleep(a.delay)

class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass

    def do_GET(self):
        b = json.dumps({"data": [{"id": "some/model",
                                  "max_model_len": 4096}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

print("serving on", a.port, flush=True)
HTTPServer(("127.0.0.1", a.port), H).serve_forever()
'''

STUB_CFG = {"alias": "stub", "hf_id": "some/model", "path": str(REPO)}


def stub_server(tmp):
    """The script that stands in for saltServe, written once per run."""
    path = Path(tmp) / "stub_server.py"
    if not path.exists():
        path.write_text(STUB_SERVER)
    return str(path)


def spawn_entry(tmp, name="w", *extra, **spawn):
    """A roster entry whose server is the stub script, not saltServe."""
    s = {"port": "auto", "ready_timeout": 30,
         "command": [sys.executable, stub_server(tmp)] + list(extra)}
    s.update(spawn)
    return R.RosterEntry(name=name, alias="stub", role="worker", spawn=s,
                         model=STUB_CFG)


def spawn_handle(tmp, name="w", *extra, **spawn):
    return WorkerHandle(spawn_entry(tmp, name, *extra, **spawn))


def attach_entry(name="att", url="http://127.0.0.1:8099"):
    return R.RosterEntry(name=name, alias="stub", role="worker",
                         server_url=url, model=STUB_CFG)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def entry(url=None, alias=SAMPLE_ALIAS, name="w", model=None, **kw):
    return R.RosterEntry(name=name, alias=alias, role="worker",
                         server_url=url, model=model, **kw)


def write_roster(path, models, version=None):
    doc = {"models": models}
    if version is not None:
        doc["version"] = version
    path.write_text(json.dumps(doc))
    return path


def refuses(path, fragment):
    """load_roster must reject this file, naming the problem."""
    try:
        R.load_roster(path)
    except R.RosterError as exc:
        assert fragment in str(exc), (
            f"roster error for {path.name} was {str(exc)!r}, expected it to "
            f"mention {fragment!r}")
        return
    raise AssertionError(f"load_roster accepted {path.name}")


def run_arm(tmp, cid, tok, mdl, roster):
    """Replay TRANSCRIPT through the real turn path. Returns the per-turn
    trace and the trie, so two arms can be compared column by column."""
    args = cli.build_parser().parse_args(["--device", "cpu", "--sync-ingest"])
    trie = SessionTrie(cid, cache_dir=tmp, model_name=BGE_MODEL,
                       budget_pct_default=args.budget_pct)
    runner = _FakeRunner(tok, REPLIES)
    state = cli.ChatState(args, tok, mdl, runner, trie, roster)
    trace = []
    try:
        with redirect_stdout(io.StringIO()):
            for line in TRANSCRIPT:
                reply = cli.chat_turn(state, line)
                trace.append({"reply": reply,
                              "prompt": runner.prompts[-1],
                              "stats": dict(state.last_stats or {}),
                              "coverage": dict(trie.coverage),
                              "tail": list(state.tail)})
    finally:
        with redirect_stdout(io.StringIO()):
            cli.close_ingest(state)
    return trace, trie


def check_validation(tmp):
    good = {"name": "w", "alias": SAMPLE_ALIAS,
            "server_url": "http://127.0.0.1:8081"}
    bad = tmp / "bad.json"

    bad.write_text("{not json")
    refuses(bad, "is not valid JSON")
    bad.write_text("[]")
    refuses(bad, "must be a JSON object")
    write_roster(bad, [])
    refuses(bad, "needs a non-empty 'models' list")
    write_roster(bad, [good], version="salt-roster/99")
    refuses(bad, "carries schema")
    bad.write_text(json.dumps({"models": [good], "extras": 1}))
    refuses(bad, "unknown top-level keys")
    write_roster(bad, [good, dict(good)])
    refuses(bad, "duplicate name")
    write_roster(bad, [{"name": "w", "alias": "a", "server_url": "http://h",
                        "spawn": {"port": 1}}])
    refuses(bad, "exactly one of server_url")
    write_roster(bad, [{"name": "w", "alias": "a"}])
    refuses(bad, "exactly one of server_url")
    write_roster(bad, [{"name": "w", "alias": "a", "server_url": "127.0.0.1"}])
    refuses(bad, "must start with http://")
    write_roster(bad, [{"name": "w!", "alias": "a", "server_url": "http://h"}])
    refuses(bad, "needs a name of letters")
    write_roster(bad, [{"name": "w", "alias": "a",
                        "spawn": {"port": 99999}}])
    refuses(bad, "spawn.port must be <= 65535")
    write_roster(bad, [{"name": "w", "alias": "a", "spawn": {"prt": 1}}])
    refuses(bad, "unknown spawn keys")
    write_roster(bad, [{"name": "w", "alias": "a", "spawn": {"port": 1},
                        "role": "boss"}])
    refuses(bad, "role must be one of")
    write_roster(bad, [{"name": "a", "alias": "x", "role": "orchestrator",
                        "spawn": {"port": 1}},
                       {"name": "b", "alias": "y", "role": "orchestrator",
                        "spawn": {"port": 2}}])
    refuses(bad, "more than one orchestrator")
    write_roster(bad, [{"name": "w", "alias": "a", "server_url": "http://h",
                        "timeout_s": 0}])
    refuses(bad, "timeout_s must be >= 1")
    for command in ("echo hi", [], ["ok", 7], ["ok", ""]):
        write_roster(bad, [{"name": "w", "alias": "a",
                            "spawn": {"port": "auto", "command": command}}])
        refuses(bad, "spawn.command must be a non-empty list of strings")
    write_roster(bad, [{"name": "w", "alias": "a",
                        "spawn": {"port": "auto", "commnad": ["x"]}}])
    refuses(bad, "unknown spawn keys ['commnad']")
    for val, fragment in (("soon", "must be a number"), (0, "must be >= 1"),
                          (True, "must be a number")):
        write_roster(bad, [{"name": "w", "alias": "a",
                            "spawn": {"port": "auto", "ready_timeout": val}}])
        refuses(bad, "spawn.ready_timeout " + fragment)
    refuses(tmp / "absent.json", "Cannot read roster")
    # spawn.command is how these checks stand a stub in for saltServe. It
    # is deliberately absent from every surface a user reads
    assert "spawn.command" not in (REPO / "docs" / "options.md").read_text(
        encoding="utf-8")
    assert '"command"' not in SAMPLE.read_text(encoding="utf-8")
    assert "command" not in R.__doc__
    print("1. roster validation: 24 malformed files each refused by name, "
          "and the test spawn hook stays out of the sample and the docs")


def check_loading():
    raw = json.loads(SAMPLE.read_text())
    assert raw["version"] == R.ROSTER_SCHEMA, "the sample carries a stale schema"
    assert raw["models"], "the shipped sample names no models"
    try:
        cfg = resolve_model(SAMPLE_ALIAS)
        registered = bool(cfg["downloaded"])
    except RegistryError:
        registered = False
    if not registered:
        print(f"2. roster loading: the shipped sample is well formed "
              f"({SAMPLE_ALIAS!r} is not installed, so registry resolution "
              f"is not exercised here)")
        return
    roster = R.load_roster(SAMPLE)
    assert roster.entries, "the sample loaded with no entries"
    first = roster.entries[0]
    assert first.attach, "the sample's first entry stopped being attach mode"
    assert first.model and first.model["hf_id"], (
        "loading did not resolve the entry against the registry")
    assert roster.workers, "the sample names no worker"
    assert roster.orchestrator is None, "the sample grew an orchestrator"
    try:
        roster.get("no-such-worker")
        raise AssertionError("Roster.get accepted an unknown name")
    except R.RosterError as exc:
        assert "known:" in str(exc), "an unknown name did not list the known"
    print(f"2. roster loading: the shipped sample resolves all "
          f"{len(roster.entries)} of its entries against the registry")


def check_probe():
    with Stub(cards=[{"id": "some/model", "max_model_len": 4096}]) as s:
        res = R.probe(entry(s.url, model={"hf_id": "some/model"}))
        assert res.state == "PROBED", f"a healthy endpoint read {res.state}"
        assert res.served_model == "some/model", "the served id was lost"
        assert res.max_model_len == 4096, "the window was lost"
        assert res.note == "serving some/model, window 4096 tokens", (
            f"the status note read {res.note!r}")
        assert R.probe(entry(s.url + "/", model={"hf_id": "some/model"})
                       ).state == "PROBED", "a trailing slash broke the probe"

    with Stub(cards=[{"id": "another/model"}]) as s:
        res = R.probe(entry(s.url))
        assert res.state == "DEAD", "a server holding another model read alive"
        assert "another/model" in res.detail and "not 'qwen05'" in res.detail, (
            f"the mismatch did not name both sides: {res.detail!r}")

    with Stub(cards=[]) as s:
        assert "serves nothing" in R.probe(entry(s.url)).detail, (
            "an empty card list did not read as nothing served")

    with Stub(raw=b"<html>not json</html>") as s:
        assert R.probe(entry(s.url)).state == "DEAD", "a non-JSON body passed"
    with Stub(raw=json.dumps({"data": "nope"}).encode()) as s:
        assert "serves nothing" in R.probe(entry(s.url)).detail, (
            "a malformed data field was not tolerated")
    with Stub(status=500) as s:
        assert R.probe(entry(s.url)).state == "DEAD", "an HTTP 500 passed"

    res = R.probe(entry(f"http://127.0.0.1:{closed_port()}"), timeout=2)
    assert res.state == DEAD, "a closed port did not read DEAD"
    assert "no server answering at" in res.detail, (
        f"a closed port said {res.detail!r}")

    with Stub(cards=[{"id": SAMPLE_ALIAS}], delay=1.0) as s:
        t0 = time.monotonic()
        res = R.probe(entry(s.url), timeout=0.2)
        waited = time.monotonic() - t0
        assert res.state == DEAD and "Timeout" in res.detail, (
            f"a hung server read {res.state} / {res.detail!r}")
        assert waited < 0.9, f"the probe hung for {waited:.2f}s past its timeout"

    spawn_only = R.RosterEntry(name="tiny", alias="t", role="worker",
                               spawn={"port": "auto"})
    assert R.probe(spawn_only).state == "UNPROBED", (
        "a spawn entry with nothing running was not left UNPROBED")
    with Stub(cards=[{"id": "t"}]) as s:
        assert R.probe(spawn_only, url=s.url).state == "PROBED", (
            "the url override did not probe a started worker")
    print("3. probe: 6 failure shapes read DEAD with a reason, and a hung "
          "server is cut off at its timeout")


def check_worker_handle():
    cfg = {"alias": "stub", "hf_id": "some/model", "path": str(REPO)}
    with Stub(cards=[{"id": "some/model", "max_model_len": 4096}],
              pieces=("he", "llo")) as s:
        h = WorkerHandle(entry(s.url, model=cfg))
        assert h.state == DECLARED and h.runner is None, (
            "building a handle opened something")
        assert h.probe().state == PROBED and h.state == PROBED, (
            "a healthy probe did not move the handle to PROBED")

    h = WorkerHandle(entry(f"http://127.0.0.1:{closed_port()}", model=cfg))
    h.probe(timeout=2)
    assert h.state == DEAD and h.last_error, "a closed port left no reason"

    h = WorkerHandle(R.RosterEntry(name="tiny", alias="t", role="worker",
                                   spawn={"port": "auto"}, model=cfg))
    try:
        h.ready()
        raise AssertionError("a spawn entry opened a client")
    except WorkerError as exc:
        assert "nothing is running" in str(exc), f"unexpected refusal: {exc}"
    h = WorkerHandle(entry("http://127.0.0.1:1"))
    try:
        h.ready()
        raise AssertionError("an unresolved entry opened a client")
    except WorkerError as exc:
        assert "no resolved model" in str(exc), f"unexpected refusal: {exc}"
    print("4. worker handles: unopened until asked, and every unusable "
          "entry refuses with the reason")


def check_worker_calls(tok_path):
    cfg = {"alias": "stub", "hf_id": "some/model", "path": tok_path}
    with Stub(cards=[{"id": "some/model", "max_model_len": 4096}],
              pieces=("he", "llo"), delay=0.0) as s:
        h = WorkerHandle(entry(s.url, model=cfg))
        msg = [{"role": "user", "content": "hi"}]
        with redirect_stdout(io.StringIO()):
            text = "".join(h.call(msg))
        assert text == "hello", f"the worker reply came back as {text!r}"
        assert h.state == READY, f"the handle settled at {h.state}"
        assert h.calls == 1 and h.busy_s > 0, "the call was not measured"
        assert isinstance(s.httpd.last_payload["prompt"], list), (
            "the worker was sent text instead of token ids")

        s.httpd.delay = 0.08
        s.httpd.pieces = list("abcde")
        seen = {}

        def one(tag):
            seen[tag] = "".join(h.call(msg))

        threads = [threading.Thread(target=one, args=(t,)) for t in "AB"]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        assert set(seen) == {"A", "B"}, "a concurrent call never finished"
        assert set(seen.values()) == {"abcde"}, "concurrent calls interleaved"
        assert s.httpd.peak == 1, (
            f"the worker saw {s.httpd.peak} requests at once, so the "
            f"single-flight lock is not holding")
        assert h.calls == 3, f"the handle counted {h.calls} calls, expected 3"
        h.close()
        assert h.runner is None, "close kept the client"
        assert h.probe().state == PROBED, "close stopped the server"
    print("5. worker calls: 2 concurrent callers serialized to 1 request at "
          "a time, and close left the server serving")


def check_abort(tok_path):
    cfg = {"alias": "stub", "hf_id": "some/model", "path": tok_path}
    with Stub(cards=[{"id": "some/model", "max_model_len": 4096}],
              pieces=["a"] + [f"t{i} " for i in range(300)], delay=0.05) as s:
        h = WorkerHandle(entry(s.url, model=cfg))
        with redirect_stdout(io.StringIO()):
            stream = h.call([{"role": "user", "content": "hi"}])
            first = next(stream)
        assert first == "a", f"the stream opened with {first!r}"
        assert h.state == BUSY, "a running call did not read BUSY"
        stream.close()
        assert h.state == READY, "abandoning a call left the handle busy"
        assert s.httpd.aborted.wait(20), (
            "walking away from a call did not sever the response, so the "
            "worker would keep generating")
        s.httpd.pieces, s.httpd.delay = ["done"], 0.0
        with redirect_stdout(io.StringIO()):
            again = "".join(h.call([{"role": "user", "content": "hi"}]))
        assert again == "done", "the handle did not take a call after an abort"
        h.close()
    print("6. abandoning a call: the response is severed, the worker aborts, "
          "and the handle takes the next call")


def check_spawning(tmp):
    assert STARTING in STATES and STATES.index(STARTING) == 1, (
        f"STARTING left its place between DECLARED and PROBED: {STATES}")

    port = free_port()
    assert 1024 < port <= 65535 and port_available(port), (
        f"free_port returned {port}, which cannot be bound")
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((HOST, 0))
    listener.listen(1)
    taken = listener.getsockname()[1]
    assert not port_available(taken), "a listening port read as available"
    listener.close()
    assert port_available(taken), (
        "a just-closed port read as taken, so the pre-bind probe is not "
        "using REUSEADDR and a restarted worker would be refused its port")
    assert len({free_port() for _ in range(20)}) > 1, (
        "free_port handed out the same port 20 times")

    e = R.RosterEntry(name="w", alias=SAMPLE_ALIAS, role="worker",
                      spawn={"port": 8081}, model=STUB_CFG)
    argv = spawn_argv(e, 8081)
    assert argv[-2:] == ["--port", "8081"] and argv[-3] == SAMPLE_ALIAS, argv
    assert "saltServe" in argv[0] or argv[1:3] == ["-m", "salt.chat.serve"], (
        f"the command line does not run saltServe: {argv}")
    full = R.RosterEntry(name="w", alias=SAMPLE_ALIAS, role="worker",
                         model=STUB_CFG,
                         spawn={"port": "auto", "gpu": "1",
                                "gpu_mem_util": 0.4, "max_model_len": 4096})
    argv = spawn_argv(full, 9000)
    for flag, val in (("--port", "9000"), ("--gpu", "1"),
                      ("--gpu-mem-util", "0.4"), ("--max-model-len", "4096")):
        assert flag in argv and argv[argv.index(flag) + 1] == val, (
            f"{flag} {val} is missing from {argv}")
    bare = R.RosterEntry(name="w", alias=SAMPLE_ALIAS, role="worker",
                         model=STUB_CFG, spawn={"port": "auto",
                                                "max_model_len": 0})
    argv = spawn_argv(bare, 9000)
    assert not [f for f in ("--gpu", "--gpu-mem-util", "--max-model-len")
                if f in argv], (
        f"an unset spawn field still reached the child, overriding "
        f"saltServe's own default: {argv}")
    injected = R.RosterEntry(name="w", alias=SAMPLE_ALIAS, role="worker",
                             model=STUB_CFG,
                             spawn={"port": "auto",
                                    "command": ["/bin/true", "x"]})
    assert spawn_argv(injected, 7000) == ["/bin/true", "x", "--port", "7000"]
    exe = serve_executable()
    assert exe and all(isinstance(part, str) for part in exe), exe

    d = Path(tmp) / "spawn" / "workers"
    h = spawn_handle(tmp)
    assert h.state == DECLARED and h.process is None and h.url is None
    assert h.endpoint == "port auto", h.endpoint
    proc = h.start(d)
    assert proc.poll() is None, "the child died immediately"
    assert h.state == STARTING, f"the handle read {h.state} after start"
    assert h.port and h.url == f"http://{HOST}:{h.port}", h.url
    assert h.endpoint == h.url, h.endpoint
    assert h.start(d) is proc, "start() spawned a second child over a live one"
    assert h.wait_ready(timeout=30, poll=0.2).state == PROBED, h.note

    attach = WorkerHandle(attach_entry("a"))
    try:
        attach.start(d)
        raise AssertionError("an attach entry was started")
    except WorkerError as exc:
        assert "nothing for this session to start" in str(exc), exc

    busy = spawn_handle(tmp, "busy", port=h.port)
    try:
        busy.start(d)
        raise AssertionError(f"a worker spawned onto occupied port {h.port}")
    except WorkerError as exc:
        assert f"port {h.port} is already taken" in str(exc), exc
        assert "server_url" in str(exc), "the refusal points at no fix"
    assert busy.process is None and busy.state == DECLARED, busy.state

    missing = WorkerHandle(spawn_entry(tmp, "nope",
                                       command=["/nonexistent/binary"]))
    try:
        missing.start(d)
        raise AssertionError("a missing executable spawned")
    except WorkerError as exc:
        assert "cannot run" in str(exc) and "/nonexistent" in str(exc), exc
    assert missing.state == DEAD and missing.last_error, "no reason recorded"
    h.stop()
    print("7. spawning: an auto port picked and served, and 3 starts refused "
          "instead (attach entry, occupied port, missing binary)")


def check_worker_files(tmp):
    """The pid record is what a later session reads to tell a live worker
    from one this machine forgot, so its shape is pinned here."""
    d = Path(tmp) / "files" / "workers"
    h = spawn_handle(tmp, "rec")
    proc = h.start(d)
    h.wait_ready(timeout=30, poll=0.2)

    record = json.loads((d / "rec.json").read_text())
    assert set(record) == {"name", "alias", "pid", "port", "url",
                           "started_at", "argv", "log"}, sorted(record)
    assert record["pid"] == proc.pid, record
    assert record["port"] == h.port and record["url"] == h.url, record
    assert record["name"] == "rec" and record["alias"] == "stub", record
    assert isinstance(record["started_at"], float), record
    assert record["started_at"] > 0, record
    assert record["argv"][-2:] == ["--port", str(h.port)], record
    assert record["log"] == str(d / "rec.log"), record
    assert h.record_path == d / "rec.json", h.record_path
    assert not list(d.glob("*.tmp")), (
        "an atomic-write temp file was left behind")

    log = (d / "rec.log").read_text()
    assert f"serving on {h.port}" in log, log
    assert h.log_path == d / "rec.log", h.log_path
    assert h.log_tail(1) == f"serving on {h.port}", repr(h.log_tail(1))

    h.stop()
    assert not (d / "rec.json").exists(), (
        "the pid record survived a clean stop, so a later session would "
        "read it as a worker still running")
    assert (d / "rec.log").exists(), "the log was removed with the record"
    h.start(d)
    h.wait_ready(timeout=30, poll=0.2)
    assert (d / "rec.log").read_text().count("loading weights") == 2, (
        "the log was truncated on restart, losing the run that came before")
    h.stop()
    print("8. worker files: <name>.json carries 8 pinned fields and is "
          "removed on a clean stop, <name>.log survives and is appended to")


def check_readiness(tmp):
    d = Path(tmp) / "ready" / "workers"
    h = spawn_handle(tmp, "slow", "--delay", "1.5")
    h.start(d)
    t0 = time.monotonic()
    result = h.wait_ready(timeout=30, poll=0.2)
    waited = time.monotonic() - t0
    assert result.state == PROBED and result.served_model == "some/model"
    assert result.max_model_len == 4096, result
    assert waited >= 1.4, f"it returned before the server answered ({waited}s)"
    assert h.state == PROBED and not h.last_error, (h.state, h.last_error)
    assert h.wait_ready(timeout=5, poll=0.2).state == PROBED, (
        "wait_ready on a worker that is already up did not return at once")
    assert h.healthy(timeout=3), f"a live worker reported unhealthy: {h.note}"
    h.process.kill()
    h.process.wait(10)
    assert not h.healthy(timeout=3), "a killed worker reported healthy"
    assert h.state == DEAD and "exited with code" in h.last_error, h.last_error
    h.stop()

    dead = spawn_handle(tmp, "oom", "--die", "0.3")
    dead.start(d)
    try:
        dead.wait_ready(timeout=30, poll=0.2)
        raise AssertionError("a server that exited was reported ready")
    except WorkerError as exc:
        msg = str(exc)
    assert "exited with code 7" in msg, msg
    assert "CUDA out of memory" in msg and "engine failed to start" in msg, (
        f"the log tail did not come with the error, so the reason for the "
        f"failed start is nowhere on screen: {msg}")
    assert str(d / "oom.log") in msg, msg
    assert dead.state == DEAD and dead.last_error == msg, dead.state
    assert len(dead.log_tail().splitlines()) == 3, dead.log_tail()

    late = spawn_handle(tmp, "never", "--delay", "60")
    late.start(d)
    t0 = time.monotonic()
    try:
        late.wait_ready(timeout=1.5, poll=0.2)
        raise AssertionError("a server that never answered was reported ready")
    except WorkerError as exc:
        msg = str(exc)
    took = time.monotonic() - t0
    assert 1.4 <= took < 12, f"the deadline was not honoured ({took:.1f}s)"
    assert "did not answer" in msg and "1.5s" in msg, msg
    assert "spawn.ready_timeout" in msg, (
        f"the timeout names no way to raise the limit: {msg}")
    late.stop()

    assert spawn_handle(tmp, "d").ready_timeout == 30, "the fixture's own"
    assert WorkerHandle(R.RosterEntry(name="x", alias="stub", role="worker",
                                      spawn={"port": "auto"}, model=STUB_CFG)
                        ).ready_timeout == READY_TIMEOUT == 180
    roster_limit = spawn_handle(tmp, "late2", "--delay", "60", ready_timeout=1)
    roster_limit.start(d)
    t0 = time.monotonic()
    try:
        roster_limit.wait_ready(poll=0.2)
        raise AssertionError("the roster's ready_timeout was not used")
    except WorkerError as exc:
        assert "within 1s" in str(exc), str(exc)
    assert time.monotonic() - t0 < 10, "the 180s default was used instead"
    roster_limit.stop()
    print("9. readiness: a slow start polled through without flapping to "
          "DEAD, a crash surfaced with its exit code and log tail, and both "
          "deadlines honoured")


def check_stopping(tmp):
    d = Path(tmp) / "stop" / "workers"
    h = spawn_handle(tmp, "term")
    h.start(d)
    h.wait_ready(timeout=30, poll=0.2)
    pid = h.process.pid
    code = h.stop()
    assert code is not None, "stop() returned no exit code for a live child"
    assert h.process is None and h.state == DECLARED, (h.process, h.state)
    assert h.url is None and h.port is None, (h.url, h.port)
    assert not alive(pid), f"pid {pid} is still alive after stop()"
    assert h.stop() is None, "a second stop did something"

    stubborn = spawn_handle(tmp, "stubborn", "--ignore-term")
    stubborn.start(d)
    stubborn.wait_ready(timeout=30, poll=0.2)
    pid = stubborn.process.pid
    t0 = time.monotonic()
    code = stubborn.stop(grace=1)
    took = time.monotonic() - t0
    assert took >= 1.0, f"SIGKILL came before the grace period ({took:.1f}s)"
    assert took < 8, f"stopping took {took:.1f}s"
    assert code == -signal.SIGKILL, f"the exit code was {code}"
    assert not alive(pid), "the child that ignored SIGTERM survived"

    attach = WorkerHandle(attach_entry("a"))
    try:
        attach.stop()
        raise AssertionError("an attached server was stopped")
    except WorkerError as exc:
        assert "must not stop it" in str(exc), str(exc)

    prog = textwrap.dedent(f'''
        import sys
        sys.path.insert(0, {str(REPO)!r})
        from salt.agents import roster as R
        from salt.agents.worker import WorkerHandle
        e = R.RosterEntry(name="ax", alias="stub", role="worker",
                          model={STUB_CFG!r},
                          spawn={{"port": "auto",
                                  "command": [sys.executable,
                                              {stub_server(tmp)!r}]}})
        h = WorkerHandle(e)
        h.start({str(d)!r})
        h.wait_ready(timeout=60, poll=0.2)
        print(h.process.pid)
        sys.stdout.flush()
    ''')
    out = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                         text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    pid = int(out.stdout.strip().splitlines()[-1])
    for _ in range(100):
        if not alive(pid):
            break
        time.sleep(0.05)
    assert not alive(pid), (
        f"the spawned server (pid {pid}) outlived the session that started "
        f"it, so a REPL exiting without /worker stop leaks a server")
    assert not (d / "ax.json").exists(), "the pid record outlived the session"

    out = subprocess.run([sys.executable, "-c", prog + "\nh.stop()\n"],
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    assert not out.stderr.strip(), (
        f"exiting after an explicit stop printed to stderr, so the atexit "
        f"hook ran a second time:\n{out.stderr}")
    print("10. stopping: SIGTERM, SIGKILL after the grace, idempotent, and a "
          "session that exits without stopping still takes its worker down")


def placement_entry(name="w", gpu=None, util=None):
    spawn = {"port": "auto"}
    if gpu is not None:
        spawn["gpu"] = gpu
    if util is not None:
        spawn["gpu_mem_util"] = util
    return R.RosterEntry(name=name, alias="stub", role="worker", spawn=spawn,
                         model=STUB_CFG)


def check_placement_rules():
    assert R.entry_cards(placement_entry(gpu="1")) == (1,)
    assert R.entry_cards(placement_entry(gpu="0,2")) == (0, 2)
    assert R.entry_cards(placement_entry()) == ()
    assert R.entry_cards(attach_entry()) == ()

    refusal, notes = check_placement(placement_entry(gpu="1"), chat_gpus=[0],
                                     bge_gpu=0)
    assert refusal is None and notes == [], (refusal, notes)
    refusal, notes = check_placement(placement_entry(gpu="1", util=0.4),
                                     chat_gpus=[0], chat_mem_util=0.85,
                                     bge_gpu=0)
    assert refusal is None and notes == [], (refusal, notes)

    refusal, _ = check_placement(placement_entry(gpu="0"), chat_gpus=[0],
                                 chat_mem_util=0.85)
    assert refusal and "GPU 0 already carries the chat model" in refusal
    assert "out of memory at load" in refusal and "gpu_mem_util" in refusal
    assert "this entry declares" in refusal, (
        f"the refusal does not say which side failed to declare a share: "
        f"{refusal}")
    refusal, notes = check_placement(placement_entry(gpu="0", util=0.10),
                                     chat_gpus=[0], chat_mem_util=0.85)
    assert refusal is None and notes == [], (refusal, notes)
    refusal, notes = check_placement(placement_entry(gpu="0", util=0.30),
                                     chat_gpus=[0], chat_mem_util=0.85)
    assert refusal is None, "an over-subscription became a refusal"
    assert len(notes) == 1 and "1.15 in total" in notes[0], notes
    assert f"{PLACEMENT_CEILING:g}" in notes[0], notes
    refusal, _ = check_placement(placement_entry(gpu="0", util=0.10),
                                 chat_gpus=[0], chat_mem_util=None)
    assert refusal and "the chat model does not declare" in refusal, refusal
    refusal, _ = check_placement(placement_entry(gpu="0,1"), chat_gpus=[1],
                                 chat_mem_util=0.8)
    assert refusal and "GPU 1" in refusal, refusal

    refusal, _ = check_placement(placement_entry("second", gpu="1"),
                                 chat_gpus=[0], running=[("first", (1,), None)])
    assert refusal and "worker 'first'" in refusal, refusal
    refusal, notes = check_placement(placement_entry("second", gpu="1",
                                                     util=0.3),
                                     chat_gpus=[0],
                                     running=[("first", (1,), 0.4)])
    assert refusal is None and notes == [], (refusal, notes)
    refusal, notes = check_placement(placement_entry("third", gpu="1",
                                                     util=0.4),
                                     running=[("first", (1,), 0.4),
                                              ("second", (1,), 0.3)])
    assert refusal is None and "1.10 in total" in notes[0], (refusal, notes)

    refusal, notes = check_placement(placement_entry(gpu="1"), chat_gpus=[0],
                                     bge_gpu=1)
    assert refusal is None, "the BGE card was refused"
    assert len(notes) == 1 and f"about {BGE_CARD_MB} MB" in notes[0], notes
    assert "BGE encoder" in notes[0], notes
    refusal, notes = check_placement(placement_entry(), chat_gpus=[0])
    assert refusal is None and len(notes) == 1, (refusal, notes)
    assert "names no gpu" in notes[0] and "spawn.gpu" in notes[0], notes

    args = cli.build_parser().parse_args(["--gpu", "0,1"])
    from salt.chat.serve import parse_gpu_list
    args.device, args.bge_device, args.gpu_mem_util, _ = \
        cli.resolve_gpu_devices(parse_gpu_list(args.gpu), args.device,
                                args.bge_device, args.gpu_mem_util)
    assert cli.cuda_index(args.device) == 0, args.device
    assert cli.cuda_index(args.bge_device) == 1, args.bge_device
    assert cli.cuda_index("cuda") == 0, "bare cuda is not read as card 0"
    assert cli.cuda_index("cuda:3") == 3
    assert cli.cuda_index("cpu") is None and cli.cuda_index(None) is None
    src = (REPO / "salt" / "chat" / "cli.py").read_text(encoding="utf-8")
    assert 'self.chat_gpus = [] if args.backend == "vllm-serve"' in src, (
        "a session whose model lives in another process still claims a card")
    assert src.index("refusal, notes = check_placement") < src.index(
        "handle.start(state.workers_dir())"), (
        "placement is checked after the spawn, which would be pointless")
    print("11. placement: a taken card refused unless both sides declare a "
          "share, over-subscription warned, and the BGE card allowed with "
          "its note")


def worker_entry(url, tok_path, timeout_s=3):
    return R.RosterEntry(name="w", alias="stub", role="worker", server_url=url,
                         model={"alias": "stub", "hf_id": "some/model",
                                "path": tok_path},
                         timeout_s=timeout_s)


def quiet_call(handle):
    with redirect_stdout(io.StringIO()):
        return "".join(handle.call([{"role": "user", "content": "hi"}]))


def failed_call(handle):
    with redirect_stdout(io.StringIO()):
        try:
            "".join(handle.call([{"role": "user", "content": "hi"}]))
        except Exception as exc:
            return exc
    raise AssertionError("the call was supposed to fail and did not")


def check_call_timeout(tok_path):
    h = WorkerHandle(worker_entry("http://x", tok_path, timeout_s=None))
    assert h.timeout_s == CALL_TIMEOUT == 300, h.timeout_s
    assert WorkerHandle(worker_entry("http://x", tok_path,
                                     timeout_s=12)).timeout_s == 12
    runner_src = (REPO / "salt" / "chat" / "runner_serve.py").read_text(
        encoding="utf-8")
    assert "timeout=(5, self.read_timeout)" in runner_src, (
        "the serve client no longer passes its read timeout to the request")
    cli_src = (REPO / "salt" / "chat" / "cli.py").read_text(encoding="utf-8")
    assert "read_timeout" not in cli_src, (
        "the chat model's own client now passes a read timeout, so a slow "
        "reply from the model the session depends on can be cut off")

    import requests
    assert is_read_timeout(requests.exceptions.ReadTimeout("x"))
    assert not is_read_timeout(requests.exceptions.ConnectTimeout("x"))
    assert is_read_timeout(requests.exceptions.ConnectionError(
        "HTTPConnectionPool: ReadTimeoutError(read timeout=1)"))
    assert not is_read_timeout(requests.exceptions.ConnectionError("refused"))
    assert not is_read_timeout(ValueError("nope"))

    # past 512 bytes, so text really reaches the caller through requests'
    # chunked line reader before the worker goes quiet
    sent = [f"word{i:03d} " for i in range(120)]
    with Stub(cards=CARDS, pieces=sent, stall=6) as s:
        h = WorkerHandle(worker_entry(s.url, tok_path, timeout_s=1))
        got = []
        t0 = time.monotonic()
        with redirect_stdout(io.StringIO()):
            try:
                for piece in h.call([{"role": "user", "content": "hi"}]):
                    got.append(piece)
                raise AssertionError("the stalled call never timed out")
            except WorkerError as exc:
                msg = str(exc)
        took = time.monotonic() - t0
        assert got and got == sent[:len(got)], (
            f"{len(got)} pieces arrived and they are not a prefix of what "
            f"the worker sent")
        assert 1.0 <= took < 8, f"the timeout took {took:.1f}s for a 1s limit"
        assert "sent nothing for 1s" in msg and "given up on" in msg, msg
        assert h.state == READY, f"a timed-out worker read {h.state}"
        assert h.failures == 0, (
            f"a stall counted as {h.failures} failures toward DEAD, but the "
            f"worker is alive and took the next call")
        assert h.last_error == msg, h.last_error
        assert s.aborted.wait(20), (
            "the timeout did not sever the response, so the worker would "
            "keep generating")
        s.httpd.stall, s.httpd.pieces = 0, ["fine"]
        assert quiet_call(h) == "fine", "the handle refused the next call"
        assert h.state == READY and h.calls == 2, (h.state, h.calls)
        h.close()
    print(f"12. call timeouts: {len(sent)} pieces streamed, then a quiet "
          f"worker was given up on at its own limit, severed, and took the "
          f"next call")


def check_retry_policy(tok_path):
    assert (CALL_RETRIES, FAILURES_TO_DEAD) == (1, 2), (
        f"the policy constants moved: {CALL_RETRIES}, {FAILURES_TO_DEAD}")
    import requests
    assert is_connection_error(requests.exceptions.ConnectionError("x"))
    assert is_connection_error(requests.exceptions.ConnectTimeout("x"))
    assert not is_connection_error(requests.exceptions.ReadTimeout("x"))
    assert not is_connection_error(requests.exceptions.ConnectionError(
        "ReadTimeoutError(read timeout=1)"))

    port = free_port()
    first = Stub(cards=CARDS, pieces=("he", "llo"), port=port)
    h = WorkerHandle(worker_entry(first.url, tok_path))
    assert quiet_call(h) == "hello"
    first.stop()
    second = Stub(cards=CARDS, pieces=("re", "tried"), port=port)
    try:
        assert quiet_call(h) == "retried", "the call after a restart failed"
        assert h.retries == 0, (
            f"{h.retries} retries spent on a restart the connection pool "
            f"already handles by dropping the dead connection")
    finally:
        h.close()
        second.stop()

    port = free_port()
    down = Stub(cards=CARDS, pieces=("re", "tried"), port=port, serving=False)
    up = Stub(cards=CARDS, port=port)
    h = WorkerHandle(worker_entry(down.url, tok_path))
    with redirect_stdout(io.StringIO()):
        h.ready()
    up.stop()
    timer = threading.Timer(RETRY_DELAY / 2, down.start)
    timer.start()
    try:
        t0 = time.monotonic()
        text = quiet_call(h)
        took = time.monotonic() - t0
        assert text == "retried", f"the retried call returned {text!r}"
        assert h.retries == 1, f"retries counted {h.retries}"
        assert took >= RETRY_DELAY, (
            f"the retry went out after {took:.2f}s, inside the {RETRY_DELAY}s "
            f"backoff, so a server coming back up gets no time")
        assert down.posts == 1, f"the server saw {down.posts} requests"
        assert h.failures == 0 and h.state == READY, (h.failures, h.state)
        assert h.calls == 1, f"the retry inflated the call count to {h.calls}"
    finally:
        timer.cancel()
        h.close()
        down.stop()

    sent = [f"word{i:03d} " for i in range(120)]
    with Stub(cards=CARDS, pieces=sent, drop=True) as s:
        h = WorkerHandle(worker_entry(s.url, tok_path))
        got = []
        with redirect_stdout(io.StringIO()):
            try:
                for piece in h.call([{"role": "user", "content": "hi"}]):
                    got.append(piece)
            except Exception:
                pass
        assert got, "nothing streamed before the drop, so this proves nothing"
        assert s.posts == 1, (
            f"the worker saw {s.posts} requests, so a half-written reply was "
            f"asked for again and its text would be duplicated")
        assert h.retries == 0, h.retries
        h.close()

    port = free_port()
    s = Stub(cards=CARDS, port=port)
    h = WorkerHandle(worker_entry(s.url, tok_path))
    assert quiet_call(h) == "hello"
    s.stop()
    failed_call(h)
    assert h.failures == 1 and h.state != DEAD, (h.failures, h.state)
    assert h.retries == 1, f"the failed call spent {h.retries} retries"
    back = Stub(cards=CARDS, port=port)
    assert quiet_call(h) == "hello"
    assert h.failures == 0, (
        f"a good call left {h.failures} failures, so DEAD would not need "
        f"two IN A ROW")
    back.stop()
    failed_call(h)
    assert h.state != DEAD and h.failures == 1, (h.state, h.failures)
    failed_call(h)
    assert h.failures == FAILURES_TO_DEAD and h.state == DEAD, h.failures
    assert h.last_error, "DEAD with no reason recorded"
    revived = Stub(cards=CARDS, pieces=("a", "live"), port=port)
    try:
        assert h.probe(timeout=3).state == PROBED, h.note
        assert h.failures == 0, f"a revived worker kept {h.failures} failures"
        assert quiet_call(h) == "alive"
    finally:
        h.close()
        revived.stop()
    print("13. retry policy: a refused connection retried once after its "
          "backoff, a half-written reply never retried, and DEAD only after "
          "two failures in a row, cleared by a probe")


class _FakeState:
    """What the /worker command reads off a session, without a chat model,
    a trie or a GPU."""

    def __init__(self, tmp, entries, chat_gpus=(), chat_mem_util=None,
                 bge_gpu=None):
        self.roster = (None if entries is None
                       else R.Roster(path=str(Path(tmp) / "r.json"),
                                     entries=tuple(entries)))
        self.workers = {}
        self._dir = Path(tmp) / "session" / "workers"
        self.chat_gpus = list(chat_gpus)
        self.chat_mem_util = chat_mem_util
        self.bge_gpu = bge_gpu

    def workers_dir(self):
        return self._dir

    worker = cli.ChatState.worker
    worker_handles = cli.ChatState.worker_handles
    running_workers = cli.ChatState.running_workers


def worker_line(state, line):
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.worker_command(state, line.split())
    return buf.getvalue()


def check_worker_commands(tmp):
    st = _FakeState(tmp, [spawn_entry(tmp, "w")])
    out = worker_line(st, "")
    assert "NAME" in out and "w" in out, out
    for bad in ("bogus", "start", "stop", "probe", "start a b", "stop --all",
                "probe a b"):
        out = worker_line(st, bad)
        assert "Usage: /worker" in out, f"{bad!r} printed {out!r}"
        assert "start --all" in out, "the usage does not name every verb"
    out = worker_line(st, "start nosuch")
    assert "No roster entry named 'nosuch'" in out and "known: w" in out, out
    assert "no roster loaded" in worker_line(_FakeState(tmp, None), "start w")

    st = _FakeState(tmp, [spawn_entry(tmp, "w", "--delay", "0.4")])
    out = worker_line(st, "start w")
    assert "starting stub at" in out, out
    assert "ready, serving some/model" in out and "window 4096" in out, out
    assert "PROBED" in out, out
    h = st.worker("w")
    assert h.state == PROBED and h.process.poll() is None, h.state
    assert (st.workers_dir() / "w.log").exists(), "no log under the session"
    assert (st.workers_dir() / "w.json").exists(), "no pid record"
    assert "already running" in worker_line(st, "start w"), (
        "starting a running worker spawned a second one")
    h.state = BUSY
    out = worker_line(st, "stop w")
    assert "answering a call right now" in out and "Ctrl-C" in out, out
    assert h.process is not None and h.process.poll() is None, (
        "a worker mid-call was stopped out from under the caller")
    h.state = PROBED
    assert "stopped, exit code" in worker_line(st, "stop w")
    assert h.process is None and h.state == DECLARED, (h.process, h.state)
    assert "nothing running" in worker_line(st, "stop w"), out

    att = _FakeState(tmp, [attach_entry()])
    out = worker_line(att, "start att")
    assert "attached to http://127.0.0.1:8099" in out, out
    assert "nothing for this session to start" in out, out
    assert att.worker("att").process is None, "an attach entry spawned"
    assert "must not stop it" in worker_line(att, "stop att")

    every = _FakeState(tmp, [spawn_entry(tmp, "a"), attach_entry("b"),
                             spawn_entry(tmp, "c", "--die", "0.1")])
    out = worker_line(every, "start --all")
    assert "Starting 2 of the roster's workers" in out, out
    assert "b:" not in out, f"the attach entry was touched: {out}"
    assert "1 of 2 ready" in out, out
    assert "exited with code 7" in out and "CUDA out of memory" in out, out
    assert every.worker("a").state == PROBED, every.worker("a").state
    assert every.worker("c").state == DEAD, every.worker("c").state
    worker_line(every, "stop a")
    assert "no spawn entries in the roster" in worker_line(
        _FakeState(tmp, [attach_entry("b")]), "start --all")

    clash = _FakeState(tmp, [placement_entry("clash", gpu="0")],
                       chat_gpus=[0], chat_mem_util=0.85, bge_gpu=0)
    out = worker_line(clash, "start clash")
    assert "already carries the chat model" in out, out
    assert clash.worker("clash").process is None, "it spawned anyway"

    assert cli.build_parser().parse_args([]).workers_autostart is False, (
        "--workers-autostart is not off by default")
    assert cli.build_parser().parse_args(
        ["--workers-autostart"]).workers_autostart is True
    src = (REPO / "salt" / "chat" / "cli.py").read_text(encoding="utf-8")
    assert "args.workers_autostart and state.roster is not None" in src
    assert src.index("start_all_workers(state)") < src.index(
        "for doc in args.doc:"), (
        "autostart runs after the --doc ingests, so a document is compressed "
        "before the workers it might be delegated to exist")
    assert 'self.trie.cache_dir / "workers"' in src, (
        "the workers dir is not anchored on this session's own cache dir, "
        "so two conversations would fight over one worker name")
    print("14. /worker commands: 7 malformed lines refused, start and stop "
          "over a real child, a BUSY worker protected, and start --all "
          "reporting 1 of 2 up")


def replayed_state(tmp, cid, tok, mdl, turns=TRANSCRIPT, roster=None,
                   flags=()):
    """A session with real memory in it, left open for inspection."""
    args = cli.build_parser().parse_args(["--device", "cpu", "--sync-ingest",
                                          *flags])
    trie = SessionTrie(cid, cache_dir=tmp, model_name=BGE_MODEL,
                       budget_pct_default=args.budget_pct)
    state = cli.ChatState(args, tok, mdl, _FakeRunner(tok, REPLIES), trie,
                          roster)
    with redirect_stdout(io.StringIO()):
        for line in turns:
            cli.chat_turn(state, line)
    return state


def trie_snapshot(trie):
    """Everything a committed turn would move."""
    return {"coverage": dict(trie.coverage), "drift_ema": trie.drift_ema,
            "keyword_weights": json.loads(json.dumps(trie.keyword_weights)),
            "n_sentences": trie.n_sentences, "alive": list(trie.alive),
            "texts": list(trie.texts), "turns": list(trie.turns)}


def check_delegation_context(tmp, tok, mdl):
    req = D.DelegationRequest(task="Summarize the sizing argument.")
    assert req.query == req.task, "the task is not its own context query"
    assert D.DelegationRequest(task="t", context_query="q").query == "q"
    assert (req.target, req.ingest, req.budget_pct) == (None, False, None), req
    assert not D.DelegationContext().text, "an empty context carries text"
    assert D.DelegationContext().empty and D.DelegationContext().n_selected == 0

    instructions = D.worker_instructions()
    assert instructions and "TASK:" in instructions, instructions[:120]
    assert instructions != D.FALLBACK_INSTRUCTIONS, (
        "the shipped worker prompt is the fallback, so the file is missing")
    real, D.INSTRUCTIONS_PATH = D.INSTRUCTIONS_PATH, Path("/nonexistent.md")
    try:
        assert D.worker_instructions() == D.FALLBACK_INSTRUCTIONS, (
            "an unreadable worker prompt takes the delegation down with it")
    finally:
        D.INSTRUCTIONS_PATH = real
    assert '"worker_instructions.md"' in (
        REPO / "pyproject.toml").read_text(encoding="utf-8"), (
        "the worker prompt is not package data, so an installed wheel would "
        "fall back to the built-in wording")

    state = replayed_state(tmp, "delegation_ctx", tok, mdl)
    try:
        assert state.tail, "the fixture built no verbatim tail to exclude"
        assert state.tail_exclude, "the fixture session has tail exclusion off"
        before, tail_before = trie_snapshot(state.trie), list(state.tail)
        with redirect_stdout(io.StringIO()):
            ctx = D.build_context(state, req)
        assert ctx.text and ctx.n_selected, "the delegation context is empty"
        assert ctx.words_used == ctx.stats["words_used"], ctx.stats
        assert ctx.stats["excluded_sent"] == 0, (
            f"{ctx.stats['excluded_sent']} sentences were held back as tail "
            f"resident, but a worker never sees the tail, so they are "
            f"ordinary context for it")

        comp = state.trie.compress(
            query=req.task, budget_pct=state.budget, tokenizer=state.bge_tok,
            model=state.bge_model, device=state.bge_device,
            coverage_half_life=state.coverage_half_life,
            coverage_decay_docs=state.coverage_decay_docs,
            shift_damping=state.shift_damping,
            shift_margin=state.shift_margin,
            shift_query_boost=state.shift_query_boost,
            per_source_themes=state.per_source_themes,
            max_words=cli.memory_word_cap(state, req.task),
            stable_keys=state.stable_coverage_keys,
            coverage_gc=state.coverage_gc,
            coverage_max_keys=state.coverage_max_keys,
            defer_commit=True, exclude_sent_idx=None)
        reference = cli.format_memory_block(state.trie,
                                            comp["selected_sent_idx"],
                                            state.turn_labels,
                                            state.conversation_map)
        assert ctx.text == reference, (
            "the delegation context is not what the same deferred compress "
            "produces, so a worker reads the conversation differently from "
            "the chat model")
        assert list(ctx.selected_idx) == comp["selected_sent_idx"], (
            ctx.selected_idx, comp["selected_sent_idx"])
        assert comp.get("commit") is not None, (
            "the deferred compress returned no commit, so this pin proves "
            "nothing about dropping it")

        with redirect_stdout(io.StringIO()):
            for _ in range(50):
                D.build_context(state, req)
        after = trie_snapshot(state.trie)
        for key in before:
            assert before[key] == after[key], (
                f"51 delegations moved trie.{key}, so delegating is visible "
                f"in the session's own memory")
        assert state.tail == tail_before, "delegating moved the verbatim tail"

        # budget_pct is a fraction, so 1.0 is the whole corpus
        wide = D.DelegationRequest(task=req.task, budget_pct=1.0)
        with redirect_stdout(io.StringIO()):
            whole = D.build_context(state, wide)
        assert whole.words_used > ctx.words_used, (
            f"a full budget selected {whole.words_used} words against "
            f"{ctx.words_used} at the session's {state.budget}, so the "
            f"request's budget_pct is not reaching the compress")
    finally:
        with redirect_stdout(io.StringIO()):
            cli.close_ingest(state)

    empty = replayed_state(tmp, "delegation_empty", tok, mdl, turns=())
    try:
        assert empty.trie.n_sentences == 0, "the empty fixture has memory"
        with redirect_stdout(io.StringIO()):
            blank = D.build_context(empty, req)
        assert blank.empty and blank.n_selected == 0 and blank.stats == {}, (
            "a session with no memory did not yield an empty context, so a "
            "pure-task delegation would fail before it started")
    finally:
        with redirect_stdout(io.StringIO()):
            cli.close_ingest(empty)
    print("15. delegation context: byte-identical to the same deferred "
          "compress, tail-resident rows kept, and 51 of them moved nothing "
          "in the session's memory")


def delegation_roster(url, tmp, **kw):
    entry = R.RosterEntry(name="w", alias="stub", role="worker",
                          server_url=url,
                          model={"alias": "stub", "hf_id": "some/model",
                                 "path": BGE_MODEL}, **kw)
    return R.Roster(path=str(Path(tmp) / "r.json"), entries=(entry,))


def run_delegation(state, **kw):
    req = D.DelegationRequest(**kw)
    with redirect_stdout(io.StringIO()):
        return req, D.delegate(state, req)


def check_delegation_call(tmp, tok, mdl):
    try:
        D.delegate(object(), D.DelegationRequest(task="t"))
        raise AssertionError("a delegation with no target was sent")
    except D.DelegationError as exc:
        assert "names none" in str(exc), exc

    with Stub(cards=CARDS, pieces=("the ", "answer")) as s:
        state = replayed_state(tmp, "delegation_call", tok, mdl,
                               roster=delegation_roster(s.url, tmp,
                                                        max_tokens=128))
        try:
            assert state.delegation_seq == 0, "a fresh session has ids spent"
            req, res = run_delegation(state, task="What size battery?",
                                      target="w")
            assert res.status == "ok" and res.text == "the answer", res
            assert res.id == 1 and res.target == "w" and res.task == req.task
            assert not res.error and res.ok, res.error
            assert res.t_end >= res.t_start and res.seconds >= 0, res
            assert set(res.usage) == {"prompt_tokens", "cached_tokens",
                                      "output_tokens"}, sorted(res.usage)
            assert res.usage["prompt_tokens"] > 0, res.usage
            assert res.usage["output_tokens"] == 2, res.usage
            assert res.context.n_selected, "the reply carries no context"
            assert s.httpd.last_payload["max_tokens"] == 128, (
                "the roster entry's max_tokens did not reach the worker")

            messages = D.build_messages(res.context, req)
            assert [m["role"] for m in messages] == ["system", "user"]
            assert messages[0]["content"] == D.worker_instructions()
            assert messages[1]["content"].endswith(
                f"{D.TASK_HEADER}{req.task}"), messages[1]["content"][-80:]
            assert messages[1]["content"].startswith(res.context.text), (
                "the context is not what the task sits under")
            bare = D.build_messages(D.DelegationContext(), req)
            assert bare[1]["content"] == f"{D.TASK_HEADER}{req.task}", (
                "a delegation with no context lost its task header")

            _, second = run_delegation(state, task="And the inverter?",
                                       target="w", max_tokens=64)
            assert second.id == 2, f"ids are not monotonic: {second.id}"
            assert s.httpd.last_payload["max_tokens"] == 64, (
                "the request's max_tokens did not win over the entry's")

            # D5: worker text is data. A reply shaped like a directive comes
            # back as the string it is, and nothing here parses it
            directive = '{"tool": "salt_compress", "args": {"budget": 0.5}}'
            s.httpd.pieces = [directive[:20], directive[20:]]
            _, echoed = run_delegation(state, task="emit a directive",
                                       target="w")
            assert echoed.text == directive, echoed.text
            assert echoed.status == "ok" and isinstance(echoed.text, str)
            assert set(echoed.usage) == {"prompt_tokens", "cached_tokens",
                                         "output_tokens"}, (
                "the result grew a field parsed out of the worker's text")
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)

    with Stub(cards=CARDS, post_status=503) as s:
        state = replayed_state(tmp, "delegation_error", tok, mdl,
                               roster=delegation_roster(s.url, tmp))
        try:
            _, res = run_delegation(state, task="anything", target="w")
            assert res.status == "error", res.status
            assert res.text == "", f"a rejected request returned {res.text!r}"
            assert "503" in res.error and "no model is loaded" in res.error, (
                f"the server's own words are missing from {res.error!r}")
            assert state.worker("w").state != DEAD, (
                "one rejected request condemned the worker")
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)

    sent = [f"word{i:03d} " for i in range(120)]
    with Stub(cards=CARDS, pieces=sent, stall=6) as s:
        state = replayed_state(tmp, "delegation_timeout", tok, mdl,
                               roster=delegation_roster(s.url, tmp))
        try:
            handle = state.worker("w")
            assert handle.timeout_s == CALL_TIMEOUT, handle.timeout_s
            t0 = time.monotonic()
            _, res = run_delegation(state, task="stall please", target="w",
                                    timeout_s=1)
            took = time.monotonic() - t0
            assert res.status == "timeout", (
                f"a worker that went quiet came back {res.status}: "
                f"{res.error}")
            assert took < 20, (
                f"the request's own timeout_s never reached the client, so "
                f"the call ran {took:.1f}s against the roster's "
                f"{handle.timeout_s}s")
            assert res.text and res.text.startswith("word000"), (
                "the partial reply was thrown away with the timeout")
            assert handle.runner.read_timeout == handle.timeout_s, (
                f"the request's timeout stuck to the client at "
                f"{handle.runner.read_timeout}, so every later call inherits "
                f"one delegation's limit")
            assert handle.state != DEAD, "a stall condemned the worker"
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)

    dead = Stub(cards=CARDS, port=free_port())
    state = replayed_state(tmp, "delegation_dead", tok, mdl,
                           roster=delegation_roster(dead.url, tmp))
    try:
        run_delegation(state, task="warm the client", target="w")
        dead.stop()
        _, first = run_delegation(state, task="anyone there", target="w")
        _, again = run_delegation(state, task="anyone there", target="w")
        assert (first.status, again.status) == ("error", "dead"), (
            f"a vanished worker read {first.status} then {again.status}, "
            f"expected one failure to be survivable and the second not")
        assert state.worker("w").state == DEAD, state.worker("w").state
        assert again.error, "a dead worker came back with no reason"
    finally:
        with redirect_stdout(io.StringIO()):
            cli.close_ingest(state)
        dead.stop()
    print("16. executing a delegation: text and usage captured, a directive "
          "shaped reply returned verbatim, and a rejection, a stall and a "
          "vanished server each named as what they are")


def offload_line(state, line):
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.offload_command(state, line.split())
    return buf.getvalue()


def check_offload_command(tmp, tok, mdl):
    with Stub(cards=CARDS, pieces=("a ", "battery")) as s:
        roster = delegation_roster(s.url, tmp)
        state = replayed_state(tmp, "offload_one", tok, mdl, roster=roster)
        try:
            for bad in ("", "@w", "   "):
                out = offload_line(state, bad)
                assert "Usage: /offload" in out, f"{bad!r} printed {out!r}"
                assert "@NAME" in out, "the usage does not show naming one"
            assert state.delegation_seq == 0, "a usage error spent an id"

            out = offload_line(state, "what size battery")
            assert "delegating to w (stub)" in out, out
            assert "of context" in out and "words]" in out, out
            assert "a battery" in out, f"the worker's reply is not shown: {out}"
            status = [ln for ln in out.splitlines() if ln.startswith("  [w]")]
            assert len(status) == 1, f"expected one status line, got {status}"
            assert " ok, " in status[0] and " in, " in status[0], status[0]
            assert status[0].rstrip().endswith("s"), status[0]
            assert state.delegation_seq == 1, state.delegation_seq
            assert s.httpd.posts == 1, s.httpd.posts

            out = offload_line(state, "@w and the inverter")
            assert "delegating to w" in out, out
            assert state.delegation_seq == 2, state.delegation_seq
            out = offload_line(state, "@nosuch anything")
            assert "known:" in out and "nosuch" in out, out
            assert state.delegation_seq == 2, "an unknown name spent an id"

            before = trie_snapshot(state.trie)
            offload_line(state, "one more question")
            assert trie_snapshot(state.trie) == before, (
                "delegating through /offload moved the session's memory")
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)

    two = R.Roster(path=str(Path(tmp) / "two.json"), entries=(
        R.RosterEntry(name="a", alias="stub", role="worker",
                      server_url="http://127.0.0.1:1", model=None),
        R.RosterEntry(name="b", alias="stub", role="worker",
                      server_url="http://127.0.0.1:2", model=None)))
    state = replayed_state(tmp, "offload_two", tok, mdl, turns=(), roster=two)
    try:
        out = offload_line(state, "who does this")
        assert "2 workers" in out and "/offload @NAME" in out, out
        assert "a, b" in out, f"the refusal lists no names: {out}"
        state.roster = None
        assert "No roster loaded" in offload_line(state, "anything")
    finally:
        with redirect_stdout(io.StringIO()):
            cli.close_ingest(state)

    with Stub(cards=CARDS, post_status=503) as s:
        state = replayed_state(tmp, "offload_fail", tok, mdl,
                               roster=delegation_roster(s.url, tmp))
        try:
            out = offload_line(state, "anything at all")
            assert "[w] error," in out, out
            assert "503" in out, f"the reason never reached the user: {out}"
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)

    # Ctrl-C mid-delegation: the REPL catches KeyboardInterrupt itself, so
    # what has to hold here is that the response was severed and the worker
    # is free for the next task
    slow = Stub(cards=CARDS, pieces=[f"t{i} " for i in range(400)], delay=0.05)
    state = replayed_state(tmp, "offload_ctrlc", tok, mdl,
                           roster=delegation_roster(slow.url, tmp))
    try:
        timer = threading.Timer(1.0, _thread.interrupt_main)
        timer.start()
        interrupted = False
        try:
            offload_line(state, "talk for a long time")
        except KeyboardInterrupt:
            interrupted = True
        finally:
            timer.cancel()
        assert interrupted, (
            "the delegation ran to completion, so this proves nothing about "
            "interrupting one")
        handle = state.worker("w")
        assert slow.aborted.wait(20), (
            "the interrupt did not sever the response, so the worker would "
            "keep generating into nothing")
        assert handle.state == READY, (
            f"an interrupted delegation left the worker {handle.state}")
        slow.httpd.pieces, slow.httpd.delay = ["fine"], 0.0
        out = offload_line(state, "still there")
        assert "fine" in out and "[w] ok," in out, (
            f"the worker did not take the next task: {out}")
    finally:
        with redirect_stdout(io.StringIO()):
            cli.close_ingest(state)
        slow.stop()
    print("17. /offload: one worker needs no naming and several refuse "
          "without @NAME, the reply and one status line are printed, and "
          "Ctrl-C severs the call and leaves the worker ready")


def _no_space(*a, **kw):
    raise OSError("no space left on device")


def ledger_lines(session_dir):
    path = L.ledger_path(session_dir)
    return [json.loads(ln) for ln
            in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def check_delegation_ledger(tmp, tok, mdl):
    with Stub(cards=CARDS, pieces=("a ", "battery")) as s:
        roster = delegation_roster(s.url, tmp)
        state = replayed_state(tmp, "ledger", tok, mdl, roster=roster)
        home = state.trie.cache_dir
        try:
            assert not L.ledger_path(home).exists(), (
                "a session that has delegated nothing already has a ledger")
            offload_line(state, "what size battery")
            offload_line(state, "@w and the inverter")
            rows = ledger_lines(home)
            assert len(rows) == 2, f"two delegations wrote {len(rows)} lines"
            rec = rows[0]
            assert tuple(rec) == L.FIELDS, (
                f"the ledger schema changed: {tuple(rec)}")
            assert rec["schema"] == "salt-delegation/1", rec
            assert (rec["id"], rec["target"], rec["status"]) == (1, "w", "ok")
            assert rec["task"] == "what size battery", rec
            assert tuple(rec["context_stats"]) == L.CONTEXT_FIELDS, rec
            assert rec["context_stats"]["n_selected"] > 0, rec
            assert rec["context_stats"]["words_used"] > 0, rec
            assert rec["usage"]["output_tokens"] > 0, rec
            assert rec["ingest"] is False, "a result was filed as ingested"
            assert rec["t_end"] >= rec["t_start"] > 0, rec
            assert rows[1]["id"] == 2, rows[1]

            # the ledger is history, so a lost line must not lose the answer
            real_append, L.append = L.append, _no_space
            try:
                out = offload_line(state, "will not be filed")
                assert "a battery" in out, (
                    f"a failed record swallowed the worker's answer: {out}")
                assert "recording #3" in out and "no space" in out, out
            finally:
                L.append = real_append
            assert len(ledger_lines(home)) == 2, "the failed append wrote"
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)

        state = replayed_state(tmp, "ledger", tok, mdl, turns=(),
                               roster=roster)
        try:
            assert state.delegation_seq == 2, (
                f"resuming restarted the ids at {state.delegation_seq}")
            offload_line(state, "one more question")
            assert [r["id"] for r in ledger_lines(home)] == [1, 2, 3], (
                "a resumed session reused an id")
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)

    with Stub(cards=CARDS, post_status=503) as s:
        state = replayed_state(tmp, "ledger_fail", tok, mdl, turns=(),
                               roster=delegation_roster(s.url, tmp))
        try:
            offload_line(state, "anything at all")
            rec = ledger_lines(state.trie.cache_dir)[0]
            assert rec["status"] == "error", (
                f"a failed delegation was filed as {rec['status']!r}")
            assert tuple(rec) == L.FIELDS, rec
            assert rec["context_stats"]["n_selected"] == 0, (
                "a session with no memory was given context")
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)

    torn = Path(tmp) / "torn"
    torn.mkdir()
    empty = L.read(torn)
    assert not empty.records and not empty.warnings and empty.last_id == 0, (
        "a missing ledger is not read as an empty one")
    older = {"schema": "salt-delegation/1", "id": 1, "target": "w",
             "task": "an earlier task",
             "context_stats": {"n_selected": 2, "words_used": 40},
             "status": "ok", "usage": {}, "t_start": 1.0, "t_end": 2.0,
             "ingest": False}
    L.ledger_path(torn).write_text(
        json.dumps(older) + "\n"
        '{"schema": "salt-delegation/2", "id": 7, "wrote": "the future"}\n'
        '{"schema": "salt-delegation/1", "id": 9, "target": "w", "ta',
        encoding="utf-8")
    found = L.read(torn)
    assert [r["id"] for r in found.records] == [1], (
        f"a line this salt cannot read was loaded anyway: {found.records}")
    assert found.last_id == 7, (
        f"the ledger hands id {found.last_id + 1} out twice after a newer "
        f"salt wrote id 7")
    assert len(found.warnings) == 2, found.warnings
    assert any("did not finish writing" in w for w in found.warnings), found
    assert any("salt-delegation/2" in w for w in found.warnings), found

    buf = io.StringIO()
    with redirect_stdout(buf):
        state = replayed_state(tmp, "torn", tok, mdl, turns=())
    out = buf.getvalue()
    assert "line 3" in out and "line 2" in out, (
        f"opening a damaged ledger said nothing about it: {out}")
    assert state.delegation_seq == 7, state.delegation_seq
    real, cli.SESSIONS_DIR = cli.SESSIONS_DIR, Path(tmp)
    try:
        with redirect_stdout(io.StringIO()):
            state.new_trie("ledger_fresh")
        assert state.delegation_seq == 0, (
            "a new session inherited the previous one's delegation ids")
        with redirect_stdout(io.StringIO()):
            state.new_trie("ledger")
        assert state.delegation_seq == 3, (
            f"switching back to a session lost its ids ({state.delegation_seq})")
    finally:
        cli.SESSIONS_DIR = real
        with redirect_stdout(io.StringIO()):
            cli.close_ingest(state)
    print("18. the delegation ledger: one line per delegation in the pinned "
          "schema, ids resume from the file, and a torn or newer line is "
          "skipped with a warning")


def bare_trie(tmp, cid, budget=0.2):
    return SessionTrie(cid, cache_dir=tmp, model_name=BGE_MODEL,
                       budget_pct_default=budget)


def check_worker_rows(tmp, tok, mdl):
    assert "worker" in VALID_ROLES, VALID_ROLES
    assert tuple(CONVERSATION_ROLES) == ("user", "assistant", "worker"), (
        f"a role changed sides between conversation and document: "
        f"{CONVERSATION_ROLES}")
    assert "doc" not in CONVERSATION_ROLES, (
        "attachments are being treated as somebody speaking")

    kw = dict(tokenizer=tok, model=mdl, device="cpu")
    trie = bare_trie(tmp, "worker_rows")
    trie.add_turn("The inverter is rated at 5 kW continuous.",
                  role="assistant", **kw)
    # a worker restating the ASSISTANT is not a duplicate of it: the gate
    # is same-role, and the roles differ
    info = trie.add_turn("the inverter is rated at 5 kW continuous",
                         role="worker", origin="w", dedup_cos=0.9, **kw)
    assert (info["added"], info["near_dups"]) == (1, 0), (
        f"a worker row was gated against another role's sentence: {info}")
    # a worker restating ITSELF is
    info = trie.add_turn("The inverter is rated at 5 kW, continuous",
                         role="worker", origin="w", dedup_cos=0.9, **kw)
    assert (info["added"], info["near_dups"]) == (0, 1), (
        f"the near-dup gate never ran for a worker row: {info}")
    assert trie.roles == ["assistant", "worker"], trie.roles
    assert trie.origins == [None, "w"], trie.origins
    assert len(trie.origins) == len(trie.texts) == len(trie.timestamps), (
        "origins is not parallel to the corpus")
    try:
        trie.add_turn("anything", role="orchestrator", **kw)
        raise AssertionError("an unknown role was accepted")
    except ValueError as exc:
        assert "role must be one of" in str(exc), exc

    trie.save()
    reloaded = bare_trie(tmp, "worker_rows")
    assert reloaded.load(), "the session did not reload"
    assert reloaded.origins == trie.origins, (
        f"origins did not survive a save: {reloaded.origins}")

    # a session written before origins existed: the key is simply absent
    sp = Path(tmp) / "worker_rows" / "state.pkl"
    state = pickle.loads(sp.read_bytes())
    del state["origins"]
    sp.write_bytes(pickle.dumps(state))
    old = bare_trie(tmp, "worker_rows")
    assert old.load(), "an older session refused to load"
    assert old.origins == [None] * len(old.texts), (
        f"the backfill guessed instead of leaving it unknown: {old.origins}")
    assert old.roles == trie.roles, "the backfill disturbed the roles"

    # worker rows are conversation rows, so a session cap retires them
    capped = bare_trie(tmp, "worker_cap")
    capped.add_turn("The panels come to 9 kW on the roof.", role="user", **kw)
    capped.add_turn("A worker said the inverter caps output at 5 kW.",
                    role="worker", origin="w", **kw)
    info = capped.add_turn("Winter evenings draw for about 4 hours.",
                           role="user", max_sentences=2, **kw)
    assert info["masked"] == 1, (
        f"the cap did not count the worker row: {info}")
    assert capped.alive == [False, True, True], capped.alive
    assert capped.n_alive == 2, capped.n_alive

    # end to end: the flag is what decides whether an answer is remembered
    answer = ("A battery of about 9 kWh ", "covers the winter evening draw.")
    with Stub(cards=CARDS, pieces=answer) as s:
        roster = delegation_roster(s.url, tmp)
        off = replayed_state(tmp, "ingest_off", tok, mdl, roster=roster)
        try:
            assert off.offload_ingest is False, "--offload-ingest is on by default"
            before = trie_snapshot(off.trie)
            offload_line(off, "what size battery")
            assert trie_snapshot(off.trie) == before, (
                "a result was remembered without --offload-ingest")
            assert ledger_lines(off.trie.cache_dir)[0]["ingest"] is False
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(off)

        on = replayed_state(tmp, "ingest_on", tok, mdl, roster=roster,
                            flags=("--offload-ingest",))
        try:
            assert on.offload_ingest is True, on.offload_ingest
            n_before, tail_before = on.trie.n_sentences, list(on.tail)
            offload_line(on, "what size battery")
            added = on.trie.n_sentences - n_before
            assert added == 1, f"the answer was not remembered ({added} rows)"
            assert on.trie.roles[-1] == "worker", on.trie.roles[-1]
            assert on.trie.origins[-1] == "w", on.trie.origins[-1]
            assert "9 kWh" in on.trie.texts[-1], on.trie.texts[-1]
            assert on.trie.texts[-1] == "".join(answer), (
                f"the answer was rewritten on the way in: "
                f"{on.trie.texts[-1]!r}")
            assert on.trie.sources[-1] is None, (
                "a delegated answer was filed as an attachment")
            assert on.tail == tail_before, (
                "a delegated answer entered the verbatim tail")
            assert ledger_lines(on.trie.cache_dir)[0]["ingest"] is True
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(on)

    with Stub(cards=CARDS, post_status=503) as s:
        bad = replayed_state(tmp, "ingest_fail", tok, mdl,
                             roster=delegation_roster(s.url, tmp),
                             flags=("--offload-ingest",))
        try:
            before = trie_snapshot(bad.trie)
            offload_line(bad, "what size battery")
            assert trie_snapshot(bad.trie) == before, (
                "a failed delegation was remembered anyway")
            assert ledger_lines(bad.trie.cache_dir)[0]["ingest"] is False, (
                "the ledger claims a failure was ingested")
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(bad)
    print("19. worker rows: a delegated answer can be remembered as its own "
          "role with the worker's name on it, gated and capped as "
          "conversation, and only when --offload-ingest asks for it")


def check_identity(tmp, tok, mdl):
    roster = R.Roster(path=str(SAMPLE), entries=(
        R.RosterEntry(name="qwen05", alias=SAMPLE_ALIAS, role="worker",
                      server_url="http://127.0.0.1:8081",
                      model={"alias": SAMPLE_ALIAS, "hf_id": "test/fake",
                             "path": "-"}),))
    off_trace, off = run_arm(tmp, "agents_off", tok, mdl, None)
    on_trace, on = run_arm(tmp, "agents_on", tok, mdl, roster)

    assert len(off_trace) == len(on_trace) == len(TRANSCRIPT), (
        "an arm did not run the whole transcript")
    for i, (a, b) in enumerate(zip(off_trace, on_trace), 1):
        assert a["prompt"] == b["prompt"], (
            f"turn {i}: a loaded roster changed the prompt the model sees")
        assert a["stats"] == b["stats"], f"turn {i}: the stats moved"
        assert a["coverage"] == b["coverage"], f"turn {i}: the coverage moved"
        assert a["tail"] == b["tail"], f"turn {i}: the verbatim tail moved"
        assert a["reply"] == b["reply"], f"turn {i}: the reply moved"
    for attr in ("texts", "roles", "turns", "sources", "n_words",
                 "keyword_weights", "coverage", "drift_ema", "alive"):
        assert getattr(off, attr) == getattr(on, attr), (
            f"trie.{attr} diverged: a loaded roster is visible in memory")
    assert np.array_equal(off.embeddings, on.embeddings), (
        "the embeddings diverged with a roster loaded")
    assert off.n_sentences > 0, "the fixture built no memory to compare"
    print(f"20. identity: {len(TRANSCRIPT)} turns over {off.n_sentences} "
          f"sentences byte-identical with and without a roster loaded "
          f"({len(off_trace)} prompts compared in full)")


def imports_pulled(module, watch):
    """Which of `watch` end up in sys.modules from importing `module`,
    measured in a fresh interpreter so this harness's own imports do not
    count. `watch` entries match a module or its dotted children."""
    code = ("import sys, importlib; importlib.import_module(%r);"
            "w = %r;"
            "print(' '.join(sorted({m for m in w for k in sys.modules"
            " if k == m or k.startswith(m + '.')})))" % (module, list(watch)))
    # importing torch here pins MKL_THREADING_LAYER for the whole process,
    # and a child that inherits it dies against libgomp before it can
    # import anything
    env = dict(os.environ)
    env.pop("MKL_THREADING_LAYER", None)
    out = subprocess.run([sys.executable, "-c", code], cwd=str(REPO),
                         capture_output=True, text=True, env=env)
    assert out.returncode == 0, (
        f"importing {module} in a fresh interpreter failed: {out.stderr}")
    return out.stdout.split()


def check_import_purity():
    heavy = ("torch", "transformers", "requests", "vllm", "salt.mcp",
             "salt.chat.runner_serve")
    for module in ("salt.agents", "salt.agents.roster", "salt.agents.worker",
                   "salt.agents.delegate"):
        pulled = imports_pulled(module, heavy)
        assert not pulled, (
            f"importing {module} pulled {pulled}: the agent layer must cost "
            f"nothing to import, so a roster can name workers a session "
            f"never uses")
    # the chat entry point carries the encoder stack either way, so only
    # the pieces this ladder could newly drag in are pinned here
    cli_pulled = imports_pulled("salt.chat.cli",
                                ("vllm", "salt.mcp",
                                 "salt.chat.runner_serve"))
    assert not cli_pulled, f"importing salt.chat.cli pulled {cli_pulled}"
    print("21. import purity: the agent layer pulls none of "
          f"{len(heavy)} heavy imports, and saltChat still reaches neither "
          f"the serve client nor an MCP server")


def check_frozen_core():
    for rel in FROZEN:
        assert (REPO / rel).is_file(), (
            f"the frozen list names {rel}, which does not exist - this "
            f"guard would pass vacuously")
    base = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--verify",
                           "-q", LADDER_BASE + "^{commit}"],
                          capture_output=True, text=True)
    if base.returncode != 0:
        print(f"22. frozen core: {LADDER_BASE} is not resolvable here, so the "
              f"{len(FROZEN)} eval files were checked for existence only")
        return
    # against the working tree, not just HEAD, so an uncommitted edit to a
    # frozen file is caught before it can be staged
    out = subprocess.run(["git", "-C", str(REPO), "diff", "--name-only",
                          base.stdout.strip(), "--", *FROZEN],
                         capture_output=True, text=True)
    touched = [ln for ln in out.stdout.splitlines() if ln.strip()]
    assert not touched, (
        f"the agent work changed frozen eval files {touched} - the eval "
        f"path must stay byte-identical across this ladder")
    print(f"22. frozen core: all {len(FROZEN)} eval files untouched since "
          f"the agent layer began")


def check_command_surfaces():
    """A REPL command lives in three places at once. /roster shipped in
    only one of them, so this pins all three against each other."""
    doc = (REPO / "docs" / "chatbot.md").read_text(encoding="utf-8")
    helped = {ln.split()[0] for ln in cli.HELP.splitlines()
              if ln.startswith("/")}
    assert helped, "HELP listed no commands - this check would be vacuous"
    missing = sorted(helped - set(cli.COMMANDS))
    assert not missing, f"{missing} are in HELP but TAB cannot complete them"
    stray = sorted(set(cli.COMMANDS) - helped)
    assert not stray, f"TAB completes {stray}, which HELP never mentions"
    for cmd in ("/roster", "/worker", "/offload"):
        assert cmd in helped, f"{cmd} left HELP"
        assert f"| `{cmd}" in doc, (
            f"{cmd} is not in the docs/chatbot.md command table")
    print(f"23. command surfaces: all {len(helped)} REPL commands are in "
          f"HELP and TAB completion, agent commands documented too")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu", help="device for the encoder")
    ap.add_argument("--keep", action="store_true",
                    help="keep the temp session dirs for inspection")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="salt_agents_regression_"))
    try:
        check_validation(tmp)
        check_loading()
        check_probe()
        check_worker_handle()
        print(f"Loading BGE encoder {BGE_MODEL} on {args.device} ...")
        tok, mdl = load_bge(BGE_MODEL, args.device)
        tok_path = BGE_MODEL
        check_worker_calls(tok_path)
        check_abort(tok_path)
        check_spawning(tmp)
        check_worker_files(tmp)
        check_readiness(tmp)
        check_stopping(tmp)
        check_placement_rules()
        check_call_timeout(tok_path)
        check_retry_policy(tok_path)
        check_worker_commands(tmp)
        check_delegation_context(tmp, tok, mdl)
        check_delegation_call(tmp, tok, mdl)
        check_offload_command(tmp, tok, mdl)
        check_delegation_ledger(tmp, tok, mdl)
        check_worker_rows(tmp, tok, mdl)
        check_identity(tmp, tok, mdl)
        check_import_purity()
        check_frozen_core()
        check_command_surfaces()
        print("PASS")
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
