# -*- coding: utf-8 -*-
"""Regression harness for the agent layer (--roster, /roster, /worker).

The fixture is a small scripted conversation plus the stub worker from
_agent_stub.py, which speaks the two routes a saltServe worker answers
on, so every check runs on CPU with no vLLM, no GPU and no chat model.

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
 20. Delegated-work labels: what a remembered answer is headed with,
     and the reading guide saying the same thing to the model.
 21. Delegation stats: what /stats reports, and the additive agent
     keys the next turn's kvtrace entry carries.
 22. Tail and template integrity: what delegating leaves the verbatim
     tail, the prompt and the kv indices looking like.
 23. Delegation budgets: what bounds the context handed over, the
     reply asked for, a prompt too big for the worker, and how long a
     quiet one is waited for.
 24. Scripted delegations: an offload item in a --turns file, what it
     writes to --turns-out, and the item shapes that are refused.
 25. Resuming a session: what carries over, what is retired as a
     claim nobody can honour, and what is deliberately not restarted.
 26. Delegation identity: a scripted run with delegated answers ends
     in the same memory inline as on the ingest thread, and a
     delegation raised mid-encode waits for the row rather than
     racing it.
 27. Identity: a scripted conversation runs byte-identically with and
     without a roster loaded, prompts and coverage included.
 28. Import purity: importing the agent layer costs nothing, and no
     entry point reaches the serve client or an MCP server on import.
 29. Frozen core: the agent work has not touched the eval files.
 30. Command surfaces: HELP, TAB completion and the docs
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
import ast
import inspect
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
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

import numpy as np

if not __debug__:
    sys.exit("this harness is assert-based - run it without python -O")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from salt.agents import delegate as D                            # noqa: E402
from salt.agents import ledger as L                              # noqa: E402
from salt.agents import roster as R                              # noqa: E402
from salt.agents import worker as W                              # noqa: E402
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
from salt.chat import runner as runner_mod                       # noqa: E402
from salt.chat.kvtrace import KVTrace                            # noqa: E402
from salt.chat.registry import RegistryError, resolve_model      # noqa: E402
from salt.engine.compressor import load_bge                      # noqa: E402
from salt.engine.session_trie import (CONVERSATION_ROLES,        # noqa: E402
                                      VALID_ROLES, SessionTrie)

from _agent_stub import Stub, closed_port, stub_server            # noqa: E402

BGE_MODEL = "BAAI/bge-small-en-v1.5"
SAMPLE = REPO / "salt" / "agents" / "roster_sample.json"
DEMO = REPO / "salt" / "agents" / "demo_turns.json"
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


STUB_CFG = {"alias": "stub", "hf_id": "some/model", "path": str(REPO)}


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
            # a thread of its own is exactly what call() refuses without
            # being told the caller can carry its own results back
            seen[tag] = "".join(h.call(msg, off_thread=True))

        refused = {}

        def unannounced():
            try:
                "".join(h.call(msg))
            except WorkerError as exc:
                refused["why"] = str(exc)

        t = threading.Thread(target=unannounced, name="salt-ingest")
        t.start()
        t.join(30)
        assert "salt-ingest" in refused.get("why", ""), (
            f"a delegation from another thread was allowed: {refused}")
        assert h.calls == 1, (
            f"the refused call still reached the worker: {h.calls}")

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
          "a time, a call from another thread refused unless it says so, "
          "and close left the server serving")


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
                   flags=(), sync=True):
    """A session with real memory in it, left open for inspection."""
    args = cli.build_parser().parse_args(
        ["--device", "cpu", *(["--sync-ingest"] if sync else []), *flags])
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


def delegation_roster(url, tmp, name="w", **kw):
    entry = R.RosterEntry(name=name, alias="stub", role="worker",
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

    # what a warm worker gets to keep between two unrelated tasks: the
    # standing instructions, which are a static file and therefore the
    # same tokens every time. The context under them is selected per task
    # and is expected to differ
    head = D.worker_instructions()
    assert head == D.worker_instructions(), "the worker prompt is not stable"
    with Stub(cards=CARDS, pieces=("an ", "answer"), usage=True) as s:
        state = replayed_state(tmp, "delegation_cache", tok, mdl,
                               roster=delegation_roster(s.url, tmp))
        try:
            offload_line(state, "What size battery?")
            offload_line(state, "Name the biggest risk.")
            runner = state.worker("w").runner
            head_tokens = len(runner.tokenizer(head).input_ids)
            first, second = [r["usage"] for r in
                             ledger_lines(state.trie.cache_dir)]
            assert first["cached_tokens"] == 0, (
                f"a cold worker reported reuse: {first}")
            kept = second["cached_tokens"]
            assert kept >= head_tokens * 0.9, (
                f"a second task to the same worker reused {kept} tokens, "
                f"less than the {head_tokens} token instruction head")
            assert kept < second["prompt_tokens"], (
                f"the whole prompt was reported as reused: {second}")
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

    assert D.STATUSES == ("ok", "timeout", "dead", "aborted", "error"), (
        f"the failure taxonomy changed: {D.STATUSES}")

    dead = Stub(cards=CARDS, port=free_port())
    # ingest on, so a failure has something to fail to leave behind
    state = replayed_state(tmp, "delegation_dead", tok, mdl,
                           roster=delegation_roster(dead.url, tmp),
                           flags=("--offload-ingest",))
    try:
        run_delegation(state, task="warm the client", target="w")
        dead.stop()
        before, tail_before = trie_snapshot(state.trie), list(state.tail)
        _, first = run_delegation(state, task="anyone there", target="w")
        _, again = run_delegation(state, task="anyone there", target="w")
        assert (first.status, again.status) == ("error", "dead"), (
            f"a vanished worker read {first.status} then {again.status}, "
            f"expected one failure to be survivable and the second not")
        assert state.worker("w").state == DEAD, state.worker("w").state
        assert again.error, "a dead worker came back with no reason"
        for res in (first, again):
            assert res.status in D.STATUSES, res.status
            assert not res.ok and res.text == "", res
        assert trie_snapshot(state.trie) == before, (
            "a delegation that failed still moved the session's memory")
        assert state.tail == tail_before, (
            "a delegation that failed reached the verbatim tail")
    finally:
        with redirect_stdout(io.StringIO()):
            cli.close_ingest(state)
        dead.stop()
    print(f"16. executing a delegation: text and usage captured, a second "
          f"task to a warm worker keeping {kept} tokens, all {head_tokens} of "
          f"the instruction head, a directive shaped reply returned verbatim, "
          f"a "
          f"rejection, a stall and a vanished server each named as what they "
          f"are, and no failure moves the session's memory")


def _interrupt(*a, **kw):
    raise KeyboardInterrupt


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

    # Ctrl-C mid-delegation: the interrupt has done its job once the
    # response is severed, so it comes back as a status rather than an
    # exception, and the worker is free for the next task
    slow = Stub(cards=CARDS, pieces=[f"t{i} " for i in range(400)], delay=0.05)
    state = replayed_state(tmp, "offload_ctrlc", tok, mdl,
                           roster=delegation_roster(slow.url, tmp),
                           flags=("--offload-ingest",))
    try:
        before, tail_before = trie_snapshot(state.trie), list(state.tail)
        timer = threading.Timer(1.0, _thread.interrupt_main)
        timer.start()
        try:
            out = offload_line(state, "talk for a long time")
        finally:
            timer.cancel()
        assert "[w] aborted," in out, (
            f"an interrupted delegation was not reported as one: {out}")
        assert "t0 t1 " in out, (
            f"what the worker had said before the interrupt was "
            f"thrown away: {out!r}")
        rec = ledger_lines(state.trie.cache_dir)[-1]
        assert rec["status"] == "aborted", (
            f"the ledger recorded an interrupted delegation as "
            f"{rec['status']!r}")
        assert rec["ingest"] is False, "an interrupted answer was remembered"
        assert trie_snapshot(state.trie) == before, (
            "an interrupted delegation moved the session's memory")
        assert state.tail == tail_before, (
            "an interrupted delegation reached the verbatim tail")
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
    # a second Ctrl-C during the cleanup: severing the response is what
    # aborts the request on the worker, so it is retried against the
    # interrupt rather than skipped
    class _Stubborn:
        def __init__(self, refusals):
            self.refusals, self.closed = refusals, 0

        def close(self):
            self.closed += 1
            if self.closed <= self.refusals:
                raise KeyboardInterrupt

    one_more = _Stubborn(1)
    assert D.close_quietly(one_more) is True, (
        "an interrupt during the cleanup left the response open")
    assert one_more.closed == 2, one_more.closed
    endless = _Stubborn(99)
    assert D.close_quietly(endless) is False, (
        "closing the response never gave the interrupt back to the caller")
    assert endless.closed == D.CLOSE_ATTEMPTS, endless.closed

    # an interrupt while the answer is being remembered still files the
    # delegation, and one before the worker was ever called files nothing
    with Stub(cards=CARDS, pieces=("A bank of about 9 kWh usable ",
                                   "covers the evening draw.")) as s:
        state = replayed_state(tmp, "offload_cleanup", tok, mdl,
                               roster=delegation_roster(s.url, tmp),
                               flags=("--offload-ingest",))
        try:
            real_ingest = cli.submit_ingest
            cli.submit_ingest = _interrupt
            try:
                offload_line(state, "what size bank")
                raise AssertionError("the interrupt never reached the caller")
            except KeyboardInterrupt:
                pass
            finally:
                cli.submit_ingest = real_ingest
            rec = ledger_lines(state.trie.cache_dir)[-1]
            assert rec["status"] == "ok" and rec["ingest"] is False, (
                f"an interrupted ingest lost or mislabeled the record: {rec}")
            assert state.trie.roles.count("worker") == 0, (
                "an interrupted ingest left half an answer in memory")

            before = len(ledger_lines(state.trie.cache_dir))
            # cli imported it by name, so that is the name to interrupt
            real_build = cli.build_context
            cli.build_context = _interrupt
            try:
                offload_line(state, "and the inverter")
                raise AssertionError("the interrupt never reached the caller")
            except KeyboardInterrupt:
                pass
            finally:
                cli.build_context = real_build
            assert len(ledger_lines(state.trie.cache_dir)) == before, (
                "a delegation that never left the session was recorded")
            assert state.delegation_seq == 1, (
                f"an id was spent before the worker was called: "
                f"{state.delegation_seq}")
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)

    print("17. /offload: one worker needs no naming and several refuse "
          "without @NAME, the reply and one status line are printed, Ctrl-C "
          "comes back as an aborted delegation with the worker still ready, "
          "and a second one during the cleanup loses neither the abort nor "
          "the record")


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


def labeled_trie(tmp, tok, mdl):
    """A session holding one turn each of user, assistant and worker."""
    kw = dict(tokenizer=tok, model=mdl, device="cpu")
    trie = bare_trie(tmp, "labels")
    trie.add_turn("We are sizing a home battery for 9 kW of panels.",
                  role="user", **kw)
    trie.add_turn("The inverter caps continuous output at 5 kW.",
                  role="assistant", **kw)
    trie.add_turn("A battery of about 9 kWh covers the winter evening draw.",
                  role="worker", origin="qwen05", **kw)
    return trie


def head_of(section):
    return section.splitlines()[0]


def check_worker_labels(tmp, tok, mdl):
    assert cli.WORKER_LABEL == (
        "[from delegated work — turn {turn}, {origin}]"), cli.WORKER_LABEL
    assert cli.WORKER_LABEL_AGE == (
        "[from delegated work — turn {turn}, {origin}, {age}]"), (
        cli.WORKER_LABEL_AGE)
    # the lockstep contract, made executable: the guide the model reads
    # has to name the header the code actually emits
    guide = (REPO / "salt" / "chat" / "instructions.md").read_text(
        encoding="utf-8")
    assert "[from delegated work — turn" in guide, (
        "instructions.md never tells the model what a delegated-work "
        "header is")
    assert "worker(<name>)" in guide, (
        "instructions.md does not explain the map's worker speaker")
    for label in (cli.CONVERSATION_LABEL, cli.CONVERSATION_MAP_LABEL):
        assert label in guide, f"{label} left the reading guide"

    trie = labeled_trie(tmp, tok, mdl)
    idxs = list(range(trie.n_sentences))
    heads = [head_of(s) for s in cli.conversation_sections(trie, idxs)]
    assert len(heads) == 3, f"one section per turn expected, got {heads}"
    assert heads[0].startswith("[from the earlier conversation — turn 0, "
                               "user, "), heads[0]
    assert heads[1].startswith("[from the earlier conversation — turn 1, "
                               "assistant, "), heads[1]
    assert heads[2].startswith("[from delegated work — turn 2, qwen05, "), (
        f"the delegated section is not labeled as one: {heads[2]}")
    assert heads[2].endswith("just now]"), heads[2]
    assert "worker" not in heads[2], (
        f"the label says the role instead of the worker: {heads[2]}")

    # an origin is what makes it a delegated section: without one the
    # generic header is the honest fallback, not an invented name
    trie.origins[-1] = None
    assert head_of(cli.conversation_sections(trie, idxs)[2]) == (
        "[from the earlier conversation — turn 2, worker, just now]"), (
        cli.conversation_sections(trie, idxs)[2])
    trie.origins[-1] = "qwen05"
    # a session resumed from a build that stored no ingest time
    trie.timestamps[-1] = None
    assert head_of(cli.conversation_sections(trie, idxs)[2]) == (
        "[from delegated work — turn 2, qwen05]"), (
        "a missing stamp invented an age")
    trie.timestamps[-1] = time.time()

    # origin is part of the grouping key, so two workers answering the
    # same turn are two sections rather than one under one name
    trie.turns[-1] = trie.turns[-2]
    trie.roles[-2] = "worker"
    trie.origins[-2] = "helper2"
    heads = [head_of(s) for s in cli.conversation_sections(trie, idxs)]
    assert len(heads) == 3, f"two origins were merged into one section: {heads}"
    assert "helper2]" in heads[1] or "helper2," in heads[1], heads[1]
    assert "qwen05]" in heads[2] or "qwen05," in heads[2], heads[2]

    trie = labeled_trie(tmp, tok, mdl)
    plain = cli.conversation_sections(trie, idxs, turn_labels=False)
    assert len(plain) == 1 and head_of(plain[0]) == cli.CONVERSATION_LABEL, (
        f"--no-turn-labels no longer falls back to the plain header: {plain}")
    assert "qwen05" not in plain[0], (
        "the unlabeled form leaked provenance it is supposed to drop")

    lines, total = cli.conversation_map(trie)
    assert total == 3 and len(lines) == 3, (lines, total)
    assert lines[0].startswith("t0 user: "), lines[0]
    assert lines[2].startswith("t2 worker(qwen05): "), (
        f"the map does not say which worker answered: {lines[2]}")
    block = cli.format_memory_block(trie, idxs, conv_map=True)
    assert "[from delegated work — turn 2, qwen05" in block, block
    assert "t2 worker(qwen05):" in block, block
    print("20. delegated-work labels: the header names the worker rather "
          "than the role, origin splits sections, and the reading guide "
          "the model gets carries the same strings")


def stats_output(state):
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.handle_command("/stats", state)
    return buf.getvalue()


def events_of(state):
    path = state.trie.cache_dir / "kvtrace" / "events.jsonl"
    return [json.loads(ln) for ln
            in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def check_delegation_stats(tmp, tok, mdl):
    rows = [{"target": "w", "status": "ok", "t_start": 1.0, "t_end": 3.0,
             "usage": {"prompt_tokens": 100, "output_tokens": 20}},
            {"target": "w", "status": "error", "t_start": 5.0, "t_end": 6.0,
             "usage": {}},
            {"target": "other", "status": "ok", "t_start": 1.0, "t_end": 1.5,
             "usage": {"prompt_tokens": 7, "output_tokens": None}}]
    summary = L.summarize(rows)
    assert summary["n"] == 3 and set(summary["workers"]) == {"w", "other"}
    w = summary["workers"]["w"]
    assert (w["calls"], w["ok"]) == (2, 1), w
    assert (w["prompt_tokens"], w["output_tokens"]) == (100, 20), w
    assert w["seconds"] == 3.0 and w["last_status"] == "error", w
    assert summary["workers"]["other"]["output_tokens"] == 0, (
        "a missing token count was not read as none")
    assert L.summarize([])["n"] == 0, "an empty ledger counted something"

    with Stub(cards=CARDS, pieces=("a ", "battery")) as s:
        roster = delegation_roster(s.url, tmp)
        state = replayed_state(tmp, "stats", tok, mdl, roster=roster)
        try:
            out = stats_output(state)
            assert "delegations:" not in out, (
                f"a session that delegated nothing reported delegations: "
                f"{out}")
            offload_line(state, "what size battery")
            offload_line(state, "@w and the inverter")
            out = stats_output(state)
            assert "delegations: 2 to 1 worker " in out, out
            line = [l for l in out.splitlines() if l.startswith("  w: ")]
            assert len(line) == 1, f"expected one worker line, got {line}"
            assert "2 calls, 2 ok, " in line[0], line[0]
            assert " in / " in line[0] and " out tokens, " in line[0], line[0]
            assert line[0].endswith("last ok"), line[0]

            # the delegations ran between turns, so the NEXT turn is where
            # they are attributed
            assert state.pending_delegations, "nothing is waiting for a turn"
            with redirect_stdout(io.StringIO()):
                cli.chat_turn(state, "so what do you recommend?")
            assert not state.pending_delegations, (
                "the turn did not take the pending delegations")
            event = events_of(state)[-1]
            assert event["v"] == 1, f"the event format version moved: {event}"
            assert event["agent_delegations"] == 2, event
            assert event["agent_workers"] == ["w"], event
            assert event["agent_delegated_tokens"]["output"] > 0, event
            assert set(event["usage"]) == {"input", "input_cached_tokens",
                                           "output", "total"}, (
                f"the usage block changed shape: {event['usage']}")

            with redirect_stdout(io.StringIO()):
                cli.chat_turn(state, "and the heat pump?")
            plain = events_of(state)[-1]
            assert not [k for k in plain if k.startswith("agent_")], (
                f"a turn with no delegation carried agent keys: {plain}")
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)

        # a resumed session reports what the ledger holds, before it has
        # delegated anything itself
        again = replayed_state(tmp, "stats", tok, mdl, turns=(), roster=roster)
        try:
            assert again.delegation_stats["n"] == 2, again.delegation_stats
            assert "delegations: 2 to 1 worker " in stats_output(again), (
                "totals did not survive the resume")
            assert not again.pending_delegations, (
                "resuming queued old delegations onto the next turn")
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(again)

    # a session whose events.jsonl predates the agent keys opens clean
    old = Path(tmp) / "pre_agent" / "kvtrace"
    old.mkdir(parents=True)
    (old / "events.jsonl").write_text(json.dumps({
        "v": 1, "turn": 4, "conversation_id": "pre_agent", "model": "m",
        "usage": {"input": 10, "input_cached_tokens": 2, "output": 5,
                  "total": 17},
        "selected_sent_idx": [0, 1], "token_rows": [0, 0]}) + "\n",
        encoding="utf-8")
    kv = KVTrace(Path(tmp) / "pre_agent", "pre_agent")
    assert kv.turn == 5, f"an older ledger did not resume: {kv.turn}"
    assert kv.totals["output"] == 5, kv.totals
    print("21. delegation stats: /stats reports per-worker totals that "
          "survive a resume, and the turn after a delegation carries "
          "additive agent keys the format version does not move for")


# A deliberately strict chat template, the shape the Mistral and Llama
# families ship: it refuses anything that is not user/assistant/user/...
# after the system message. Applied to the encoder's tokenizer purely as
# a judge, so the pin needs no chat model and no GPU.
STRICT_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' %}"
    "{%- set body = messages[1:] %}{%- else %}{%- set body = messages %}"
    "{%- endif %}"
    "{%- for m in body %}"
    "{%- if (m['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{- raise_exception('roles must alternate user/assistant') }}"
    "{%- endif %}"
    "{{- '<' + m['role'] + '>' + m['content'] }}"
    "{%- endfor %}")


@contextmanager
def strict_template(tok):
    """Lend the encoder's tokenizer a strict chat template for the length
    of a block, then give it back exactly as it was."""
    prior = getattr(tok, "chat_template", None)
    tok.chat_template = STRICT_TEMPLATE
    try:
        yield
    finally:
        tok.chat_template = prior


def saved_tail(state):
    return json.loads(
        (state.trie.cache_dir / "tail.json").read_text(encoding="utf-8"))


def resolved_indices(event, trie):
    """Every sentence index one kv event names, and the text it points at.
    Indices are permanent, so this mapping must never move."""
    idx = sorted(set(event.get("selected_sent_idx", []))
                 | set(event.get("read_sent_idx", []))
                 | set(event.get("write_sent_idx", [])))
    assert all(0 <= i < trie.n_sentences for i in idx), (
        f"turn {event['turn']} names a sentence the session does not have: "
        f"{idx} against {trie.n_sentences} rows")
    return {i: trie.texts[i] for i in idx}


def check_tail_integrity(tmp, tok, mdl):
    # the judge has to be strict, or every assertion under it is vacuous
    with strict_template(tok):
        try:
            tok.apply_chat_template([{"role": "user", "content": "a"},
                                     {"role": "user", "content": "b"}],
                                    tokenize=False)
            raise AssertionError("the strict template accepted two users "
                                 "in a row, so it proves nothing")
        except AssertionError:
            raise
        except Exception as exc:
            assert "alternate" in str(exc), exc

    with Stub(cards=CARDS, pieces=("A battery of about 9 kWh ",
                                   "covers the evening draw.")) as s:
        state = replayed_state(tmp, "tail_pins", tok, mdl, turns=(),
                               roster=delegation_roster(s.url, tmp),
                               flags=("--offload-ingest",))
        try:
            before = {}
            for n, line in enumerate(TRANSCRIPT[:3]):
                with redirect_stdout(io.StringIO()):
                    cli.chat_turn(state, line)
                offload_line(state, f"summarize point {n}")
                if n == 0:
                    before = resolved_indices(events_of(state)[-1], state.trie)
            assert state.delegation_seq == 3, state.delegation_seq
            assert any(r == "worker" for r in state.trie.roles), (
                "no worker row was ingested, so this proves nothing")

            roles = [m["role"] for m in state.tail]
            assert roles == ["user", "assistant"] * (len(roles) // 2), (
                f"the tail stopped alternating after 3 delegations: {roles}")
            assert saved_tail(state) == state.tail, (
                "tail.json and the live tail disagree")
            worker_rows = [state.trie.texts[i]
                           for i, r in enumerate(state.trie.roles)
                           if r == "worker"]
            joined = json.dumps(state.tail, ensure_ascii=False)
            for text in worker_rows:
                assert text not in joined, (
                    f"a delegated answer reached the verbatim tail: {text!r}")

            block = cli.format_memory_block(
                state.trie, list(range(state.trie.n_sentences)))
            messages = cli.build_messages(block, state.tail, "and after that?")
            with strict_template(tok):
                rendered, used = runner_mod.render_prompt(tok, messages)
            assert used is True, (
                "the strict template refused the prompt, so a model whose "
                "template alternates could not be asked this turn")
            assert rendered.startswith("<user>"), rendered[:40]
            assert rendered.count("<user>") == rendered.count("<assistant>") + 1
            assert "delegated work" in rendered, (
                "the memory block lost its delegated section on the way "
                "into the prompt")

            # indices are permanent: rows added since turn 0, worker rows
            # among them, must not have moved what its event pointed at
            after = resolved_indices(events_of(state)[0], state.trie)
            assert after == before, (
                "kv indices no longer resolve to the same sentences after "
                "worker rows entered the session")
            for event in events_of(state):
                resolved_indices(event, state.trie)
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)
    print("22. tail and template integrity: 3 delegations later the tail "
          "still alternates and holds no delegated text, a strict chat "
          "template accepts the prompt, and every kv index still resolves")


class _WindowRunner:
    """A worker client with a real tokenizer and a small window, which is
    all the budget code reads off one."""

    def __init__(self, tokenizer, max_input_len):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.cfg = {"gen": {"max_new_tokens": 64}}

    def input_budget(self, max_new_tokens=None):
        return runner_mod.input_budget_for(self.max_input_len, self.cfg["gen"],
                                           max_new_tokens)


def check_delegation_budgets(tmp, tok, mdl):
    entry = R.RosterEntry(name="w", alias="stub", role="worker",
                          server_url="http://127.0.0.1:1", model=None)
    capped = R.RosterEntry(name="w", alias="stub", role="worker",
                           server_url="http://127.0.0.1:1", model=None,
                           max_tokens=128)
    small = _WindowRunner(tok, 400)
    assert D.window_max_tokens(small) == 200, (
        "the window backstop is not half the window")
    assert D.window_max_tokens(_WindowRunner(tok, None)) is None, (
        "an unknown window invented a cap")
    req = D.DelegationRequest(task="t", target="w")
    assert D.call_overrides(entry, req) == {}, (
        "a call with nothing to say about length said something")
    assert D.call_overrides(entry, req, small)["max_new_tokens"] == 200, (
        "the worker's window did not act as the backstop")
    assert D.call_overrides(capped, req, small)["max_new_tokens"] == 128, (
        "the window overrode the roster entry")
    asked = D.DelegationRequest(task="t", target="w", max_tokens=16)
    assert D.call_overrides(capped, asked, small)["max_new_tokens"] == 16, (
        "the request did not win over the roster entry")

    # a context that does not fit is trimmed from its head, never its task
    task = "name the single biggest risk in one sentence"
    req = D.DelegationRequest(task=task, target="w")
    ctx = D.DelegationContext(text="filler sentence about batteries. " * 200,
                              selected_idx=(0,))
    messages = D.build_messages(ctx, req)
    over = D.count_tokens(small, messages[0]["content"]
                          + messages[-1]["content"])
    assert over > small.input_budget(), "the fixture already fits, so it "\
        "would prove nothing"
    fitted, note = D.fit_messages(small, messages, req)
    assert note and "task in full" in note, note
    body = fitted[-1]["content"]
    assert body.endswith(f"{D.TASK_HEADER}{task}"), (
        f"the task did not survive the trim: {body[-120:]!r}")
    assert fitted[0] == messages[0], "the instructions were trimmed"
    assert len(body) < len(messages[-1]["content"]), "nothing was trimmed"
    assert (D.count_tokens(small, fitted[0]["content"] + body)
            <= small.input_budget()), "the trimmed prompt still overflows"

    roomy = _WindowRunner(tok, 8192)
    same, note = D.fit_messages(roomy, messages, req)
    assert same == messages and note == "", (
        "a prompt that fits was trimmed anyway")
    blind, note = D.fit_messages(object(), messages, req)
    assert blind == messages and note == "", (
        "a client with no window to read was trimmed against a guess")

    # the word cap bounds the context the session hands over
    with Stub(cards=CARDS, pieces=("ok",)) as s:
        roster = delegation_roster(s.url, tmp)
        wide = replayed_state(tmp, "budget_wide", tok, mdl, roster=roster)
        try:
            # the whole corpus, so the cap has something to bite on
            full = D.build_context(wide, D.DelegationRequest(
                task="what did we say about the battery", target="w",
                budget_pct=1.0))
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(wide)
        tight = replayed_state(tmp, "budget_tight", tok, mdl, roster=roster,
                               flags=("--offload-context-cap", "12"))
        try:
            assert tight.offload_context_cap == 12, tight.offload_context_cap
            cut = D.build_context(tight, D.DelegationRequest(
                task="what did we say about the battery", target="w",
                budget_pct=1.0))
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(tight)
    assert full.words_used > cut.words_used, (
        f"the cap changed nothing: {full.words_used} vs {cut.words_used}")
    assert cut.words_used <= 12, (
        f"the cap let {cut.words_used} words through against 12")
    assert cut.n_selected >= 1, "the cap starved the context entirely"

    # the two session flags: neutral by default, and the roster keeps its
    # own say on how long a given worker is waited for
    plain = cli.build_parser().parse_args(["--device", "cpu"])
    assert (plain.offload_timeout, plain.offload_budget_pct) == (None, None), (
        "the offload flags are not neutral by default")
    with Stub(cards=CARDS, pieces=("ok",)) as s:
        state = replayed_state(tmp, "budget_flags", tok, mdl, turns=(),
                               roster=delegation_roster(s.url, tmp),
                               flags=("--offload-timeout", "1",
                                      "--offload-budget-pct", "0.5"))
        try:
            handle = state.worker("w")
            assert cli.session_timeout(state, handle) == 1, (
                "the session timeout never reached a worker that has none")
            spoken = WorkerHandle(worker_entry(s.url, BGE_MODEL, timeout_s=42))
            assert cli.session_timeout(state, spoken) is None, (
                "a launch flag overruled a roster entry's own timeout")
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)

    # end to end: a worker that goes quiet is given up on at the session's
    # limit, not at the five minute default
    quiet = [f"word{i:03d} " for i in range(120)]
    with Stub(cards=CARDS, pieces=quiet, stall=6) as s:
        state = replayed_state(tmp, "budget_stall", tok, mdl, turns=(),
                               roster=delegation_roster(s.url, tmp),
                               flags=("--offload-timeout", "1"))
        try:
            t0 = time.monotonic()
            out = offload_line(state, "talk until you stop")
            took = time.monotonic() - t0
            assert "[w] timeout," in out, out
            assert took < CALL_TIMEOUT, (
                f"the session timeout was ignored, the call took {took:.0f}s")
            assert "word000" in out, (
                "what the worker did say before going quiet was thrown away")
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)
    print(f"23. delegation budgets: the word cap held a {full.words_used} "
          f"word context to {cut.words_used}, the worker's window backstops "
          f"the reply length, an over-window prompt loses context head "
          f"rather than its task, and a quiet worker is given up on at the "
          f"session's limit")


def turns_file(tmp, name, items):
    path = Path(tmp) / name
    path.write_text(json.dumps(items), encoding="utf-8")
    return str(path)


def refused_turns(tmp, name, items, fragment):
    try:
        cli.load_turns(turns_file(tmp, name, items))
    except ValueError as exc:
        assert fragment in str(exc), f"{items} raised {exc}"
        return
    raise AssertionError(f"{items} was accepted")


def check_scripted_offload(tmp, tok, mdl):
    # a plain item is read exactly as it was before delegations existed
    plain = cli.load_turns(turns_file(tmp, "plain.json", [
        "bare string", {"id": "x", "question": "an object"}]))
    assert [(t.id, t.text, t.offload) for t in plain] == [
        (None, "bare string", None), ("x", "an object", None)], plain

    spec = cli.load_turns(turns_file(tmp, "spec.json", [
        {"offload": "just the task"},
        {"id": "d1", "offload": {"task": "T", "target": "w",
                                 "ingest": True}}]))
    assert spec[0].text == "just the task" and spec[0].offload == {
        "task": "just the task"}, spec[0]
    assert spec[1].id == "d1" and spec[1].text == "T", spec[1]
    refused_turns(tmp, "bad1.json", [{"offload": {"task": "T", "worker": "w"}}],
                  "unknown keys ['worker']")
    refused_turns(tmp, "bad2.json", [{"offload": {"target": "w"}}],
                  "names no task")
    refused_turns(tmp, "bad3.json", [{"offload": ["T"]}],
                  "neither a task string nor an object")
    # an item with no text and no offload is still the old ambiguity error
    refused_turns(tmp, "bad4.json", [{"a": "1", "b": "2"}], "no obvious message")

    with Stub(cards=CARDS, pieces=("A battery of about 9 kWh ",
                                   "covers the evening draw.")) as s:
        roster = delegation_roster(s.url, tmp)
        state = replayed_state(tmp, "scripted", tok, mdl, turns=(),
                               roster=roster)
        out_path = Path(tmp) / "turns_out.jsonl"
        script = turns_file(tmp, "mixed.json", [
            {"id": "c1", "question": "The evening draw is 6 kWh."},
            {"id": "d1", "offload": {"task": "size the bank",
                                     "ingest": True}},
            {"id": "c2", "question": "And the inverter is 5 kW."},
            {"offload": {"task": "name the risk", "target": "w"}},
            {"id": "d3", "offload": {"task": "anything", "target": "nope"}}])
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                cli.run_turns(state, cli.load_turns(script), str(out_path))
            out = buf.getvalue()
            assert out.count("offload> ") == 3, out
            assert out.count("you> ") == 2, out
            assert "[turn d3 failed:" in out and "nope" in out, (
                f"an unknown worker did not fail its item alone: {out}")

            rows = [json.loads(l) for l in
                    out_path.read_text(encoding="utf-8").splitlines()]
            assert len(rows) == 5, rows
            chat = [r for r in rows if "kind" not in r]
            assert [r["id"] for r in chat] == ["c1", "c2"], chat
            assert set(chat[0]) == {"id", "turn", "question", "answer"}, (
                f"a chat row changed shape: {sorted(chat[0])}")
            assert chat[0]["answer"] == REPLIES[0], chat[0]
            done = [r for r in rows if r.get("kind") == "offload"]
            assert [r["turn"] for r in done] == [1, 3, 4], done
            assert [r["status"] for r in done] == ["ok", "ok", "error"], done
            assert [r["worker"] for r in done] == ["w", "w", "nope"], done
            assert done[0]["question"] == "size the bank", done[0]
            assert done[0]["answer"] == ("A battery of about 9 kWh covers "
                                         "the evening draw."), done[0]
            assert done[2]["answer"] is None, done[2]

            assert s.httpd.posts == 2, (
                f"the failed item still reached a worker: {s.httpd.posts}")
            assert state.delegation_seq == 2, state.delegation_seq
            recs = ledger_lines(state.trie.cache_dir)
            assert [r["ingest"] for r in recs] == [True, False], (
                f"the item's ingest flag did not decide: {recs}")
            # the session was launched WITHOUT --offload-ingest, so only the
            # item that asked for it is remembered, and as a worker row
            assert state.trie.roles.count("worker") == 1, state.trie.roles
            assert "w" in state.trie.origins, state.trie.origins
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)
    # the shipped demo conversation is what a reader copies, so it has to
    # parse, name workers the sample roster really has, and replay
    demo = cli.load_turns(DEMO)
    named = {e["name"] for e in json.loads(SAMPLE.read_text())["models"]}
    delegated = [t for t in demo if t.offload]
    assert len(demo) > len(delegated) > 0, "the demo stopped being mixed"
    for t in delegated:
        assert t.offload["target"] in named, (
            f"the demo delegates to {t.offload['target']!r}, which the "
            f"sample roster does not name: {sorted(named)}")
    with Stub(cards=CARDS, pieces=("About 9 kWh ",
                                   "covers the evening draw.")) as s:
        target = delegated[0].offload["target"]
        state = replayed_state(tmp, "demo", tok, mdl, turns=(),
                               roster=delegation_roster(s.url, tmp,
                                                        name=target))
        try:
            with redirect_stdout(io.StringIO()):
                cli.run_turns(state, demo)
            recs = ledger_lines(state.trie.cache_dir)
            assert [r["status"] for r in recs] == ["ok"] * len(delegated), recs
            assert s.httpd.posts == len(delegated), s.httpd.posts
            assert state.trie.roles.count("worker") == 1, (
                f"the demo remembered {state.trie.roles.count('worker')} "
                f"worker answers, and one item asks for it")
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)
    print(f"24. scripted delegations: an offload item goes to a worker "
          f"instead of the chat model, its --turns-out row names the worker "
          f"and how it ended, a plain row keeps the shape it always had, a "
          f"misspelled key is refused before the model loads, and the "
          f"shipped demo replays its {len(delegated)} delegations")


def dead_pid():
    """A pid that named a process and no longer does."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def write_record(d, name, pid, port=9999):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(json.dumps({
        "name": name, "alias": "stub", "pid": pid, "port": port,
        "url": f"http://127.0.0.1:{port}", "started_at": 1.0,
        "argv": ["saltServe"], "log": str(d / f"{name}.log")}),
        encoding="utf-8")


def check_resume(tmp, tok, mdl):
    with Stub(cards=CARDS, pieces=("A battery of about 9 kWh ",
                                   "covers the evening draw.")) as s:
        roster = delegation_roster(s.url, tmp)
        state = replayed_state(tmp, "resume", tok, mdl, roster=roster,
                               flags=("--offload-ingest",))
        home = state.trie.cache_dir
        try:
            for n in range(3):
                offload_line(state, f"summarize point {n}")
            assert state.delegation_seq == 3, state.delegation_seq
            origins = [o for o in state.trie.origins if o]
            assert origins == ["w"] * len(origins) and origins, (
                f"no delegated row carried its worker: {state.trie.origins}")
            n_rows, before = state.trie.n_sentences, stats_output(state)
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)

        # the session is reopened the way a new launch would open it
        back = replayed_state(tmp, "resume", tok, mdl, turns=(), roster=roster,
                              flags=("--offload-ingest",))
        try:
            assert back.delegation_seq == 3, (
                f"ids restarted at {back.delegation_seq}")
            assert back.delegation_stats["n"] == 3, back.delegation_stats
            after = stats_output(back)
            assert "delegations: 3 to 1 worker " in after, after
            for line in before.splitlines():
                if line.startswith("  w: "):
                    assert line in after, (
                        f"the per-worker line changed across a resume:\n"
                        f"  before {line}\n  after  "
                        f"{[l for l in after.splitlines() if l.startswith('  w: ')]}")
            assert back.trie.n_sentences == n_rows, (
                "the corpus changed size across a resume")
            assert [o for o in back.trie.origins if o] == origins, (
                f"origins did not survive the resume: {back.trie.origins}")
            assert back.trie.roles.count("worker") == len(origins), (
                "a worker row came back under a different role")
            offload_line(back, "one more")
            assert [r["id"] for r in ledger_lines(home)] == [1, 2, 3, 4], (
                "a resumed session reused an id")
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(back)

    # pid records: the gone are retired, the living are left alone
    workers = Path(tmp) / "resume_pids" / "workers"
    gone, mine = dead_pid(), os.getpid()
    write_record(workers, "ghost", gone)
    write_record(workers, "alive", mine, port=9998)
    live, archived = W.check_records(workers)
    assert [r["name"] for r in archived] == ["ghost"], archived
    assert [r["name"] for r in live] == ["alive"], live
    assert archived[0]["pid"] == gone and archived[0]["argv"] == ["saltServe"], (
        "the archive lost what the record said")
    assert not (workers / "ghost.json").exists(), "the dead claim still stands"
    assert (workers / "ghost.json.stale").exists(), "nothing was archived"
    assert (workers / "alive.json").exists(), "a live worker was retired"
    assert W.check_records(workers)[1] == [], (
        "a second pass archived something twice")
    assert W.check_records(Path(tmp) / "nowhere") == ([], []), (
        "a session that never spawned anything reported records")
    assert W.pid_alive(mine) and not W.pid_alive(gone)
    assert not W.pid_alive(None) and not W.pid_alive("not a pid")

    # and a session opening on that directory says so, and starts nothing.
    # the checks above already retired the first ghost, so leave another
    write_record(workers, "ghost", gone)
    spawn = R.Roster(path=str(Path(tmp) / "r.json"), entries=(
        spawn_entry(tmp, "ghost"), spawn_entry(tmp, "alive")))
    buf = io.StringIO()
    with redirect_stdout(buf):
        state = replayed_state(tmp, "resume_pids", tok, mdl, turns=(),
                               roster=spawn)
    out = buf.getvalue()
    try:
        assert "'ghost' from an earlier run is gone" in out, out
        assert "'alive' from an earlier run is still up" in out, out
        assert all(h.process is None for h in state.worker_handles()), (
            "resuming restarted a worker instead of waiting to be asked")
        assert all(h.state == DECLARED for h in state.worker_handles()), (
            "a resumed handle claims a state nobody established")
    finally:
        with redirect_stdout(io.StringIO()):
            cli.close_ingest(state)
    print("25. resuming a session: ids, totals and origins all continue "
          "where they stopped, a dead worker record is archived rather "
          "than believed, and nothing is restarted unasked")


MIXED_SCRIPT = [
    {"id": "c1", "question": "The evening draw is about 6 kWh."},
    {"id": "d1", "offload": {"task": "size the bank", "target": "a",
                             "ingest": True}},
    {"id": "c2", "question": "The inverter is rated 5 kW continuous."},
    {"id": "d2", "offload": {"task": "name the risk", "target": "b",
                             "ingest": True}},
]


def two_worker_roster(tmp, first, second):
    def one(name, url):
        return R.RosterEntry(name=name, alias="stub", role="worker",
                             server_url=url,
                             model={"alias": "stub", "hf_id": "some/model",
                                    "path": BGE_MODEL})
    return R.Roster(path=str(Path(tmp) / "two_workers.json"),
                    entries=(one("a", first), one("b", second)))


def scripted_arm(tmp, cid, tok, mdl, roster, sync):
    state = replayed_state(tmp, cid, tok, mdl, turns=(), sync=sync,
                           roster=roster)
    script = turns_file(tmp, f"{cid}.json", MIXED_SCRIPT)
    with redirect_stdout(io.StringIO()):
        cli.run_turns(state, cli.load_turns(script))
        cli.close_ingest(state)
    return state.trie


def functions_containing(path, needle):
    """Which functions of a module have `needle` in their source."""
    src = Path(path).read_text(encoding="utf-8")
    lines = src.splitlines()
    found = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = "\n".join(lines[node.lineno - 1:node.end_lineno])
            if needle in body:
                found.add(node.name)
    return found


def check_delegation_identity(tmp, tok, mdl):
    # the barrier story, as source facts rather than as a comment: the
    # agent layer runs no thread of its own, one function reads the trie
    # and it is the one that drains first, and one function sends
    agents = REPO / "salt" / "agents"
    for mod in sorted(agents.glob("*.py")):
        assert "Thread(" not in mod.read_text(encoding="utf-8"), (
            f"{mod.name} starts a thread, so the delegation path is no "
            f"longer the session's own thread")
    touching = functions_containing(agents / "delegate.py", "state.trie")
    assert touching == {"build_context"}, (
        f"the trie is read outside build_context: {sorted(touching)}")
    assert "state.ingest.drain()" in (agents / "delegate.py").read_text(
        encoding="utf-8"), "build_context stopped draining before it reads"
    sending = functions_containing(agents / "delegate.py", "handle.call(")
    assert sending == {"delegate"}, (
        f"a worker is called outside delegate(): {sorted(sending)}")

    # the same scripted run, once inline and once on the ingest thread. Two
    # workers with answers of their own, so each delegated row is a row of
    # its own rather than the near-dup of the one before it
    with Stub(cards=CARDS, pieces=("A bank of about 9 kWh usable ",
                                   "covers the evening draw.")) as a, \
         Stub(cards=CARDS, pieces=("The inverter clips at 5 kW, ",
                                   "so a kettle and a kiln together trip "
                                   "it.")) as b:
        roster = two_worker_roster(tmp, a.url, b.url)
        inline = scripted_arm(tmp, "sync_arm", tok, mdl, roster, True)
        background = scripted_arm(tmp, "async_arm", tok, mdl, roster, False)
    assert inline.roles.count("worker") == 2, inline.roles
    assert [o for o in inline.origins if o] == ["a", "b"], inline.origins
    for attr in ("texts", "roles", "origins", "turns", "sources", "n_words",
                 "keyword_weights", "coverage", "drift_ema", "alive"):
        assert getattr(inline, attr) == getattr(background, attr), (
            f"trie.{attr} diverged: a delegated answer is remembered "
            f"differently on the ingest thread than inline")
    assert np.array_equal(inline.embeddings, background.embeddings), (
        "the embeddings diverged between the two ingest modes")

    # a delegation raised while an encode is still in flight: build_context
    # drains first, so it reads a trie nobody is writing to
    late = "The battery bank was finally sized at 9 kWh usable."
    with Stub(cards=CARDS, pieces=("noted.",)) as s:
        state = replayed_state(tmp, "race", tok, mdl, sync=False,
                               roster=delegation_roster(s.url, tmp))
        try:
            running = threading.Event()

            def slow_encode():
                running.set()
                time.sleep(0.6)
                state.trie.add_turn(late, role="user", tokenizer=tok,
                                    model=mdl, device="cpu")

            state.ingest.submit(slow_encode, "gate")
            assert running.wait(10), "the gate job never started"
            before = state.trie.n_sentences
            assert state.ingest.pending == 1, state.ingest.pending
            t0 = time.monotonic()
            ctx = D.build_context(state, D.DelegationRequest(
                task="how big is the battery bank", target="w"))
            took = time.monotonic() - t0
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)
    assert took >= 0.3, (
        f"the delegation did not wait for the encode in flight ({took:.2f}s)")
    assert state.ingest.pending == 0, "a job was still pending after the drain"
    assert state.trie.n_sentences == before + 1, (
        f"the row being written never landed: {before} -> "
        f"{state.trie.n_sentences}")
    assert "9 kWh usable" in ctx.text, (
        f"the context was selected from a trie mid-write: {ctx.text!r}")
    assert (len(state.trie.texts) == len(state.trie.origins) ==
            len(state.trie.roles) == len(state.trie.embeddings)), (
        "the corpus and its parallel lists disagree after the drain")
    print(f"26. delegation identity: the agent layer runs on the session's "
          f"own thread and reads the trie in one place, a scripted run "
          f"with delegated answers "
          f"builds the same {inline.n_sentences} sentence memory inline and "
          f"on the ingest thread, and a delegation raised mid-encode waited "
          f"{took:.1f}s for the row before selecting it")


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
    print(f"27. identity: {len(TRANSCRIPT)} turns over {off.n_sentences} "
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
    print("28. import purity: the agent layer pulls none of "
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
        print(f"29. frozen core: {LADDER_BASE} is not resolvable here, so the "
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
    print(f"29. frozen core: all {len(FROZEN)} eval files untouched since "
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
    print(f"30. command surfaces: all {len(helped)} REPL commands are in "
          f"HELP and TAB completion, agent commands documented too")


def offload_again(state, line):
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.offload_command(state, line.split(), again=True)
    return buf.getvalue()


def check_offload_ergonomics(tmp, tok, mdl):
    with Stub(cards=CARDS, pieces=("nine ", "kWh")) as first, \
            Stub(cards=CARDS, pieces=("about ", "ten kWh")) as second:
        roster = two_worker_roster(tmp, first.url, second.url)
        state = replayed_state(tmp, "ergonomics", tok, mdl, roster=roster)
        try:
            assert cli.worker_completions(state, "@") == ["@a", "@b"], (
                cli.worker_completions(state, "@"))
            assert cli.worker_completions(state, "@b") == ["@b"]
            assert cli.worker_completions(state, "@z") == []

            assert "Usage: /offload!" in offload_again(state, ""), (
                "repeating without a name did not say so")
            assert "takes no text" in offload_again(state, "@b a new task")
            assert "Nothing has been delegated" in offload_again(state, "@b"), (
                "repeating before any delegation invented a task")
            assert state.delegation_seq == 0, "a usage error spent an id"

            offload_line(state, "@a what size battery")
            out = offload_again(state, "@b")
            assert "again: what size battery" in out, out
            assert "delegating to b" in out and "ten kWh" in out, out
            tasks = [r["task"] for r in ledger_lines(state.trie.cache_dir)]
            assert tasks == ["what size battery"] * 2, tasks
            assert [r["target"] for r in
                    ledger_lines(state.trie.cache_dir)] == ["a", "b"]
            assert first.httpd.posts == second.httpd.posts == 1, (
                "the repeat went to the wrong worker")

            state.roster = None
            recipe = offload_line(state, "anything")
            for fragment in ("saltServe", "salt-roster/1", "--roster"):
                assert fragment in recipe, (
                    f"the enable recipe never mentions {fragment}: {recipe}")
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)
    print("31. offload ergonomics: @NAME completes from the roster, "
          "/offload! puts the same task to a second worker, and asking "
          "with no roster prints the recipe for having one")


def worker_turn_line(state, line):
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.worker_turn(state, line)
    return buf.getvalue()


def check_worker_turns(tmp, tok, mdl):
    # the seam's default is what keeps every other turn in this suite
    # honest: all of it None is the path that was there before
    seam = ["reply_fn", "reply_model_id", "reply_tokenizer", "reply_label"]
    params = inspect.signature(cli.chat_turn).parameters
    for name in seam:
        assert params[name].default is None, (
            f"chat_turn.{name} defaults to {params[name].default!r}, so an "
            f"ordinary turn no longer takes the ordinary path")

    answer = "A 9 kWh bank covers the evening. Winter is the binding case."
    with Stub(cards=CARDS, pieces=(answer[:20], answer[20:])) as s:
        state = replayed_state(tmp, "worker_turn", tok, mdl,
                               roster=delegation_roster(s.url, tmp))
        try:
            for bad in ("@w", "@w   "):
                assert "Usage: @NAME" in worker_turn_line(state, bad), bad
            assert "known:" in worker_turn_line(state, "@nope anything"), (
                "an unknown worker was not refused by name")
            assert s.httpd.posts == 0, "a refused line still reached a worker"

            prompts_before = len(state.runner.prompts)
            out = worker_turn_line(state,
                                   "@w what size battery does that argue for")
            assert f"w> {answer}" in out, (
                f"the turn was not labeled with the worker that spoke: {out}")
            assert s.httpd.posts == 1, s.httpd.posts
            assert len(state.runner.prompts) == prompts_before, (
                "the chat model answered a turn that was not its own")
            # the serve client sends token ids, so the prompt is read back
            # through the worker's own tokenizer
            worker_runner = state.worker("w").runner
            sent = worker_runner.tokenizer.decode(
                s.httpd.last_payload["prompt"]).lower()
            assert "what size battery does that argue for" in sent, (
                "the worker was not given this turn's own question")
            assert "salt memory" in sent, (
                "the worker answered without the turn's memory block")

            assert [t["role"] for t in state.tail[-2:]] == [
                "user", "assistant"], state.tail[-2:]
            assert state.tail[-1]["content"] == answer, state.tail[-1]
            assert state.tail[-2]["content"].startswith("what size battery"), (
                state.tail[-2])
            assert state.trie.roles[-1] == "assistant", (
                f"the answer was remembered as {state.trie.roles[-1]!r} "
                f"rather than as this session's own assistant turn")
            assert not L.ledger_path(state.trie.cache_dir).exists(), (
                "a turn the worker answered was filed as a delegation")
            assert state.delegation_seq == 0, state.delegation_seq

            spoke = events_of(state)[-1]["model"]
            assert spoke == "some/model", (
                f"the turn was stamped {spoke!r} rather than the worker's "
                f"model")
            with redirect_stdout(io.StringIO()):
                cli.chat_turn(state, "and what about the inverter?")
            assert events_of(state)[-1]["model"] == "test/fake", (
                "the next turn did not go back to the chat model")
            assert len(state.runner.prompts) == prompts_before + 1, (
                "the chat model did not take the turn after it")
            assert s.httpd.posts == 1, "a plain turn reached the worker"
        finally:
            with redirect_stdout(io.StringIO()):
                cli.close_ingest(state)
    print("32. worker-answered turns: @NAME hands this turn's own prompt "
          "to a worker and keeps the answer as the session's own, stamped "
          "with the model that gave it, and the next turn is the chat "
          "model's again")


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
        check_worker_labels(tmp, tok, mdl)
        check_delegation_stats(tmp, tok, mdl)
        check_tail_integrity(tmp, tok, mdl)
        check_delegation_budgets(tmp, tok, mdl)
        check_scripted_offload(tmp, tok, mdl)
        check_resume(tmp, tok, mdl)
        check_delegation_identity(tmp, tok, mdl)
        check_identity(tmp, tok, mdl)
        check_import_purity()
        check_frozen_core()
        check_command_surfaces()
        check_offload_ergonomics(tmp, tok, mdl)
        check_worker_turns(tmp, tok, mdl)
        print("PASS")
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
