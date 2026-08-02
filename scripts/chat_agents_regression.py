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
  7. Identity: a scripted conversation runs byte-identically with and
     without a roster loaded, prompts and coverage included.
  8. Import purity: importing the agent layer costs nothing, and no
     entry point reaches the serve client or an MCP server on import.
  9. Frozen core: the agent work has not touched the eval files.

Needs only the salt install and the BGE encoder (downloaded to the HF
cache on first use). Assert-based: refuses to run under python -O.

Usage:
    python scripts/chat_agents_regression.py
"""

import argparse
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
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

from salt.agents import roster as R                              # noqa: E402
from salt.agents.worker import (BUSY, DECLARED, DEAD, PROBED, READY,   # noqa: E402
                                WorkerError, WorkerHandle)
from salt.chat import cli                                        # noqa: E402
from salt.chat.registry import RegistryError, resolve_model      # noqa: E402
from salt.engine.compressor import load_bge                      # noqa: E402
from salt.engine.session_trie import SessionTrie                 # noqa: E402

BGE_MODEL = "BAAI/bge-small-en-v1.5"
SAMPLE = REPO / "salt" / "agents" / "roster_sample.json"
SAMPLE_ALIAS = "qwen05"

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

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.server.last_payload = json.loads(self.rfile.read(n) or b"{}")
        with self.server.gauge:
            self.server.inflight += 1
            self.server.peak = max(self.server.peak, self.server.inflight)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in self.server.pieces:
                frame = {"choices": [{"text": piece}]}
                self.wfile.write(b"data: " + json.dumps(frame).encode()
                                 + b"\n\n")
                self.wfile.flush()
                if self.server.delay:
                    time.sleep(self.server.delay)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            self.server.aborted.set()
        finally:
            with self.server.gauge:
                self.server.inflight -= 1


class Stub:
    """One /v1/models plus /v1/completions endpoint with a scripted answer.

    Port 0 so a real server on the sample roster's port never collides
    with, or silently satisfies, one of these checks.
    """

    def __init__(self, cards=(), pieces=("he", "llo"), delay=0.0,
                 status=200, raw=None):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.httpd.cards, self.httpd.pieces = list(cards), list(pieces)
        self.httpd.delay, self.httpd.status, self.httpd.raw = delay, status, raw
        self.httpd.last_payload = None
        self.httpd.aborted = threading.Event()
        self.httpd.inflight, self.httpd.peak = 0, 0
        self.httpd.gauge = threading.Lock()
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


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
    refuses(tmp / "absent.json", "Cannot read roster")
    print("1. roster validation: 16 malformed files each refused by name")


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
    print(f"7. identity: {len(TRANSCRIPT)} turns over {off.n_sentences} "
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
    for module in ("salt.agents", "salt.agents.roster", "salt.agents.worker"):
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
    print("8. import purity: the agent layer pulls none of "
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
        print(f"9. frozen core: {LADDER_BASE} is not resolvable here, so the "
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
    print(f"9. frozen core: all {len(FROZEN)} eval files untouched since "
          f"the agent layer began")


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
        check_identity(tmp, tok, mdl)
        check_import_purity()
        check_frozen_core()
        print("PASS")
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
