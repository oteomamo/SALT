# -*- coding: utf-8 -*-
"""The stub worker the regression harnesses drive instead of a real one.

Two shapes of the same fake. `Stub` is an in-process HTTP endpoint
speaking the two routes a saltServe worker answers on, scripted down to
the pieces it streams and the ways it can misbehave. `stub_server()`
writes the same thing as a standalone script, for the checks that need a
worker in a process of its own.

A test helper, not a tool: nothing here is installed, and it deliberately
lives outside the package so a fake OpenAI endpoint never ships beside
the real client. Import it from a script in this directory.
"""

import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from salt.agents.worker import HOST                              # noqa: E402


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

    def _usage_frame(self, prompt, kept):
        """What a server with a prefix cache reports about the prompt it
        was just sent: how long it was, and how much of it it already
        had. Exact here rather than block-aligned, so a check can name a
        number instead of a range."""
        self.wfile.write(b"data: " + json.dumps(
            {"choices": [], "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(self.server.pieces),
                "prompt_tokens_details": {"cached_tokens": kept}}}).encode()
            + b"\n\n")
        self.wfile.flush()

    def _reuse(self, prompt):
        """How much of this prompt the previous one already covered."""
        prior, self.server.last_prompt = self.server.last_prompt, list(prompt)
        kept = 0
        for a, b in zip(prior or [], prompt):
            if a != b:
                break
            kept += 1
        return kept

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
            if self.server.usage:
                prompt = self.server.last_payload.get("prompt") or []
                self._usage_frame(prompt, self._reuse(prompt))
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
                 serving=True, post_status=200, usage=False):
        self.cfg = dict(cards=list(cards), pieces=list(pieces), delay=delay,
                        status=status, raw=raw, stall=stall, drop=drop,
                        post_status=post_status, usage=usage)
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
        self.httpd.last_prompt = None
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


def stub_server(tmp):
    """The script that stands in for saltServe, written once per run."""
    path = Path(tmp) / "stub_server.py"
    if not path.exists():
        path.write_text(STUB_SERVER)
    return str(path)
