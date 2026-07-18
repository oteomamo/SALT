# -*- coding: utf-8 -*-
"""Regression harness for persistent serving (saltServe + --backend vllm-serve).

The harness owns a private vllm serve process on a scratch port and covers
the serve seam end to end:

  1. Off-path: importing the CLI imports neither vllm nor the serve client.
  2. Multi-GPU plumbing (no GPU needed): a --gpu list yields
     --tensor-parallel-size and a joined CUDA_VISIBLE_DEVICES in PCI order,
     a lone card yields neither, the memory cap defaults to 0.80 across
     cards but 0.90 on one, saltChat resolves the model and BGE devices in
     PCI order, the hf backend shards via device_map, and duplicate indices
     are rejected.
  3. Launcher refusals: unknown model, bad --vllm-bin, a bad port, and a
     bad --gpu token fail with actionable messages before anything starts.
  4. Stub fault injection (a local fake server, no GPU): mid-stream error
     frames surface as errors instead of a silently truncated reply,
     U+2028-class codepoints stream intact, closing the stream severs the
     request server-side, and an over-window prompt sends exactly the
     last budget token ids.
  5. saltServe boots the server: /v1/models answers under the alias and
     carries the context window.
  6. Client errors: a dead port and a wrong model fail with messages that
     name the fix.
  7. Prompt parity: the serve client's post-truncation token counts match a
     direct local render, including the over-window keep-the-tail path,
     and the server's usage echoes the same count (replies are never
     compared).
  8. Streaming + abort: pieces arrive incrementally, and closing the
     stream mid-reply leaves the client and the server healthy.
  9. APC over the wire: a scripted REPL session records engine_backend
     vllm-serve with real cache hits from turn 2, format v1, additive keys.
 10. Warm resume: a second REPL process on the same conversation renders
     its restored tail (the first prompt grows by at least the tail),
     served mostly from the still-warm cache; tail.json holds the
     alternating exchanges.
 11. Resume stability: attachment order and the saved tail reload exactly;
     malformed tail files fall back to the empty-tail behavior.
 12. The server outlives its clients: after every client exited,
     /v1/models still answers.

Skips with exit 0 when vLLM, a GPU, the qwen05 registry entry, or the
scratch port is unavailable, so default HF-only environments stay green
(check 2 is pure and runs regardless).
Assert-based: refuses to run under python -O.
"""

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import requests

MODEL_ALIAS = "qwen05"


class _StubHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _frame(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False)
                         .encode("utf-8") + b"\n\n")
        self.wfile.flush()

    def do_GET(self):
        body = json.dumps({"data": [{"id": MODEL_ALIAS,
                                     "max_model_len": 4096}]}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(
            int(self.headers.get("Content-Length", 0))))
        self.server.last_prompt = req.get("prompt")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        mode = self.server.mode
        if mode == "unicode-then-error":
            self._frame({"choices": [{"text": "line1\u2028line2"}]})
            self._frame({"error": {"message": "engine boom"}})
            self.wfile.write(b"data: [DONE]\n\n")
        elif mode == "echo":
            self._frame({"choices": [{"text": "ok"}]})
            self._frame({"usage": {"prompt_tokens": len(req["prompt"])},
                         "choices": []})
            self.wfile.write(b"data: [DONE]\n\n")
        elif mode == "endless":
            try:
                for i in range(300):
                    self._frame({"choices": [{"text": f"t{i} "}]})
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                self.server.aborted.set()

if not __debug__:
    sys.exit("chat_serve_regression is assert-based; run without -O")


def skip(reason):
    print(f"SKIP: {reason}")
    sys.exit(0)


def run_repl(lines, conversation_id, gpu, server_url):
    script = "".join(l + "\n" for l in lines) + "/exit\n"
    proc = subprocess.run(
        [sys.executable, "-m", "salt.chat.cli", "--model", MODEL_ALIAS,
         "--backend", "vllm-serve", "--server-url", server_url,
         "--gpu", str(gpu), "--conversation-id", conversation_id],
        input=script, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout


def load_events(session_dir):
    with open(session_dir / "kvtrace" / "events.jsonl") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def serve_cmd(*extra):
    return subprocess.run(
        [sys.executable, "-m", "salt.chat.serve", *extra],
        capture_output=True, text=True, timeout=60)


def check_multi_gpu():
    """Command construction is pure and GPU-free: run it before the vllm/GPU
    gate so HF-only environments still cover the tensor-parallel plumbing."""
    from salt.chat.serve import (build_cmd, build_env, default_gpu_mem_util,
                                 parse_gpu_list)
    assert parse_gpu_list("0,1") == ["0", "1"]
    assert parse_gpu_list(" 0 , 1 ") == ["0", "1"]
    assert parse_gpu_list("0") == ["0"]
    assert parse_gpu_list(None) is None and parse_gpu_list("") is None
    try:
        parse_gpu_list("0,x")
        raise AssertionError("parse_gpu_list accepted a non-index token")
    except ValueError:
        pass
    assert default_gpu_mem_util(["0", "1"]) == 0.80
    assert default_gpu_mem_util(["0"]) == 0.90
    assert default_gpu_mem_util(None) == 0.90
    assert default_gpu_mem_util(["0", "1"], single=0.85) == 0.80
    fake = {"path": "/w", "alias": "m"}
    multi = build_cmd("vllm", fake, "127.0.0.1", 8000, "bfloat16", 0.80,
                      ["0", "1"], 4096, ["--trust-remote-code"])
    assert multi[multi.index("--tensor-parallel-size") + 1] == "2"
    assert multi[multi.index("--gpu-memory-utilization") + 1] == "0.8"
    assert multi[multi.index("--max-model-len") + 1] == "4096"
    assert multi[-1] == "--trust-remote-code"
    solo = build_cmd("vllm", fake, "127.0.0.1", 8000, "bfloat16", 0.90,
                     ["0"], 0, [])
    assert "--tensor-parallel-size" not in solo
    assert "--max-model-len" not in solo
    env = build_env({"KEEP": "1"}, ["0", "1"])
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" and env["KEEP"] == "1"
    assert "CUDA_VISIBLE_DEVICES" not in build_env({}, None)
    # a repeat would misset tensor-parallel size and point two ranks at one
    # card, so it is rejected rather than silently hiding a card
    for dup in ("0,0", "1,0,1"):
        try:
            parse_gpu_list(dup)
            raise AssertionError(f"parse_gpu_list accepted duplicate {dup!r}")
        except ValueError:
            pass
    # saltChat side: model on the first card, BGE on the last, PCI order
    # pinned for the whole process so both halves name the same physical card
    from salt.chat.cli import resolve_gpu_devices
    assert resolve_gpu_devices(["0", "1"], None, None, None) == \
        ("cuda:0", "cuda:1", 0.80, "PCI_BUS_ID")
    assert resolve_gpu_devices(["1"], None, None, None) == \
        ("cuda:1", "cuda:1", 0.85, "PCI_BUS_ID")
    assert resolve_gpu_devices(None, None, None, None) == \
        ("cuda", None, 0.85, None)
    # explicit flags win, but a --gpu list still pins PCI order
    assert resolve_gpu_devices(["0", "1"], "cuda:3", "cpu", 0.5) == \
        ("cuda:3", "cpu", 0.5, "PCI_BUS_ID")
    # hf backend placement: several cards -> device_map balanced + a
    # per-card memory cap keyed by PCI index; one card (or none) keeps the
    # plain device string unchanged
    from salt.chat.runner import hf_placement
    gib = 1024 ** 3
    assert hf_placement(None, "cuda", 0.80, lambda i: 24 * gib) == \
        ("cuda", None)
    assert hf_placement(["1"], "cuda:1", 0.80, lambda i: 24 * gib) == \
        ("cuda:1", None)
    # distinct per-card totals prove the cap is keyed to each card's own
    # memory, not a shared constant
    dmap, mmem = hf_placement(["0", "1"], "cuda:0", 0.80,
                              lambda i: (16 if i == 0 else 24) * gib)
    assert dmap == "balanced"
    assert mmem == {0: int(16 * gib * 0.80), 1: int(24 * gib * 0.80)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0, help="CUDA GPU index")
    ap.add_argument("--port", type=int, default=18077,
                    help="scratch port for the private server")
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}"

    # 1. off-path: the CLI must not pull vllm or the serve client
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys, salt.chat.cli; "
         "bad = [m for m in sys.modules if m == 'vllm' "
         "or m.startswith('vllm.') or m.endswith('runner_serve')]; "
         "sys.exit(1 if bad else 0)"])
    assert probe.returncode == 0, "salt.chat.cli import pulled vllm/serve"
    print("1. off-path: CLI import pulls neither vllm nor the serve client")

    # 2. multi-GPU command construction (pure, GPU-free)
    check_multi_gpu()
    print("2. multi-GPU: a --gpu list yields --tensor-parallel-size and a "
          "joined CUDA_VISIBLE_DEVICES, a lone card yields neither, cap "
          "defaults 0.80 across cards, model/BGE resolve in PCI order, "
          "hf shards via device_map, duplicates rejected")

    try:
        import vllm  # noqa: F401
    except ImportError:
        skip("vllm not installed (pip install vllm==0.11.0)")
    import torch
    if not torch.cuda.is_available():
        skip("no CUDA device")
    from salt.chat.registry import RegistryError, resolve_model
    try:
        cfg = resolve_model(MODEL_ALIAS)
    except RegistryError:
        skip(f"model {MODEL_ALIAS!r} not registered")
    if not cfg["downloaded"]:
        skip(f"model {MODEL_ALIAS!r} has no weights")
    try:
        with socket.socket() as s:
            # REUSEADDR matches the server's own bind, so a just-finished
            # run's TIME_WAIT socket does not skip the next run
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", args.port))
    except OSError:
        try:
            resp = requests.get(f"{url}/v1/models", timeout=3)
            stale = (resp.status_code == 200
                     and resp.json()["data"][0]["id"] == MODEL_ALIAS)
        except Exception:
            stale = False
        if stale:
            sys.exit(f"port {args.port} is held by a stale {MODEL_ALIAS} "
                     f"server from an earlier run - kill it and rerun")
        skip(f"port {args.port} is busy")

    r = serve_cmd("no-such-model")
    assert r.returncode != 0 and "No registered model" in r.stderr, r.stderr
    r = serve_cmd(MODEL_ALIAS, "--vllm-bin", "/no/such/vllm")
    assert r.returncode != 0 and "not an executable" in r.stderr, r.stderr
    r = serve_cmd(MODEL_ALIAS, "--port", "99999")
    assert r.returncode != 0 and "--port" in r.stderr, r.stderr
    r = serve_cmd(MODEL_ALIAS, "--gpu", "0,x")
    assert r.returncode != 0 and "--gpu takes GPU indices" in r.stderr, r.stderr
    print("3. launcher refusals: unknown model, bad --vllm-bin, bad port, "
          "bad --gpu")

    from salt.chat.cli import ChatState, SESSIONS_DIR
    from salt.chat.runner import make_runner, render_prompt

    stub = ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
    stub.mode, stub.last_prompt = "unicode-then-error", None
    stub.aborted = threading.Event()
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    stub_url = f"http://127.0.0.1:{stub.server_address[1]}"
    sr = make_runner(cfg, "cuda", "vllm-serve", server_url=stub_url)
    pieces, err = [], None
    try:
        for p in sr.stream_chat([{"role": "user", "content": "hi"}],
                                max_new_tokens=8, temperature=0.0,
                                do_sample=False):
            pieces.append(p)
    except RuntimeError as exc:
        err = str(exc)
    assert "\u2028" in "".join(pieces), "U+2028 sheared the SSE frame"
    assert err and "engine boom" in err, "mid-stream error frame swallowed"
    stub.mode = "echo"
    long_msgs = [{"role": "user", "content": "pad " * 6000 + "Say OK."}]
    list(sr.stream_chat(long_msgs, max_new_tokens=8, temperature=0.0,
                        do_sample=False))
    text, used = render_prompt(sr.tokenizer, long_msgs)
    ids = sr.tokenizer(text, add_special_tokens=not used).input_ids
    budget = sr.input_budget(8)
    assert sr.last_prompt_tokens == budget
    assert stub.last_prompt == ids[-budget:], \
        "client did not send exactly the last budget token ids"
    stub.mode = "endless"
    gen = sr.stream_chat([{"role": "user", "content": "go"}],
                         max_new_tokens=8, temperature=0.0, do_sample=False)
    next(gen), next(gen)
    gen.close()
    assert stub.aborted.wait(10), "closing the stream did not sever it"
    sr.unload()
    stub.shutdown()
    print("4. stub fault injection: error frames surface, U+2028 streams "
          "intact, abort severs the request, truncation keeps the tail")

    log = tempfile.NamedTemporaryFile(prefix="saltserve-reg-", suffix=".log",
                                      delete=False)
    server = subprocess.Popen(
        [sys.executable, "-m", "salt.chat.serve", MODEL_ALIAS,
         "--gpu", str(args.gpu), "--port", str(args.port),
         "--gpu-mem-util", "0.45"],
        stdout=log, stderr=log)
    try:
        deadline = time.time() + 240
        card = None
        while time.time() < deadline:
            if server.poll() is not None:
                raise AssertionError(
                    "server exited early; log tail:\n"
                    + Path(log.name).read_text()[-2000:])
            try:
                resp = requests.get(f"{url}/v1/models", timeout=3)
                if resp.status_code == 200:
                    card = resp.json()["data"][0]
                    break
            except requests.RequestException:
                pass
            time.sleep(3)
        assert card, ("server never became ready; log tail:\n"
                      + Path(log.name).read_text()[-2000:])
        assert card["id"] == MODEL_ALIAS and card.get("max_model_len")
        print(f"5. saltServe serves {card['id']} "
              f"(window {card['max_model_len']})")

        try:
            make_runner(cfg, "cuda", "vllm-serve",
                        server_url="http://127.0.0.1:9")
            raise AssertionError("dead port did not raise")
        except RuntimeError as exc:
            assert f"saltServe {MODEL_ALIAS}" in str(exc), exc
        wrong = dict(cfg, alias="other-model", hf_id="org/other",
                     path="/nonexistent")
        try:
            make_runner(wrong, "cuda", "vllm-serve", server_url=url)
            raise AssertionError("wrong model did not raise")
        except RuntimeError as exc:
            assert MODEL_ALIAS in str(exc) and "other-model" in str(exc), exc
        print("6. client errors: dead port and wrong model are actionable")

        runner = make_runner(cfg, "cuda", "vllm-serve", server_url=url)
        turns = [
            [{"role": "system", "content": "You are a careful assistant."},
             {"role": "user", "content": "Name two rivers."}],
            [{"role": "user", "content": "pad " * 40000 + "Say OK."}],
        ]
        for msgs in turns:
            list(runner.stream_chat(msgs, max_new_tokens=8,
                                    temperature=0.0, do_sample=False))
            text, used = render_prompt(runner.tokenizer, msgs)
            ids = runner.tokenizer(text,
                                   add_special_tokens=not used).input_ids
            budget = runner.input_budget(8)
            expect = min(len(ids), budget) if budget else len(ids)
            assert runner.last_prompt_tokens == expect, \
                (runner.last_prompt_tokens, expect)
            assert runner.last_engine_stats["apc_prompt_tokens"] == expect
        print(f"7. prompt parity: local render, truncation to "
              f"input_budget, and the server's count all agree ({expect})")

        gen = runner.stream_chat(
            [{"role": "user", "content": "Count from 1 to 40, one per line."}],
            temperature=0.0, do_sample=False)
        next(gen), next(gen), next(gen)
        gen.close()
        after = "".join(runner.stream_chat(
            [{"role": "user", "content": "Say OK."}],
            temperature=0.0, do_sample=False, max_new_tokens=8))
        assert after.strip(), "client dead after aborted stream"
        runner.unload()
        print("8. streaming + abort: incremental pieces, clean recovery")

        cid = "servereg-apc"
        session = SESSIONS_DIR / cid
        shutil.rmtree(session, ignore_errors=True)
        fox = "The quick brown fox jumps over the lazy dog. " * 8
        run_repl([f"My favorite color is teal. {fox}Remember it.",
                  "name one planet, one word"], cid, args.gpu, url)
        ev = load_events(session)
        assert all(e["v"] == 1 for e in ev)
        assert all(e["engine_backend"] == "vllm-serve" for e in ev)
        assert all("usage" in e and "input_cached_tokens" in e["usage"]
                   for e in ev)
        frac = ev[1]["apc_cached_tokens"] / ev[1]["apc_prompt_tokens"]
        assert frac >= 0.3, f"turn-1 reuse only {frac:.0%}"
        print(f"9. APC over the wire: turn-1 reuse {frac:.0%}, "
              f"format v1, additive keys")

        tail = json.loads((session / "tail.json").read_text())
        assert [m["role"] for m in tail] == \
            ["user", "assistant"] * (len(tail) // 2)
        run_repl(["what is my favorite color, one word"],
                 cid, args.gpu, url)
        ev2 = load_events(session)
        resumed = ev2[len(ev)]
        # a dropped tail cannot fake this: without it the resumed prompt
        # is SHORTER than turn 0 (whose question was the long fox line)
        delta = resumed["apc_prompt_tokens"] - ev[0]["apc_prompt_tokens"]
        assert delta >= 40, f"restored tail added only {delta} tokens"
        rfrac = resumed["apc_cached_tokens"] / resumed["apc_prompt_tokens"]
        assert rfrac >= 0.5, f"warm resume only {rfrac:.0%}"
        print(f"10. warm resume: restored tail adds {delta} prompt tokens, "
              f"{rfrac:.0%} served from the warm cache")
        # kept on assert failure so events.jsonl stays inspectable; the
        # next run's pre-clean removes it
        shutil.rmtree(session, ignore_errors=True)

        with tempfile.TemporaryDirectory() as tmp:
            st = ChatState.__new__(ChatState)
            class _T:  # noqa: E301
                cache_dir = Path(tmp)
            st.trie = _T()
            st.tail, st.tail_min, st.tail_max = [], 4, 8
            st.full_attachments = {}
            st.save_full_attachment("b_paper", "BBB")
            st.save_full_attachment("a_notes", "AAA")
            st.tail = [{"role": "user", "content": "q"},
                       {"role": "assistant", "content": "a"}]
            st.save_tail()
            st2 = ChatState.__new__(ChatState)
            st2.trie = _T()
            st2.tail, st2.tail_min, st2.tail_max = [], 4, 8
            st2.load_full_attachments()
            st2.load_tail()
            assert list(st2.full_attachments) == ["b_paper", "a_notes"]
            assert st2.tail == st.tail
            (Path(tmp) / "tail.json").write_text("garbage{")
            st3 = ChatState.__new__(ChatState)
            st3.trie = _T()
            st3.tail, st3.tail_min, st3.tail_max = [], 4, 8
            st3.load_tail()
            assert st3.tail == []
        print("11. resume stability: attach order and tail reload exactly, "
              "malformed tail falls back to empty")

        assert requests.get(f"{url}/v1/models", timeout=5).status_code == 200
        print("12. the server outlived every client")
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()
        Path(log.name).unlink(missing_ok=True)

    print("PASS")


if __name__ == "__main__":
    main()
