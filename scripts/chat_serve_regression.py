# -*- coding: utf-8 -*-
"""Regression harness for persistent serving (saltServe + --backend vllm-serve).

The harness owns a private vllm serve process on a scratch port and covers
the serve seam end to end:

  1. Off-path: importing the CLI imports neither vllm nor the serve client.
  2. Launcher refusals: unknown model, bad --vllm-bin, and a bad port fail
     with actionable messages before anything starts.
  3. saltServe boots the server: /v1/models answers under the alias and
     carries the context window.
  4. Client errors: a dead port and a wrong model fail with messages that
     name the fix.
  5. Prompt parity: the serve client's post-truncation token counts match a
     direct local render, including the over-window keep-the-tail path,
     and the server's usage echoes the same count (replies are never
     compared).
  6. Streaming + abort: pieces arrive incrementally, and closing the
     stream mid-reply leaves the client and the server healthy.
  7. APC over the wire: a scripted REPL session records engine_backend
     vllm-serve with real cache hits from turn 2, format v1, additive keys.
  8. Warm resume: a second REPL process on the same conversation renders
     its restored tail (larger first prompt) and is served mostly from the
     still-warm cache; tail.json holds the alternating exchanges.
  9. Resume stability: attachment order and the saved tail reload exactly;
     malformed tail files fall back to the empty-tail behavior.
 10. The server outlives its clients: after every client exited,
     /v1/models still answers.

Skips with exit 0 when vLLM, a GPU, the qwen05 registry entry, or the
scratch port is unavailable, so default HF-only environments stay green.
Assert-based: refuses to run under python -O.
"""

import argparse
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

MODEL_ALIAS = "qwen05"

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
        skip(f"port {args.port} is busy")

    r = serve_cmd("no-such-model")
    assert r.returncode != 0 and "No registered model" in r.stderr, r.stderr
    r = serve_cmd(MODEL_ALIAS, "--vllm-bin", "/no/such/vllm")
    assert r.returncode != 0 and "not an executable" in r.stderr, r.stderr
    r = serve_cmd(MODEL_ALIAS, "--port", "99999")
    assert r.returncode != 0 and "--port" in r.stderr, r.stderr
    print("2. launcher refusals: unknown model, bad --vllm-bin, bad port")

    from salt.chat.cli import ChatState, SESSIONS_DIR
    from salt.chat.runner import make_runner, render_prompt

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
        assert card, "server never became ready"
        assert card["id"] == MODEL_ALIAS and card.get("max_model_len")
        print(f"3. saltServe serves {card['id']} "
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
        print("4. client errors: dead port and wrong model are actionable")

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
        print(f"5. prompt parity: local render, truncation to "
              f"input_budget, and the server's count all agree ({expect})")

        gen = runner.stream_chat(
            [{"role": "user", "content": "Count from 1 to 40, one per line."}],
            temperature=0.0, do_sample=False)
        pieces = [next(gen), next(gen), next(gen)]
        gen.close()
        assert all(pieces)
        after = "".join(runner.stream_chat(
            [{"role": "user", "content": "Say OK."}],
            temperature=0.0, do_sample=False, max_new_tokens=8))
        assert after.strip(), "client dead after aborted stream"
        runner.unload()
        print("6. streaming + abort: incremental pieces, clean recovery")

        cid = "servereg-apc"
        session = SESSIONS_DIR / cid
        shutil.rmtree(session, ignore_errors=True)
        try:
            run_repl(["my favorite color is teal, remember it",
                      "name one planet, one word"], cid, args.gpu, url)
            ev = load_events(session)
            assert all(e["v"] == 1 for e in ev)
            assert all(e["engine_backend"] == "vllm-serve" for e in ev)
            assert all("usage" in e and "input_cached_tokens" in e["usage"]
                       for e in ev)
            frac = ev[1]["apc_cached_tokens"] / ev[1]["apc_prompt_tokens"]
            assert frac >= 0.3, f"turn-1 reuse only {frac:.0%}"
            print(f"7. APC over the wire: turn-1 reuse {frac:.0%}, "
                  f"format v1, additive keys")

            tail = json.loads((session / "tail.json").read_text())
            assert [m["role"] for m in tail] == \
                ["user", "assistant"] * (len(tail) // 2)
            run_repl(["what is my favorite color, one word"],
                     cid, args.gpu, url)
            ev2 = load_events(session)
            resumed = ev2[len(ev)]
            assert resumed["apc_prompt_tokens"] > ev[0]["apc_prompt_tokens"], \
                "resumed prompt did not include the restored tail"
            rfrac = resumed["apc_cached_tokens"] / resumed["apc_prompt_tokens"]
            assert rfrac >= 0.5, f"warm resume only {rfrac:.0%}"
            print(f"8. warm resume: restored tail in the prompt, "
                  f"{rfrac:.0%} of it served from the warm cache")
        finally:
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
        print("9. resume stability: attach order and tail reload exactly, "
              "malformed tail falls back to empty")

        assert requests.get(f"{url}/v1/models", timeout=5).status_code == 200
        print("10. the server outlived every client")
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
