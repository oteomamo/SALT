# -*- coding: utf-8 -*-
"""Regression harness for the saltChat vLLM backend (--backend vllm).

Covers the seam from both sides:

  1. Off-path: importing the CLI never imports vllm.
  2. Prompt parity: the HF and vLLM runners render identical prompt text and
     count identical post-truncation prompt tokens for the same turns
     (replies are NOT compared - kernels differ at the logits margin).
  3. Interrupt: dropping the stream mid-reply aborts the request and the
     engine serves the next turn.
  4. Teardown: unload() returns the GPU to its pre-load footprint.
  5. APC reality: a scripted REPL session records real prefix-cache hits
     from turn 1 on (stable system prompt), while the ledger's
     content-overlap split stays untouched - the two are different numbers
     recorded side by side.
  6. Ledger contract: events stay format v1, engine fields are additive,
     and a session recorded without them resumes and appends cleanly.

Skips with exit 0 when vLLM, a GPU, or the qwen05 registry entry is
missing, so default HF-only environments stay green.
"""

import argparse
import json
import shutil
import subprocess
import sys

MODEL_ALIAS = "qwen05"


def skip(reason):
    print(f"SKIP: {reason}")
    sys.exit(0)


def gpu_used_mib(index):
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits", "-i", str(index)])
    return int(out.strip())


def run_repl(lines, conversation_id, gpu, backend="vllm"):
    script = "".join(l + "\n" for l in lines) + "/exit\n"
    proc = subprocess.run(
        [sys.executable, "-m", "salt.chat.cli", "--model", MODEL_ALIAS,
         "--backend", backend, "--gpu", str(gpu),
         "--conversation-id", conversation_id],
        input=script, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    return proc.stdout


def load_events(session_dir):
    with open(session_dir / "kvtrace" / "events.jsonl") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0,
                    help="CUDA GPU index")
    args = ap.parse_args()

    # 1. off-path: the CLI must be importable without touching vllm
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys, salt.chat.cli; "
         "bad = [m for m in sys.modules if m == 'vllm' "
         "or m.startswith('vllm.')]; sys.exit(1 if bad else 0)"])
    assert probe.returncode == 0, "importing salt.chat.cli imported vllm"
    print("1. off-path: CLI import does not import vllm")

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

    from salt.chat.cli import SESSIONS_DIR
    from salt.chat.runner import make_runner, render_prompt

    turns = [
        [{"role": "system", "content": "You are a careful assistant."},
         {"role": "user", "content": "Name two rivers."}],
        [{"role": "system", "content": "You are a careful assistant."},
         {"role": "user", "content": "Name two rivers."},
         {"role": "assistant", "content": "The Nile and the Amazon."},
         {"role": "user", "content": "[SALT memory]\nfacts here.\n\nLonger?"}],
        # over-window turn: exercises the shared keep-the-tail truncation
        [{"role": "user", "content": "pad " * 40000 + "Say OK."}],
    ]

    def drive(runner):
        counts = []
        for msgs in turns:
            list(runner.stream_chat(msgs, max_new_tokens=8,
                                    temperature=0.0, do_sample=False))
            counts.append(runner.last_prompt_tokens)
        rendered = [render_prompt(runner.tokenizer, m)[0] for m in turns]
        return counts, rendered

    device = f"cuda:{args.gpu}"
    hf = make_runner(cfg, device=device, backend="hf")
    hf_counts, hf_prompts = drive(hf)
    hf.unload()

    base_mib = gpu_used_mib(args.gpu)
    vl = make_runner(cfg, device=device, backend="vllm")
    vl_counts, vl_prompts = drive(vl)
    assert vl_prompts == hf_prompts, "rendered prompt text diverged"
    assert vl_counts == hf_counts, \
        f"prompt token counts diverged: hf {hf_counts} vs vllm {vl_counts}"
    budget = vl.input_budget(8)
    assert vl_counts[-1] == budget, \
        f"truncation parity: expected {budget}, got {vl_counts[-1]}"
    print(f"2. prompt parity: identical text and token counts {vl_counts} "
          f"(last = truncated to budget)")

    gen = vl.stream_chat(
        [{"role": "user", "content": "Count from 1 to 40, one per line."}],
        temperature=0.0, do_sample=False)
    next(gen), next(gen)
    gen.close()
    after = "".join(vl.stream_chat([{"role": "user", "content": "Say OK."}],
                                   temperature=0.0, do_sample=False))
    assert after.strip(), "engine dead after aborted stream"
    print("3. interrupt: aborted mid-stream, next turn healthy")

    vl.unload()
    freed = gpu_used_mib(args.gpu)
    assert freed <= base_mib + 500, \
        f"unload leaked GPU memory: {base_mib} -> {freed} MiB"
    print(f"4. teardown: {base_mib} MiB before load, {freed} MiB after "
          f"unload")

    cid = "vllmreg-apc"
    session = SESSIONS_DIR / cid
    shutil.rmtree(session, ignore_errors=True)
    try:
        run_repl(["tell me about the sun", "what about the moon",
                  "and mars"], cid, args.gpu)
        ev = load_events(session)
        assert all(e["v"] == 1 for e in ev)
        assert all("apc_cached_tokens" in e and "apc_prompt_tokens" in e
                   and e["engine_backend"] == "vllm" for e in ev)
        for e in ev[1:]:
            assert e["apc_cached_tokens"] > 0, f"no reuse on turn {e['turn']}"
        frac = ev[1]["apc_cached_tokens"] / ev[1]["apc_prompt_tokens"]
        assert frac >= 0.3, f"turn-1 reuse only {frac:.0%}"
        gap = [e for e in ev
               if e["apc_cached_tokens"] != e["usage"]["input_cached_tokens"]]
        assert gap, "APC hits and selection overlap never diverged"
        print(f"5. APC reality: turn-1 reuse {frac:.0%}; engine hits and "
              f"ledger overlap are distinct numbers on {len(gap)}/{len(ev)} "
              f"turns")
    finally:
        shutil.rmtree(session, ignore_errors=True)

    cid = "vllmreg-resume"
    session = SESSIONS_DIR / cid
    shutil.rmtree(session, ignore_errors=True)
    try:
        run_repl(["hello there, remember the number 41"], cid, args.gpu,
                 backend="hf")
        run_repl(["what number did I mention?"], cid, args.gpu)
        ev = load_events(session)
        assert [e["turn"] for e in ev] == [0, 1]
        assert "apc_cached_tokens" not in ev[0]
        assert "apc_cached_tokens" in ev[1]
        print("6. ledger contract: pre-feature session resumed, engine "
              "fields additive on the new turn only")
    finally:
        shutil.rmtree(session, ignore_errors=True)

    print("PASS")


if __name__ == "__main__":
    main()
