# -*- coding: utf-8 -*-
"""Regression harness for whether a turn searches memory at all.

Pins the question the chat model is asked, how its answer becomes a
skip probability, the fixed threshold, and the record and census each
decision leaves. Groups:

  A. THE QUESTION - the instructions match their pinned hash; the user
     message carries the request with an empty history, keys in order,
     non-ASCII kept.
  B. THE ANSWER - Yes mass against No mass over the top tokens and the
     chosen one, case, spacing and tokenizer markers ignored, a token
     counted once; Yes alone gives 1.0, neither gives 0.0; missing
     fields read as no answer.
  C. THE DECISION - under off nothing is asked; under auto a fake
     runner's answer skips at the threshold and above and searches
     below it; a runner that cannot score, an error or a malformed
     answer searches and says why; an interrupt is not swallowed; the
     record and the census carry the asserted keys and print as
     promised.
  D. THE SERVED RUNNER - against a stub server: the request carries the
     served model's name, the fixed fields and the model's template
     settings, and nothing else; a Yes answer skips and a No answer
     searches; a server error, an answer without log probabilities and
     a server that does not answer all search and say why.
  E. THE FLAG - real sessions through the chat turn against a fake
     runner: the default is off and asks nothing; a searching answer
     leaves the turn byte-identical to off (prompts, stats); a skipping
     answer builds no block, runs no search and commits nothing, keeps
     a summary mark for the next turn and ledgers the decision; an
     empty session and a turn another model answers ask nothing; a
     backend that cannot score and an uncalibrated model are said once
     and the turn searches.
  F. THE COMMAND AND THE CENSUS - /gate shows and sets, refuses
     anything else, the census counts every decision, /stats prints the
     decision and the census, and a new conversation starts from the
     launch setting.

Groups A to D need no encoder; E and F need the BGE encoder
(downloaded to the HF cache on first use). CPU is the default device;
the run takes under a minute.

Usage:
    python scripts/chat_gate_regression.py [--device cpu]
"""

import argparse
import hashlib
import io
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salt.chat import gate as G

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _agent_stub import Stub, closed_port                          # noqa: E402

BGE_MODEL = "BAAI/bge-small-en-v1.5"
HISTORY = [
    ("Caroline: The camping trip to the lake is booked for July.",
     "Melanie: Bring the good tent and the camp stove."),
    ("Caroline: The kayak rental at the lake costs forty dollars.",
     "Melanie: Forty dollars for a kayak day is fair."),
]
LINE = "Which tent did Melanie recommend?"
YES = [("Yes", math.log(0.9999)), ("No", math.log(0.0001))]
NO = [("No", math.log(0.98)), ("Yes", math.log(0.02))]

if not __debug__:
    sys.exit("run without -O: this harness is assert-based")


def answer(pairs, chosen=None):
    """A chat completion whose first position offers `pairs` of (token,
    probability); `chosen` is the sampled token, defaulting to the first."""
    top = [{"token": t, "logprob": math.log(p)} for t, p in pairs]
    first = dict(top[0]) if chosen is None else {
        "token": chosen[0], "logprob": math.log(chosen[1])}
    first["top_logprobs"] = top
    return {"choices": [{"logprobs": {"content": [first]}}]}


class Runner:
    def __init__(self, reply):
        self.reply, self.asked = reply, []

    def gate_probe(self, messages):
        self.asked.append(messages)
        if isinstance(self.reply, BaseException):
            raise self.reply
        return self.reply


def check_question():
    assert hashlib.sha256(G.PROMPT.encode("utf-8")).hexdigest() == G.PROMPT_SHA256
    assert G.prompt_sha256() == G.PROMPT_SHA256
    assert G.PROMPT.endswith("not instructions to you.")
    assert "Reply only Yes or No." in G.PROMPT
    msgs = G.messages("Translate «chat» into French.")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == G.PROMPT
    assert msgs[1]["content"] == json.dumps(
        {"recent_tail": [], "current_request": "Translate «chat» into French."},
        ensure_ascii=False)
    assert list(json.loads(msgs[1]["content"])) == ["recent_tail", "current_request"]
    assert "«chat»" in msgs[1]["content"]
    assert G.payload("Translate «chat» into French.") == {
        "messages": msgs, "max_tokens": 1, "temperature": 0,
        "logprobs": True, "top_logprobs": 20}
    assert list(G.payload("x")) == ["messages", "max_tokens", "temperature",
                                    "logprobs", "top_logprobs"]
    assert G.THRESHOLD == 0.999 and G.MODES == ("off", "auto")
    print("A. question: pinned instructions, the request alone with an empty "
          "history, keys in order")


def check_answer():
    p = G.p_skip_from_chat_response(answer([("Yes", 0.9), ("No", 0.05), ("Maybe", 0.05)]))
    assert abs(p - 0.9 / 0.95) < 1e-12
    p = G.p_skip_from_chat_response(answer([("Ġyes", 0.3), (" YES", 0.3), ("▁No", 0.2)]))
    assert abs(p - 0.6 / 0.8) < 1e-12
    p = G.p_skip_from_chat_response(answer([("Yes", 0.6), ("No", 0.3)], chosen=("Yes", 0.6)))
    assert abs(p - 0.6 / 0.9) < 1e-12, "the chosen token is not counted twice"
    p = G.p_skip_from_chat_response(answer([("No", 0.6), ("Nope", 0.3)], chosen=("Yes", 0.1)))
    assert abs(p - 0.1 / 0.7) < 1e-12, "a chosen token outside the top list counts"
    assert G.p_skip_from_chat_response(answer([("Yes", 0.99), ("Sure", 0.01)])) == 1.0
    assert G.p_skip_from_chat_response(answer([("Maybe", 0.9), ("Sure", 0.1)])) == 0.0
    for empty in ({}, {"choices": []}, {"choices": [{}]},
                  {"choices": [{"logprobs": None}]},
                  {"choices": [{"logprobs": {"content": []}}]}):
        assert G.p_skip_from_chat_response(empty) == 0.0, empty
    print("B. answer: Yes against No over the top tokens, markers and case "
          "ignored, Yes alone 1.0, neither 0.0")


def check_decision():
    r = Runner(answer([("Yes", 0.9999), ("No", 0.0001)]))
    assert G.decide("off", "x", r) == (False, None) and r.asked == []
    skip, rec = G.decide("auto", "What is 2 + 2?", r)
    assert skip and rec["fired"] and tuple(rec) == G.RECORD_KEYS
    assert rec["mode"] == "auto" and rec["error"] is None and rec["p_skip"] > 0.999
    assert r.asked == [G.payload("What is 2 + 2?")]
    edge = Runner(answer([("Yes", 0.999), ("No", 0.001)]))
    skip, rec = G.decide("auto", "x", edge)
    assert rec["p_skip"] == G.THRESHOLD and skip, "exactly the threshold skips"
    real = G.p_skip_from_chat_response
    try:
        G.p_skip_from_chat_response = lambda data: math.nextafter(G.THRESHOLD, 0)
        assert not G.decide("auto", "x", edge)[0], "one step under the threshold searches"
    finally:
        G.p_skip_from_chat_response = real
    below = Runner(answer([("Yes", 0.998), ("No", 0.002)]))
    skip, rec = G.decide("auto", "x", below)
    assert not skip and not rec["fired"] and rec["p_skip"] < G.THRESHOLD
    skip, rec = G.decide("auto", "x", object())
    assert not skip and rec["mode"] == G.UNAVAILABLE and rec["error"] == G.NO_SCORING
    skip, rec = G.decide("auto", "x", Runner(RuntimeError("HTTP 500")))
    assert not skip and rec["error"] == "RuntimeError: HTTP 500" and rec["p_skip"] is None
    skip, rec = G.decide("auto", "x", Runner(RuntimeError("e" * 1000)))
    assert not skip and len(rec["error"]) == 300
    skip, rec = G.decide("auto", "x", Runner({"choices": "bad"}))
    assert not skip and rec["error"]
    skip, rec = G.decide("auto", "x", Runner({}))
    assert not skip and rec["p_skip"] == 0.0 and rec["error"] is None
    try:
        G.decide("auto", "x", Runner(KeyboardInterrupt()))
        raise AssertionError("an interrupt was swallowed")
    except KeyboardInterrupt:
        pass
    c = G.census()
    assert tuple(c) == G.CENSUS_KEYS
    for rec in (G.decide("auto", "x", r)[1], G.decide("auto", "x", below)[1],
                G.decide("auto", "x", Runner(ValueError("v")))[1],
                G.decide("auto", "x", object())[1], None):
        G.count(c, rec)
    assert c == {"asked": 4, "fired": 1, "consulted": 3, "errors": 1,
                 "unavailable": 1}, c
    assert G.census_lines(None) == []
    assert G.census_lines(c) == [
        "search census: 4 turns decided, 1 answered without memory, 3 searched, "
        "1 questions failed, 1 not askable on this backend"]
    assert G.lines(None) == []
    ln = G.lines(G.decide("auto", "x", r)[1])
    assert len(ln) == 1 and ln[0].startswith("search (auto): skipped, P(skip) 0.999900 against 0.999")
    assert G.lines(G.decide("auto", "x", object())[1]) == [
        f"search (unavailable): searched, {G.NO_SCORING}"]
    print("C. decision: off asks nothing, auto skips at the threshold and "
          "above, every failure searches and says why, record and census as "
          "promised")


def served(url, gen=None):
    import requests
    from salt.chat.runner_serve import VLLMServeChatRunner
    r = object.__new__(VLLMServeChatRunner)
    r.cfg = {"alias": "m", "gen": gen or {}}
    r.server_url, r.served_model, r.http = url, "some/model", requests.Session()
    return r


def check_served():
    yes = [("Yes", math.log(0.9999)), ("No", math.log(0.0001))]
    with Stub(cards=[{"id": "some/model"}], logprobs=yes) as s:
        skip, rec = G.decide("auto", "What is 2 + 2?", served(s.url))
        assert skip and rec["error"] is None, rec
        body = s.httpd.last_payload
        assert body == {"model": "some/model", **G.payload("What is 2 + 2?")}, body
        G.decide("auto", "x", served(s.url, {"chat_template_kwargs":
                                              {"enable_thinking": False},
                                              "temperature": 0.7}))
        body = s.httpd.last_payload
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["temperature"] == 0 and "top_p" not in body, body
    no = [("No", math.log(0.98)), ("Yes", math.log(0.02))]
    with Stub(cards=[{"id": "some/model"}], logprobs=no) as s:
        skip, rec = G.decide("auto", "What did I say about Paris?", served(s.url))
        assert not skip and abs(rec["p_skip"] - 0.02) < 1e-9, rec
    with Stub(cards=[{"id": "some/model"}]) as s:
        skip, rec = G.decide("auto", "x", served(s.url))
        assert not skip and rec["p_skip"] == 0.0 and rec["error"] is None
    with Stub(cards=[{"id": "some/model"}], post_status=503, logprobs=yes) as s:
        skip, rec = G.decide("auto", "x", served(s.url))
        assert not skip and rec["error"].startswith("RuntimeError: HTTP 503"), rec
    skip, rec = G.decide("auto", "x", served(f"http://127.0.0.1:{closed_port()}"))
    assert not skip and rec["error"].startswith("ConnectionError"), rec
    print("D. served runner: the request is the fixed one under the served "
          "model's name, Yes skips and No searches, an error, a bare answer "
          "and a silent server all search")


class FakeChat:
    kind = "fake"

    def __init__(self, tokenizer, logprobs=None, hf_id="meta-llama/Llama-3.1-8B-Instruct"):
        self.tokenizer = tokenizer
        self.alias = "fake"
        self.cfg = {"alias": "fake", "hf_id": hf_id, "path": "-"}
        self.max_input_len = 4096
        self.last_prompt_tokens = None
        self.last_engine_stats = None
        self.prompts, self.asked = [], []
        if logprobs is not None:
            self.logprobs = logprobs
            self.gate_probe = self._probe

    def _probe(self, payload):
        self.asked.append(payload)
        return answer([(t, math.exp(lp)) for t, lp in self.logprobs])

    def input_budget(self, max_new_tokens=None):
        return self.max_input_len

    def stream_chat(self, messages, **overrides):
        self.prompts.append(json.loads(json.dumps(messages)))
        yield "Noted."


def session(root, tok, mdl, device, flags, runner, empty=False):
    from salt.chat import cli
    from salt.engine.session_trie import SessionTrie
    args = cli.build_parser().parse_args(
        ["--device", device, "--sync-ingest", "--budget-pct", "0.9", *flags])
    trie = SessionTrie("gate_flag", cache_dir=root, model_name=BGE_MODEL,
                       budget_pct_default=args.budget_pct)
    if not empty:
        for user, assistant in HISTORY:
            trie.add_turn(user, "user", tokenizer=tok, model=mdl,
                          device=device, save=False)
            trie.add_turn(assistant, "assistant", tokenizer=tok, model=mdl,
                          device=device, save=False)
    return cli.ChatState(args, tok, mdl, runner, trie)


def turn(state, line=LINE, **kw):
    from contextlib import redirect_stdout
    from salt.chat import cli
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.chat_turn(state, line, **kw)
    stats = {k: v for k, v in (state.last_stats or {}).items()
             if not any(t in k for t in ("time", "elapsed", "seconds"))}
    notes = [ln for ln in buf.getvalue().splitlines() if ln.startswith("search gate:")]
    return state.runner.prompts[-1] if state.runner.prompts else None, stats, notes


def counting(state):
    calls = {"compress": 0, "commit": 0}
    real = state.trie.compress

    def compress(*a, **k):
        calls["compress"] += 1
        out = real(*a, **k)
        inner = out.get("commit")
        if inner is not None:
            def commit(*ca, **ck):
                calls["commit"] += 1
                return inner(*ca, **ck)
            out["commit"] = commit
        return out
    state.trie.compress = compress
    return calls


def ledger(state):
    from salt.chat import cli
    return cli.build_stats(state)["kv"]["last_event"] or {}


def check_flag(tmp, tok, mdl, device):
    from salt.chat import cli
    assert cli.build_parser().parse_args([]).gate == "off"
    for bad in (["--gate", "on"], ["--gate-threshold", "0.5"]):
        try:
            with __import__("contextlib").redirect_stderr(io.StringIO()):
                cli.build_parser().parse_args(bad)
            raise AssertionError(f"{bad} parsed")
        except SystemExit:
            pass
    off = session(tmp / "off", tok, mdl, device, [], FakeChat(tok, YES))
    p_off, s_off, _ = turn(off)
    assert off.runner.asked == [] and off.last_gate is None
    assert not any(k.startswith("gate_") for k in ledger(off))
    searched = session(tmp / "no", tok, mdl, device, ["--gate", "auto"], FakeChat(tok, NO))
    p_no, s_no, _ = turn(searched)
    assert searched.runner.asked == [G.payload(LINE)]
    assert p_no == p_off and s_no == s_off, "a searching answer changed the turn"
    assert searched.last_gate["fired"] is False
    on = session(tmp / "yes", tok, mdl, device, ["--gate", "auto"], FakeChat(tok, YES))
    calls = counting(on)
    on.summary_next = True
    p_on, s_on, out = turn(on)
    assert calls == {"compress": 0, "commit": 0}, calls
    assert on.last_gate["fired"] and on.last_stats is None and s_on == {}
    assert p_on[-1] == {"role": "user", "content": LINE}, p_on[-1]
    assert p_on[:-1] == p_off[:-1], "a skip changed the system prompt or the history"
    assert on.summary_next is True, "a skip consumed the summary mark"
    lg = ledger(on)
    assert lg["gate_fired"] is True and lg["gate_mode"] == "auto"
    assert lg["gate_p_skip"] > 0.999 and lg["gate_seconds"] >= 0
    assert lg["gate_error"] is None
    assert out == [], out
    on.runner.logprobs = NO
    _, _, _ = turn(on, "And what did the kayak cost?")
    assert calls["compress"] == 1 and calls["commit"] == 1 and on.summary_next is False
    empty = session(tmp / "empty", tok, mdl, device, ["--gate", "auto"],
                    FakeChat(tok, YES), empty=True)
    turn(empty)
    assert empty.runner.asked == [] and empty.last_gate is None
    other = session(tmp / "other", tok, mdl, device, ["--gate", "auto"], FakeChat(tok, YES))

    def reply_fn(state, messages, block):
        yield "From another model."
    turn(other, reply_fn=reply_fn)
    assert other.runner.asked == [] and other.last_gate is None
    blind = session(tmp / "blind", tok, mdl, device, ["--gate", "auto"], FakeChat(tok))
    calls = counting(blind)
    _, _, out1 = turn(blind)
    _, _, out2 = turn(blind, "And the kayak?")
    assert calls["compress"] == 2 and blind.last_gate["mode"] == G.UNAVAILABLE
    assert ledger(blind)["gate_error"] == G.NO_SCORING
    assert out1 == ["search gate: this backend cannot score the answer, "
                    "so every turn searches (--backend vllm-serve can)"], out1
    assert out2 == [], "the backend note was said twice"
    qwen = session(tmp / "qwen", tok, mdl, device, ["--gate", "auto"],
                   FakeChat(tok, YES, hf_id="Qwen/Qwen3-4B"))
    _, _, out1 = turn(qwen)
    _, _, out2 = turn(qwen, "And the kayak?")
    assert out1 == ["search gate: the threshold was set on "
                    "meta-llama/Llama-3.1-8B-Instruct, not on fake"], out1
    assert out2 == [] and qwen.last_gate["fired"], "the model note was said twice"
    assert G.calibrated({"hf_id": "meta-llama/Llama-3.1-8B-Instruct"})
    assert not G.calibrated({"hf_id": "Qwen/Qwen3-4B"}) and not G.calibrated({})
    print("E. flag: off by default and asks nothing, a searching answer is "
          "byte-identical to off, a skip builds no block, searches and commits "
          "nothing and keeps the summary mark, an empty session and another "
          "model's turn ask nothing, an unscoring backend and an uncalibrated "
          "model are said once")


def check_command_census(tmp, tok, mdl, device):
    from contextlib import redirect_stdout
    from salt.chat import cli
    st = session(tmp / "f", tok, mdl, device, [], FakeChat(tok, YES))
    assert cli.gate_command(st, []) == "search gate: off"
    assert cli.build_stats(st)["gate_census"] is None
    assert cli.gate_command(st, ["sometimes"]) == "search gate: 'sometimes' is not one of off, auto"
    assert cli.gate_command(st, ["AUTO"]) == "search gate: auto"
    turn(st)
    st.runner.logprobs = NO
    turn(st, "And the kayak?")
    assert st.gate_stats == {"asked": 2, "fired": 1, "consulted": 1,
                             "errors": 0, "unavailable": 0}, st.gate_stats
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.print_stats(st)
    text = buf.getvalue().splitlines()
    assert any(ln.startswith("search (auto): searched, P(skip) 0.020000 against 0.999")
               for ln in text), text
    assert ("search census: 2 turns decided, 1 answered without memory, "
            "1 searched, 0 questions failed, 0 not askable on this backend") in text, text
    assert cli.gate_command(st, ["off"]) == "search gate: off"
    turn(st, "And the stove?")
    assert st.last_gate is None and st.gate_stats["asked"] == 2
    assert cli.build_stats(st)["gate_census"] == st.gate_stats
    st.gate_mode = "auto"
    with redirect_stdout(io.StringIO()):
        st.new_trie("gate_other")
    assert st.gate_mode == "off" and st.gate_stats["asked"] == 0 and st.last_gate is None
    assert "/gate" in cli.COMMANDS and "/gate" in cli.HELP
    print("F. command and census: /gate shows, sets and refuses, the census "
          "counts every decision, /stats prints both, and a new conversation "
          "starts from the launch setting")


def main():
    import shutil
    import tempfile
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    check_question()
    check_answer()
    check_decision()
    check_served()
    tmp = Path(tempfile.mkdtemp(prefix="salt_gate_"))
    try:
        from transformers import AutoModel, AutoTokenizer
        print(f"Loading BGE encoder {BGE_MODEL} on {args.device} ...")
        tok = AutoTokenizer.from_pretrained(BGE_MODEL)
        mdl = AutoModel.from_pretrained(BGE_MODEL, output_attentions=True).eval()
        mdl.to(args.device)
        check_flag(tmp, tok, mdl, args.device)
        check_command_census(tmp, tok, mdl, args.device)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS")


if __name__ == "__main__":
    main()
