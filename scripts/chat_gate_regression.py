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

No encoder and no model; the run takes a few seconds.

Usage:
    python scripts/chat_gate_regression.py
"""

import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salt.chat import gate as G

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _agent_stub import Stub, closed_port                          # noqa: E402

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


def main():
    check_question()
    check_answer()
    check_decision()
    check_served()
    print("PASS")


if __name__ == "__main__":
    main()
