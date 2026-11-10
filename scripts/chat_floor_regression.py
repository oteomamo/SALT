# -*- coding: utf-8 -*-
"""Regression harness for the memory floor and the memory hold.

Pins the least a selection reads and the most it reads when memory holds
nothing for the query. The budget is a fraction of the living words, so
a young session selects through a block of a few lines and a long one
through thousands of words whether or not any bear on the query; the
floor raises the first and the hold cuts the second. Groups:

  A. FLOOR OFF - compress with no floor and with memory_floor=None give
     the same selection, the same context and the same stats, and the
     stats say the budget was not raised.
  B. RAISED - a budget fraction smaller than the floor is raised to the
     floor, the stats say so, and more is selected; a floor larger than
     the session selects every living row; rows kept out of candidacy
     do not count toward what can be read; max_words still clamps the
     raise; a fraction already at or above the floor is left alone and
     selects exactly as it does with no floor; a floor of zero is no
     floor.
  C. FLAG - `--memory-floor` parses off, auto and a number of words and
     refuses anything else at launch; through the real chat turn
     against a fake runner, a session launched without it never raises
     a block and prints no floor line, and a session launched with it
     raises the block of a young session, says so in the turn's stats
     and prints the floor in /stats.
  D. EVIDENCE - on a hand-built session with known vectors (no
     encoder): the mean cosine of the three closest rows and the share
     of the query's term information found in the rows match
     hand-computed values; fewer than three rows use what is there; no
     row or no vector gives no cosine; no term gives a zero share; the
     result carries EVIDENCE_KEYS and nothing on the session changes.
  E. HELD - compress with no hold and with memory_hold=None select the
     same; a memory under HOLD_MIN_WORDS is never judged; past it a
     query the rows hold nothing for is cut to the hold and the stats
     say so with the evidence read; rows kept out of candidacy do not
     count toward the size; a line that names nothing, hold_ok=False
     and a query the rows hold evidence for are never cut, the last
     selecting exactly as with no hold; the cut never goes under the
     floor, and a ceiling already under the hold leaves nothing to cut.

Groups A to C and E need the BGE encoder (downloaded to the HF cache on
first use). CPU is the default device; the run takes about a minute.

Usage:
    python scripts/chat_floor_regression.py [--device cpu] [--budget 0.2]
"""
import argparse
import io
import json
import math
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salt.engine import session_trie
from salt.engine.session_trie import (EVIDENCE_KEYS, EVIDENCE_MATCH,
                                      EVIDENCE_TOP3, SessionTrie)

if not __debug__:
    sys.exit("run without -O: this harness is assert-based")

BGE_MODEL = "BAAI/bge-small-en-v1.5"
HOLD_STATS = ("word_budget_held", "evidence_top3", "evidence_match")

EXCHANGES = [
    ("I want to plan a rooftop solar installation for my house.",
     "A south facing roof suits solar panels well. Size the array around "
     "your daily kilowatt usage and pick an inverter to match it."),
    ("How many solar panels would I need for thirty kilowatt hours?",
     "Around twenty panels at typical output. The array size decides what "
     "inverter capacity the installation needs."),
    ("Different topic: I want to learn baking sourdough bread at home.",
     "Sourdough starts with a starter of flour and water fermented until "
     "it rises predictably. Feed the starter daily for about a week."),
    ("What hydration should my first sourdough dough be?",
     "Start around seventy percent hydration so the dough stays "
     "manageable. Higher hydration gives an open crumb but sticky dough."),
    ("Back to the roof: which inverter did you suggest for the array?",
     "A string inverter is the cheaper choice under even sun. Microinverters "
     "cost more and help when part of the roof is shaded."),
    ("And how often should the solar panels be cleaned?",
     "Twice a year is enough in most climates. Dust and pollen cut the "
     "output of a solar array by a few percent."),
]
QUERY = "Which inverter suits the solar array on the roof?"
UNRELATED = "Name the capital city of Australia and the river that crosses it."


def build_session(name, tmp, tok, mdl, device):
    trie = SessionTrie(name, cache_dir=tmp)
    for user, assistant in EXCHANGES:
        trie.add_turn(user, "user", tokenizer=tok, model=mdl, device=device,
                      save=False)
        trie.add_turn(assistant, "assistant", tokenizer=tok, model=mdl,
                      device=device, save=False)
    return trie


def comparable(stats):
    return {k: v for k, v in stats.items()
            if not any(t in k for t in ("time", "elapsed", "seconds"))}


def check_floor_off(tmp, tok, mdl, device, budget):
    trie = build_session("floor_off", tmp, tok, mdl, device)
    kw = dict(budget_pct=budget, tokenizer=tok, model=mdl, device=device,
              defer_commit=True)
    a = trie.compress(QUERY, **kw)
    b = trie.compress(QUERY, memory_floor=None, **kw)
    assert a["selected_sent_idx"] == b["selected_sent_idx"]
    assert a["context"] == b["context"]
    assert comparable(a["stats"]) == comparable(b["stats"]), (
        {k: (a["stats"].get(k), b["stats"].get(k))
         for k in set(a["stats"]) | set(b["stats"])
         if a["stats"].get(k) != b["stats"].get(k)})
    assert a["stats"]["word_budget_raised"] is False, a["stats"]
    assert a["stats"]["word_budget"] == int(trie.live_words * budget)
    print("A. floor off: no floor and memory_floor=None select the same "
          "rows with the same context and stats, and nothing is raised")


def check_raised(tmp, tok, mdl, device, budget):
    trie = build_session("floor_raised", tmp, tok, mdl, device)
    kw = dict(budget_pct=budget, tokenizer=tok, model=mdl, device=device,
              defer_commit=True)
    plain = int(trie.live_words * budget)
    floor = plain + 40
    assert floor < trie.live_words, "the fixture no longer exercises the raise"
    base = trie.compress(QUERY, **kw)

    up = trie.compress(QUERY, memory_floor=floor, **kw)
    s = up["stats"]
    assert s["word_budget_raised"] is True and s["word_budget"] == floor, s
    assert plain < s["words_used"] <= floor, s
    assert len(up["selected_sent_idx"]) > len(base["selected_sent_idx"])

    whole = trie.compress(QUERY, memory_floor=10 ** 6, **kw)
    assert whole["stats"]["word_budget"] == trie.live_words, whole["stats"]
    assert len(whole["selected_sent_idx"]) == trie.n_alive

    tail = set(range(trie.n_sentences - 6, trie.n_sentences))
    left = sum(trie.n_words[i] for i in range(trie.n_sentences)
               if i not in tail)
    cut = trie.compress(QUERY, memory_floor=10 ** 6, exclude_sent_idx=tail,
                        **kw)
    assert cut["stats"]["word_budget"] == left, (cut["stats"], left)
    assert not set(cut["selected_sent_idx"]) & tail

    capped = trie.compress(QUERY, memory_floor=floor, max_words=plain + 10,
                           **kw)
    assert capped["stats"]["word_budget"] == plain + 10, capped["stats"]
    assert capped["stats"]["word_budget_raised"] is True, capped["stats"]

    for small in (plain, plain - 5, 0):
        same = trie.compress(QUERY, memory_floor=small, **kw)
        assert same["stats"]["word_budget_raised"] is False, same["stats"]
        assert same["stats"]["word_budget"] == plain, same["stats"]
        assert same["selected_sent_idx"] == base["selected_sent_idx"]
        assert same["context"] == base["context"]
    print("B. raised: a fraction under the floor is raised to it or to "
          "everything selectable, kept-out rows do not count, max_words "
          "clamps the raise, and a floor at or under the fraction changes "
          "nothing")


class _FakeRunner:
    """Answers from the script above and keeps every prompt."""

    kind = "fake"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.alias = "fake"
        self.cfg = {"alias": "fake", "hf_id": "test/fake", "path": "-"}
        self.max_input_len = 4096
        self.last_prompt_tokens = None
        self.last_engine_stats = None
        self.prompts = []

    def input_budget(self, max_new_tokens=None):
        return self.max_input_len

    def stream_chat(self, messages, **overrides):
        self.prompts.append(json.loads(json.dumps(messages)))
        n = len(self.prompts)
        yield EXCHANGES[n - 1][1] if n <= len(EXCHANGES) else f"noted {n}."

    def unload(self):
        pass


def chat_session(root, tok, mdl, device, flags):
    from salt.chat import cli
    args = cli.build_parser().parse_args(
        ["--device", device, "--sync-ingest", "--tail", "2", *flags])
    trie = SessionTrie("fl_flag", cache_dir=root, model_name=BGE_MODEL,
                       budget_pct_default=args.budget_pct)
    state = cli.ChatState(args, tok, mdl, _FakeRunner(tok), trie)
    with redirect_stdout(io.StringIO()):
        for user, _ in EXCHANGES:
            cli.chat_turn(state, user)
        cli.chat_turn(state, QUERY)
    return state


def stats_text(state):
    from salt.chat import cli
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.print_stats(state)
    return buf.getvalue().splitlines()


def check_flag(tmp, tok, mdl, device):
    from salt.chat import cli
    parse = cli.build_parser().parse_args
    assert parse([]).memory_floor is None
    assert parse(["--memory-floor", "off"]).memory_floor is None
    assert parse(["--memory-floor", "auto"]).memory_floor == (
        cli.MEMORY_FLOOR_WORDS) == 400
    assert parse(["--memory-floor", "250"]).memory_floor == 250
    for bad in ("0", "-3", "many", "1.5"):
        try:
            with redirect_stderr(io.StringIO()):
                parse(["--memory-floor", bad])
        except SystemExit:
            continue
        raise AssertionError(f"--memory-floor accepted {bad!r}")

    off = chat_session(tmp / "off", tok, mdl, device, [])
    assert off.memory_floor is None
    assert off.last_stats["word_budget_raised"] is False, off.last_stats
    assert not any(ln.startswith("memory floor") for ln in stats_text(off))

    on = chat_session(tmp / "on", tok, mdl, device,
                      ["--memory-floor", "auto"])
    assert on.memory_floor == 400
    s = on.last_stats
    assert s["word_budget_raised"] is True, s
    assert s["word_budget"] > int(on.trie.live_words * 0.2), s
    assert s["words_used"] > off.last_stats["words_used"], (
        s["words_used"], off.last_stats["words_used"])
    assert on.runner.prompts[-1] != off.runner.prompts[-1]
    assert on.runner.prompts[0] == off.runner.prompts[0], (
        "the first turn has no memory to raise")
    text = stats_text(on)
    assert ("memory floor: 400 words - the last block was raised to it"
            in text), text
    print("C. --memory-floor: off, auto and a word count parse and anything "
          "else is refused at launch; without the flag nothing is raised "
          "or printed; with it a young session's block is raised and "
          "/stats says so")


def hand_built(tmp):
    """Five rows with vectors chosen by hand, so every number the
    evidence read returns can be worked out on paper."""
    trie = SessionTrie("floor_hand", cache_dir=tmp)
    rows = [
        ("The zeppelin mast gasket was replaced yesterday.", (1, 0, 0, 0)),
        ("A south facing roof suits solar panels well.", (0.8, 0.6, 0, 0)),
        ("Size the solar array around daily usage.", (0.6, 0.8, 0, 0)),
        ("The bulk fermentation ran five hours.", (0, 1, 0, 0)),
        ("We talked about bread dough for a while.", (0, 0, 1, 0)),
    ]
    for text, _ in rows:
        trie.texts.append(text)
        trie.roles.append("user")
        trie.turns.append(0)
        trie.sources.append(None)
        trie.origins.append(None)
        trie.timestamps.append(None)
        trie.n_words.append(len(text.split()))
        trie.keyword_weights.append({})
        trie.alive.append(True)
    trie.embeddings = np.asarray([v for _, v in rows], dtype=np.float32)
    trie.dim = 4
    return trie


def check_evidence(tmp):
    trie = hand_built(tmp)
    q = np.asarray((1, 0, 0, 0), dtype=np.float32)
    before = (list(trie.texts), trie.embeddings.copy(), dict(trie.coverage))
    ev = trie.evidence(range(5), q, {"solar", "gasket", "harbor"},
                       {"Zeppelin"})
    assert tuple(ev) == EVIDENCE_KEYS, ev
    assert ev["rows"] == 5 and ev["terms"] == 4, ev
    # the three closest rows sit at 1.0, 0.8 and 0.6
    assert ev["top3"] == round((1.0 + 0.8 + 0.6) / 3, 4) == 0.8, ev
    # solar is in two rows, gasket and zeppelin in one each, harbor in none
    n = 5
    found = math.log((n + 1) / 3) + 2 * math.log((n + 1) / 2)
    assert ev["match"] == round(found / (4 * math.log((n + 1) / 2)), 4), ev
    assert 0.0 < ev["match"] < 1.0, ev

    two = trie.evidence([0, 3], q)
    assert two == {"rows": 2, "terms": 0, "top3": 0.5, "match": 0.0}, two
    assert trie.evidence([], q, {"solar"}) == {
        "rows": 0, "terms": 1, "top3": None, "match": 0.0}
    assert trie.evidence(range(5), None, {"solar"})["top3"] is None
    every = trie.evidence(range(5), q, {"gasket"}, {"Zeppelin"})
    assert every["match"] == 1.0, every
    assert before[0] == trie.texts and (before[1] == trie.embeddings).all()
    assert before[2] == trie.coverage
    print("D. evidence: the asserted keys, the mean of the three closest "
          "rows and the surprisal share by hand, short and empty row sets, "
          "no vector, and a pure read")


def check_held(tmp, tok, mdl, device, budget):
    trie = build_session("floor_held", tmp, tok, mdl, device)
    kw = dict(budget_pct=budget, tokenizer=tok, model=mdl, device=device,
              defer_commit=True)
    plain = int(trie.live_words * budget)
    hold = 12
    assert plain > hold + 8, "the fixture no longer exercises the hold"
    untouched = [False, None, None]

    off = trie.compress(UNRELATED, **kw)
    same = trie.compress(UNRELATED, memory_hold=None, hold_ok=True, **kw)
    assert same["selected_sent_idx"] == off["selected_sent_idx"]
    assert same["context"] == off["context"]
    assert comparable(same["stats"]) == comparable(off["stats"])
    assert [off["stats"][k] for k in HOLD_STATS] == untouched, off["stats"]

    real = session_trie.HOLD_MIN_WORDS
    assert real == 5000 and trie.live_words < real, (real, trie.live_words)
    small = trie.compress(UNRELATED, memory_hold=hold, **kw)
    assert small["selected_sent_idx"] == off["selected_sent_idx"]
    assert [small["stats"][k] for k in HOLD_STATS] == untouched, (
        "a memory under the size the hold needs was judged", small["stats"])

    try:
        session_trie.HOLD_MIN_WORDS = trie.live_words + 1
        under = trie.compress(UNRELATED, memory_hold=hold, **kw)
        assert [under["stats"][k] for k in HOLD_STATS] == untouched

        session_trie.HOLD_MIN_WORDS = trie.live_words
        held = trie.compress(UNRELATED, memory_hold=hold, **kw)
        s = held["stats"]
        assert s["evidence_top3"] < EVIDENCE_TOP3 and (
            s["evidence_match"] < EVIDENCE_MATCH), (
            "the unrelated request finds evidence in the fixture", s)
        assert s["word_budget_held"] is True and s["word_budget"] == hold, s
        assert s["words_used"] <= hold < off["stats"]["words_used"], s
        assert set(held["selected_sent_idx"]) != set(off["selected_sent_idx"])

        tail = set(range(trie.n_sentences - 6, trie.n_sentences))
        cut = trie.compress(UNRELATED, memory_hold=hold,
                            exclude_sent_idx=tail, **kw)
        assert [cut["stats"][k] for k in HOLD_STATS] == untouched, (
            "rows kept out of candidacy counted toward the size",
            cut["stats"])

        for none in (0, None):
            flat = trie.compress(UNRELATED, memory_hold=none, **kw)
            assert flat["selected_sent_idx"] == off["selected_sent_idx"]
            assert [flat["stats"][k] for k in HOLD_STATS] == untouched

        blank = trie.compress("why is that so?", memory_hold=hold, **kw)
        assert blank["stats"]["word_budget_held"] is False, (
            "a line that names nothing was cut", blank["stats"])
        assert blank["stats"]["word_budget"] == plain, blank["stats"]

        free = trie.compress(UNRELATED, memory_hold=hold, hold_ok=False,
                             **kw)
        assert free["stats"]["word_budget_held"] is False, free["stats"]
        assert free["stats"]["word_budget"] == plain, free["stats"]
        assert free["selected_sent_idx"] == off["selected_sent_idx"]
        assert free["stats"]["evidence_top3"] == s["evidence_top3"]
        assert free["stats"]["evidence_match"] == s["evidence_match"]

        base = trie.compress(QUERY, **kw)
        kept = trie.compress(QUERY, memory_hold=hold, **kw)
        k = kept["stats"]
        assert k["evidence_top3"] >= EVIDENCE_TOP3 or (
            k["evidence_match"] >= EVIDENCE_MATCH), (
            "the fixture's own question finds no evidence", k)
        assert k["word_budget_held"] is False and k["word_budget"] == plain
        assert kept["selected_sent_idx"] == base["selected_sent_idx"]
        assert kept["context"] == base["context"]

        floored = trie.compress(UNRELATED, memory_hold=hold,
                                memory_floor=hold + 8, **kw)
        f = floored["stats"]
        assert f["word_budget_held"] is True and (
            f["word_budget"] == hold + 8), f
        raised = trie.compress(UNRELATED, memory_hold=hold,
                               memory_floor=plain + 40, **kw)
        r = raised["stats"]
        assert r["word_budget_raised"] is True and (
            r["word_budget_held"] is False), r
        assert r["word_budget"] == plain + 40, r

        capped = trie.compress(UNRELATED, memory_hold=hold,
                               max_words=hold - 2, **kw)
        assert capped["stats"]["word_budget"] == hold - 2, capped["stats"]
        assert [capped["stats"][k] for k in HOLD_STATS] == untouched
    finally:
        session_trie.HOLD_MIN_WORDS = real
    print("E. held: no hold and memory_hold=None select the same, a small "
          "memory is never judged, past the size a query the rows hold "
          "nothing for is cut to the hold, kept-out rows do not count, a "
          "line that names nothing, hold_ok=False and a query with "
          "evidence are never cut, the cut stays above the floor, and a "
          "ceiling under the hold leaves nothing to cut")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--budget", type=float, default=0.2)
    args = ap.parse_args()
    from salt.engine.compressor import load_bge
    tok, mdl = load_bge(BGE_MODEL, args.device)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        check_floor_off(tmp / "a", tok, mdl, args.device, args.budget)
        check_raised(tmp / "b", tok, mdl, args.device, args.budget)
        check_flag(tmp / "c", tok, mdl, args.device)
        check_evidence(tmp / "d")
        check_held(tmp / "e", tok, mdl, args.device, args.budget)
    print("PASS")


if __name__ == "__main__":
    main()
