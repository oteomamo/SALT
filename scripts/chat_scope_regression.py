# -*- coding: utf-8 -*-
"""Regression harness for the branch seams of the session trie.

Pins what a session reports about the branches of its trie for one
query and the seam that lets a caller encode the query once. Groups:

  A. BRANCH SCORES - on a hand-built session with known vectors (no
     encoder): one row per attached source and one for the
     conversation, in that order, every row carrying BRANCH_KEYS; the
     centroid and peak cosines match hand-computed values; keyword and
     proper-noun hits count distinct query terms found in the branch;
     a source below the fold threshold is counted with the
     conversation; a masked row is not read and can drop a source under
     the threshold; an empty session reports nothing.
  B. PASS-THROUGH IDENTITY - the same transcript compressed with the
     query encoded inside the call and with the vector handed in gives
     the same selection, the same context, the same stats, the same
     coverage and the same drift state, and the handed-in vector costs
     the compressor no encoder call.
  C. BRANCH STATS FLAG - the same session driven through the chat turn
     with --branch-stats off and on, against a fake runner: every prompt,
     the selection stats and the kvtrace event keys are identical, the
     off session reports no branch rows, the on session reports one row
     per branch with the asserted keys, and /stats differs only by the
     branch block.
  D. SCOPE CANDIDACY - compress(scope_sources=...) never selects a row
     outside the named branches, takes the budget fraction of the words
     inside them floored so a short branch is still read and never
     above the unscoped budget, reports the rows it kept out, composes with tail
     exclusion, and under stable_keys a coverage key carried only by an
     out-of-scope branch survives the commit untouched (the commit-
     universe union).
  E. SCOPE IDENTITY - naming every branch selects exactly what no scope
     selects, and a scope no living row belongs to is ignored outright;
     stats agree except for the three scope fields.
  F. SCOPE RULE - on declared rows (no encoder): one file about the
     question beside three that mention it and six cold ones keeps the
     one; tied peaks keep every tied file; the file with the most name
     hits stays in on names alone; tighter margins never widen; nothing
     to route gives None; a failing rule is reported, never raised; the
     record and its /stats lines carry the asserted keys.
  G. SCOPE FLAG - real sessions through the chat turn: --scope auto on a
     session without attachments is byte-identical (prompts, stats,
     kvtrace keys, no scoped block); with one file the file is kept and
     the prompts still match; with two files under zero margins the out
     file leaves the prompt and the scoped block, the stats and the
     kvtrace keys agree; a rule that raises leaves the turn unscoped and
     says so.
  H. SCOPE CENSUS - under --scope auto the session counts every turn the
     rule was asked about and how it came out, the counts follow the
     records turn by turn, a failing rule counts as not applied, a
     session without attachments counts plain turns, /stats prints the
     census and a new session starts it over; with scope off there is
     no census.

Groups B to H need the BGE encoder (downloaded to the HF cache on first
use). CPU is the default device; the run takes about two minutes.

Usage:
    python scripts/chat_scope_regression.py [--device cpu] [--budget 0.2]
"""

import argparse
import copy
import io
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import salt.engine.session_trie as st_mod
from salt.chat import scope as scope_module
from salt.engine.session_trie import (BRANCH_KEYS, FILE_TOKEN_PREFIX,
                                      SCOPE_FLOOR_WORDS, SessionTrie)

if not __debug__:
    sys.exit("run without -O: this harness is assert-based")

BGE_MODEL = "BAAI/bge-small-en-v1.5"

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
]
DOC_NAME = "irrigation-notes.txt"
DOC_TEXT = (
    "The garden irrigation system uses a drip line on each vegetable bed. "
    "A timer valve opens the drip line for twenty minutes at dawn. "
    "Rain sensors pause the irrigation schedule after heavy rainfall. "
    "The pump pressure for the drip system stays near two bar.")
QUERY = "What did we decide about the solar inverter?"
DOC2_NAME = "mooring-notes.txt"
DOC2_TEXT = (
    "The zeppelin mooring mast stands at the north end of the airfield. "
    "Its hydrogen manifold uses a nitrile gasket that is replaced yearly. "
    "Ground crews check the mast winch cable before every mooring. "
    "A red beacon on the mast warns aircraft after dusk.")
DOC_QUERY = "How long does the timer valve open the drip line?"
SCOPE_STATS = ("scope_branches", "scope_words", "scope_excluded")


def hand_built(tmp):
    """A session filled field by field: four unit axes stand in for the
    encoder, so every cosine below is known before the code runs."""
    trie = SessionTrie("scope_hand", cache_dir=tmp)
    rows = [
        # text, source, vector
        ("The zeppelin mast gasket was replaced yesterday.", None, (0, 0, 1, 0)),
        ("We talked about bread dough hydration for a while.", None, (0, 0, 0, 1)),
        ("The bulk fermentation ran five hours.", None, (0, 0, 1, 0)),
        ("A south facing roof suits solar panels well.", "solar.txt", (1, 0, 0, 0)),
        ("Size the solar array around daily usage.", "solar.txt", (0, 1, 0, 0)),
        ("A string inverter is cheaper under even sun.", "solar.txt", (1, 0, 0, 0)),
        ("Tiny note one.", "tiny.txt", (0, 0, 0, 1)),
        ("Tiny note two.", "tiny.txt", (0, 0, 0, 1)),
    ]
    vecs = []
    for text, source, vec in rows:
        trie.texts.append(text)
        trie.roles.append("user")
        trie.turns.append(0)
        trie.sources.append(source)
        trie.origins.append(None)
        trie.timestamps.append(None)
        trie.n_words.append(len(text.split()))
        trie.keyword_weights.append({})
        trie.alive.append(True)
        vecs.append(vec)
    trie.embeddings = np.asarray(vecs, dtype=np.float32)
    trie.dim = 4
    return trie


def check_branch_scores(tmp):
    trie = hand_built(tmp)
    q = np.asarray((1, 0, 0, 0), dtype=np.float32)
    rows = trie.branch_scores(q, {"solar", "inverter", "gasket"}, {"Zeppelin"})
    assert [tuple(r) for r in rows] == [BRANCH_KEYS] * 2, rows
    conv, solar = rows
    assert conv["name"] is None and solar["name"] == "solar.txt", rows
    # the two-row source folds into the conversation: 3 + 2 rows
    assert conv["rows"] == 5 and conv["words"] == 7 + 9 + 6 + 3 + 3, conv
    assert conv["centroid"] == 0.0 and conv["peak"] == 0.0, conv
    assert conv["terms"] == 1 and conv["names"] == 1, conv
    assert solar["rows"] == 3 and solar["words"] == 8 + 7 + 8, solar
    # mean of (1,0,0,0), (0,1,0,0), (1,0,0,0) normalized, dotted with q
    assert solar["centroid"] == round(2 / 5 ** 0.5, 4) == 0.8944, solar
    assert solar["peak"] == 1.0, solar
    assert solar["terms"] == 2 and solar["names"] == 0, solar

    # a query with no lexical side still scores the vectors
    bare = trie.branch_scores(q)
    assert [(r["terms"], r["names"]) for r in bare] == [(0, 0), (0, 0)], bare
    assert [r["centroid"] for r in bare] == [0.0, 0.8944], bare

    # masking one solar row drops that source under the fold threshold:
    # its two living rows join the conversation, which now carries the
    # peak and the solar hits, and the masked row's inverter is unread
    trie.alive[5] = False
    rows = trie.branch_scores(q, {"solar", "inverter", "gasket"}, {"Zeppelin"})
    assert len(rows) == 1 and rows[0]["name"] is None, rows
    only = rows[0]
    assert only["rows"] == 7 and only["words"] == 28 + 8 + 7, only
    assert only["peak"] == 1.0 and only["terms"] == 2, only
    trie.alive[5] = True

    empty = SessionTrie("scope_empty", cache_dir=tmp)
    assert empty.branch_scores(q) == [], "an empty session must report nothing"
    print("A. branch scores: one row per branch with the asserted keys, "
          "hand-computed cosines, distinct-term hits, the fold under the "
          "threshold, masked rows unread, empty session empty")


def build_session(name, tmp, tok, mdl, device):
    trie = SessionTrie(name, cache_dir=tmp)
    trie.add_turn(DOC_TEXT, "user", tokenizer=tok, model=mdl, device=device,
                  source=DOC_NAME, save=False)
    for user, assistant in EXCHANGES:
        trie.add_turn(user, "user", tokenizer=tok, model=mdl, device=device,
                      save=False)
        trie.add_turn(assistant, "assistant", tokenizer=tok, model=mdl,
                      device=device, save=False)
    return trie


def comparable(stats):
    return {k: v for k, v in stats.items()
            if not any(t in k for t in ("time", "elapsed", "seconds"))}


def check_passthrough(tmp, tok, mdl, device, budget):
    from salt.engine.trie_core import embed_query
    calls = []
    real = st_mod.embed_query

    def counted(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    st_mod.embed_query = counted
    try:
        a = build_session("scope_a", tmp, tok, mdl, device)
        b = build_session("scope_b", tmp, tok, mdl, device)
        calls.clear()
        out_a = a.compress(QUERY, budget, tokenizer=tok, model=mdl,
                           device=device)
        assert len(calls) == 1, f"the plain call must encode once, {len(calls)}"
        vec = embed_query(QUERY, tok, mdl, device)
        calls.clear()
        out_b = b.compress(QUERY, budget, tokenizer=tok, model=mdl,
                           device=device, query_embedding=vec)
        assert calls == [], "a handed-in vector must cost no encoder call"
    finally:
        st_mod.embed_query = real
    assert out_a["selected_sent_idx"] == out_b["selected_sent_idx"], (
        out_a["selected_sent_idx"], out_b["selected_sent_idx"])
    assert out_a["context"] == out_b["context"]
    assert comparable(out_a["stats"]) == comparable(out_b["stats"]), (
        {k: (out_a["stats"].get(k), out_b["stats"].get(k))
         for k in set(out_a["stats"]) | set(out_b["stats"])
         if out_a["stats"].get(k) != out_b["stats"].get(k)})
    assert a.coverage == b.coverage and a.coverage_turn == b.coverage_turn
    assert a.drift_ema == b.drift_ema and a._n_compress == b._n_compress
    print("B. pass-through identity: the handed-in query vector gives the "
          "same selection, context, stats, coverage and drift state, and "
          "costs the compressor no encoder call")


def check_scope_candidacy(tmp, tok, mdl, device, budget):
    trie = build_session("scope_cand", tmp, tok, mdl, device)
    kw = dict(budget_pct=budget, tokenizer=tok, model=mdl, device=device,
              defer_commit=True)
    doc_rows = {i for i in range(trie.n_sentences)
                if trie.sources[i] == DOC_NAME}
    conv_rows = set(range(trie.n_sentences)) - doc_rows
    assert doc_rows and conv_rows
    conv_words = sum(trie.n_words[i] for i in conv_rows)
    doc_words = sum(trie.n_words[i] for i in doc_rows)
    plain = int(trie.live_words * budget)

    def scoped_budget(words):
        return min(max(int(words * budget), min(words, SCOPE_FLOOR_WORDS)),
                   plain)

    assert int(doc_words * budget) < min(trie.n_words[i] for i in doc_rows), (
        "the fixture no longer exercises the floor: the document's "
        "fraction must be smaller than its shortest sentence")

    c = trie.compress(QUERY, scope_sources={None}, **kw)
    assert c["selected_sent_idx"], "a conversation-only scope selected nothing"
    assert not set(c["selected_sent_idx"]) & doc_rows, (
        "a scoped turn selected outside its scope")
    s = c["stats"]
    assert (s["scope_branches"], s["scope_excluded"], s["scope_words"]) == (
        1, len(doc_rows), conv_words), s
    assert s["word_budget"] == scoped_budget(conv_words), (s, plain)

    d = trie.compress(DOC_QUERY, scope_sources={DOC_NAME}, **kw)
    assert d["selected_sent_idx"] and set(d["selected_sent_idx"]) <= doc_rows
    assert d["stats"]["scope_words"] == doc_words, d["stats"]
    assert d["stats"]["word_budget"] == scoped_budget(doc_words) <= plain, (
        d["stats"], plain)

    excl = set(sorted(conv_rows)[-4:])
    e = trie.compress(QUERY, scope_sources={None}, exclude_sent_idx=excl, **kw)
    assert not set(e["selected_sent_idx"]) & (doc_rows | excl), (
        "scope and tail exclusion did not compose")
    assert (e["stats"]["scope_excluded"], e["stats"]["excluded_sent"]) == (
        len(doc_rows), len(excl)), e["stats"]

    t2 = build_session("scope_keys", tmp, tok, mdl, device)
    t2.compress(DOC_QUERY, budget, tokenizer=tok, model=mdl, device=device,
                stable_keys=True)
    held = {k: v for k, v in t2.coverage.items()
            if any(x.startswith(FILE_TOKEN_PREFIX) for x in k)}
    assert held, "the priming turn persisted no document key"
    out = t2.compress(QUERY, budget, tokenizer=tok, model=mdl, device=device,
                      stable_keys=True, scope_sources={None})
    assert out["stats"]["scope_excluded"] == len(doc_rows), out["stats"]
    for k, v in held.items():
        assert k in t2.coverage and abs(t2.coverage[k] - v) < 1e-9, (
            f"a document key did not survive a conversation-only scope: "
            f"{sorted(k)}")
    print("D. scope candidacy: nothing selected outside the scope, the "
          "budget taken of the words in scope with the short-branch floor "
          "under the unscoped budget, the kept-out rows reported, tail "
          "exclusion composes, and out-of-scope keys survive the "
          "stable-keys commit")


def check_scope_identity(tmp, tok, mdl, device, budget):
    a = build_session("scope_id_a", tmp, tok, mdl, device)
    b = build_session("scope_id_b", tmp, tok, mdl, device)
    c = build_session("scope_id_c", tmp, tok, mdl, device)
    kw = dict(budget_pct=budget, tokenizer=tok, model=mdl, device=device)
    out_a = a.compress(QUERY, **kw)
    out_b = b.compress(QUERY, scope_sources={None, DOC_NAME}, **kw)
    out_c = c.compress(QUERY, scope_sources={"nothing.txt"}, **kw)

    def core(stats):
        return {k: v for k, v in comparable(stats).items()
                if k not in SCOPE_STATS}

    for out, what in ((out_b, "every branch named"),
                      (out_c, "a scope no row belongs to")):
        assert out["selected_sent_idx"] == out_a["selected_sent_idx"], what
        assert out["context"] == out_a["context"], what
        assert core(out["stats"]) == core(out_a["stats"]), what
    assert out_a["stats"]["scope_branches"] is None
    assert out_a["stats"]["scope_words"] is None
    assert out_a["stats"]["scope_excluded"] == 0
    assert out_b["stats"]["scope_words"] == a.live_words, out_b["stats"]
    assert out_b["stats"]["scope_excluded"] == 0, out_b["stats"]
    assert out_c["stats"]["scope_words"] is None, out_c["stats"]
    assert out_c["stats"]["scope_excluded"] == 0, out_c["stats"]
    assert a.coverage == b.coverage == c.coverage
    print("E. scope identity: every branch named and an empty scope both "
          "select exactly what no scope selects, with only the scope "
          "fields differing in stats")


class _FakeRunner:
    """Answers from a script and keeps every prompt: the turn path needs
    a tokenizer, a window and a stream, nothing else."""

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
        yield f"noted point {len(self.prompts)}."

    def unload(self):
        pass


def chat_session(root, tok, mdl, device, flags, docs=1):
    """One session under the given flags, driven through the real chat
    turn: `docs` documents attached, four questions, then the query."""
    from salt.chat import cli
    args = cli.build_parser().parse_args(
        ["--device", device, "--sync-ingest", *flags])
    trie = SessionTrie("scope_flag", cache_dir=root, model_name=BGE_MODEL,
                       budget_pct_default=args.budget_pct)
    for name, text in ((DOC_NAME, DOC_TEXT), (DOC2_NAME, DOC2_TEXT))[:docs]:
        trie.add_turn(text, "user", tokenizer=tok, model=mdl, device=device,
                      source=name, save=False)
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
    return [ln for ln in buf.getvalue().splitlines()
            if not ln.startswith("ingest")]


def check_branch_stats_flag(tmp, tok, mdl, device):
    from salt.chat import cli
    off = chat_session(tmp / "off", tok, mdl, device, [])
    on = chat_session(tmp / "on", tok, mdl, device, ["--branch-stats"])
    assert off.runner.prompts == on.runner.prompts, (
        "--branch-stats changed a prompt")
    assert off.last_stats == on.last_stats, "--branch-stats changed selection"
    d_off, d_on = cli.build_stats(off), cli.build_stats(on)
    assert set(d_off["kv"]["last_event"] or {}) == set(
        d_on["kv"]["last_event"] or {}), "--branch-stats changed kvtrace keys"
    assert off.last_branch_scores is None and d_off["branches"] is None
    rows = on.last_branch_scores
    assert rows and [tuple(r) for r in rows] == [BRANCH_KEYS] * len(rows), rows
    assert [r["name"] for r in rows] == [None, DOC_NAME], rows
    assert all(0.0 <= r["centroid"] <= 1.0 and 0.0 <= r["peak"] <= 1.0
               for r in rows), rows
    assert d_on["branches"] == rows
    # ingest carries busy time and kv its event timestamps; the kv event's
    # key set is compared above, its values move with the clock
    volatile = {"ingest", "branches", "kv"}
    assert {k: v for k, v in d_off.items() if k not in volatile} == {
        k: v for k, v in d_on.items() if k not in volatile}, (
        "--branch-stats changed a /stats section other than its own")
    lines = cli.branch_lines(rows)
    assert lines[0].startswith("branches") and len(lines) == 1 + len(rows)
    text_off, text_on = stats_text(off), stats_text(on)
    assert all(ln in text_on for ln in lines), text_on
    assert [ln for ln in text_on if ln not in set(lines)] == text_off, (
        text_off, text_on)
    assert cli.branch_lines(None) == []
    assert cli.branch_lines([])[1:] == ["  (no living rows at the last query)"]
    print("C. --branch-stats: prompts, selection and kvtrace keys identical "
          "off and on, no rows off, one asserted row per branch on, and "
          "/stats differs only by the branch block")


def check_scope_rule():
    def row(name, c, p, n=0):
        return {"name": name, "rows": 10, "words": 100, "centroid": c,
                "peak": p, "terms": 0, "names": n}
    conv = row(None, 0.40, 0.50)
    about = row("computers.pdf", 0.72, 0.80)
    mentions = [row(f"mention{i}.pdf", 0.62, 0.69) for i in range(3)]
    cold = [row(f"cold{i}.pdf", 0.52, 0.58) for i in range(6)]
    keep = scope_module.scope_of([conv, about, *mentions, *cold])
    assert keep == {None, "computers.pdf"}, keep
    tied = [row(f"t{i}.pdf", 0.55, 0.78 + 0.005 * i) for i in range(4)]
    keep = scope_module.scope_of([conv, *tied, *cold])
    assert keep == {None} | {f"t{i}.pdf" for i in range(4)}, keep
    named = row("people.pdf", 0.50, 0.55, n=2)
    keep = scope_module.scope_of([conv, about, named, *cold])
    assert keep == {None, "computers.pdf", "people.pdf"}, keep
    wide = scope_module.scope_of([conv, about, *mentions], 0.2, 0.2)
    tight = scope_module.scope_of([conv, about, *mentions], 0.0, 0.0)
    assert tight == {None, "computers.pdf"} and len(wide) == 5, (tight, wide)
    assert tight <= wide, "a tighter margin widened the scope"
    assert scope_module.scope_of([conv]) is None
    assert scope_module.decide("off", [conv, about]) == (None, None)
    assert scope_module.decide("auto", None) == (None, None)
    assert scope_module.decide("auto", [conv, about]) == (
        {None, "computers.pdf"}, None)
    sources, note = scope_module.decide("auto", [{"name": "x.pdf"}])
    assert sources is None and note and "KeyError" in note, (sources, note)
    stats = {"scope_words": 300, "word_budget": 60, "scope_excluded": 60}
    rec = scope_module.record("auto", [conv, about, *cold],
                              {None, "computers.pdf"}, None, stats)
    assert tuple(rec) == scope_module.SCOPED_KEYS, rec
    assert rec["kept"] == ["computers.pdf"] and len(rec["out"]) == 6, rec
    assert (rec["words"], rec["budget"], rec["excluded"]) == (300, 60, 60)
    assert scope_module.record("auto", [conv], None, None, {}) is None
    failed = scope_module.record("auto", [conv, about], None,
                                 "KeyError: 'peak'", {})
    assert failed["kept"] == [] and failed["out"] == ["computers.pdf"]
    assert scope_module.lines(None) == []
    assert scope_module.lines(failed)[0].startswith(
        "scope (auto): not applied"), scope_module.lines(failed)
    text = scope_module.lines(rec)
    assert len(text) == 2 and "'computers.pdf'" in text[0], text
    assert "60 rows held back" in text[1], text
    print("F. scope rule: the about-it file alone beats three mentions and "
          "six cold files, tied peaks all stay, the most-names file stays, "
          "tighter never widens, nothing to route is None, a failing rule "
          "is reported not raised, record and lines carry the keys")


def check_scope_flag(tmp, tok, mdl, device):
    from salt.chat import cli
    off = chat_session(tmp / "g_off", tok, mdl, device, [], docs=0)
    on = chat_session(tmp / "g_on", tok, mdl, device, ["--scope", "auto"],
                      docs=0)
    assert off.runner.prompts == on.runner.prompts, (
        "--scope auto changed a prompt of a session without attachments")
    assert off.last_stats == on.last_stats
    assert on.last_scope is None and cli.build_stats(on)["scoped"] is None
    d_off, d_on = cli.build_stats(off), cli.build_stats(on)
    assert set(d_off["kv"]["last_event"] or {}) == set(
        d_on["kv"]["last_event"] or {}), "--scope auto changed kvtrace keys"
    assert not any(ln.startswith("scope (") for ln in stats_text(on))

    one_off = chat_session(tmp / "g_one_off", tok, mdl, device, [])
    one = chat_session(tmp / "g_one", tok, mdl, device, ["--scope", "auto"])
    assert one_off.runner.prompts == one.runner.prompts, (
        "a single attached file is its own best, so nothing may narrow")
    rec = one.last_scope
    assert rec and tuple(rec) == scope_module.SCOPED_KEYS, rec
    assert (rec["mode"], rec["kept"], rec["out"], rec["note"]) == (
        "auto", [DOC_NAME], [], None), rec
    s = one.last_stats
    assert rec["excluded"] == 0 and s["scope_excluded"] == 0, (rec, s)
    assert (rec["words"], rec["budget"]) == (s["scope_words"], s["word_budget"])
    assert s["scope_branches"] == 2, s

    two = chat_session(tmp / "g_two", tok, mdl, device,
                       ["--scope", "auto", "--scope-margin", "0",
                        "--scope-peak-margin", "0"], docs=2)
    with redirect_stdout(io.StringIO()):
        cli.chat_turn(two, DOC_QUERY)
    rec = two.last_scope
    assert rec and set(rec["kept"]) | set(rec["out"]) == {DOC_NAME, DOC2_NAME}
    assert rec["kept"] and rec["out"] == [DOC2_NAME], (
        "the irrigation question did not keep the irrigation notes alone")
    s = two.last_stats
    assert (s["scope_branches"], s["scope_excluded"] > 0) == (2, True), s
    assert (rec["words"], rec["budget"], rec["excluded"]) == (
        s["scope_words"], s["word_budget"], s["scope_excluded"]), (rec, s)
    prompt = two.runner.prompts[-1][-1]["content"]
    assert f"[from attached file '{DOC2_NAME}'" not in prompt, (
        "an out-of-scope file reached the prompt")
    assert cli.build_stats(two)["scoped"] == rec
    text = stats_text(two)
    assert any(ln.startswith("scope (auto): searched") for ln in text), text
    ev = cli.build_stats(two)["kv"]["last_event"] or {}
    assert ev.get("scope_kept") == rec["kept"] and ev.get("scope_out") == 1
    assert ev.get("scope_words") == rec["words"], ev
    assert "scope_kept" not in (d_on["kv"]["last_event"] or {})

    real = scope_module.scope_of

    def broken(*a, **kw):
        raise RuntimeError("no rule today")

    scope_module.scope_of = broken
    try:
        with redirect_stdout(io.StringIO()):
            cli.chat_turn(two, QUERY)
    finally:
        scope_module.scope_of = real
    rec = two.last_scope
    assert rec and rec["note"] == "RuntimeError: no rule today", rec
    assert two.last_stats["scope_branches"] is None, two.last_stats
    assert "scope_kept" not in (cli.build_stats(two)["kv"]["last_event"] or {})
    assert any(ln.startswith("scope (auto): not applied")
               for ln in stats_text(two)), stats_text(two)
    print("G. --scope auto: byte-identical without attachments and with one "
          "file, a two-file question keeps its file and the other leaves "
          "the prompt, stats and kvtrace agree, and a failing rule leaves "
          "the turn unscoped and reported")


def check_scope_census(tmp, tok, mdl, device):
    from salt.chat import cli
    off = chat_session(tmp / "h_off", tok, mdl, device, [], docs=2)
    assert cli.build_stats(off)["scope_census"] is None
    assert not any(ln.startswith("scope census") for ln in stats_text(off))

    st = chat_session(tmp / "h_on", tok, mdl, device,
                      ["--scope", "auto", "--scope-margin", "0",
                       "--scope-peak-margin", "0"], docs=2)
    c = st.scope_stats
    assert tuple(c) == scope_module.CENSUS_KEYS, c
    assert c["asked"] == 5 == c["scoped"] and c["plain"] == c["guarded"] == 0, c
    before = copy.deepcopy(c)
    with redirect_stdout(io.StringIO()):
        cli.chat_turn(st, DOC_QUERY)
    rec, after = st.last_scope, st.scope_stats
    assert after["asked"] == 6 and after["scoped"] == 6, after
    n = len(rec["kept"])
    assert after["branches"][n] == before["branches"].get(n, 0) + 1, after
    for name in rec["kept"]:
        assert after["kept"][name] == before["kept"].get(name, 0) + 1, after
    assert sum(after["branches"].values()) == after["scoped"], after
    assert sum(after["kept"].values()) == sum(
        k * v for k, v in after["branches"].items()), after

    real = scope_module.scope_of

    def broken(*a, **kw):
        raise RuntimeError("no rule today")

    scope_module.scope_of = broken
    try:
        with redirect_stdout(io.StringIO()):
            cli.chat_turn(st, QUERY)
    finally:
        scope_module.scope_of = real
    assert (st.scope_stats["asked"], st.scope_stats["guarded"]) == (7, 1), (
        st.scope_stats)
    assert cli.build_stats(st)["scope_census"] == st.scope_stats
    text = stats_text(st)
    head = [ln for ln in text if ln.startswith("scope census:")]
    assert head == ["scope census: 7 turns asked, 6 scoped, 0 with nothing "
                    "to route, 1 not applied"], text
    assert any(ln.startswith("  files kept per scoped turn:") for ln in text)
    assert any(ln.startswith("  kept:") for ln in text), text

    sessions, cli.SESSIONS_DIR = cli.SESSIONS_DIR, tmp / "h_sessions"
    try:
        with redirect_stdout(io.StringIO()):
            cli.handle_command("/new h_fresh", st)
    finally:
        cli.SESSIONS_DIR = sessions
    assert st.scope_stats == scope_module.census(), st.scope_stats

    bare = chat_session(tmp / "h_bare", tok, mdl, device, ["--scope", "auto"],
                        docs=0)
    c = bare.scope_stats
    assert (c["asked"], c["plain"], c["scoped"]) == (4, 4, 0), c
    assert cli.scope_module.census_lines(c)[0] == (
        "scope census: 4 turns asked, 0 scoped, 4 with nothing to route, "
        "0 not applied")
    print("H. scope census: counts follow the records turn by turn, a "
          "failing rule counts as not applied, plain turns count on a "
          "session without attachments, /stats prints it, a new session "
          "starts it over, and scope off keeps no census")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--budget", type=float, default=0.2)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="salt_scope_regression_"))
    try:
        check_branch_scores(tmp)
        check_scope_rule()
        from salt.engine.compressor import load_bge
        print(f"Loading BGE encoder {BGE_MODEL} on {args.device} ...")
        tok, mdl = load_bge(BGE_MODEL, args.device)
        check_passthrough(tmp, tok, mdl, args.device, args.budget)
        check_branch_stats_flag(tmp, tok, mdl, args.device)
        check_scope_candidacy(tmp, tok, mdl, args.device, args.budget)
        check_scope_identity(tmp, tok, mdl, args.device, args.budget)
        check_scope_flag(tmp, tok, mdl, args.device)
        check_scope_census(tmp, tok, mdl, args.device)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS")


if __name__ == "__main__":
    main()
