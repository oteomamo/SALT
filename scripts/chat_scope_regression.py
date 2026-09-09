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

Group B needs the BGE encoder (downloaded to the HF cache on first use).
CPU is the default device; the run takes well under a minute.

Usage:
    python scripts/chat_scope_regression.py [--device cpu] [--budget 0.2]
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import salt.engine.session_trie as st_mod
from salt.engine.session_trie import BRANCH_KEYS, SessionTrie

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--budget", type=float, default=0.2)
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="salt_scope_regression_"))
    try:
        check_branch_scores(tmp)
        from salt.engine.compressor import load_bge
        print(f"Loading BGE encoder {BGE_MODEL} on {args.device} ...")
        tok, mdl = load_bge(BGE_MODEL, args.device)
        check_passthrough(tmp, tok, mdl, args.device, args.budget)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS")


if __name__ == "__main__":
    main()
