# -*- coding: utf-8 -*-
"""Regression harness for saltChat's background ingestion (IngestWorker).

Replays a scripted transcript through SessionTrie three ways - direct
calls, an inline synchronous worker (--sync-ingest), and a background
worker mirroring chat_turn's order - and asserts:

  1. submit() returns while a job runs, drain() blocks, strict FIFO.
  2. All three runs end identical: corpus, embeddings, coverage, drift
     EMA, selections, near-dup decisions. Pre-turn rows never move
     while a job is in flight (record_turn's index-stability).
  3. A failing job is reported at the next drain(), the worker survives.
  4. Failures are journaled verbatim in ingest_failures.jsonl and the
     `journaled` flag tells the truth even when the write fails.
  5. Sync mode runs inline, journals, and re-raises at the submit site.
     KeyboardInterrupt passes through unlogged.
  6. close() finishes queued work, reports, is idempotent, rejects
     later submits.
  7. A leaked pending counter self-heals at the next drain().
  8. stats counts jobs, failures, and busy_s exactly.
  9. atexit drains the queue on exits that bypass close() (subprocess).
 10. A turn whose generation fails still has the user line in the trie.
 11. Background mode saves once per turn and reloads identical to the
     per-call-saved direct run. An error turn stays dirty until the
     boundary save persists it.
 12. The stored corpus is faithful to what was typed: generics, table
     rows and rescued link sentences appear in trie.texts verbatim.
 13. A torn save (embeddings.npy vs state.pkl length mismatch) is
     repaired on load: orphan rows drop byte-exact, an over-long corpus
     truncates with hashes withdrawn, the turn clock never rewinds, and
     a clean session reloads as a strict no-op.
 14. compress(max_words=None) is byte-identical to omitting the arg;
     with a cap the selected word count never exceeds it and the budget
     plateaus, while the uncapped budget keeps growing with the corpus.

Needs the BGE encoder (fetched to the HF cache on first use). The CPU
run takes under a minute. Refuses to run under `python -O`.

Usage:
    python scripts/chat_ingest_regression.py [--device cpu]
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

if not __debug__:
    sys.exit("this harness is assert-based - run it without python -O")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from salt.chat.ingest import IngestWorker
from salt.engine.compressor import load_bge
from salt.engine.session_trie import SessionTrie

BGE_MODEL = "BAAI/bge-small-en-v1.5"
DEDUP_COS = 0.92

R_ORIG = ("The reservoir holds four million liters and feeds the eastern "
          "irrigation district through the whole dry season.")
R_RESTATE = ("Remember, the reservoir holds four million liters and feeds "
             "the eastern irrigation district through the whole dry season.")

LONG_PASTE = " ".join(
    f"Paragraph {i} of the survey report covers the {w} network in detail, "
    f"including its maintenance backlog, seasonal load profile, and the "
    f"interconnects with the neighboring districts."
    for i, w in enumerate(
        "canal pumping filtration telemetry drainage storage".split()))

TRANSCRIPT = [
    ("How is the eastern irrigation district supplied through the summer?",
     R_ORIG + " Gravity feed covers the upper terraces without pumping."),
    (LONG_PASTE,
     "Noted - the survey flags the pumping and drainage networks as the "
     "two with the largest maintenance backlog."),
    ("Which network had the largest seasonal load swing in that survey?",
     R_RESTATE + " The pumping network shows the widest seasonal swing."),
    ("What would reduce the pumping network's summer load?",
     "Night irrigation windows and lining the upper canal would cut summer "
     "pumping load by roughly a quarter."),
]


def run_transcript(mode, cache_dir, tok, mdl, device, budget):
    """One full session in the given mode, mirroring chat_turn's order.
    Returns (trie, per-turn log, worker-or-None)."""
    trie = SessionTrie(f"ingest-{mode}", cache_dir=cache_dir,
                       model_name=BGE_MODEL)
    kw = dict(tokenizer=tok, model=mdl, device=device, dedup_cos=DEDUP_COS)
    worker = None
    if mode != "direct":
        worker = IngestWorker(
            journal_path=trie.cache_dir / "ingest_failures.jsonl",
            synchronous=(mode == "sync"))
    per_turn = []
    for user, assistant in TRANSCRIPT:
        if worker is not None:
            fails = worker.drain()          # the dispatch/chat_turn barrier
            assert not fails, f"{mode}: unexpected ingest failures {fails}"
        sel = []
        if trie.n_sentences:
            comp = trie.compress(query=user, budget_pct=budget,
                                 tokenizer=tok, model=mdl, device=device)
            sel = list(comp["selected_sent_idx"])
        if mode == "background":
            # chat_turn's order: the user line encodes during generation
            # while the main thread re-reads pre-turn rows, exactly like
            # kvtrace.record_turn
            n_pre = trie.n_sentences
            pre_rows = [(trie.texts[i], trie.n_words[i])
                        for i in range(n_pre)]
            worker.submit(lambda u=user: trie.add_turn(u, "user",
                                                       save=False, **kw),
                          label="user-message ingest", payload=user)
            while True:
                busy = worker.pending > 0
                for i in range(n_pre):
                    assert (trie.texts[i], trie.n_words[i]) == pre_rows[i], (
                        "a pre-turn row changed while the user-line job "
                        "was in flight - record_turn's index-stability "
                        "assumption is broken")
                if not busy:            # one full settled pass, then stop
                    break
            worker.submit(lambda a=assistant: trie.add_turn(a, "assistant",
                                                            save=False,
                                                            **kw),
                          label="assistant-message ingest", payload=assistant)
            worker.submit(lambda: trie.save() if trie.dirty else None,
                          label="session save")
        elif mode == "sync":
            worker.submit(lambda u=user: trie.add_turn(u, "user", **kw),
                          label="user-message ingest", payload=user)
            worker.submit(lambda a=assistant: trie.add_turn(a, "assistant",
                                                            **kw),
                          label="assistant-message ingest", payload=assistant)
        else:
            trie.add_turn(user, "user", **kw)
            trie.add_turn(assistant, "assistant", **kw)
        per_turn.append({"sel": sel})
    if worker is not None:
        assert worker.drain() == [], f"{mode}: failures at final drain"
    return trie, per_turn, worker


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--budget", type=float, default=0.35)
    ap.add_argument("--keep", action="store_true",
                    help="keep the temp session dirs for inspection")
    args = ap.parse_args()

    # assert 1: overlap + FIFO, no encoder needed
    w = IngestWorker()
    gate, started, order = threading.Event(), threading.Event(), []
    w.submit(lambda: (started.set(), gate.wait(), order.append(-1)),
             label="gated")
    assert started.wait(5) and not order, (
        "submit did not return while the job was pending - the prompt-"
        "return property is broken")
    assert w.pending == 1
    for i in range(100):
        w.submit(lambda i=i: order.append(i))
    threading.Timer(0.2, gate.set).start()
    t0 = time.monotonic()
    assert w.drain() == []
    assert time.monotonic() - t0 >= 0.15, "drain returned before the gated job"
    assert order == [-1] + list(range(100)), (
        "queued jobs did not run strictly FIFO - the single-mutator "
        "ordering is broken")
    w.close()
    print("overlap + FIFO: submit non-blocking, drain exact, 101 jobs in order")

    # --turns file parsing (pure, no encoder): json array, jsonl, bare
    # strings, field auto-detect + override, and the ambiguous-item refusal
    from salt.chat.cli import load_turns, _turn_text
    with tempfile.TemporaryDirectory() as td:
        arr = Path(td) / "a.json"
        arr.write_text(json.dumps([{"id": "test-0", "puzzle": "solve me"},
                                   {"id": "test-1", "puzzle": "and me"}]))
        assert load_turns(str(arr)) == [("test-0", "solve me"),
                                        ("test-1", "and me")]
        jl = Path(td) / "b.jsonl"
        jl.write_text('{"id":"a","question":"Q1"}\n'
                      '{"id":"b","question":"Q2"}\n')
        assert load_turns(str(jl)) == [("a", "Q1"), ("b", "Q2")]
        strs = Path(td) / "c.json"
        strs.write_text(json.dumps(["hi", "there"]))
        assert load_turns(str(strs)) == [(None, "hi"), (None, "there")]
    assert _turn_text({"id": "x", "body": "B"}, "body", 0) == "B"
    assert _turn_text({"prompt": "P", "text": "T"}, None, 0) == "P"
    try:
        _turn_text({"a": "1", "b": "2"}, None, 0)
        raise AssertionError("ambiguous turn item did not raise")
    except ValueError:
        pass
    print("--turns parsing: json array, jsonl, bare strings, field "
          "auto-detect and override, ambiguous item rejected")

    # short-turn predicate (pure, no encoder): the four W9 utterances pass,
    # junk still drops, and the filter without keep= still drops them all
    from salt.chat.shortturn import is_short_user_unit
    from salt.engine.sentence_filter import filter_texts
    for t in ("yes", "Go with option B.", "the second one",
              "no, use PostgreSQL"):
        assert is_short_user_unit(t), f"short decisive turn rejected: {t!r}"
    for t in ("https://example.com/decision", "?!.,",
              "This ordinary sentence is long enough to pass the junk "
              "filter's length gates entirely on its own."):
        assert not is_short_user_unit(t), f"non-target unit kept: {t!r}"
    kept, *_ = filter_texts(["yes", "no, use PostgreSQL"], aggressive=True,
                            remove_urls=True, deduplicate=True,
                            strip_urls=True, lenient=True)
    assert kept == [], (
        "chat-ingest filter defaults now keep short turns WITHOUT keep= - "
        "sentence_filter's frozen behavior changed")
    print("short turns: predicate keeps the four W9 utterances, drops "
          "junk, filter still drops them without keep=")

    # fuse helpers (pure): acks are classified, the fused unit carries
    # both the utterance and the question it answers
    from salt.chat.shortturn import acknowledgement_only, fuse_with_question
    for t in ("yes", "ok sure", "the second one"):
        assert acknowledgement_only(t), f"ack not classified: {t!r}"
    for t in ("no, use PostgreSQL", "go with option B"):
        assert not acknowledgement_only(t), f"non-ack classified: {t!r}"
    fused = fuse_with_question("the second one",
                               "Two choices.\nRed option, or the blue one?")
    assert "the second one" in fused and "blue one?" in fused, fused
    assert "Two choices" not in fused, (
        "fuse quoted more than the last sentence of the reply")
    print("short turns: fuse quotes the answered question, acks classified")

    # memory cap plumbing (pure, no encoder): parse, off, auto with a
    # stub runner, explicit int conversion, and the floor
    import types as _types
    from salt.chat.cli import (MEMORY_CAP_FLOOR_WORDS, memory_word_cap,
                               parse_memory_cap, prompt_fixed_tokens)
    assert parse_memory_cap("off") == "off"
    assert parse_memory_cap("auto") == "auto"
    assert parse_memory_cap("4000") == 4000
    assert parse_memory_cap("-3") is None and parse_memory_cap("x") is None

    class _StubRunner:
        alias = "stub"
        def __init__(self, budget):
            self._b = budget
        def input_budget(self):
            return self._b

    stub = _types.SimpleNamespace(
        runner=_StubRunner(100000), full_attachments={"a.txt": "alpha beta"},
        tail=[{"role": "user", "content": "one two"},
              {"role": "assistant", "content": "three four five"}],
        trie=_types.SimpleNamespace(attached_sources=[], n_sentences=0),
        count_tokens=lambda text: len(text.split()),
        _fixed_tokens_cache=None, memory_cap="off", tokens_per_word=1.6)
    assert memory_word_cap(stub) is None, "off must disable the cap"
    fixed = prompt_fixed_tokens(stub)
    assert fixed and fixed > 5, "fixed cost missed the prompt components"
    stub.memory_cap = "auto"
    wide = memory_word_cap(stub, "the user line")
    assert wide and wide > MEMORY_CAP_FLOOR_WORDS, wide
    stub.runner = _StubRunner(fixed + 100)
    stub._fixed_tokens_cache = None
    tight = memory_word_cap(stub, "the user line")
    assert tight == MEMORY_CAP_FLOOR_WORDS, (
        f"a tight window must land on the floor, got {tight}")
    stub.memory_cap = 320
    assert memory_word_cap(stub) == int(320 / 1.6)
    print("memory cap: off/auto/int parse and convert, auto fits the "
          "window, floor holds when the window is tight")

    print(f"Loading BGE encoder {BGE_MODEL} on {args.device} ...")
    tok, mdl = load_bge(BGE_MODEL, args.device)
    tmp = Path(tempfile.mkdtemp(prefix="salt_ingest_regression_"))
    try:
        # assert 2: three-way identity - direct vs sync worker vs background
        runs = {m: run_transcript(m, tmp, tok, mdl, args.device, args.budget)
                for m in ("direct", "sync", "background")}
        base, base_turns, _ = runs["direct"]
        assert base.n_near_dups >= 1, (
            "the engineered restatement did not fire the near-dup gate - "
            "the gate-through-the-worker check is void (encoder change?)")
        for mode in ("sync", "background"):
            t, turns, _ = runs[mode]
            for attr in ("texts", "roles", "turns", "sources", "n_words",
                         "keyword_weights", "coverage", "drift_ema",
                         "n_near_dups", "_seen_hashes"):
                assert getattr(t, attr) == getattr(base, attr), (
                    f"{mode}: trie.{attr} diverged from the direct run - "
                    f"backgrounding is visible in selection state")
            assert np.array_equal(t.embeddings, base.embeddings), (
                f"{mode}: embeddings diverged from the direct run")
            assert [x["sel"] for x in turns] == [x["sel"] for x in base_turns], (
                f"{mode}: per-turn selections diverged from the direct run")
            log_a = (base.cache_dir / "near_dups.jsonl").read_text()
            log_b = (t.cache_dir / "near_dups.jsonl").read_text()
            assert log_a == log_b, (
                f"{mode}: near-dup log records differ from the direct run")
        print(f"identity: direct == sync == background over "
              f"{len(TRANSCRIPT)} turns ({base.n_sentences} sentences, "
              f"{base.n_near_dups} near-dup suppressed, embeddings "
              f"byte-identical)")

        # assert 11 (persistence half)
        disk_direct = SessionTrie("ingest-direct", cache_dir=tmp,
                                  model_name=BGE_MODEL)
        for mode in ("sync", "background"):
            rl = SessionTrie(f"ingest-{mode}", cache_dir=tmp,
                             model_name=BGE_MODEL)
            assert rl.is_loaded, f"{mode}: session did not reload"
            for attr in ("texts", "roles", "turns", "sources", "n_words",
                         "keyword_weights", "coverage", "drift_ema",
                         "n_near_dups", "n_turns", "_seen_hashes"):
                assert getattr(rl, attr) == getattr(disk_direct, attr), (
                    f"{mode}: reloaded {attr} differs from the direct "
                    f"run's - coalesced saving lost state")
            assert np.array_equal(rl.embeddings, disk_direct.embeddings), (
                f"{mode}: reloaded embeddings differ from the direct run's")
        print("persistence: coalesced background disk state reloads "
              "identical to per-call-saved direct")

        # short turns through the real ingest path: keep= admits the unit,
        # default ingest still drops it
        st = SessionTrie("shortturn", cache_dir=tmp, model_name=BGE_MODEL)
        info = st.add_turn("no, use PostgreSQL", role="user", tokenizer=tok,
                           model=mdl, device=args.device)
        assert info["added"] == 0, (
            "a short user turn entered the trie with keep=None - the "
            "default-off contract is broken")
        info = st.add_turn("no, use PostgreSQL", role="user", tokenizer=tok,
                           model=mdl, device=args.device,
                           keep=is_short_user_unit)
        assert info["added"] == 1 and st.roles[-1] == "user", (
            "keep=is_short_user_unit did not ingest the short user turn")
        print("short turns: add_turn keeps the decision under keep=, "
              "drops it by default")

        # fuse through the real add_to_trie wiring: an ack is stored fused
        # with its question, a non-ack short turn stays verbatim
        import types
        from salt.chat.cli import add_to_trie
        ns = types.SimpleNamespace(trie=st, bge_tok=tok, bge_model=mdl,
                                   bge_device=args.device, dedup_cos=None,
                                   short_turns="fuse")
        info = add_to_trie(ns, "the second one", "user", save=False,
                           context="Do you want the red option, or the "
                                   "blue option?")
        assert info["added"] == 1, "fused ack did not enter the trie"
        assert ("the second one" in st.texts[-1]
                and "blue option?" in st.texts[-1]), st.texts[-1]
        info = add_to_trie(ns, "no, use MariaDB", "user", save=False,
                           context="SQLite then?")
        assert info["added"] == 1 and st.texts[-1] == "no, use MariaDB", (
            "a content-bearing short turn was not stored verbatim in "
            "fuse mode")
        print("short turns: fuse stores ack+question, content-bearing "
              "short turns stay verbatim")

        # assert 12: the CORPUS (not merely the embedding input) keeps
        # code, tables and link sentences as typed
        tf = SessionTrie("faithful", cache_dir=tmp, model_name=BGE_MODEL)
        tf.add_turn("The cache is a HashMap<String, Vec<u8>> guarded by "
                    "one mutex per shard.\n"
                    "| Model | Score | Latency |\n"
                    "| tiny | 61.2 | 12ms |\n"
                    "The eviction notes live at https://ex.io/cache so "
                    "read them before tomorrow.",
                    role="user", tokenizer=tok, model=mdl,
                    device=args.device)
        assert any("HashMap<String, Vec<u8>>" in x for x in tf.texts), (
            "generics were mangled on the way into the corpus")
        assert "| Model | Score | Latency |" in tf.texts, (
            "the table header row did not reach the corpus verbatim")
        assert any(x.startswith("The eviction notes live at <url>")
                   for x in tf.texts), (
            "the link sentence did not survive with its prose")
        print("faithful corpus: generics, table rows and the <url> "
              "sentence stored as typed")

        # assert 10: failed-generation survival (background order)
        line = ("Please remember that the maintenance window moves to "
                "Sunday night for the pumping stations.")
        tg = SessionTrie("genfail", cache_dir=tmp, model_name=BGE_MODEL)
        wg = IngestWorker(journal_path=tg.cache_dir / "f.jsonl")
        kw = dict(tokenizer=tok, model=mdl, device=args.device,
                  dedup_cos=DEDUP_COS)
        try:
            assert wg.drain() == []
            wg.submit(lambda: tg.add_turn(line, "user", save=False, **kw),
                      label="user-message ingest", payload=line)
            raise RuntimeError("simulated generation failure")
        except RuntimeError:
            pass
        assert wg.drain() == []
        assert any("maintenance window" in t for t in tg.texts), (
            "the user line was lost on a failed generation - the "
            "rides-the-generation-window fix does not hold")
        # assert 11 (error-turn half): dirty until the boundary save
        assert tg.dirty, "an unsaved error turn did not mark the trie dirty"
        ghost = SessionTrie("genfail", cache_dir=tmp, model_name=BGE_MODEL)
        assert not ghost.is_loaded, (
            "an unsaved error turn already reached disk - the save=False "
            "path is writing")
        tg.save()                       # what exit and /new run when dirty
        assert not tg.dirty
        recovered = SessionTrie("genfail", cache_dir=tmp,
                                model_name=BGE_MODEL)
        assert recovered.is_loaded and any(
            "maintenance window" in t for t in recovered.texts), (
            "the boundary save did not persist the error turn's ingest")
        wg.close()
        print("failed generation: the user line survives the abort, "
              "dirty until the boundary save persists it")

        # assert 13: torn-save reconciliation (embeddings vs state)
        tt = SessionTrie("tornsave", cache_dir=tmp, model_name=BGE_MODEL)
        for s in ("The turbine hall inspection is booked for Thursday "
                  "morning with the vendor.",
                  "Grid frequency stayed inside the band for the whole "
                  "afternoon test window.",
                  "The relay firmware rollback finished without any "
                  "alarms raised overnight."):
            tt.add_turn(s, "user", tokenizer=tok, model=mdl,
                        device=args.device)
        assert tt.load_repair is None and not tt.dirty
        clean_emb = np.array(tt.embeddings, copy=True)
        n_clean = tt.n_sentences
        turn_clock = tt._next_turn_index
        ep = tt.cache_dir / "embeddings.npy"

        c0 = SessionTrie("tornsave", cache_dir=tmp, model_name=BGE_MODEL)
        assert c0.load_repair is None and not c0.dirty, (
            "a healthy session was repaired - the no-op contract broke")
        assert c0.texts == tt.texts and np.array_equal(c0.embeddings,
                                                       clean_emb)

        with open(ep, "wb") as fh:
            np.save(fh, np.vstack([clean_emb,
                                   np.zeros((2, clean_emb.shape[1]),
                                            dtype=clean_emb.dtype)]))
        ta = SessionTrie("tornsave", cache_dir=tmp, model_name=BGE_MODEL)
        assert ta.load_repair == {"kept": n_clean, "orphan_rows": 2,
                                  "dropped_sentences": 0}, ta.load_repair
        assert ta.dirty, "a repaired session must be dirty until saved"
        assert len(ta.texts) == ta.embeddings.shape[0] == n_clean
        assert np.array_equal(ta.embeddings, clean_emb), (
            "surviving rows changed - the repair must be byte-exact")
        ta.save()
        t2 = SessionTrie("tornsave", cache_dir=tmp, model_name=BGE_MODEL)
        assert t2.load_repair is None, (
            "a persisted repair was repaired again - not idempotent")

        with open(ep, "wb") as fh:
            np.save(fh, clean_emb[:-2])
        tb = SessionTrie("tornsave", cache_dir=tmp, model_name=BGE_MODEL)
        assert tb.load_repair == {"kept": n_clean - 2, "orphan_rows": 0,
                                  "dropped_sentences": 2}, tb.load_repair
        assert (len(tb.texts) == len(tb.roles) == len(tb.turns)
                == len(tb.sources) == len(tb.timestamps)
                == len(tb.n_words) == len(tb.keyword_weights)
                == tb.embeddings.shape[0] == n_clean - 2), (
            "the seven corpus lists and the matrix disagree after repair")
        for t in tt.texts[-2:]:
            assert tt._norm_hash(t) not in tb._seen_hashes, (
                "a dropped sentence's hash was not withdrawn")
        assert tb._next_turn_index == turn_clock, (
            "a repair rewound the turn clock")
        print("torn save: orphan rows dropped byte-exact, truncated "
              "matrix shrinks the corpus, hashes withdrawn, repair "
              "persists and is idempotent")

        # assert 14: the word-budget cap (W20)
        ckw = dict(tokenizer=tok, model=mdl, device=args.device)
        ta14 = SessionTrie("cap-a", cache_dir=tmp, model_name=BGE_MODEL)
        tb14 = SessionTrie("cap-b", cache_dir=tmp, model_name=BGE_MODEL)
        for t14 in (ta14, tb14):
            for i in range(5):
                t14.add_turn(f"Seed sentence {i} covers the intake filter "
                             f"swap and the flow calibration for line {i}.",
                             "user", **ckw)
        ca = ta14.compress(query="what covers the filters?", budget_pct=0.3,
                           **ckw)
        cb = tb14.compress(query="what covers the filters?", budget_pct=0.3,
                           max_words=None, **ckw)
        assert ca["context"] == cb["context"] and (
            ca["stats"]["word_budget"] == cb["stats"]["word_budget"]), (
            "max_words=None diverged from omitting the argument")
        assert ca["stats"]["word_budget_capped"] is False

        CAP = 40
        tu = SessionTrie("cap-grow-u", cache_dir=tmp, model_name=BGE_MODEL)
        tc = SessionTrie("cap-grow-c", cache_dir=tmp, model_name=BGE_MODEL)
        budgets_u, budgets_c, capped_seen = [], [], False
        for i in range(30):
            s14 = (f"Maintenance item {i} covers the pump gasket swap and "
                   f"the pressure flow test for line {i}.")
            tu.add_turn(s14, "user", save=False, **ckw)
            tc.add_turn(s14, "user", save=False, **ckw)
            cu = tu.compress(query="pump maintenance status?",
                             budget_pct=0.2, **ckw)
            cc = tc.compress(query="pump maintenance status?",
                             budget_pct=0.2, max_words=CAP, **ckw)
            budgets_u.append(cu["stats"]["word_budget"])
            budgets_c.append(cc["stats"]["word_budget"])
            sel_words = sum(tc.n_words[j] for j in cc["selected_sent_idx"])
            assert sel_words <= CAP, (
                f"turn {i}: capped selection used {sel_words} words > {CAP}")
            capped_seen = capped_seen or cc["stats"]["word_budget_capped"]
        assert budgets_u == sorted(budgets_u) and budgets_u[-1] > budgets_u[0], (
            "the uncapped budget stopped growing with the corpus - the "
            "growth premise of this check is broken")
        assert capped_seen and budgets_c[-1] == CAP == max(budgets_c), (
            "the capped budget never plateaued at the cap")
        assert budgets_u[-1] > CAP, (
            "the uncapped run never outgrew the cap, so the comparison "
            "is vacuous")
        print(f"word cap: None identical to omitted, capped run plateaus "
              f"at {CAP} words (uncapped reached {budgets_u[-1]})")

        # asserts 3+4: deferred failure report + journal recovery
        jdir = tmp / "failsession"
        jdir.mkdir()
        wf = IngestWorker(journal_path=jdir / "ingest_failures.jsonl")
        ran = []
        wf.submit(lambda: ran.append("before"), label="pre")

        def boom():
            raise RuntimeError("simulated encode failure")

        wf.submit(boom, label="user-message ingest",
                  payload="the pasted message that failed to encode")
        wf.submit(lambda: ran.append("after"), label="post")
        fails = wf.drain()
        assert ran == ["before", "after"], (
            "jobs after a failure did not run - the worker died")
        assert len(fails) == 1 and fails[0]["label"] == "user-message ingest" \
            and "simulated encode failure" in fails[0]["error"], (
            f"failure not reported at drain: {fails}")
        assert wf.drain() == [], "a failure was delivered twice"
        assert fails[0]["journaled"] is True, (
            "a journaled failure was not marked journaled - the REPL "
            "report would under-claim preservation")
        rec = json.loads(
            (jdir / "ingest_failures.jsonl").read_text().strip())
        assert rec["payload"] == "the pasted message that failed to encode", (
            "the journal did not preserve the failed payload verbatim")
        assert "journaled" not in rec, (
            "the self-referential journaled flag leaked into the file")
        wf.close()
        wu = IngestWorker(journal_path=tmp / "no_dir" / "unwritable.jsonl")
        wu.submit(boom, label="journal-less", payload="still reported")
        fails = wu.drain()
        assert len(fails) == 1 and fails[0]["payload"] == "still reported", (
            "an unwritable journal cost the drain report - delivery must "
            "not depend on the journal write")
        assert fails[0]["journaled"] is False, (
            "a failed journal write was reported as preserved - the REPL "
            "report would lie about recovery")
        wu.close()
        print("failure path: deferred report, worker survives, journal "
              "verbatim, report independent of journal")

        # assert 5: sync-mode contract - inline, journaled, re-raised
        js = jdir / "sync_failures.jsonl"
        ws = IngestWorker(journal_path=js, synchronous=True)
        seen = []
        ws.submit(lambda: seen.append(1), label="inline")
        assert seen == [1] and ws._thread is None and ws.pending == 0, (
            "sync mode did not run the job inline")
        try:
            ws.submit(boom, label="sync-fail", payload="sync payload")
        except RuntimeError:
            seen.append("raised")
        assert seen == [1, "raised"], (
            "a sync-mode failure did not re-raise at the submit site - "
            "--sync-ingest would no longer abort the turn like today")
        assert json.loads(js.read_text().strip())["payload"] == "sync payload"
        assert ws.drain() == [], (
            "a re-raised sync failure was also queued for drain - one "
            "failure would be reported twice")
        try:
            ws.submit(lambda: (_ for _ in ()).throw(KeyboardInterrupt()),
                      label="ctrl-c")
            raise AssertionError("KeyboardInterrupt did not propagate")
        except KeyboardInterrupt:
            pass
        assert ws.stats["failures"] == 1 and len(
            js.read_text().strip().splitlines()) == 1, (
            "a KeyboardInterrupt was logged as an ingest failure")
        ws.close()
        print("sync mode: inline, journal + re-raise at the submit site, "
              "Ctrl-C passes through unlogged")

        # assert 6: close semantics
        wc = IngestWorker()
        slow = []
        wc.submit(lambda: (time.sleep(0.2), slow.append(1)), label="slow")
        wc.submit(boom, label="fail-at-close")
        fails = wc.close()
        assert slow == [1], "close did not finish queued work first"
        assert len(fails) == 1, "close did not return undelivered failures"
        assert wc.close() == [], "close is not idempotent"
        try:
            wc.submit(lambda: None)
            raise AssertionError("submit after close did not raise")
        except RuntimeError:
            pass
        print("close: drains first, reports, idempotent, rejects submits")

        # assert 7: pending self-heal (the interrupted-submit leak)
        wp = IngestWorker()
        with wp._lock:
            wp._pending += 1            # what a Ctrl-C mid-submit leaves
        assert wp.pending == 1 and wp.drain() == [] and wp.pending == 0, (
            "drain did not reset a leaked pending counter")
        wp.close()

        # assert 8: bookkeeping (background queues a third job per turn)
        for mode, per_turn in (("sync", 2), ("background", 3)):
            wk = runs[mode][2]
            assert wk.stats["jobs"] == per_turn * len(TRANSCRIPT) \
                and wk.stats["failures"] == 0, (
                f"{mode}: stats disagree with the transcript "
                f"({wk.stats['jobs']} jobs)")
            assert wk.stats["busy_s"] > 0, f"{mode}: busy_s never accumulated"
            wk.close()
        print("bookkeeping: pending self-heals; jobs/failures/busy_s exact")

        # assert 9: atexit safety net, proven in a subprocess
        marker = tmp / "atexit_marker.txt"
        code = (
            "import sys, time\n"
            f"sys.path.insert(0, {str(REPO)!r})\n"
            "from salt.chat.ingest import IngestWorker\n"
            "w = IngestWorker()\n"
            "w.submit(lambda: (time.sleep(0.3), "
            f"open({str(marker)!r}, 'w').write('drained')))\n"
            "sys.exit(0)  # no close(): the atexit hook must drain\n")
        subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
        assert marker.exists() and marker.read_text() == "drained", (
            "an exit that bypassed close() hard-killed the worker mid-job "
            "- the atexit safety net is not draining")
        print("atexit: a close()-less exit still drained the queue")
        print("PASS")
    finally:
        if args.keep:
            print(f"session dirs kept under {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
