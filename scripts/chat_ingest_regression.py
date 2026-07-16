# -*- coding: utf-8 -*-
"""Regression harness for background ingestion (the IngestWorker that takes
saltChat's per-turn keyword/embedding passes off the REPL's critical path).

Replays a scripted transcript through SessionTrie three ways — direct calls
(the pre-feature code path), an inline `synchronous=True` worker
(--sync-ingest), and a real background worker — mirroring saltChat's
chat_turn ordering exactly: drain barrier, compress(), the user line
submitted BEFORE the model generates (background mode; its encode
overlaps generation, with no drain until the next turn's barrier), then
the assistant ingest queued behind it FIFO and one coalesced
conditional save job per turn (each background ingest runs save=False);
sync and direct run both sides at the pre-feature post-reply position
with the pre-feature save-per-call durability (the CLI barrier
placement lives in salt/chat/cli.py, outside this harness). The transcript includes a long
pasted message and an engineered restatement so the near-dup gate
(--dedup-cos) is exercised THROUGH the worker. Asserts:

  1. Overlap: submit returns while a gated job is still running (the
     prompt-return property) and drain blocks until it finishes; a queued
     burst executes strictly FIFO — the trie keeps a single mutator in
     submission order.
  2. Identity: after every turn's barrier, all three runs agree — corpus
     (texts, roles, turns, sources, word counts, keyword weights),
     embeddings (byte-identical), persisted coverage, drift EMA,
     selections, and the near-dup gate's decisions (same suppression
     count >= 1, same near_dups.jsonl records). Backgrounding must be
     invisible to selection: this is the feature's core invariant. The
     background run also re-reads every pre-turn row from the main
     thread WHILE the user-line job is in flight — kvtrace.record_turn's
     documented index-stable exception, pinned as an assert.
  3. Deferred failure reporting: a job that raises is captured, not
     thrown; the failure comes back at the NEXT drain with its label and
     error; later jobs still ran (the worker survived) and the queue
     stays usable.
  4. Journal recovery: each failure is one parseable ingest_failures.jsonl
     record carrying the payload text VERBATIM — an ingest error can never
     silently lose a user's words. Delivery survives an unwritable journal
     path: the drain report must not depend on the journal write, and each
     record's `journaled` field tells the truth about whether its line
     reached the file (True on the writable path, False on the unwritable
     one), so the REPL's failure report never claims preservation that
     did not happen.
  5. Sync-mode contract: `synchronous=True` runs the job inline (no
     thread, effects visible immediately); a failing job is journaled and
     then RE-RAISED at the submit site — today's control flow, so
     --sync-ingest aborts a turn exactly where the direct call did; a
     KeyboardInterrupt propagates untouched and is NOT logged as a
     failure.
  6. close(): finishes queued work first, returns undelivered failure
     records, is idempotent, and rejects later submits with RuntimeError
     — a job aimed at a torn-down session fails loudly, never silently.
  7. Counter self-heal: an interrupted submit (simulated) leaves `pending`
     nonzero with an empty queue; the next drain returns normally and
     resets it — the /stats reading cannot drift for a session's life.
  8. Bookkeeping: stats jobs/failures match ground truth per run and
     busy_s accumulates the real encode time.
  9. atexit safety net (subprocess): an exit path that never calls
     close() still drains the queue before the daemon worker dies — a
     soft crash cannot hard-kill a half-written ingest.
 10. Failed-generation survival: in chat_turn's background order the
     user line is submitted before generation, so a turn whose
     generation raises (aborting before the assistant ingest) still has
     the user's message in the trie at the next barrier — the
     lost-user-message half of the old compress-commits-early failure
     mode is closed. (--sync-ingest keeps today's abort purely by
     cli.py statement order — the post-reply submit is never reached —
     which lives outside this harness's scope, with the barrier
     placement.)
 11. Save coalescing: the background run persists once per turn (a
     queued conditional save after both save=False ingests) and its
     reloaded on-disk session is IDENTICAL to the direct run's
     per-call-saved one; the error turn of assert 10 — whose save job
     was never queued — leaves the trie dirty and its line OFF disk,
     and the boundary save (what exit and /new run) then persists it.

Needs the BGE encoder (downloaded to the HF cache on first use). CPU is
the default device; the run takes well under a minute. The checks are
assert statements: the harness refuses to run under `python -O`.

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
    """One full session in the given mode; returns (trie, per-turn log,
    worker-or-None). Mirrors chat_turn: barrier, compress, then the two
    ingests — direct calls in 'direct' mode, submitted jobs otherwise."""
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
            # chat_turn's order: the user line rides the generation
            # window (no drain in between — a drain there would put the
            # tail of a big paste's encode back on the prompt path).
            # WHILE the job is in flight, the main thread re-reads every
            # pre-turn row exactly like kvtrace.record_turn does — the
            # documented index-stable I2 exception, pinned here as an
            # assert instead of assumed (encode takes long enough on CPU
            # that these reads genuinely interleave with the append).
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

    print(f"Loading BGE encoder {BGE_MODEL} on {args.device} ...")
    tok, mdl = load_bge(BGE_MODEL, args.device)
    tmp = Path(tempfile.mkdtemp(prefix="salt_ingest_regression_"))
    try:
        # assert 2: three-way identity — direct vs sync worker vs background
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

        # assert 11 (persistence half): the coalesced background session
        # reloads identical to the per-call-saved direct session
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

        # assert 10: failed-generation survival (background order; the
        # sync abort is cli.py statement order, out of harness scope)
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
            # the abort skips the assistant submit AND the coalesced
            # save job, exactly like chat_turn
        except RuntimeError:
            pass
        assert wg.drain() == []
        assert any("maintenance window" in t for t in tg.texts), (
            "the user line was lost on a failed generation - the "
            "rides-the-generation-window fix does not hold")
        # assert 11 (error-turn half): the ingest is RAM-only and marked
        # dirty; the boundary save persists it
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

        # assert 5: sync-mode contract — inline, journaled, re-raised
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

        # assert 8: bookkeeping on the identity runs (background queues a
        # third job per turn: the coalesced session save)
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
