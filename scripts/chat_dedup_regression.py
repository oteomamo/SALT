# -*- coding: utf-8 -*-
"""Regression harness for the chat-mode near-duplicate ingest gate
(--dedup-cos).

Replays two scripted transcripts through SessionTrie with the gate off
(dedup_cos=None) and on (0.92, the saltChat suggestion): a CLEAN transcript
of distinct exchanges, and a RESTATE transcript engineered with the shapes
the gate exists for — an assistant restating one sentence of an earlier
reply, a user re-asking their own question with one word changed, a
near-copy that crosses roles, a batch-internal repeat inside one message,
and a whole message that is one big near-duplicate. compress() runs before
every exchange, mirroring saltChat's chat_turn ordering (the CLI flag
plumbing lives in salt/chat/cli.py, outside this harness). Asserts:

  1. Pair sanity: every engineered duplicate pair really embeds at or above
     the threshold and every engineered keep pair below it, so a future
     encoder change fails HERE with the cosines printed, not downstream as
     a mystery.
  2. Off-path identity on the CLEAN transcript: the FULL corpus state
     (texts, roles, turns, word counts, keyword weights, embeddings),
     selections, and persisted coverage are IDENTICAL between the
     gate-off and gate-on runs, and the gated run suppresses nothing —
     no false positives at the suggested threshold on an ordinary
     conversation, and no state perturbation on the keep path.
  3. Restatement suppression on the RESTATE transcript: the assistant's
     restated sentence and the user's re-asked question are absent from
     the gated corpus (present in the ungated one); the rest of each
     message survives. Keyword mass for the restated content is strictly
     lower in the gated run — the df-inflation motivation, measured.
  4. Role scoping: a user sentence that near-duplicates an ASSISTANT
     sentence is kept, and the harness proves the pair was over-threshold,
     so only same-role similarity can suppress.
  5. Batch keep-first: of two near-identical sentences inside one message,
     exactly the first survives.
  6. All-suppressed turn: added == 0 with the turn index still advancing
     and the embedding matrix untouched; the next compress works.
  7. Doc exemption: a document containing a NEAR-copy (not an exact copy,
     which the hash dedupe would eat before the gate) of a conversation
     sentence, plus an internal near-repeat, ingests with zero
     suppressions in full.
  8. Re-sends are re-judged, never hash-poisoned: re-sending the exact
     text of a suppressed sentence while the gate is on suppresses it
     AGAIN (counted and logged — the event stays visible), and the same
     text ingests normally on a later call with the gate off — the
     escape hatch for a deliberately repeated correction.
  9. Bookkeeping: n_near_dups equals the per-call sum and survives a
     save/load round trip; a pre-feature state.pkl backfills 0; every
     suppression is one parseable near_dups.jsonl record whose cosine is
     at or above the threshold and whose turn, role, and matched text
     point at the real kept sentence of the same role.
 10. Parallel-array integrity after partial suppression: embeddings stay
     row-aligned with the corpus (len == n_sentences, also after reload),
     and every KEPT sentence's keyword weights and embedding are
     bit-identical to the ungated run's rows for the same text — the
     "gate runs AFTER the model passes" invariant and all three
     keep_rows reindex slices are asserted, not assumed.

Needs the BGE encoder (downloaded to the HF cache on first use). CPU is
the default device; the run takes well under a minute. The checks are
assert statements: the harness refuses to run under `python -O`.

Usage:
    python scripts/chat_dedup_regression.py [--device cpu] [--threshold 0.92]
"""

import argparse
import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

if not __debug__:
    sys.exit("this harness is assert-based - run it without python -O")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salt.engine.compressor import load_bge
from salt.engine.session_trie import SessionTrie
from salt.engine.trie_core import get_bge_sentence_embeddings

BGE_MODEL = "BAAI/bge-small-en-v1.5"

CLEAN = [  # distinct exchanges: the gate must not fire once
    ("How should I plan the vegetable beds for a small garden this spring?",
     "Group the beds by watering needs and keep the tall crops on the north "
     "side so they never shade the low ones."),
    ("What soil mix works best for raised beds?",
     "A third compost, a third topsoil, and a third coarse sand drains well "
     "and feeds the plants through the season."),
    ("When do tomato seedlings go outside?",
     "Move tomatoes out two weeks after the last frost date, once night "
     "temperatures stay above ten degrees."),
    ("How often should the drip irrigation run in summer?",
     "Twenty minutes at dawn every second day is plenty for mulched beds; "
     "sandy soil may need a daily cycle."),
]

# The RESTATE transcript's engineered sentences, named so the pair-sanity
# check and the assertions reference the same strings.
A_ORIG = ("The launch window opens at dawn on Tuesday and lasts about "
          "ninety minutes for the crew capsule.")
A_RESTATE = ("Remember, the launch window opens at dawn on Tuesday and "
             "lasts roughly ninety minutes for the crew capsule.")
U_ASK = "When does the crew capsule take off this week?"
U_REASK = "When does the crew capsule take off again this week?"
U_CROSSROLE = ("You mentioned the launch window opens at dawn on Tuesday "
               "and lasts about ninety minutes for the crew capsule.")
B_FIRST = ("The booster uses nine engines for ascent and relights three "
           "of them for the landing burn.")
B_SECOND = ("The booster uses nine engines for ascent and then relights "
            "three of them for its landing burn.")
B_OTHER = "Also, tell me how the heat shield tiles are attached."

# NEAR-copy of conversation content arriving doc-side: an exact copy would
# be eaten by the hash dedupe before the gate and test nothing
DOC_NEARCOPY = ("The launch window opens at dawn on Tuesday and lasts "
                "almost ninety minutes for the crew capsule.")
DOC_NAME = "mission-notes.txt"
DOC_TEXT = (DOC_NEARCOPY + " "
            "The recovery ship holds position two hundred miles downrange "
            "of the pad. "
            "The recovery ship keeps position two hundred miles downrange "
            "of the pad. "  # doc-internal near-repeat
            "Suit checks finish before the access arm retracts.")

# (dup pair, keep pair) sanity sets: (text_a, text_b, must_reach_threshold)
PAIRS = [
    (A_ORIG, A_RESTATE, True),
    (U_ASK, U_REASK, True),
    (A_ORIG, U_CROSSROLE, True),   # over threshold — only role scoping keeps it
    (B_FIRST, B_SECOND, True),
    (U_ASK, U_CROSSROLE, False),   # the cross-role probe's same-role neighbor
    (A_ORIG, A_ORIG.replace("ninety minutes", "ninety-five minutes"), True),
    (A_ORIG, DOC_NEARCOPY, True),  # over threshold — only the doc exemption keeps it
]


def run_clean(cache_dir, cid, dedup_cos, tok, mdl, device, budget):
    trie = SessionTrie(cid, cache_dir=cache_dir, model_name=BGE_MODEL)
    per_turn = []
    for user, assistant in CLEAN:
        comp = trie.compress(query=user, budget_pct=budget, tokenizer=tok,
                             model=mdl, device=device)
        iu = trie.add_turn(user, role="user", tokenizer=tok, model=mdl,
                           device=device, dedup_cos=dedup_cos)
        ia = trie.add_turn(assistant, role="assistant", tokenizer=tok,
                           model=mdl, device=device, dedup_cos=dedup_cos)
        per_turn.append({"sel": list(comp["selected_sent_idx"]),
                         "cov": dict(trie.coverage),
                         "near_dups": iu["near_dups"] + ia["near_dups"]})
    return trie, per_turn


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threshold", type=float, default=0.92,
                    help="gate threshold for the on runs (the value the "
                         "saltChat --dedup-cos help suggests)")
    ap.add_argument("--budget", type=float, default=0.35)
    ap.add_argument("--keep", action="store_true",
                    help="keep the temp session dirs for inspection")
    args = ap.parse_args()

    print(f"Loading BGE encoder {BGE_MODEL} on {args.device} ...")
    tok, mdl = load_bge(BGE_MODEL, args.device)

    # assert 1: pair sanity — the engineered relationships must hold in the
    # encoder actually installed, with the readings printed either way
    texts = sorted({t for a, b, _ in PAIRS for t in (a, b)})
    emb = {t: np.asarray(e) for t, e in zip(
        texts, get_bge_sentence_embeddings(texts, tok, mdl, args.device))}
    print("engineered pair cosines (threshold "
          f"{args.threshold:g}; > must reach it, < must stay under):")
    for a, b, is_dup in PAIRS:
        c = float(np.dot(emb[a], emb[b]))
        print(f"  {'>' if is_dup else '<'} cos {c:.4f}  "
              f"{a[:44]!r} / {b[:44]!r}")
        if is_dup:
            assert c >= args.threshold, (
                f"engineered duplicate pair embeds at {c:.4f} < "
                f"{args.threshold} - retune the transcript or threshold")
        else:
            assert c < args.threshold, (
                f"engineered keep pair embeds at {c:.4f} >= "
                f"{args.threshold} - the false-positive check is void")

    tmp = Path(tempfile.mkdtemp(prefix="salt_dedup_regression_"))
    try:
        # assert 2: off-path identity + no false positives on CLEAN
        off_trie, off_turns = run_clean(tmp, "clean-off", None, tok, mdl,
                                        args.device, args.budget)
        on_trie, on_turns = run_clean(tmp, "clean-on", args.threshold, tok,
                                      mdl, args.device, args.budget)
        assert on_trie.n_near_dups == 0 and all(
            t["near_dups"] == 0 for t in on_turns), (
            "the gate suppressed sentences of an ordinary conversation - "
            "false positives at the suggested threshold")
        assert off_trie.texts == on_trie.texts
        assert off_trie.roles == on_trie.roles
        assert off_trie.turns == on_trie.turns
        assert off_trie.n_words == on_trie.n_words
        assert off_trie.keyword_weights == on_trie.keyword_weights, (
            "CLEAN: keyword weights diverged with zero suppressions - the "
            "keep path is perturbing state")
        assert np.array_equal(off_trie.embeddings, on_trie.embeddings), (
            "CLEAN: embeddings diverged with zero suppressions - the keep "
            "path is perturbing state")
        for i, (a, b) in enumerate(zip(off_turns, on_turns)):
            assert a["sel"] == b["sel"] and a["cov"] == b["cov"], (
                f"CLEAN exchange {i + 1}: gate-on run diverged from gate-off "
                f"without any suppression")
        print(f"CLEAN: {on_trie.n_sentences} sentences, 0 suppressed, "
              f"gate-on identical to gate-off")

        # ── RESTATE transcript, gated and ungated ─────────────────────────
        A_VARIANT = A_ORIG.replace("ninety minutes", "ninety-five minutes")
        expected = []   # (label, role, info, expected near_dups), gated run
        for label, cid, cos in (("ungated", "restate-off", None),
                                ("gated", "restate-on", args.threshold)):
            trie = SessionTrie(cid, cache_dir=tmp, model_name=BGE_MODEL)
            kw = dict(tokenizer=tok, model=mdl, device=args.device,
                      dedup_cos=cos)
            calls = []
            trie.compress(query=U_ASK, budget_pct=args.budget, tokenizer=tok,
                          model=mdl, device=args.device)
            calls.append(("e1 user ask", "user",
                          trie.add_turn(U_ASK, "user", **kw), 0))
            calls.append(("e1 assistant", "assistant",
                          trie.add_turn(A_ORIG, "assistant", **kw), 0))
            # e2: restatement rides inside a longer, otherwise-new reply
            calls.append(("e2 user", "user", trie.add_turn(
                "And what happens if the weather turns bad before liftoff?",
                "user", **kw), 0))
            calls.append(("e2 assistant restate", "assistant", trie.add_turn(
                "If the weather turns bad the launch scrubs and moves to the "
                "backup window on Thursday morning. " + A_RESTATE,
                "assistant", **kw), 1))
            # e3: the user re-asks their own question
            calls.append(("e3 user re-ask", "user",
                          trie.add_turn(U_REASK, "user", **kw), 1))
            calls.append(("e3 assistant", "assistant", trie.add_turn(
                "Correct - and the crew boards the capsule two hours before "
                "that window begins.", "assistant", **kw), 0))
            # e4: near-copy of an ASSISTANT sentence arriving as USER text
            calls.append(("e4 user cross-role", "user", trie.add_turn(
                U_CROSSROLE, "user", **kw), 0))
            # e5: batch-internal repeat inside one message
            calls.append(("e5 user batch", "user", trie.add_turn(
                f"{B_FIRST} {B_SECOND} {B_OTHER}", "user", **kw), 1))
            # e6: a whole message that is one near-duplicate sentence
            n_before = trie.n_sentences
            rows_before = 0 if trie.embeddings is None else len(trie.embeddings)
            turn_before = trie.n_turns
            calls.append(("e6 all-suppressed", "assistant",
                          trie.add_turn(A_VARIANT, "assistant", **kw), 1))
            if cos is not None:
                info = calls[-1][2]
                assert info["added"] == 0 and trie.n_sentences == n_before \
                    and len(trie.embeddings) == rows_before, (
                    "the all-suppressed turn changed the corpus")
                assert trie.n_turns == turn_before + 1, (
                    "the all-suppressed turn did not advance the turn index")
            # e7: exact re-send of e6's suppressed text - must be RE-JUDGED
            # (suppressed again, visibly), never silently hash-dropped
            calls.append(("e7 exact re-send", "assistant",
                          trie.add_turn(A_VARIANT, "assistant", **kw), 1))
            if cos is not None:
                info = calls[-1][2]
                assert info["added"] == 0 and info["near_dups"] == 1, (
                    "an exact re-send of a suppressed sentence was not "
                    "re-judged - suppressed hashes are poisoning the dedupe")
                # e8: the escape hatch - with the gate off (per-call kwarg,
                # exactly as a later flag-less launch runs), the repeated
                # correction ingests normally
                off_kw = dict(kw); off_kw["dedup_cos"] = None
                rec = trie.add_turn(A_VARIANT, "assistant", **off_kw)
                assert rec["added"] >= 1, (
                    "a suppressed sentence stayed uningestible after the "
                    "gate was turned off - no recovery path")
            # doc exemption: near-copies ingest untouched
            calls.append(("doc", "doc", trie.add_turn(
                DOC_TEXT, "doc", source=DOC_NAME, **kw), 0))
            comp = trie.compress(query="When does the capsule take off?",
                                 budget_pct=args.budget, tokenizer=tok,
                                 model=mdl, device=args.device)
            assert comp["selected_sent_idx"], "post-gate compress selected nothing"
            if cos is None:
                ungated = trie
            else:
                gated, expected = trie, calls

        # assert 3/4/5/7/9: per-call counts, suppressed content, scoping
        total = 0
        for label, role, info, want in expected:
            assert info["near_dups"] == want, (
                f"{label}: suppressed {info['near_dups']}, expected {want}")
            total += want
        assert gated.n_near_dups == total, (
            f"n_near_dups {gated.n_near_dups} != per-call sum {total}")
        conv_gated = [t for t, s in zip(gated.texts, gated.sources)
                      if s is None]
        conv_ungated = [t for t, s in zip(ungated.texts, ungated.sources)
                        if s is None]
        for t in (A_RESTATE, U_REASK, B_SECOND):
            assert t not in conv_gated, f"suppressed text in gated corpus: {t!r}"
            assert t in conv_ungated, f"{t!r} missing from the UNGATED corpus"
        for t in (U_CROSSROLE, B_FIRST, B_OTHER, U_ASK, A_ORIG):
            assert t in conv_gated, f"kept text missing from gated corpus: {t!r}"
        doc_gated = [t for t, s in zip(gated.texts, gated.sources) if s]
        doc_ungated = [t for t, s in zip(ungated.texts, ungated.sources) if s]
        assert doc_gated == doc_ungated and len(doc_gated) >= 4, (
            "doc ingest differed between runs - attachments are being gated")
        assert DOC_NEARCOPY in doc_gated, (
            "the doc-side near-copy of a conversation sentence was not "
            "ingested - the doc exemption is not holding against "
            "conversation priors")
        print(f"RESTATE: {gated.n_near_dups} suppressed "
              f"({ungated.n_sentences} sentences ungated -> "
              f"{gated.n_sentences} gated), doc branch identical")

        # assert 10: parallel arrays stay row-aligned after partial
        # suppression, and every KEPT row is bit-identical to the ungated
        # run's row for the same text — the reindex slices and the
        # gate-after-the-model-passes invariant, asserted directly
        for name, t in (("gated", gated), ("ungated", ungated)):
            assert len(t.embeddings) == t.n_sentences, (
                f"{name}: {len(t.embeddings)} embedding rows for "
                f"{t.n_sentences} sentences - parallel arrays desynced")
        row_un = {t: i for i, t in enumerate(ungated.texts)
                  if ungated.sources[i] is None}
        for i, t in enumerate(gated.texts):
            if gated.sources[i] is not None:
                continue
            j = row_un[t]
            assert gated.roles[i] == ungated.roles[j]
            assert gated.keyword_weights[i] == ungated.keyword_weights[j], (
                f"a kept sentence's keywords differ from the ungated run's "
                f"- the gate is changing surviving rows: {t!r}")
            assert np.array_equal(gated.embeddings[i],
                                  ungated.embeddings[j]), (
                f"a kept sentence's embedding differs from the ungated "
                f"run's - rows are misassigned after suppression: {t!r}")
        print(f"kept-row fidelity: {len(conv_gated)} gated conversation "
              f"rows bit-identical to their ungated counterparts")

        # assert 3 (df motivation): keyword mass for the restated content
        # is strictly lower once its duplicates are gone
        marks = ("launch", "window", "capsule")
        n_un = sum(1 for kw in ungated.keyword_weights
                   if any(m in kw for m in marks))
        n_ga = sum(1 for kw in gated.keyword_weights
                   if any(m in kw for m in marks))
        assert n_ga < n_un, (
            f"restated-content keyword rows did not drop ({n_un} -> {n_ga})")
        print(f"keyword rows carrying the restated content: {n_un} ungated "
              f"-> {n_ga} gated")

        # assert 9: log records match the suppressions one-to-one, each
        # pointing at the real kept same-role sentence it matched, stamped
        # with the suppressing call's turn — the log is the threshold-tuning
        # artifact, so record fidelity is part of the contract
        log = [json.loads(l) for l in
               (gated.cache_dir / "near_dups.jsonl").read_text().splitlines()]
        assert len(log) == gated.n_near_dups
        sup_calls = [(info["turn"], role) for label, role, info, want
                     in expected if want]
        assert len(log) == len(sup_calls)
        conv_role = {t: r for t, r, s in zip(gated.texts, gated.roles,
                                             gated.sources) if s is None}
        for rec, (turn, role) in zip(log, sup_calls):
            assert rec["cos"] >= args.threshold and rec["text"]
            assert rec["turn"] == turn and rec["role"] == role, (
                f"log record stamped ({rec['turn']}, {rec['role']}), "
                f"suppression happened at ({turn}, {role})")
            assert conv_role.get(rec["matched"]) == rec["role"], (
                f"log 'matched' is not a kept conversation sentence of the "
                f"same role - the record points at the wrong sentence: "
                f"{rec['matched']!r}")
        assert not (ungated.cache_dir / "near_dups.jsonl").exists(), (
            "the ungated run wrote a near-dup log")
        print("suppression log: " + "; ".join(
            f"t{r['turn']} cos {r['cos']:.3f} [{r['role']}]" for r in log))

        # assert 9: persistence round trip + pre-feature backfill
        reloaded = SessionTrie("restate-on", cache_dir=tmp,
                               model_name=BGE_MODEL)
        assert reloaded.is_loaded and reloaded.n_near_dups == total, (
            "n_near_dups did not survive a save/load round trip")
        assert len(reloaded.embeddings) == reloaded.n_sentences, (
            "reloaded session has misaligned embeddings on disk")
        sp = gated.cache_dir / "state.pkl"
        state = pickle.loads(sp.read_bytes())
        state.pop("n_near_dups")
        sp.write_bytes(pickle.dumps(state))
        legacy = SessionTrie("restate-on", cache_dir=tmp,
                             model_name=BGE_MODEL)
        assert legacy.is_loaded and legacy.n_near_dups == 0, (
            "a pre-feature state.pkl did not backfill n_near_dups = 0")
        print("counter round trip + pre-feature backfill OK")
        print("PASS")
    finally:
        if args.keep:
            print(f"session dirs kept under {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
