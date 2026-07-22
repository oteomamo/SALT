# -*- coding: utf-8 -*-
"""Regression harness for the bounded-session mask (--max-sentences).

Replays one scripted transcript through SessionTrie three ways — cap off,
cap wide enough never to bite, and a cap that bites — plus an attachment
larger than the cap. The mask retires the oldest conversation rows from
selection without deleting anything, so the checks are mostly about what
must NOT move: row indices, stored text, attachment rows, and the ledger
references pointing at them. The CLI flag plumbing lives in
salt/chat/cli.py, outside this harness. Asserts:

  1. Off-path identity: a cap wider than the corpus reproduces the
     cap-off run exactly — texts, roles, turns, sources, word counts,
     keyword weights, embeddings, selections and persisted coverage.
     Nothing about the mask perturbs a session that never reaches it.
  2. Ingest is untouched by the cap: the capped run stores the SAME rows
     as the uncapped one. The cap decides what memory shows, never what
     the session recorded.
  3. Oldest-first, conversation-only: living conversation rows equal the
     cap exactly, the dead ones are precisely the oldest, and the count
     add_turn reports as `masked` sums to the rows actually retired.
  4. Attachments are never masked, even when the attachment alone is
     larger than the cap.
  5. Indices never renumber: every row of the capped run — masked rows
     included — still holds the uncapped run's text, role, turn, source,
     word count and embedding at the SAME index. This is the contract
     kvtrace and the persisted coverage keys both depend on.
  6. Selection sees living rows only, comes back in document order, and
     its sent_idx set is non-contiguous once rows are masked (CELF
     orders records positionally, so the gaps have to be harmless).
  7. Ledger references survive masking: a turn recorded through KVTrace
     BEFORE the cap bit still resolves every selected_sent_idx to the
     same text afterwards.
  8. The budget base follows the mask: live_words equals the living rows'
     word sum, drops when rows are masked, and word_budget is that base
     times the requested fraction.
  9. Persistence: the mask survives a save/load round trip, and a
     pre-feature state.pkl with no mask at all backfills every row alive.
 10. Load repair truncates the mask WITH the other corpus columns: a
     session whose embeddings.npy is short reloads with alive the same
     length as texts. An untruncated mask would silently misalign every
     later row for the rest of the session's life.
 11. The near-duplicate gate compares against living rows only, so a
     restatement of a masked sentence is still ingestible — masking
     bounds what memory shows, it must not become a way to lose content
     the user supplied again.
 12. A masked sentence re-sent WORD FOR WORD, with the near-duplicate
     gate off (the default path), ingests again and lands in a living
     row. Masking withdraws the row's verbatim-dedupe hash, so the
     always-on dedupe no longer drops the re-send against a dead row.

Needs the BGE encoder (downloaded to the HF cache on first use). CPU is
the default device; the run takes well under a minute. The checks are
assert statements: the harness refuses to run under `python -O`.

Usage:
    python scripts/chat_evict_regression.py [--device cpu] [--cap 4]
"""

import argparse
import pickle
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

if not __debug__:
    sys.exit("this harness is assert-based - run it without python -O")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salt.chat.kvtrace import KVTrace
from salt.engine.compressor import load_bge
from salt.engine.session_trie import SessionTrie

BGE_MODEL = "BAAI/bge-small-en-v1.5"
BUDGET = 0.5

TRANSCRIPT = [
    ("How does the inverter handle a battery brownout on the solar array?",
     "The inverter drops to a reduced duty cycle and the battery bank "
     "carries the load until the panels recover their rated output."),
    ("What about the kilowatt ceiling on the grid tie during peak hours?",
     "The grid tie clamps export at the utility kilowatt ceiling, so "
     "surplus generation is diverted into the battery bank instead."),
    ("Can the piezometer network share the same telemetry backhaul?",
     "The piezometer network multiplexes onto the same backhaul radio, "
     "though the borehole sensors need their own polling schedule."),
    ("Does stratigraphy logging change the recharge estimate much?",
     "Stratigraphy logging refines the aquifer porosity term, which "
     "moves the recharge estimate by a few percent either way."),
    ("Which lysimeter stations reported turbidity outside the range?",
     "Two lysimeter stations flagged turbidity spikes after the storm, "
     "both of them downstream of the sediment plume front."),
    ("How often does the anemometer mast cluster need recalibration?",
     "The northern mast cluster drifts about two percent a season, so "
     "it is recalibrated whenever the humidity alarms trip twice."),
]

DOC_NAME = "survey.txt"
DOC_TEXT = (
    "The survey archived turbidity and hydraulic conductivity readings for "
    "every basin grid cell in the district. "
    "Lysimeter stations reported sediment plume drift across the whole "
    "recharge zone during the wet season. "
    "Borehole logs were re-cut against the revised stratigraphy datum "
    "before the porosity term was fitted. "
    "Telemetry gaps longer than one hour were backfilled from the "
    "secondary radio path and flagged in the record. "
    "Grid cells along the northern boundary were resampled twice because "
    "the first pass predated the datum revision. "
    "Every archived reading carries the calibration serial of the probe "
    "that produced it, so a drifting instrument can be traced.")

# a restatement of the FIRST user question, for the near-dup interaction
RESTATED = ("How does the inverter deal with a battery brownout on the "
            "solar array?")


def build(cache_dir, cid, cap, tok, mdl, device, with_doc=True):
    """One run of the transcript. Returns (trie, per-call add_turn records,
    per-turn selections)."""
    trie = SessionTrie(cid, cache_dir=cache_dir, model_name=BGE_MODEL,
                       budget_pct_default=BUDGET)
    if with_doc:
        trie.add_turn(DOC_TEXT, role="doc", source=DOC_NAME, tokenizer=tok,
                      model=mdl, device=device)
    records, selections = [], []
    for user, assistant in TRANSCRIPT:
        comp = trie.compress(query=user, budget_pct=BUDGET, tokenizer=tok,
                             model=mdl, device=device)
        selections.append(list(comp["selected_sent_idx"]))
        records.append(trie.add_turn(user, role="user", tokenizer=tok,
                                     model=mdl, device=device,
                                     max_sentences=cap))
        records.append(trie.add_turn(assistant, role="assistant",
                                     tokenizer=tok, model=mdl,
                                     device=device, max_sentences=cap))
    return trie, records, selections


def corpus_of(trie):
    return (trie.texts, trie.roles, trie.turns, trie.sources, trie.n_words,
            trie.keyword_weights)


def conv_rows(trie):
    return [i for i in range(trie.n_sentences) if trie.sources[i] is None]


def doc_rows(trie):
    return [i for i in range(trie.n_sentences) if trie.sources[i] is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    cap = args.cap

    print(f"Loading BGE encoder {BGE_MODEL} on {args.device} ...")
    tok, mdl = load_bge(BGE_MODEL, args.device)
    tmp = Path(tempfile.mkdtemp(prefix="salt_evict_"))
    try:
        off, off_recs, off_sel = build(tmp, "off", None, tok, mdl, args.device)
        wide, _, wide_sel = build(tmp, "wide", 10_000, tok, mdl, args.device)
        on, on_recs, on_sel = build(tmp, "on", cap, tok, mdl, args.device)

        # 1. a cap that never bites is the cap-off run, exactly
        assert corpus_of(off) == corpus_of(wide), (
            "a cap wider than the corpus perturbed the stored corpus")
        assert np.array_equal(off.embeddings, wide.embeddings), (
            "a cap wider than the corpus perturbed the embeddings")
        assert off_sel == wide_sel, "a cap that never bites changed selection"
        assert off.coverage == wide.coverage, (
            "a cap that never bites changed the persisted coverage")
        assert wide.n_masked == 0 and off.n_masked == 0
        print(f"off-path identity: cap 10000 reproduced the uncapped run "
              f"over {off.n_sentences} rows and {len(off_sel)} turns")

        # 2. the cap changes memory, never the record
        assert on.texts == off.texts, "the cap changed what was ingested"
        assert on.roles == off.roles and on.turns == off.turns
        assert on.sources == off.sources and on.n_words == off.n_words
        assert on.n_sentences == off.n_sentences
        print(f"ingest untouched: capped run stored the same "
              f"{on.n_sentences} rows as the uncapped run")

        # 3. oldest-first, conversation-only
        conv = conv_rows(on)
        live_conv = [i for i in conv if on.alive[i]]
        dead_conv = [i for i in conv if not on.alive[i]]
        assert len(live_conv) == cap, (
            f"{len(live_conv)} living conversation rows, expected {cap}")
        assert dead_conv == conv[:len(dead_conv)], (
            "masking was not oldest-first")
        assert live_conv == conv[-cap:], "the survivors are not the newest"
        assert sum(r["masked"] for r in on_recs) == len(dead_conv), (
            "add_turn's reported masked counts do not sum to the rows masked")
        print(f"cap {cap}: masked the {len(dead_conv)} oldest conversation "
              f"rows {dead_conv}, kept {live_conv}")

        # 4. attachments never masked, even when larger than the cap
        docs = doc_rows(on)
        assert len(docs) > cap, (
            "fixture too small: the attachment must exceed the cap")
        assert all(on.alive[i] for i in docs), "an attachment row was masked"
        print(f"attachment exempt: {len(docs)} rows from {DOC_NAME!r} all "
              f"alive under a cap of {cap}")

        # 5. indices never renumber
        for i in range(on.n_sentences):
            assert on.texts[i] == off.texts[i], f"row {i} text moved"
            assert on.roles[i] == off.roles[i], f"row {i} role moved"
            assert on.turns[i] == off.turns[i], f"row {i} turn moved"
            assert on.sources[i] == off.sources[i], f"row {i} source moved"
            assert on.n_words[i] == off.n_words[i], f"row {i} words moved"
            assert np.array_equal(on.embeddings[i], off.embeddings[i]), (
                f"row {i} embedding moved")
        print(f"indices stable: all {on.n_sentences} rows, masked included, "
              f"still hold the uncapped run's data at the same index")

        # 6. selection sees living rows only, in order, with gaps
        probe = on.compress(query=TRANSCRIPT[0][0], budget_pct=BUDGET,
                            tokenizer=tok, model=mdl, device=args.device)
        sel = probe["selected_sent_idx"]
        assert sel, "the probe selected nothing"
        assert not (set(sel) & set(dead_conv)), (
            f"a masked row was selected: {sorted(set(sel) & set(dead_conv))}")
        assert all(on.alive[i] for i in sel), "a dead row was selected"
        assert sel == sorted(sel), "selection lost document order"
        assert probe["n_total_sentences"] == on.n_sentences, (
            "the reported corpus size followed the mask")
        gapped = any(b - a > 1 for a, b in zip(sel, sel[1:]))
        assert gapped, "fixture too small: selection never skips a row"
        print(f"selection: {sel} - living rows only, ascending, "
              f"non-contiguous")

        # 7. ledger references recorded before the cap bit still resolve
        kv = KVTrace(off.cache_dir, "off")
        early = off_sel[1] or off_sel[0]
        assert early, "no early selection to record"
        kv.record_turn(tokenizer=tok, trie=off, selected_idx=early,
                       reply_text="recorded before the cap bit",
                       model_id="test", ts_start="t0", ts_end="t1")
        recorded = kv.last_event["selected_sent_idx"]
        for i in recorded:
            assert i < on.n_sentences, f"ledger index {i} is out of range"
            assert on.texts[i] == off.texts[i], (
                f"ledger index {i} no longer resolves to its own text")
        print(f"ledger stable: {len(recorded)} recorded indices {recorded} "
              f"still resolve after {len(dead_conv)} rows were masked")

        # 8. the budget base follows the mask
        assert off.live_words == sum(off.n_words), (
            "live_words diverged with nothing masked")
        expect_live = sum(w for w, a in zip(on.n_words, on.alive) if a)
        assert on.live_words == expect_live, "live_words ignored the mask"
        assert on.live_words < off.live_words, (
            "masking did not shrink the budget base")
        assert probe["stats"]["word_budget"] == int(on.live_words * BUDGET), (
            "the word budget was not taken from the living rows")
        print(f"budget base: {off.live_words} words uncapped -> "
              f"{on.live_words} living, word_budget "
              f"{probe['stats']['word_budget']} at {BUDGET:.0%}")

        # 9. the mask persists, and a pre-feature state backfills alive
        on.save()
        reloaded = SessionTrie("on", cache_dir=tmp, model_name=BGE_MODEL)
        assert reloaded.is_loaded and reloaded.alive == on.alive, (
            "the mask did not survive a save/load round trip")
        assert reloaded.n_masked == on.n_masked
        sp = reloaded.cache_dir / "state.pkl"
        state = pickle.loads(sp.read_bytes())
        state.pop("alive")
        sp.write_bytes(pickle.dumps(state))
        legacy = SessionTrie("on", cache_dir=tmp, model_name=BGE_MODEL)
        assert legacy.is_loaded and len(legacy.alive) == legacy.n_sentences
        assert all(legacy.alive) and legacy.n_masked == 0, (
            "a pre-feature state.pkl did not backfill every row alive")
        print(f"persistence: mask round-tripped ({on.n_masked} masked), "
              f"pre-feature state backfilled {legacy.n_sentences} rows alive")

        # 10. load repair truncates the mask with the other columns
        rep, _, _ = build(tmp, "repair", cap, tok, mdl, args.device)
        rep.save()
        keep_rows = rep.n_sentences - 2
        ep = rep.cache_dir / "embeddings.npy"
        np.save(ep, np.load(ep)[:keep_rows])
        repaired = SessionTrie("repair", cache_dir=tmp, model_name=BGE_MODEL)
        assert repaired.load_repair is not None, "no repair was recorded"
        assert repaired.n_sentences == keep_rows, "texts were not truncated"
        assert len(repaired.alive) == keep_rows, (
            f"the mask kept {len(repaired.alive)} entries for "
            f"{keep_rows} rows - it was left out of the truncation")
        assert len(repaired.alive) == len(repaired.roles) == len(
            repaired.timestamps) == len(repaired.keyword_weights), (
            "corpus columns disagree on length after repair")
        print(f"load repair: {rep.n_sentences} rows truncated to "
              f"{keep_rows}, mask truncated with them")

        # 11. the near-dup gate compares against living rows only
        gated, _, _ = build(tmp, "gated", cap, tok, mdl, args.device)
        first_row = conv_rows(gated)[0]
        assert not gated.alive[first_row], (
            "fixture: the first conversation row should be masked by now")
        before = gated.n_sentences
        rec = gated.add_turn(RESTATED, role="user", tokenizer=tok, model=mdl,
                             device=args.device, dedup_cos=0.92,
                             max_sentences=cap)
        assert rec["added"] == 1 and rec["near_dups"] == 0, (
            "a restatement of a MASKED sentence was suppressed by the "
            "near-dup gate - masked content would be unrecoverable")
        assert gated.n_sentences == before + 1
        print(f"near-dup scope: restatement of masked row {first_row} "
              f"ingested (gate saw living rows only)")

        # 12. a masked sentence re-sent VERBATIM, near-dup gate OFF (the
        # default path). The verbatim dedupe used to keep the masked row's
        # hash, so the re-send returned added 0 and the text lived in no
        # living row. It must ingest again and land alive.
        revb, _, _ = build(tmp, "revb", cap, tok, mdl, args.device)
        masked_row = conv_rows(revb)[0]
        assert not revb.alive[masked_row], (
            "fixture: the first conversation row should be masked by now")
        masked_text = revb.texts[masked_row]
        assert not any(revb.texts[i] == masked_text and revb.alive[i]
                       for i in range(revb.n_sentences)), (
            "fixture: the masked text should live in no living row yet")
        before = revb.n_sentences
        rec = revb.add_turn(masked_text, role=revb.roles[masked_row],
                            tokenizer=tok, model=mdl, device=args.device,
                            max_sentences=cap)
        assert rec["added"] == 1 and rec["filtered"] == 0, (
            "a verbatim re-send of a masked sentence was dropped by the "
            "stale dedupe hash - the text would live in no living row")
        assert revb.n_sentences == before + 1
        assert any(revb.texts[i] == masked_text and revb.alive[i]
                   for i in range(revb.n_sentences)), (
            "the re-sent masked sentence did not land in a living row")
        print(f"verbatim re-send: masked row {masked_row} re-ingested with "
              f"the near-dup gate off (stale hash discarded on masking)")

        print("PASS")
    finally:
        if args.keep:
            print(f"session dirs kept under {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
