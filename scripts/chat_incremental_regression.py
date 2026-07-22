# -*- coding: utf-8 -*-
"""Regression harness for the carried-forward per-turn caches.

Replays one scripted transcript through SessionTrie twice — once normally,
and once with both caches dropped before every compress so the session
re-derives everything the way it used to. The two runs must agree turn for
turn: these caches exist to remove repeated work, so any difference in
what memory returns is a bug by definition. The rest of the checks are
about the caches being MAINTAINED rather than quietly rebuilt on every
read, which is the failure that costs nothing but the whole point. The CLI
plumbing lives in salt/chat/cli.py, outside this harness. Asserts:

  1. Off-path identity: dropping both caches before every compress
     reproduces the carried run exactly — per-turn context, stats,
     selected rows, and the persisted coverage at the end.
  2. The carried keyword count equals a full profile_themes recount over
     the living corpus after every ingest, every masking and every
     reload — the mapping and the theme set, tuple for tuple.
  3. It is maintained, not rebuilt on read: after each ingest and each
     masking the count already describes the corpus BEFORE any accessor
     runs. A missing maintenance site otherwise hides behind the rebuild.
  4. Masking removes keys instead of zeroing them: no count survives at
     zero, and a keyword only masked rows carried is gone entirely. The
     theme cutoff is an index into the count of keys, so a zero left
     behind would move it.
  5. The `>=` cutoff boundary matches profile_themes: keywords whose
     document frequency is exactly the threshold are admitted.
  6. A reload in place drops the carried count rather than trusting it.
     Row counts alone cannot catch a corpus that changed underneath a
     session, which is why the count is dropped rather than checked.
  7. The profile seam answers for the records it is handed: a filtered
     list gets a profile of that list, not of whatever the session
     happens to hold, and per-source profiling keeps its own full
     recompute either way.
  8. The carried count is never what a caller mutates: the compress path
     injects the §file: tokens into what it gets back, and none of them
     reach the count.
  9. Every living sentence's cached lexical tokens equal a fresh
     expansion, and a corpus assembled field by field — no ingest, so no
     cache was ever filled — compresses to the identical result. The
     cache is derived state, so a miss must cost time and nothing else.
 10. A living corpus that mints no keyword at all (table rows carry no
     content words) profiles to an empty pair and still compresses.
 11. Resume identity: a session saved and reopened selects exactly what
     the session that never left memory selects.

Needs the BGE encoder (downloaded to the HF cache on first use). CPU is
the default device; the run takes well under a minute. The checks are
assert statements: the harness refuses to run under `python -O`.

Usage:
    python scripts/chat_incremental_regression.py [--device cpu]
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

if not __debug__:
    sys.exit("this harness is assert-based - run it without python -O")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salt.engine.compressor import load_bge
from salt.engine.session_trie import SessionTrie, FILE_TOKEN_PREFIX
from salt.engine.trie_core import (profile_themes, clean_text_words,
                                   expand_with_stems)

BGE_MODEL = "BAAI/bge-small-en-v1.5"
BUDGET = 0.4
CAP = 5

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
    ("Which lysimeter stations reported turbidity outside the range?",
     "Two lysimeter stations flagged turbidity spikes after the storm, "
     "both of them downstream of the sediment plume front."),
    ("Does stratigraphy logging change the recharge estimate much?",
     "Stratigraphy logging refines the aquifer porosity term, which "
     "moves the recharge estimate by a few percent either way."),
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
    "secondary radio path and flagged in the record.")

# rows whose words are no content words, so they mint no keywords at all
TABLE_ROWS = ["| 12 | 34 | 56 |", "| 78 | 90 | 11 |"]

COLUMNS = ("texts", "roles", "turns", "sources", "timestamps", "n_words",
           "keyword_weights", "alive")
CARRIED = ("coverage", "coverage_turn", "kw_order", "theme_admitted")


def recount(trie):
    """Keyword document frequency over the living rows, the long way."""
    df = {}
    for kw, alive in zip(trie.keyword_weights, trie.alive):
        if alive:
            for k in kw:
                df[k] = df.get(k, 0) + 1
    return df


def drop_caches(trie):
    trie._lex = {}
    trie._kw_df = trie._kw_df_rows = None


def check_profile(trie, where):
    """The two assertions that have to hold at every single step."""
    if trie._kw_df is not None:
        assert trie._kw_df_rows == (trie.n_sentences, trie.n_alive), (
            f"{where}: the count describes {trie._kw_df_rows} rows but the "
            f"corpus has {(trie.n_sentences, trie.n_alive)} - a maintenance "
            f"site is missing and the rebuild is hiding it")
        assert trie._kw_df == recount(trie), (
            f"{where}: the carried count is stale before anything read it")
    sd = trie._sent_data()
    want = profile_themes(sd, theme_percentile=trie.config["theme_percentile"])
    assert trie._profile(sd, per_source=False) == want, (
        f"{where}: the carried profile diverged from profile_themes")


def build(cache_dir, cid, tok, mdl, device, cold=False, cap=CAP):
    """One run of the transcript. `cold` drops both caches before every
    compress, which is the pre-cache behavior. Returns (trie, turns)."""
    trie = SessionTrie(cid, cache_dir=cache_dir, model_name=BGE_MODEL,
                       budget_pct_default=BUDGET)
    trie.add_turn(DOC_TEXT, role="doc", source=DOC_NAME, tokenizer=tok,
                  model=mdl, device=device)
    check_profile(trie, f"{cid}: after the attachment")
    turns = []
    for n, (user, assistant) in enumerate(TRANSCRIPT):
        if cold:
            drop_caches(trie)
        out = trie.compress(query=user, budget_pct=BUDGET, tokenizer=tok,
                            model=mdl, device=device)
        turns.append((out["context"], out["stats"],
                      list(out["selected_sent_idx"])))
        trie.add_turn(user, role="user", tokenizer=tok, model=mdl,
                      device=device, max_sentences=cap)
        check_profile(trie, f"{cid}: turn {n} user")
        trie.add_turn(assistant, role="assistant", tokenizer=tok, model=mdl,
                      device=device, max_sentences=cap)
        check_profile(trie, f"{cid}: turn {n} assistant")
    return trie, turns


def mirror(trie, cid, cache_dir):
    """The same corpus in a session that never ran ingest, so nothing ever
    filled a cache for it."""
    hand = SessionTrie(cid, cache_dir=cache_dir, model_name=BGE_MODEL,
                       budget_pct_default=BUDGET)
    for attr in COLUMNS + CARRIED:
        setattr(hand, attr, getattr(trie, attr))
    hand.embeddings = trie.embeddings
    hand.dim = trie.dim
    hand._n_compress = trie._n_compress
    hand._next_sentence_index = trie._next_sentence_index
    hand._next_turn_index = trie._next_turn_index
    return hand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    print(f"Loading BGE encoder {BGE_MODEL} on {args.device} ...")
    tok, mdl = load_bge(BGE_MODEL, args.device)
    tmp = Path(tempfile.mkdtemp(prefix="salt_incremental_"))
    try:
        warm, warm_turns = build(tmp, "warm", tok, mdl, args.device)
        cold, cold_turns = build(tmp, "cold", tok, mdl, args.device, cold=True)

        # 1. re-deriving everything every turn changes nothing
        assert len(warm_turns) == len(cold_turns) == len(TRANSCRIPT)
        for n, (a, b) in enumerate(zip(warm_turns, cold_turns)):
            assert a[0] == b[0], f"turn {n}: the compressed context moved"
            assert a[1] == b[1], f"turn {n}: the stats moved"
            assert a[2] == b[2], f"turn {n}: the selected rows moved"
        assert warm.coverage == cold.coverage, "the persisted coverage moved"
        assert warm.texts == cold.texts and warm.alive == cold.alive
        selected = sum(len(t[2]) for t in warm_turns)
        print(f"off-path identity: {len(warm_turns)} turns, {selected} "
              f"selected rows and the coverage dict all reproduced with "
              f"both caches dropped before every compress")

        # 2 and 3 ran after every ingest inside build(); the mask path too
        assert warm.n_masked > 0, (
            "fixture too small: no row was ever masked, so the count's "
            "decrement path was never exercised")
        print(f"maintained through {warm.n_sentences} rows and "
              f"{warm.n_masked} maskings: the count matched a full recount "
              f"at every step, before any accessor could rebuild it")

        # 4. masking removes keys, it does not zero them
        df = warm._live_kw_df()
        assert all(v > 0 for v in df.values()), "a count survived at zero"
        every_row = {}
        for kw in warm.keyword_weights:
            for k in kw:
                every_row[k] = every_row.get(k, 0) + 1
        gone = set(every_row) - set(df)
        assert gone, ("fixture too small: masking retired no keyword, so "
                      "delete-at-zero is untested here")
        assert all(k not in df for k in gone)
        print(f"delete at zero: {len(gone)} keywords of {len(every_row)} "
              f"left with the masked rows, none of them left behind at 0")

        # 5. the cutoff boundary is >=, exactly as profile_themes has it
        kw_df, themes = warm._profile(warm._sent_data(), per_source=False)
        values = sorted(kw_df.values())
        idx = int(len(values) * warm.config["theme_percentile"])
        threshold = values[min(idx, len(values) - 1)]
        on_edge = [k for k, v in kw_df.items() if v == threshold]
        assert on_edge, ("fixture: no keyword sits exactly on the cutoff, "
                         "so the boundary is untested")
        assert all(k in themes for k in on_edge), (
            f"keywords at the cutoff df {threshold} were not admitted")
        print(f"cutoff: df {threshold} at the {warm.config['theme_percentile']}"
              f" percentile, {len(on_edge)} keywords on the edge, all admitted")

        # 6. a reload in place cannot trust the count it already had
        warm.save()
        stolen = dict(warm._live_kw_df())
        other = SessionTrie("other", cache_dir=tmp, model_name=BGE_MODEL,
                            budget_pct_default=BUDGET)
        other.add_turn(DOC_TEXT, role="doc", source=DOC_NAME, tokenizer=tok,
                       model=mdl, device=args.device)
        other.save()
        warm.cache_dir = other.cache_dir            # point it at another corpus
        warm.load()
        assert warm._kw_df is None, (
            "the carried count survived a reload onto another corpus")
        assert warm._live_kw_df() == recount(warm) != stolen, (
            "the reloaded session profiled the corpus it used to hold")
        print(f"reload in place: dropped a count of {len(stolen)} keywords "
              f"and recounted {len(warm._live_kw_df())} for the new corpus")

        # 7. the seam profiles what it is given, per-source included
        sd = cold._sent_data()
        subset = [r for r in sd if r["sent_idx"] % 2 == 0]
        assert 0 < len(subset) < len(sd), "fixture: the subset is the corpus"
        assert cold._profile(subset, per_source=False) == profile_themes(
            subset, theme_percentile=cold.config["theme_percentile"]), (
            "the carried count answered for a corpus it was not handed")
        per_warm = cold._profile(sd, per_source=True)
        drop_caches(cold)
        assert cold._profile(sd, per_source=True) == per_warm, (
            "per-source profiling changed with the count dropped")
        print(f"scope: a {len(subset)} of {len(sd)} row subset profiled as "
              f"itself, per-source {len(per_warm[1])} theme keywords "
              f"identical with the count warm and dropped")

        # 8. the count is not the dict callers mutate
        live = cold._live_kw_df()
        assert not any(k.startswith(FILE_TOKEN_PREFIX) for k in live), (
            "a §file: token injected at compress time reached the count")
        handed, _ = cold._profile(cold._sent_data(), per_source=False)
        assert handed is not cold._kw_df, "the count itself was handed out"
        handed["totally-not-a-keyword"] = 99
        assert "totally-not-a-keyword" not in cold._live_kw_df()
        print(f"ownership: {len(live)} keywords carried, no §file: token "
              f"among them, and a caller's edit did not reach them")

        # 9. tokens equal a fresh expansion, and a hand-built corpus works
        for i in range(warm.n_sentences):
            if warm.alive[i]:
                text = warm.texts[i]
                assert warm._lex_tokens(text) == expand_with_stems(
                    clean_text_words(text)), (
                    f"row {i}: the cached tokens are not what the selector "
                    f"would have derived")
        hand = mirror(cold, "hand", tmp)
        assert not hand._lex, "the mirrored session started with a warm cache"
        q = TRANSCRIPT[0][0]
        a = cold.compress(query=q, budget_pct=BUDGET, tokenizer=tok,
                          model=mdl, device=args.device)
        b = hand.compress(query=q, budget_pct=BUDGET, tokenizer=tok,
                          model=mdl, device=args.device)
        assert a["context"] == b["context"] and a["stats"] == b["stats"], (
            "a corpus assembled field by field compressed differently - a "
            "cache miss must cost time, never accuracy")
        print(f"derived only: {cold.n_alive} living rows matched a fresh "
              f"expansion, and a corpus that never ran ingest compressed "
              f"to the same {len(b['selected_sent_idx'])} rows")

        # 10. a corpus that mints no keywords at all
        bare = SessionTrie("bare", cache_dir=tmp, model_name=BGE_MODEL,
                           budget_pct_default=BUDGET)
        bare.add_turn("\n".join(TABLE_ROWS), role="user", tokenizer=tok,
                      model=mdl, device=args.device)
        assert bare.n_alive > 0, (
            "fixture: the table rows were filtered out, so the "
            "no-keyword corpus never existed")
        assert not any(bare.keyword_weights), (
            "fixture: the table rows minted keywords after all")
        assert bare._profile(bare._sent_data(), per_source=False) == ({}, set())
        out = bare.compress(query="12", budget_pct=1.0, tokenizer=tok,
                            model=mdl, device=args.device)
        print(f"no keywords: {bare.n_alive} table rows profiled empty and "
              f"compressed to {len(out['selected_sent_idx'])} rows")

        # 11. a resumed session selects what the live one selects
        warm2, _ = build(tmp, "resume", tok, mdl, args.device)
        warm2.save()
        resumed = SessionTrie("resume", cache_dir=tmp, model_name=BGE_MODEL,
                              budget_pct_default=BUDGET)
        assert resumed.is_loaded and resumed._kw_df is None
        live_out = warm2.compress(query=q, budget_pct=BUDGET, tokenizer=tok,
                                  model=mdl, device=args.device)
        back_out = resumed.compress(query=q, budget_pct=BUDGET, tokenizer=tok,
                                    model=mdl, device=args.device)
        assert live_out["context"] == back_out["context"], (
            "a resumed session compressed differently from the live one")
        assert live_out["stats"] == back_out["stats"]
        assert np.array_equal(warm2.embeddings, resumed.embeddings)
        print(f"resume: reopened session selected the same "
              f"{len(back_out['selected_sent_idx'])} rows as the live one")

        print("PASS")
    finally:
        if args.keep:
            print(f"session dirs kept under {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
