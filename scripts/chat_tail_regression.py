# -*- coding: utf-8 -*-
"""Regression harness for tail-aware memory selection (--tail-exclude).

Replays a fixed scripted transcript -- one attached doc, three solar
exchanges, three sourdough exchanges, a solar recap and a uniquely-worded
zeppelin exchange (the last two model the verbatim tail) -- and pins the
exclusion seam end to end. Groups:

  A. IDENTITY - the same transcript run with exclude_sent_idx=None and
     with set() is identical turn by turn (selection, coverage, stamps,
     drift EMA), under default flags AND under stable_keys + decay.
     This is the byte-identity gate for the default path.
  B. FILE TOKENS - excluding early conversation rows never disturbs the
     per-file branch: doc sentences still select for a doc question and
     the incremented coverage keys still carry the file token (pins the
     index-safe per-file injection loop).
  C. EXCLUSION HONORED - selection never returns an excluded index.
  D. BUDGET REALLOCATION - the word budget stat is unchanged and the
     freed words are respent on other rows, not surrendered.
  E. SPREAD - with the tail rows excluded, the selection covers strictly
     more distinct conversation turns.
  F. NO PHANTOM STAMPS - a theme carried only by excluded rows gains no
     coverage on the excluding turn (non-vacuous: the same query without
     exclusion does increment it).
  G. STARVATION GUARD - an exclusion covering every candidate is ignored
     and selection matches the unexcluded run exactly.
  H. KEY SURVIVAL - under stable_keys, a coverage key whose only living
     carriers are excluded survives the orphan reconcile untouched (the
     commit-universe union).
  I. HELPER - tail_resident_sent_idx unit checks with no model: word
     boundaries ("yes" never matches inside "yesterday"), attachment and
     masked-row immunity, the all-living guard, cross-message immunity.

Needs the BGE encoder (downloaded to the HF cache on first use). CPU is
the default device; the run takes about a minute.

Usage:
    python scripts/chat_tail_regression.py [--device cpu] [--budget 0.2]
"""

import argparse
import shutil
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salt.chat.cli import tail_resident_sent_idx
from salt.engine.compressor import load_bge
from salt.engine.session_trie import FILE_TOKEN_PREFIX, SessionTrie

BGE_MODEL = "BAAI/bge-small-en-v1.5"

DOC_NAME = "irrigation-notes.txt"
DOC_TEXT = (
    "The garden irrigation system uses a drip line on each vegetable bed. "
    "A timer valve opens the drip line for twenty minutes at dawn. "
    "Rain sensors pause the irrigation schedule after heavy rainfall. "
    "The pump pressure for the drip system stays near two bar.")

EXCHANGES = [
    ("I want to plan a rooftop solar installation for my house. The roof "
     "faces south and gets sun most of the day.",
     "A south facing roof suits solar panels well. Size the array around "
     "your daily kilowatt usage and pick an inverter to match it."),
    ("How many solar panels would I need for thirty kilowatt hours per day?",
     "Around twenty panels at typical output. The array size also decides "
     "what inverter capacity the installation needs."),
    ("Should I pick a string inverter or microinverters for the panels?",
     "A string inverter is cheaper when the whole roof gets even sun. "
     "Microinverters only pay off under partial shade."),
    ("Different topic: I want to learn baking sourdough bread at home.",
     "Sourdough starts with a starter of flour and water fermented until "
     "it rises predictably. Feed the starter daily for about a week."),
    ("What hydration should my first sourdough dough be?",
     "Start around seventy percent hydration so the dough stays "
     "manageable. Higher hydration gives an open crumb but sticky dough."),
    ("How long should bulk fermentation take for the dough?",
     "Bulk fermentation runs four to six hours at room temperature. The "
     "dough should grow by half and hold air bubbles."),
    # the two exchanges below model the verbatim tail
    ("Back to the solar project, what did we settle on for panels and "
     "the inverter?",
     "The plan is twenty panels with a string inverter near eight "
     "kilowatts and a ten kilowatt hour battery for backup."),
    ("One more thing, the zeppelin mooring mast needs a hydrogen manifold "
     "gasket replaced.",
     "A zeppelin mooring gasket should be nitrile and the hydrogen "
     "manifold bolts torque to twelve newton meters."),
]
N_TAIL_EXCHANGES = 2

PROBE = "Remind me what we decided about the solar panels and the inverter."
DOC_PROBE = "What happens to the irrigation schedule after heavy rainfall?"
ZEP_PROBE = "What torque do the zeppelin hydrogen manifold bolts need?"
ZEP_WORDS = {"zeppelin", "mooring", "manifold", "gasket", "nitrile"}


def build_session(cache_dir, cid, tok, mdl, device, budget, exclude_arm,
                  stable=False, half_life=None):
    """Replay the transcript, compressing before each exchange the way
    chat_turn does. Returns (trie, per-turn trace, tail turn ids)."""
    trie = SessionTrie(cid, cache_dir=cache_dir, model_name=BGE_MODEL)
    trie.add_turn(DOC_TEXT, role="doc", tokenizer=tok, model=mdl,
                  device=device, source=DOC_NAME)
    trace = []
    tail_turns = set()
    for xi, (user, assistant) in enumerate(EXCHANGES):
        comp = trie.compress(query=user, budget_pct=budget, tokenizer=tok,
                             model=mdl, device=device,
                             stable_keys=stable,
                             coverage_half_life=half_life,
                             exclude_sent_idx=exclude_arm)
        trace.append({"selected": list(comp["selected_sent_idx"]),
                      "coverage": dict(trie.coverage),
                      "stamps": dict(trie.coverage_turn),
                      "drift_ema": trie.drift_ema,
                      "excluded_sent": comp["stats"].get("excluded_sent")})
        iu = trie.add_turn(user, role="user", tokenizer=tok, model=mdl,
                           device=device)
        ia = trie.add_turn(assistant, role="assistant", tokenizer=tok,
                           model=mdl, device=device)
        if xi >= len(EXCHANGES) - N_TAIL_EXCHANGES:
            tail_turns |= {iu["turn"], ia["turn"]}
    return trie, trace, tail_turns


def clone_session(tmp, base_cid, new_cid):
    shutil.copytree(tmp / base_cid, tmp / new_cid)
    return SessionTrie(new_cid, cache_dir=tmp, model_name=BGE_MODEL)


def model_tail(tail_turns, trie):
    """The tail as chat messages: the last exchanges' original text."""
    msgs = []
    for user, assistant in EXCHANGES[-N_TAIL_EXCHANGES:]:
        msgs.append({"role": "user", "content": user})
        msgs.append({"role": "assistant", "content": assistant})
    expected = {i for i in range(trie.n_sentences)
                if trie.alive[i] and trie.sources[i] is None
                and trie.turns[i] in tail_turns}
    return msgs, expected


def changed_keys(before, after):
    keys = set(before) | set(after)
    return {k for k in keys
            if abs(after.get(k, 0.0) - before.get(k, 0.0)) > 1e-9}


def helper_unit_checks():
    ns = types.SimpleNamespace
    # word boundaries: "yes" not in "yesterday", "ok" not in "token"
    t = ns(n_sentences=3, alive=[True] * 3, sources=[None] * 3,
           texts=["yes", "OK", "use port 8080"], n_alive=3)
    got = tail_resident_sent_idx(
        t, [{"role": "user", "content": "I looked at it yesterday"},
            {"role": "assistant",
             "content": "The token broker can reuse port 8080."}])
    assert got == set(), f"helper: boundary over-match {got}"
    # true positives at message start and mid-message, case-insensitive
    t = ns(n_sentences=3, alive=[True] * 3, sources=[None] * 3,
           texts=["yes", "Use port 8080.", "an older unrelated sentence"],
           n_alive=3)
    got = tail_resident_sent_idx(
        t, [{"role": "user", "content": "YES"},
            {"role": "assistant", "content": "Fine, use port 8080. Done."}])
    assert got == {0, 1}, f"helper: expected {{0, 1}}, got {got}"
    # attachment rows are immune, masked rows are skipped
    t = ns(n_sentences=2, alive=[True, False], sources=["doc.txt", None],
           texts=["exact tail text here", "exact tail text here"], n_alive=1)
    got = tail_resident_sent_idx(
        t, [{"role": "user", "content": "exact tail text here"}])
    assert got == set(), f"helper: immunity violated {got}"
    # guard: every living row tail-resident means no exclusion
    t = ns(n_sentences=2, alive=[True] * 2, sources=[None] * 2,
           texts=["alpha beta gamma", "delta epsilon zeta"], n_alive=2)
    got = tail_resident_sent_idx(
        t, [{"role": "user", "content": "alpha beta gamma"},
            {"role": "assistant", "content": "delta epsilon zeta"}])
    assert got == set(), f"helper: guard did not fire {got}"
    # a needle never matches across two messages
    t = ns(n_sentences=2, alive=[True] * 2, sources=[None] * 2,
           texts=["gamma delta", "another sentence kept"], n_alive=2)
    got = tail_resident_sent_idx(
        t, [{"role": "user", "content": "alpha beta gamma"},
            {"role": "assistant", "content": "delta epsilon"}])
    assert got == set(), f"helper: cross-message match {got}"
    assert tail_resident_sent_idx(t, []) == set(), "helper: empty tail"
    print("helper: boundary, immunity, guard and span checks pass")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--budget", type=float, default=0.20)
    ap.add_argument("--keep", action="store_true",
                    help="keep the temp session dirs for inspection")
    args = ap.parse_args()

    helper_unit_checks()

    print(f"Loading BGE encoder {BGE_MODEL} on {args.device} ...")
    tok, mdl = load_bge(BGE_MODEL, args.device)
    tmp = Path(tempfile.mkdtemp(prefix="salt_tail_regression_"))
    try:
        # --- group A: identity, default flags and stable+decay ---
        for label, stable, hl in (("default", False, None),
                                  ("stable+decay", True, 2.0)):
            t_none, tr_none, tail_turns = build_session(
                tmp, f"none_{stable}", tok, mdl, args.device, args.budget,
                None, stable, hl)
            t_empty, tr_empty, _ = build_session(
                tmp, f"empty_{stable}", tok, mdl, args.device, args.budget,
                set(), stable, hl)
            assert t_none.texts == t_empty.texts, "corpora diverged"
            for i, (a, b) in enumerate(zip(tr_none, tr_empty)):
                assert a == b, (
                    f"identity[{label}] exchange {i + 1}: None vs set() "
                    f"diverged")
            print(f"identity[{label}]: {len(tr_none)} turns byte-identical "
                  f"across None and set()")

        base_cid = "none_False"
        base = SessionTrie(base_cid, cache_dir=tmp, model_name=BGE_MODEL)
        tail_msgs, expected = model_tail(tail_turns, base)
        excl = tail_resident_sent_idx(base, tail_msgs)
        assert excl == expected, (
            f"helper vs transcript: expected rows {sorted(expected)}, "
            f"got {sorted(excl)}")
        assert excl, "no tail-resident rows found - every group is vacuous"

        # non-vacuity for F/H: the zeppelin vocabulary must be themes
        df = base._live_kw_df()
        zep_present = {w for w in ZEP_WORDS if w in df}
        themes = base._themes_from_df(df)[1]
        zep_themes = zep_present & themes
        assert zep_themes, (
            f"zeppelin vocabulary minted no theme (present: {zep_present}) "
            f"- groups F and H would be vacuous (encoder change?)")

        # --- probe pair: plain vs excluded from identical state ---
        t_plain = clone_session(tmp, base_cid, "probe_plain")
        t_excl = clone_session(tmp, base_cid, "probe_excl")
        c_plain = t_plain.compress(query=PROBE, budget_pct=args.budget,
                                   tokenizer=tok, model=mdl,
                                   device=args.device)
        before = dict(t_excl.coverage)
        c_excl = t_excl.compress(query=PROBE, budget_pct=args.budget,
                                 tokenizer=tok, model=mdl,
                                 device=args.device, exclude_sent_idx=excl)
        sel_p, sel_e = set(c_plain["selected_sent_idx"]), set(
            c_excl["selected_sent_idx"])
        tail_hits = sel_p & excl
        assert tail_hits, (
            "the plain run selected no tail-resident row - the transcript "
            "no longer reproduces the double-exposure and groups C/D/E "
            "are vacuous")
        # C: exclusion honored
        assert not (sel_e & excl), f"excluded rows selected: {sel_e & excl}"
        assert c_excl["stats"]["excluded_sent"] == len(excl)
        # D: same budget, freed words respent
        assert (c_plain["stats"]["word_budget"]
                == c_excl["stats"]["word_budget"]), "word budget moved"
        max_row = max(w for w, a in zip(t_excl.n_words, t_excl.alive) if a)
        used_p = c_plain["stats"]["words_used"]
        used_e = c_excl["stats"]["words_used"]
        assert used_e >= used_p - max_row, (
            f"budget surrendered, not respent: {used_p} -> {used_e} "
            f"(max row {max_row})")
        # E: spread
        turns_p = {t_plain.turns[i] for i in sel_p
                   if t_plain.sources[i] is None}
        turns_e = {t_excl.turns[i] for i in sel_e
                   if t_excl.sources[i] is None}
        assert len(turns_e) > len(turns_p), (
            f"no spread gain: {len(turns_p)} -> {len(turns_e)} turns")
        print(f"probe: {len(tail_hits)} tail rows re-shown by the plain "
              f"run, 0 by the excluded run; words {used_p} -> {used_e}; "
              f"conversation turns {len(turns_p)} -> {len(turns_e)}")
        # F: no phantom stamps for tail-only themes
        zep_changed = {k for k in changed_keys(before, t_excl.coverage)
                       if set(k) & ZEP_WORDS}
        assert not zep_changed, (
            f"tail-only theme keys moved on the excluding turn: "
            f"{[sorted(k) for k in zep_changed]}")
        t_zep = clone_session(tmp, base_cid, "zep_prime")
        zb = dict(t_zep.coverage)
        t_zep.compress(query=ZEP_PROBE, budget_pct=args.budget,
                       tokenizer=tok, model=mdl, device=args.device)
        zep_moved = {k for k in changed_keys(zb, t_zep.coverage)
                     if set(k) & ZEP_WORDS}
        assert zep_moved, (
            "the zeppelin query incremented no zeppelin key even without "
            "exclusion - group F is vacuous")
        print(f"phantom stamps: 0 zeppelin keys moved under exclusion, "
              f"{len(zep_moved)} moved without it")

        # --- group G: total exclusion is ignored ---
        t_ga = clone_session(tmp, base_cid, "guard_all")
        t_gb = clone_session(tmp, base_cid, "guard_ref")
        all_rows = {i for i in range(t_ga.n_sentences) if t_ga.alive[i]}
        c_ga = t_ga.compress(query=PROBE, budget_pct=args.budget,
                             tokenizer=tok, model=mdl, device=args.device,
                             exclude_sent_idx=all_rows)
        c_gb = t_gb.compress(query=PROBE, budget_pct=args.budget,
                             tokenizer=tok, model=mdl, device=args.device)
        assert c_ga["selected_sent_idx"] == c_gb["selected_sent_idx"], (
            "total exclusion changed selection instead of being ignored")
        assert c_ga["stats"]["excluded_sent"] == 0
        assert t_ga.coverage == t_gb.coverage, "guard turn coverage moved"
        print("starvation guard: total exclusion ignored, selection intact")

        # --- group B: file branch undisturbed by exclusion ---
        t_doc = clone_session(tmp, base_cid, "doc_excl")
        early = {i for i in range(t_doc.n_sentences)
                 if t_doc.alive[i] and t_doc.sources[i] is None
                 and t_doc.turns[i] <= 6}
        db = dict(t_doc.coverage)
        c_doc = t_doc.compress(query=DOC_PROBE, budget_pct=args.budget,
                               tokenizer=tok, model=mdl,
                               device=args.device, exclude_sent_idx=early)
        doc_sel = [i for i in c_doc["selected_sent_idx"]
                   if t_doc.sources[i] is not None]
        assert doc_sel, "doc question selected no doc sentence"
        assert not (set(c_doc["selected_sent_idx"]) & early)
        doc_keys = {k for k in changed_keys(db, t_doc.coverage)
                    if any(w.startswith(FILE_TOKEN_PREFIX) for w in k)}
        assert doc_keys, (
            "no file-token coverage key moved - the per-file branch lost "
            "its token under exclusion")
        print(f"file branch: {len(doc_sel)} doc sentences selected, "
              f"{len(doc_keys)} file-token keys advanced under exclusion")

        # --- group H: stable_keys key survival (the universe union) ---
        stable_cid = "none_True"
        t_h = clone_session(tmp, stable_cid, "stable_survive")
        t_h.compress(query=ZEP_PROBE, budget_pct=args.budget,
                     tokenizer=tok, model=mdl, device=args.device,
                     stable_keys=True)
        zep_keys = {k for k in t_h.coverage if set(k) & ZEP_WORDS}
        assert zep_keys, "priming turn persisted no zeppelin key"
        held = {k: t_h.coverage[k] for k in zep_keys}
        c_h = t_h.compress(query=PROBE, budget_pct=args.budget,
                           tokenizer=tok, model=mdl, device=args.device,
                           stable_keys=True, exclude_sent_idx=excl)
        for k, v in held.items():
            assert k in t_h.coverage, (
                f"stable_keys reconcile collected a tail-carried key: "
                f"{sorted(k)}")
            assert abs(t_h.coverage[k] - v) < 1e-9, (
                f"tail-carried key changed while excluded: {sorted(k)}")
        assert c_h["stats"]["coverage_orphans_dropped"] == 0 or all(
            k in t_h.coverage for k in held), "orphan drop hit held keys"
        print(f"key survival: {len(held)} tail-carried keys intact "
              f"through the stable_keys reconcile")
        print("PASS")
    finally:
        if args.keep:
            print(f"session dirs kept under {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
