# -*- coding: utf-8 -*-
"""Regression harness for chat-mode coverage decay (--coverage-half-life).

Replays a fixed scripted transcript -- one attached doc, 5 exchanges on an
early topic (solar), 8 exchanges on a second topic (sourdough), then a probe
question returning to the early topic -- through two SessionTries: decay off
(legacy accumulate-forever coverage) and decay on. compress() runs before
every exchange, mirroring saltChat's chat_turn ordering (the CLI flag
plumbing itself lives in salt/chat/cli.py, outside this harness). Asserts:

  1. Across the detour and the probe combined, the decay run surfaces
     strictly more early-topic conversation sentences than the legacy run,
     and the probe itself still surfaces some. Cumulative, not probe-only:
     decay makes the old topic resurface DURING the detour too (the whole
     point), and each resurfacing re-increments its coverage -- so at any
     single turn the two runs can tie or flip. The default half-life of 2
     is deliberately aggressive: on this 14-exchange transcript it keeps
     the margin wide; gentler values
     (e.g. 8, the suggested saltChat feel) shrink it toward one sentence.
  2. The decayed run carries strictly less total conversation-coverage
     suppression than the legacy run (the mechanism itself).
  3. After every compress, persisted coverage == decayed-base + non-negative
     integer increments (catches the permanent-scaling bug class: a scaled
     seed written back wholesale).
  4. With doc decay off (default), no '§file:' coverage key ever loses mass.
  5. Silent turns at an aggressive half-life garbage-collect long-unselected
     conversation keys via the floor (the dict stays bounded), while
     doc-branch keys survive untouched.
  6. The same silent turns with coverage_decay_docs=True drain the
     doc-branch keys too (the opt-in path actually decays).
  7. _profile(per_source=False) equals a direct profile_themes call
     exactly, tuple for tuple - the flag-off path is the frozen path.
  8. Under a dominating attachment, per-source profiling readmits at
     least one conversation-only keyword the pooled profile evicted,
     and at least one conversation sentence with an empty pooled theme
     intersection gains a non-empty one (path mass restored).
  9. Flag-off drift guard: the legacy run's persisted coverage dict
     matches the recorded key count and total mass exactly (an encoder
     or default change trips here before it ships silently).
 10. Orphan GC (coverage_gc=True): orphaned keys drain to zero after
     the grace window, stamps never outnumber keys, live attachment
     keys survive while attachment orphans are collected.

Needs the BGE encoder (downloaded to the HF cache on first use). CPU is the
default device; the run takes well under a minute.

Usage:
    python scripts/chat_theme_regression.py [--device cpu] [--half-life 2]
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salt.engine.compressor import load_bge
from salt.engine.session_trie import (SessionTrie, FILE_TOKEN_PREFIX,
                                      COVERAGE_DECAY_FLOOR)
from salt.engine.trie_core import profile_themes

BGE_MODEL = "BAAI/bge-small-en-v1.5"

DOC_NAME = "irrigation-notes.txt"
DOC_TEXT = (
    "The garden irrigation system uses a drip line on each vegetable bed. "
    "A timer valve opens the drip line for twenty minutes at dawn. "
    "Rain sensors pause the irrigation schedule after heavy rainfall. "
    "The pump pressure for the drip system stays near two bar.")

TOPIC_X = [  # solar -- the early topic the probe returns to
    ("I want to plan a rooftop solar installation for my house. The roof "
     "faces south and gets sun most of the day.",
     "A south-facing roof is ideal for solar panels. For a typical home you "
     "start by sizing the array around your daily kilowatt usage, then pick "
     "an inverter to match."),
    ("How many solar panels would I need for about 30 kilowatt hours per day?",
     "At roughly 1.6 kilowatt hours per panel per day you would need around "
     "18 to 20 panels. The array size also decides what inverter capacity "
     "the installation needs."),
    ("Should the inverter be a string inverter or microinverters on each panel?",
     "Microinverters cost more but tolerate shade on individual panels. A "
     "string inverter is cheaper and fine when the whole roof array gets "
     "even sun."),
    ("What about a battery so the solar system keeps working during outages?",
     "A battery around 10 kilowatt hours covers an evening of household "
     "load. The battery must be compatible with the inverter, so pick the "
     "pair together."),
    ("Let's decide: string inverter with a 10 kilowatt hour battery, and "
     "about 20 panels.",
     "Good decision. So the plan is a 20 panel rooftop array, one string "
     "inverter sized near 8 kilowatts, and a 10 kilowatt hour battery for "
     "backup."),
]

TOPIC_Y = [  # sourdough -- the long detour that buries topic X
    ("Different topic: I want to learn baking sourdough bread at home.",
     "Sourdough starts with a starter - flour and water fermented until it "
     "rises predictably. Feed the starter daily and it will be ready in "
     "about a week."),
    ("How much flour and water do I feed the starter each day?",
     "A common feeding is equal weights of flour and water, say 50 grams of "
     "each. Discard half the starter before feeding so the jar never "
     "overflows."),
    ("What hydration should my first sourdough dough be?",
     "Start around 70 percent hydration so the dough stays manageable. "
     "Higher hydration gives a more open crumb but a stickier dough."),
    ("How long should bulk fermentation take for the dough?",
     "Bulk fermentation usually runs 4 to 6 hours at room temperature. The "
     "dough should grow by half and hold air bubbles."),
    ("When do I shape the dough, and what is scoring the loaf for?",
     "Shape after bulk fermentation, then proof in a basket. Scoring the "
     "loaf lets steam escape so the bread expands evenly in the oven."),
    ("What oven temperature bakes the best sourdough crust?",
     "Bake at 230 Celsius in a covered dutch oven, then uncover for the "
     "last 15 minutes. Steam early on is what makes the crust crackle."),
    ("My crumb came out dense - what did I do wrong with the dough?",
     "A dense crumb usually means underproofed dough or a weak starter. "
     "Make sure the starter doubles within 6 hours of feeding before you "
     "bake."),
    ("Can I keep the starter in the fridge between bakes?",
     "Yes, a fridge slows the starter down so weekly feeding is enough. "
     "Refresh it at room temperature a day before baking bread."),
]

# Deliberately vague: a keyword-rich probe ("what inverter did we pick?")
# resurfaces topic X through the query channels, which are never seeded with
# coverage -- both runs then look alike and the comparison is noise. The
# decay feature targets the DOCUMENT channel, where legacy suppression
# persists; a vague callback is exactly the turn it exists for.
PROBE = ("Let's get back to the earlier home project we discussed - can "
         "you summarize what we decided?")


def manual_decay(cov, half_life, decay_docs):
    """Independent reimplementation of the engine's decay step, used to
    verify what compress() persists (decayed base + increments only)."""
    if not half_life:
        return dict(cov)
    f = 0.5 ** (1.0 / half_life)
    out = {}
    for k, v in cov.items():
        if not decay_docs and any(t.startswith(FILE_TOKEN_PREFIX) for t in k):
            out[k] = v
            continue
        nv = v * f
        if nv >= COVERAGE_DECAY_FLOOR:
            out[k] = nv
    return out


def check_persisted_coverage(before, after, half_life, decay_docs, where):
    """Assert 3 + 4: persisted == decayed-base + non-negative ~integer
    increments, and (with doc decay off) doc keys never lose mass."""
    base = manual_decay(before, half_life, decay_docs)
    for k, v in base.items():
        assert k in after, f"{where}: key vanished after selection: {set(k)}"
        inc = after[k] - v
        assert inc > -1e-9, (
            f"{where}: persisted count below decayed base (scaled seed "
            f"written back?): {set(k)}: {after[k]} < {v}")
        assert abs(inc - round(inc)) < 1e-6, (
            f"{where}: non-integer increment {inc} on {set(k)}")
    for k, v in after.items():
        if k not in base:
            assert v > 0, f"{where}: fresh key with non-positive count {v}"
    if not decay_docs:
        for k, v in after.items():
            if any(t.startswith(FILE_TOKEN_PREFIX) for t in k):
                assert v >= before.get(k, 0.0) - 1e-9, (
                    f"{where}: doc-branch key lost mass with doc decay "
                    f"off: {set(k)}: {before.get(k)} -> {v}")


def run_session(cache_dir, cid, half_life, decay_docs, tok, mdl, device,
                budget):
    trie = SessionTrie(cid, cache_dir=cache_dir, model_name=BGE_MODEL)
    trie.add_turn(DOC_TEXT, role="doc", tokenizer=tok, model=mdl,
                  device=device, source=DOC_NAME)

    x_turns = set()
    x_per_exchange = []
    for xi, (user, assistant) in enumerate(TOPIC_X + TOPIC_Y):
        where = f"[hl={half_life}] exchange {xi + 1}"
        before = dict(trie.coverage)
        comp = trie.compress(query=user, budget_pct=budget, tokenizer=tok,
                             model=mdl, device=device,
                             coverage_half_life=half_life,
                             coverage_decay_docs=decay_docs)
        check_persisted_coverage(before, trie.coverage, half_life,
                                 decay_docs, where)
        x_per_exchange.append(
            sum(1 for i in comp["selected_sent_idx"]
                if trie.sources[i] is None and trie.turns[i] in x_turns))
        iu = trie.add_turn(user, role="user", tokenizer=tok, model=mdl,
                           device=device)
        ia = trie.add_turn(assistant, role="assistant", tokenizer=tok,
                           model=mdl, device=device)
        if xi < len(TOPIC_X):
            x_turns |= {iu["turn"], ia["turn"]}

    before = dict(trie.coverage)
    comp = trie.compress(query=PROBE, budget_pct=budget, tokenizer=tok,
                         model=mdl, device=device,
                         coverage_half_life=half_life,
                         coverage_decay_docs=decay_docs)
    check_persisted_coverage(before, trie.coverage, half_life, decay_docs,
                             f"[hl={half_life}] probe")
    x_selected = sum(1 for i in comp["selected_sent_idx"]
                     if trie.sources[i] is None and trie.turns[i] in x_turns)
    x_per_exchange.append(x_selected)
    return {"trie": trie, "x_selected": x_selected,
            "x_per_exchange": x_per_exchange,
            "coverage": dict(trie.coverage), "stats": comp["stats"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--half-life", type=float, default=2.0,
                    help="half-life for the behavioral comparison; "
                         "aggressive by default so the cumulative-exposure "
                         "margin stays wide on a 14-exchange transcript")
    ap.add_argument("--budget", type=float, default=0.20)
    ap.add_argument("--keep", action="store_true",
                    help="keep the temp session dirs for inspection")
    args = ap.parse_args()

    print(f"Loading BGE encoder {BGE_MODEL} on {args.device} ...")
    tok, mdl = load_bge(BGE_MODEL, args.device)

    tmp = Path(tempfile.mkdtemp(prefix="salt_decay_regression_"))
    try:
        legacy = run_session(tmp, "legacy", None, False, tok, mdl,
                             args.device, args.budget)
        decayed = run_session(tmp, "decayed", args.half_life, False, tok, mdl,
                              args.device, args.budget)

        assert legacy["trie"].texts == decayed["trie"].texts, (
            "corpora diverged between runs - the comparison is meaningless")

        print(f"early-topic sentences selected per exchange "
              f"(last = probe):\n  legacy  {legacy['x_per_exchange']}"
              f"\n  decayed {decayed['x_per_exchange']}")
        lx, dx = legacy["x_selected"], decayed["x_selected"]
        lw = sum(legacy["x_per_exchange"][len(TOPIC_X):])
        dw = sum(decayed["x_per_exchange"][len(TOPIC_X):])
        print(f"early-topic exposure detour..probe - legacy {lw}, "
              f"decayed {dw}  (probe alone: {lx} vs {dx})")
        assert dw > lw, (
            f"decay run surfaced no more early-topic content over the "
            f"detour + probe ({dw}) than legacy ({lw})")
        assert dx >= 1, "probe surfaced no early-topic sentences at all"

        def conv_mass(cov):
            return sum(v for k, v in cov.items()
                       if not any(t.startswith(FILE_TOKEN_PREFIX) for t in k))

        lm, dm = conv_mass(legacy["coverage"]), conv_mass(decayed["coverage"])
        print(f"final conversation-coverage mass - "
              f"legacy {lm:.1f}, decayed {dm:.1f}")
        assert dm < lm, (
            f"decayed run does not carry less suppression ({dm} >= {lm}) - "
            f"decay had no effect")

        s = decayed["stats"]
        assert s.get("coverage_half_life") == args.half_life
        assert s.get("coverage_keys") == len(decayed["coverage"])

        # Floor GC: run the decayed trie through "silent" turns (empty
        # query, near-zero budget so selection adds ~no increments) at an
        # aggressive half-life. Conversation keys must decay below the
        # floor and drop out; exempt doc-branch keys must all survive.
        trie = decayed["trie"]

        def split_keys(cov):
            docs = {k for k in cov
                    if any(t.startswith(FILE_TOKEN_PREFIX) for t in k)}
            return docs, set(cov) - docs

        docs0, conv0 = split_keys(trie.coverage)
        assert docs0, ("no doc-branch coverage key exists - the transcript "
                       "no longer selects any doc sentence, so the doc-"
                       "exemption assertions would all pass vacuously")
        for _ in range(30):
            before = dict(trie.coverage)
            trie.compress(query="", budget_pct=0.01, tokenizer=tok,
                          model=mdl, device=args.device,
                          coverage_half_life=2.0)
            check_persisted_coverage(before, trie.coverage, 2.0, False,
                                     "[floor-gc]")
        docs1, conv1 = split_keys(trie.coverage)
        print(f"floor GC after 30 silent turns at half-life 2: "
              f"conversation keys {len(conv0)} -> {len(conv1)}, "
              f"doc keys {len(docs0)} -> {len(docs1)}")
        assert len(conv1) < len(conv0), (
            "floor never garbage-collected any conversation key - the "
            "coverage dict is not bounded")
        assert docs1 >= docs0, "doc-branch keys were garbage-collected"

        # The opt-in: the same silent loop with coverage_decay_docs=True
        # must now drain the doc-branch keys as well.
        for _ in range(30):
            before = dict(trie.coverage)
            trie.compress(query="", budget_pct=0.01, tokenizer=tok,
                          model=mdl, device=args.device,
                          coverage_half_life=2.0, coverage_decay_docs=True)
            check_persisted_coverage(before, trie.coverage, 2.0, True,
                                     "[doc-decay]")
        docs2, _ = split_keys(trie.coverage)
        print(f"doc decay opt-in: doc keys {len(docs1)} -> {len(docs2)}")
        assert len(docs2) < len(docs1), (
            "coverage_decay_docs=True never decayed a doc-branch key")
        print("per-turn persisted-coverage invariant held on all "
              f"{len(TOPIC_X) + len(TOPIC_Y) + 1} compress calls of both runs")

        # group 7: flag-off _profile is the frozen path, exactly
        t7 = SessionTrie("scope7", cache_dir=tmp, model_name=BGE_MODEL)
        for u, a in TOPIC_X:
            t7.add_turn(u, "user", tokenizer=tok, model=mdl,
                        device=args.device)
            t7.add_turn(a, "assistant", tokenizer=tok, model=mdl,
                        device=args.device)
        t7.add_turn(DOC_TEXT, role="doc", source=DOC_NAME, tokenizer=tok,
                    model=mdl, device=args.device)
        sd7 = t7._sent_data()
        assert t7._profile(sd7, per_source=False) == profile_themes(
            sd7, theme_percentile=t7.config["theme_percentile"]), (
            "the flag-off _profile path diverged from profile_themes")
        print("theme scope 7: flag-off profile identical to the pooled call")

        # group 8: a dominating attachment evicts conversation themes from
        # the pooled profile; the per-source profile readmits them
        vocab = ["aquifer", "porosity", "recharge", "plume", "sediment",
                 "piezometer", "borehole", "stratigraphy", "turbidity",
                 "hydraulic", "conductivity", "lysimeter"]
        big_doc = " ".join(
            "The %s and %s field measurements near the %s and %s station "
            "were archived for basin grid cell %d yesterday." % (
                vocab[i % 12], vocab[(i + 1) % 12], vocab[(i + 2) % 12],
                vocab[(i + 3) % 12], i)
            for i in range(60))
        t8 = SessionTrie("scope8", cache_dir=tmp, model_name=BGE_MODEL)
        for u, a in TOPIC_X:
            t8.add_turn(u, "user", tokenizer=tok, model=mdl,
                        device=args.device)
            t8.add_turn(a, "assistant", tokenizer=tok, model=mdl,
                        device=args.device)
        t8.add_turn(big_doc, role="doc", source="survey.txt", tokenizer=tok,
                    model=mdl, device=args.device)
        sd8 = t8._sent_data()
        pct = t8.config["theme_percentile"]
        gdf, gthemes = profile_themes(sd8, theme_percentile=pct)
        sdf, sthemes = t8._profile(sd8, per_source=True)
        conv_kws, doc_kws = set(), set()
        for i, src in enumerate(t8.sources):
            (conv_kws if src is None else doc_kws).update(
                t8.keyword_weights[i])
        recovered = ((conv_kws - doc_kws) & sthemes) - gthemes
        print(f"theme scope 8: pooled evicted "
              f"{len((conv_kws - doc_kws) & sthemes)} conversation theme "
              f"keywords, per-source readmits e.g. "
              f"{sorted(recovered)[:4]}")
        assert recovered, (
            "per-source profiling readmitted no conversation-only keyword "
            "the pooled profile evicted - the fix is inert on this corpus")
        regained = 0
        for i, src in enumerate(t8.sources):
            if src is None:
                kws = set(t8.keyword_weights[i])
                if not (kws & gthemes) and (kws & sthemes):
                    regained += 1
        assert regained >= 1, (
            "no conversation sentence went from an empty pooled theme "
            "intersection to a non-empty per-source one - no path mass "
            "was restored")
        print(f"theme scope 8: {regained} conversation sentences regained "
              f"path mass under per-source profiling")

        # group 9: flag-off drift guard - the legacy run's persisted
        # dict is pinned (an encoder or default change trips here)
        LEGACY_KEYS, LEGACY_MASS = 32, 98.0
        lc = legacy["coverage"]
        assert (len(lc), round(sum(lc.values()), 1)) == (LEGACY_KEYS,
                                                         LEGACY_MASS), (
            f"flag-off coverage drifted: {len(lc)} keys, mass "
            f"{round(sum(lc.values()), 1)} (pinned {LEGACY_KEYS}/"
            f"{LEGACY_MASS}; encoder change?)")
        print(f"coverage pin: legacy dict still {LEGACY_KEYS} keys / "
              f"mass {LEGACY_MASS}")

        # group 10: orphan GC under coverage_gc=True
        from salt.engine.session_trie import COVERAGE_GC_GRACE
        tg = SessionTrie("covgc", cache_dir=tmp, model_name=BGE_MODEL)
        tg.add_turn(DOC_TEXT, role="doc", tokenizer=tok, model=mdl,
                    device=args.device, source=DOC_NAME)
        big2 = " ".join(
            f"Attachment beta clause {i} covers invoice retention audit "
            f"sampling and the ledger reconciliation window for "
            f"account {i}." for i in range(25))
        orphan_seen, gc_total = 0, 0
        for xi, (user, assistant) in enumerate(TOPIC_X + TOPIC_Y):
            if xi == 5:
                tg.add_turn(big2, role="doc", source="beta.txt",
                            tokenizer=tok, model=mdl, device=args.device)
            s9 = tg.compress(query=user, budget_pct=args.budget,
                             tokenizer=tok, model=mdl, device=args.device,
                             coverage_gc=True)["stats"]
            assert len(tg.coverage_turn) <= len(tg.coverage), (
                f"gc exchange {xi + 1}: stamps outnumber keys")
            orphan_seen = max(orphan_seen, s9["coverage_persisted_orphans"])
            gc_total += s9["coverage_gc_dropped"]
            tg.add_turn(user, "user", tokenizer=tok, model=mdl,
                        device=args.device)
            tg.add_turn(assistant, "assistant", tokenizer=tok, model=mdl,
                        device=args.device)
        assert orphan_seen > 0, (
            "the gc transcript minted no orphans - the collection "
            "assertions below are vacuous (encoder change?)")
        for _ in range(COVERAGE_GC_GRACE + 2):
            s9 = tg.compress(query="", budget_pct=args.budget,
                             tokenizer=tok, model=mdl, device=args.device,
                             coverage_gc=True)["stats"]
            gc_total += s9["coverage_gc_dropped"]
        assert s9["coverage_persisted_orphans"] == 0, (
            f"orphans survived {COVERAGE_GC_GRACE + 2} turns of gc: "
            f"{s9['coverage_persisted_orphans']}")
        assert gc_total > 0, "gc never dropped anything"
        doc_alive = sum(1 for k in tg.coverage
                        if any(t.startswith(FILE_TOKEN_PREFIX) for t in k))
        assert doc_alive > 0, (
            "gc collected live attachment keys - the doc exemption for "
            "live branches is broken")
        print(f"coverage gc: peak {orphan_seen} orphans drained to 0 "
              f"({gc_total} collected), {doc_alive} live attachment keys "
              f"kept")
        print("PASS")
    finally:
        if args.keep:
            print(f"session dirs kept under {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
