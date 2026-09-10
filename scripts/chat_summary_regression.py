# -*- coding: utf-8 -*-
"""Regression harness for the per-call theme and discount overrides of
the session trie's compressor.

A compress call may profile themes at a percentile of its own and
discount a filling branch at a lam of its own, for that call only.
Groups:

  A. IDENTITY - a call with no overrides, a call naming the session's
     own values and a call after an override give the same selection,
     the same context and the same stats; every result reports the
     percentile and the lam it used; the session's persisted values
     never move.
  B. OVERRIDES - a lower percentile admits more keywords as themes, a
     lower lam changes what the same budget selects, and a value
     outside its range is refused by name.
  C. EVERY PROFILE PATH - the override reaches the per-source profile
     and the role-weighted profile the same way it reaches the pooled
     one.

Needs the BGE encoder (downloaded to the HF cache on first use). CPU is
the default device; the run takes under a minute.

Usage:
    python scripts/chat_summary_regression.py [--device cpu]
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salt.engine.session_trie import SessionTrie

if not __debug__:
    sys.exit("run without -O: this harness is assert-based")

BGE_MODEL = "BAAI/bge-small-en-v1.5"
BUDGET = 0.2
QUERY = "Summarize what we have talked about so far."

DOC_NAME = "orchard.txt"
DOC_TEXT = (
    "The orchard holds forty apple trees and twelve pear trees on the "
    "south slope. Apple trees are pruned in late winter before the buds "
    "swell. Pear trees tolerate wetter ground than apple trees do. The "
    "orchard's drip line runs from the pond pump along the south slope. "
    "Codling moth traps hang in the apple trees from May onward. The "
    "pear harvest starts two weeks after the apple harvest. Windfall "
    "apples go to the cider press behind the barn. The barn roof was "
    "replaced the year the pond pump failed.")

EXCHANGES = [
    ("My telescope mount drifts when I track Jupiter for more than a minute.",
     "A drifting telescope mount usually needs its polar alignment redone, "
     "and Jupiter is bright enough to check the alignment on."),
    ("The sourdough starter smells like acetone by the evening.",
     "An acetone smell means the sourdough starter is hungry, so feed the "
     "starter twice a day while the kitchen stays warm."),
    ("The irrigation timer opens the drip line at dawn for twenty minutes.",
     "Twenty minutes at dawn suits a drip line in spring, and the timer can "
     "add an evening cycle once the summer heat arrives."),
    ("Can I deduct the home office on this year's taxes?",
     "The home office deduction on your taxes needs a room used only for "
     "work, measured against the whole floor area."),
    ("The kayak takes on water at the rear hatch after an hour.",
     "A kayak hatch that lets water in needs a new gasket, and the rear "
     "hatch gasket is a standard size at any paddling shop."),
    ("Jupiter's moons looked sharp last night through the telescope.",
     "Sharp moons mean the telescope's collimation is fine, so the drift is "
     "the mount and not the optics."),
    ("The sourdough loaf came out dense with a tight crumb.",
     "A dense sourdough loaf with a tight crumb was under-proofed, so give "
     "the shaped loaf another hour before the oven."),
    ("The drip line pressure dropped after I added the third bed.",
     "Three beds on one drip line drop the pressure, so split the line at "
     "the timer or fit a second valve."),
    ("Do the taxes allow the new laptop as an expense?",
     "A laptop used for work is an expense on your taxes, spread over three "
     "years or claimed at once under the small-item rule."),
    ("The kayak paddle feathering angle feels wrong in a crosswind.",
     "A smaller feathering angle helps a kayak paddle in a crosswind, and "
     "most paddles adjust at the shaft joint."),
    ("The telescope eyepiece fogs up after midnight.",
     "A fogging telescope eyepiece needs a dew heater band, or keep the "
     "eyepiece in a warm pocket between views."),
    ("The starter doubled in four hours today.",
     "Doubling in four hours means the sourdough starter is ready, so mix "
     "the dough tonight and bake tomorrow."),
    ("The irrigation pump hums but the drip line stays dry.",
     "A humming irrigation pump with a dry drip line has lost its prime, "
     "so refill the pump housing before restarting it."),
    ("My taxes were filed late last year, what is the penalty?",
     "Late taxes carry a penalty of five percent per month on the unpaid "
     "amount, capped at twenty-five percent."),
]


def build_session(name, tmp, tok, mdl, device, doc=False):
    trie = SessionTrie(name, cache_dir=tmp)
    if doc:
        trie.add_turn(DOC_TEXT, "user", tokenizer=tok, model=mdl,
                      device=device, source=DOC_NAME, save=False)
    for user, assistant in EXCHANGES:
        trie.add_turn(user, "user", tokenizer=tok, model=mdl, device=device,
                      save=False)
        trie.add_turn(assistant, "assistant", tokenizer=tok, model=mdl,
                      device=device, save=False)
    return trie


def comparable(stats):
    return {k: v for k, v in stats.items()
            if not any(t in k for t in ("time", "elapsed", "seconds"))}


def compress(trie, tok, mdl, device, **kw):
    return trie.compress(QUERY, BUDGET, tokenizer=tok, model=mdl,
                         device=device, defer_commit=True, **kw)


def check_identity(tmp, tok, mdl, device):
    trie = build_session("identity", tmp, tok, mdl, device)
    pct, lam = trie.config["theme_percentile"], trie.config["lam"]
    base = compress(trie, tok, mdl, device)
    assert base["stats"]["theme_percentile_used"] == pct, base["stats"]
    assert base["stats"]["lam_used"] == lam, base["stats"]
    named = compress(trie, tok, mdl, device, theme_percentile=pct, lam=lam)
    assert named["selected_sent_idx"] == base["selected_sent_idx"]
    assert named["context"] == base["context"]
    assert comparable(named["stats"]) == comparable(base["stats"])
    moved = compress(trie, tok, mdl, device, theme_percentile=0.4, lam=0.2)
    assert moved["stats"]["theme_percentile_used"] == 0.4
    assert moved["stats"]["lam_used"] == 0.2
    assert (trie.config["theme_percentile"], trie.config["lam"]) == (pct, lam), (
        "an override moved the session's persisted values")
    again = compress(trie, tok, mdl, device)
    assert again["selected_sent_idx"] == base["selected_sent_idx"]
    assert comparable(again["stats"]) == comparable(base["stats"])
    print("A. identity: no overrides, the session's own values named, and a "
          "call after an override select the same and report the same, "
          "every result names the percentile and lam it used, the "
          "persisted values never move")


def check_overrides(tmp, tok, mdl, device):
    trie = build_session("overrides", tmp, tok, mdl, device)
    base = compress(trie, tok, mdl, device)
    wider = compress(trie, tok, mdl, device, theme_percentile=0.4)
    assert (wider["stats"]["theme_keywords_total"]
            > base["stats"]["theme_keywords_total"]), (
        base["stats"]["theme_keywords_total"],
        wider["stats"]["theme_keywords_total"])
    assert wider["stats"]["lam_used"] == trie.config["lam"]
    flatter = compress(trie, tok, mdl, device, lam=0.2)
    assert flatter["stats"]["theme_percentile_used"] == trie.config["theme_percentile"]
    assert set(flatter["selected_sent_idx"]) != set(base["selected_sent_idx"]), (
        "a lam of 0.2 selected exactly what 0.5 selects")
    for bad in ({"lam": 1.0}, {"lam": 0.0}, {"theme_percentile": 1.0},
                {"theme_percentile": -0.1}):
        try:
            compress(trie, tok, mdl, device, **bad)
        except ValueError as exc:
            assert next(iter(bad)) in str(exc), exc
        else:
            raise AssertionError(f"{bad} was accepted")
    print("B. overrides: a lower percentile admits more themes, a lower lam "
          "changes the selection, and a value outside its range is refused "
          "by name")


def check_profile_paths(tmp, tok, mdl, device):
    trie = build_session("paths", tmp, tok, mdl, device, doc=True)
    for extra in ({"per_source_themes": True}, {"assistant_weight": 0.5},
                  {"per_source_themes": True, "assistant_weight": 0.5}):
        base = compress(trie, tok, mdl, device, **extra)
        wider = compress(trie, tok, mdl, device, theme_percentile=0.4, **extra)
        assert (wider["stats"]["theme_keywords_total"]
                >= base["stats"]["theme_keywords_total"]), (extra, base["stats"],
                                                             wider["stats"])
        assert wider["stats"]["theme_percentile_used"] == 0.4, extra
        assert base["stats"]["theme_percentile_used"] == trie.config["theme_percentile"]
    print("C. every profile path: the override reaches the per-source and the "
          "role-weighted profiles as it reaches the pooled one, and each "
          "result reports it")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    from transformers import AutoModel, AutoTokenizer
    print(f"Loading BGE encoder {BGE_MODEL} on {args.device} ...")
    tok = AutoTokenizer.from_pretrained(BGE_MODEL)
    mdl = AutoModel.from_pretrained(BGE_MODEL, output_attentions=True).eval()
    mdl.to(args.device)
    tmp = Path(tempfile.mkdtemp(prefix="salt_summary_"))
    try:
        check_identity(tmp, tok, mdl, args.device)
        check_overrides(tmp, tok, mdl, args.device)
        check_profile_paths(tmp, tok, mdl, args.device)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS")


if __name__ == "__main__":
    main()
