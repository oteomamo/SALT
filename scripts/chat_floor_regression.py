# -*- coding: utf-8 -*-
"""Regression harness for the memory floor.

Pins the least a selection reads. The budget is a fraction of the living
words, so a young session selects through a block of a few lines; the
floor raises that budget. Groups:

  A. FLOOR OFF - compress with no floor and with memory_floor=None give
     the same selection, the same context and the same stats, and the
     stats say the budget was not raised.
  B. RAISED - a budget fraction smaller than the floor is raised to the
     floor, the stats say so, and more is selected; a floor larger than
     the session selects every living row; rows kept out of candidacy
     do not count toward what can be read; max_words still clamps the
     raise; a fraction already at or above the floor is left alone and
     selects exactly as it does with no floor; a floor of zero is no
     floor.

Both groups need the BGE encoder (downloaded to the HF cache on first
use). CPU is the default device; the run takes under a minute.

Usage:
    python scripts/chat_floor_regression.py [--device cpu] [--budget 0.2]
"""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salt.engine.session_trie import SessionTrie

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
    ("Back to the roof: which inverter did you suggest for the array?",
     "A string inverter is the cheaper choice under even sun. Microinverters "
     "cost more and help when part of the roof is shaded."),
    ("And how often should the solar panels be cleaned?",
     "Twice a year is enough in most climates. Dust and pollen cut the "
     "output of a solar array by a few percent."),
]
QUERY = "Which inverter suits the solar array on the roof?"


def build_session(name, tmp, tok, mdl, device):
    trie = SessionTrie(name, cache_dir=tmp)
    for user, assistant in EXCHANGES:
        trie.add_turn(user, "user", tokenizer=tok, model=mdl, device=device,
                      save=False)
        trie.add_turn(assistant, "assistant", tokenizer=tok, model=mdl,
                      device=device, save=False)
    return trie


def comparable(stats):
    return {k: v for k, v in stats.items()
            if not any(t in k for t in ("time", "elapsed", "seconds"))}


def check_floor_off(tmp, tok, mdl, device, budget):
    trie = build_session("floor_off", tmp, tok, mdl, device)
    kw = dict(budget_pct=budget, tokenizer=tok, model=mdl, device=device,
              defer_commit=True)
    a = trie.compress(QUERY, **kw)
    b = trie.compress(QUERY, memory_floor=None, **kw)
    assert a["selected_sent_idx"] == b["selected_sent_idx"]
    assert a["context"] == b["context"]
    assert comparable(a["stats"]) == comparable(b["stats"]), (
        {k: (a["stats"].get(k), b["stats"].get(k))
         for k in set(a["stats"]) | set(b["stats"])
         if a["stats"].get(k) != b["stats"].get(k)})
    assert a["stats"]["word_budget_raised"] is False, a["stats"]
    assert a["stats"]["word_budget"] == int(trie.live_words * budget)
    print("A. floor off: no floor and memory_floor=None select the same "
          "rows with the same context and stats, and nothing is raised")


def check_raised(tmp, tok, mdl, device, budget):
    trie = build_session("floor_raised", tmp, tok, mdl, device)
    kw = dict(budget_pct=budget, tokenizer=tok, model=mdl, device=device,
              defer_commit=True)
    plain = int(trie.live_words * budget)
    floor = plain + 40
    assert floor < trie.live_words, "the fixture no longer exercises the raise"
    base = trie.compress(QUERY, **kw)

    up = trie.compress(QUERY, memory_floor=floor, **kw)
    s = up["stats"]
    assert s["word_budget_raised"] is True and s["word_budget"] == floor, s
    assert plain < s["words_used"] <= floor, s
    assert len(up["selected_sent_idx"]) > len(base["selected_sent_idx"])

    whole = trie.compress(QUERY, memory_floor=10 ** 6, **kw)
    assert whole["stats"]["word_budget"] == trie.live_words, whole["stats"]
    assert len(whole["selected_sent_idx"]) == trie.n_alive

    tail = set(range(trie.n_sentences - 6, trie.n_sentences))
    left = sum(trie.n_words[i] for i in range(trie.n_sentences)
               if i not in tail)
    cut = trie.compress(QUERY, memory_floor=10 ** 6, exclude_sent_idx=tail,
                        **kw)
    assert cut["stats"]["word_budget"] == left, (cut["stats"], left)
    assert not set(cut["selected_sent_idx"]) & tail

    capped = trie.compress(QUERY, memory_floor=floor, max_words=plain + 10,
                           **kw)
    assert capped["stats"]["word_budget"] == plain + 10, capped["stats"]
    assert capped["stats"]["word_budget_raised"] is True, capped["stats"]

    for small in (plain, plain - 5, 0):
        same = trie.compress(QUERY, memory_floor=small, **kw)
        assert same["stats"]["word_budget_raised"] is False, same["stats"]
        assert same["stats"]["word_budget"] == plain, same["stats"]
        assert same["selected_sent_idx"] == base["selected_sent_idx"]
        assert same["context"] == base["context"]
    print("B. raised: a fraction under the floor is raised to it or to "
          "everything selectable, kept-out rows do not count, max_words "
          "clamps the raise, and a floor at or under the fraction changes "
          "nothing")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--budget", type=float, default=0.2)
    args = ap.parse_args()
    from salt.engine.compressor import load_bge
    tok, mdl = load_bge(BGE_MODEL, args.device)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        check_floor_off(tmp / "a", tok, mdl, args.device, args.budget)
        check_raised(tmp / "b", tok, mdl, args.device, args.budget)
    print("PASS")


if __name__ == "__main__":
    main()
