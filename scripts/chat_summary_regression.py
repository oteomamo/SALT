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
  D. THE LEXICON - the wordings that ask for a summary are recognized,
     ordinary questions that share their words are not, and a turn is a
     summary when marked by hand whatever the mode.
  E. THE FLAG - real sessions through the chat turn: --summary auto
     against off on ordinary questions is byte-identical (prompts,
     stats, no record, no ledger fields); a summary ask under auto
     selects at the two knobs, records what it selected under, carries
     the two values into the ledger, and the next ordinary turn selects
     as before; under off the same ask selects as before.
  F. THE COMMAND AND THE CENSUS - /summary shows, sets and marks; a
     marked turn selects as a summary under off; the census counts
     every turn looked at, the summaries, the marked ones and the plain
     ones, and /stats prints the record and the census.

Needs the BGE encoder (downloaded to the HF cache on first use). CPU is
the default device; the run takes about a minute.

Usage:
    python scripts/chat_summary_regression.py [--device cpu]
"""

import argparse
import io
import json
import shutil
import sys
import tempfile
from contextlib import redirect_stdout
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


SUMMARY_ASKS = (
    "Can you summarize what we have talked about so far?",
    "Give me a recap of the conversation.",
    "What did we discuss about the kayak and the taxes?",
    "Catch me up on everything we covered.",
    "tl;dr of this chat please",
    "What are the main points so far?",
    "Sum it up for me.",
)
PLAIN_ASKS = (
    "What feathering angle did you suggest for the paddle?",
    "How far does the drip line run?",
    "Is the starter ready to bake with?",
    "Summit day on the hike is Sunday, what should I pack?",
    "So far the pump has not failed, should I still refill it?",
)


class _FakeRunner:
    kind = "fake"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.alias = "fake"
        self.cfg = {"alias": "fake", "hf_id": "test/fake", "path": "-"}
        self.max_input_len = 4096
        self.last_prompt_tokens = None
        self.last_engine_stats = None
        self.prompts = []

    def input_budget(self, max_new_tokens=None):
        return self.max_input_len

    def stream_chat(self, messages, **overrides):
        self.prompts.append(json.loads(json.dumps(messages)))
        yield "Noted, and here is what I make of it all together."


def chat_session(root, tok, mdl, device, flags):
    from salt.chat import cli
    args = cli.build_parser().parse_args(
        ["--device", device, "--sync-ingest", *flags])
    trie = SessionTrie("summary_flag", cache_dir=root, model_name=BGE_MODEL,
                       budget_pct_default=args.budget_pct)
    for user, assistant in EXCHANGES:
        trie.add_turn(user, "user", tokenizer=tok, model=mdl, device=device,
                      save=False)
        trie.add_turn(assistant, "assistant", tokenizer=tok, model=mdl,
                      device=device, save=False)
    return cli.ChatState(args, tok, mdl, _FakeRunner(tok), trie)


def turn(state, line):
    from salt.chat import cli
    with redirect_stdout(io.StringIO()):
        cli.chat_turn(state, line)
    return state.runner.prompts[-1], comparable(state.last_stats)


def ledger_keys(state):
    from salt.chat import cli
    return set(cli.build_stats(state)["kv"]["last_event"] or {})


def stats_text(state):
    from salt.chat import cli
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.print_stats(state)
    return buf.getvalue().splitlines()


def check_lexicon():
    from salt.chat import summary as S
    for ask in SUMMARY_ASKS:
        assert S.is_summary_ask(ask), ask
    for ask in PLAIN_ASKS:
        assert not S.is_summary_ask(ask), ask
    assert S.decide("auto", SUMMARY_ASKS[0]) == (True, S.WORDED)
    assert S.decide("auto", PLAIN_ASKS[0]) == (False, None)
    assert S.decide("off", SUMMARY_ASKS[0]) == (False, None)
    assert S.decide("off", PLAIN_ASKS[0], marked=True) == (True, S.MARKED)
    assert S.record(False, None, {}) is None
    print("D. lexicon: the wordings that ask for a summary are recognized, "
          "ordinary questions sharing their words are not, a marked turn is "
          "a summary whatever the mode")


def check_flag(tmp, tok, mdl, device):
    from salt.chat import summary as S
    off = chat_session(tmp / "e_off", tok, mdl, device, ["--summary", "off"])
    on = chat_session(tmp / "e_on", tok, mdl, device, ["--summary", "auto"])
    for ask in PLAIN_ASKS[:2]:
        p_off, s_off = turn(off, ask)
        p_on, s_on = turn(on, ask)
        assert p_off == p_on and s_off == s_on, ask
        assert on.last_summary is None and off.last_summary is None
        assert s_on["theme_percentile_used"] == on.trie.config["theme_percentile"]
    base_keys = ledger_keys(on)
    assert not any(k.startswith("summary_") for k in base_keys), base_keys
    p_off, s_off = turn(off, SUMMARY_ASKS[0])
    assert off.last_summary is None
    assert s_off["theme_percentile_used"] == off.trie.config["theme_percentile"]
    p_on, s_on = turn(on, SUMMARY_ASKS[0])
    rec = on.last_summary
    assert rec is not None and tuple(rec) == S.RECORD_KEYS, rec
    assert rec["why"] == S.WORDED and rec["themes"] == S.SUMMARY_THEMES
    assert rec["lam"] == S.SUMMARY_LAM, rec
    assert s_on["theme_percentile_used"] == S.SUMMARY_THEMES
    assert s_on["lam_used"] == S.SUMMARY_LAM, s_on
    assert s_on["theme_keywords_total"] >= s_off["theme_keywords_total"], (s_on, s_off)
    assert ledger_keys(on) >= base_keys | {"summary_themes", "summary_lam"}, ledger_keys(on)
    p_on2, s_on2 = turn(on, PLAIN_ASKS[2])
    assert on.last_summary is None
    assert s_on2["theme_percentile_used"] == on.trie.config["theme_percentile"]
    assert s_on2["lam_used"] == on.trie.config["lam"], s_on2
    assert not any(k.startswith("summary_") for k in ledger_keys(on))
    knobs = chat_session(tmp / "e_knobs", tok, mdl, device,
                         ["--summary", "auto", "--summary-themes", "0.3",
                          "--summary-lam", "0.15"])
    _, s_k = turn(knobs, SUMMARY_ASKS[1])
    assert (s_k["theme_percentile_used"], s_k["lam_used"]) == (0.3, 0.15), s_k
    from salt.chat import cli
    assert cli.build_parser().parse_args([]).summary == "off"
    assert cli.main(["--summary-themes", "1.0"]) == 1
    assert cli.main(["--summary-lam", "0"]) == 1
    print("E. flag: --summary auto is byte-identical to off on ordinary "
          "questions, a summary ask under auto selects at the two knobs and "
          "records and ledgers them, the next ordinary turn selects as "
          "before, under off the ask selects as before, the knobs reach the "
          "turn, and the launch refuses a knob outside its range")


def check_command_census(tmp, tok, mdl, device):
    from salt.chat import cli
    from salt.chat import summary as S
    st = chat_session(tmp / "f", tok, mdl, device, ["--summary", "off"])
    assert cli.summary_command(st, []) == "summary: off"
    assert cli.build_stats(st)["summary_census"] is None
    assert "next" in cli.summary_command(st, ["next"])
    assert cli.summary_command(st, []) == "summary: off, the next turn is marked"
    turn(st, PLAIN_ASKS[0])
    assert st.last_summary is not None and st.last_summary["why"] == S.MARKED
    assert not st.summary_next
    turn(st, PLAIN_ASKS[1])
    assert st.last_summary is None
    c = st.summary_stats
    assert tuple(c) == S.CENSUS_KEYS and c == {"asked": 1, "summary": 1,
                                               "marked": 1, "plain": 0}, c
    assert cli.build_stats(st)["summary_census"] == c
    assert cli.summary_command(st, ["auto"]) == "summary: auto"
    turn(st, PLAIN_ASKS[2])
    turn(st, SUMMARY_ASKS[2])
    assert st.summary_stats == {"asked": 3, "summary": 2, "marked": 1,
                                "plain": 1}, st.summary_stats
    text = stats_text(st)
    assert any(ln.startswith("summary turn: the question asks") for ln in text), text
    assert any(ln.startswith("summary census: 3 turns looked at, 2 selected "
                             "as summaries (1 marked by hand), 1 plain")
               for ln in text), text
    assert cli.summary_command(st, ["off"]) == "summary: off"
    assert cli.summary_command(st, ["sideways"]).startswith("Usage:")
    turn(st, SUMMARY_ASKS[3])
    assert st.last_summary is None and st.summary_stats["asked"] == 3
    assert cli.build_stats(st)["summary_census"] == st.summary_stats
    assert "/summary" in cli.COMMANDS and "/summary" in cli.HELP
    print("F. command and census: /summary shows, sets and marks, a marked "
          "turn selects as a summary under off, the census counts what was "
          "looked at, and /stats prints the record and the census")


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
        check_lexicon()
        check_flag(tmp, tok, mdl, args.device)
        check_command_census(tmp, tok, mdl, args.device)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS")


if __name__ == "__main__":
    main()
