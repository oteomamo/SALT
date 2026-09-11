# -*- coding: utf-8 -*-
"""Regression harness for the time a turn searches.

Pins what a line can name as a time, the window each resolves to, and
the conversation rows a window holds out. Groups:

  A. THE PARSER - against a fixed moment of asking: a day written as
     day-month, month-day or ISO, with or without a year; a month with
     a year or after a preposition; a year after a preposition; and
     the relative spans (today, yesterday, the day before, N days or
     weeks or months ago, the last N days, last or this week, month
     and year, last or on a weekday). A modal "may", a bare month
     that is a verb, and an ordinary question name nothing.
  B. THE ROWS - on a hand-built session with timestamps (no encoder):
     a day window holds out the other days' conversation rows and
     never a file row or a row without a time; a day without a year
     takes the conversation's own year; a time with no rows is
     reported as not applied; the mode and a pinned window are
     honored; the record and the census carry the asserted keys and
     print as promised.
  C. THE FLAG - real sessions through the chat turn against a fake
     runner: --when auto against off on an undated line is
     byte-identical (prompts, stats, no record, no ledger fields); a
     dated line under auto holds the other days out so the block holds
     that day's rows and the attached file's, records the window and
     carries it into the ledger; a time with no rows selects as off
     does and says so; under off a dated line selects as before.
  D. THE COMMAND AND THE CENSUS - /when shows, sets and pins; a pinned
     window holds out under off and clears on a mode; a time that does
     not read is refused; the census counts every turn looked at; and
     /stats prints the window and the census.

Groups A and B need no encoder; C and D need the BGE encoder
(downloaded to the HF cache on first use). CPU is the default device;
the run takes about a minute.

Usage:
    python scripts/chat_when_regression.py [--device cpu]
"""

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from salt.chat import when as W
from salt.engine.session_trie import SessionTrie

if not __debug__:
    sys.exit("run without -O: this harness is assert-based")

ASK = datetime(2026, 9, 10, 15, 0)          # a Thursday
ANCHOR = ASK.timestamp()


def day(y, m, d):
    s = datetime(y, m, d)
    return s.timestamp(), (s + timedelta(days=1)).timestamp()


def resolved(line, year=None):
    spec = W.parse(line, ANCHOR)
    if spec is None:
        return None
    w = W.window(spec, year)
    return None if w is None else (w[0], w[1], w[2])


def check_parser():
    cases = {
        "Summarize what Caroline and Melanie talked about on 8 May 2023, in about 100 words.":
            (*day(2023, 5, 8), "8 May 2023"),
        "What did we decide on May 8, 2023?": (*day(2023, 5, 8), "8 May 2023"),
        "Notes from 2023-05-08 please": (*day(2023, 5, 8), "8 May 2023"),
        "the 8th of May 2023": (*day(2023, 5, 8), "8 May 2023"),
        "what happened in May 2023": (datetime(2023, 5, 1).timestamp(),
                                      datetime(2023, 6, 1).timestamp(), "May 2023"),
        "what did we cover during 2023": (datetime(2023, 1, 1).timestamp(),
                                          datetime(2024, 1, 1).timestamp(), "2023"),
        "what did I tell you yesterday?": (*day(2026, 9, 9), "9 September 2026"),
        "and the day before yesterday": (*day(2026, 9, 8), "8 September 2026"),
        "recap today": (*day(2026, 9, 10), "10 September 2026"),
        "three days ago we spoke about it": (*day(2026, 9, 7), "7 September 2026"),
        "what came up last week": (datetime(2026, 8, 31).timestamp(),
                                   datetime(2026, 9, 7).timestamp(), "31 August to 6 September 2026"),
        "everything this week": (datetime(2026, 9, 7).timestamp(),
                                 datetime(2026, 9, 14).timestamp(), "7 to 13 September 2026"),
        "the last 3 days": (datetime(2026, 9, 8).timestamp(),
                            datetime(2026, 9, 11).timestamp(), "8 to 10 September 2026"),
        "over the past two weeks": (datetime(2026, 8, 27).timestamp(),
                                    datetime(2026, 9, 11).timestamp(), "27 August to 10 September 2026"),
        "what did we do last month": (datetime(2026, 8, 1).timestamp(),
                                      datetime(2026, 9, 1).timestamp(), "August 2026"),
        "two weeks ago": (datetime(2026, 8, 24).timestamp(),
                          datetime(2026, 8, 31).timestamp(), "24 to 30 August 2026"),
        "what did we say last monday": (*day(2026, 9, 7), "7 September 2026"),
        "on friday you mentioned a gasket": (*day(2026, 9, 4), "4 September 2026"),
        "last year's plans": (datetime(2025, 1, 1).timestamp(),
                              datetime(2026, 1, 1).timestamp(), "2025"),
    }
    for line, want in cases.items():
        got = resolved(line)
        assert got == want, (line, got, want)
    # a day or a month without a year waits for one
    spec = W.parse("what did we discuss on 8 May?", ANCHOR)
    assert spec == {"kind": "day", "day": 8, "month": 5, "year": None}, spec
    assert W.window(spec, None) is None
    assert W.window(spec, 2023)[2] == "8 May 2023"
    assert W.parse("what happened in may", ANCHOR) == {"kind": "month", "month": 5, "year": None}
    assert W.window({"kind": "day", "day": 31, "month": 2, "year": None}, 2024) is None
    for line in ("we may have discussed it already", "can you summarize?",
                 "march the troops home", "the 8 may be too many",
                 "what is 2023 minus 5", "nothing dated here", ""):
        assert W.parse(line, ANCHOR) is None, line
    print("A. parser: days, months, years and relative spans resolve to the "
          "asserted windows against a fixed moment, a day or month without "
          "a year waits for one, and modal or bare words name nothing")


def hand_built(tmp):
    trie = SessionTrie("when_hand", cache_dir=tmp)
    may8 = datetime(2023, 5, 8, 13, 56).timestamp()
    may25 = datetime(2023, 5, 25, 13, 14).timestamp()
    rows = [
        ("Caroline: I went to the support group.", None, may8),
        ("Melanie: That sounds like it helped.", None, may8 + 30),
        ("Caroline: I want to study counseling.", None, may8 + 60),
        ("Caroline: The camping trip is booked.", None, may25),
        ("Melanie: Bring the good tent.", None, may25 + 30),
        ("The orchard holds forty apple trees.", "orchard.txt", may8 + 5),
        ("An old row with no time on it.", None, None),
    ]
    for text, source, ts in rows:
        trie.texts.append(text)
        trie.roles.append("user")
        trie.turns.append(0)
        trie.sources.append(source)
        trie.origins.append(None)
        trie.timestamps.append(ts)
        trie.n_words.append(len(text.split()))
        trie.keyword_weights.append({})
        trie.alive.append(True)
    return trie


def check_rows(tmp):
    trie = hand_built(tmp)
    start, end = day(2023, 5, 8)
    out, n_in = W.rows_outside(trie, start, end)
    assert out == {3, 4} and n_in == 3, (out, n_in)
    assert W.years_of(trie) == [2023]
    held, rec = W.decide("auto", "What did we talk about on 8 May 2023?", trie, ANCHOR)
    assert held == {3, 4} and tuple(rec) == W.RECORD_KEYS, rec
    assert rec["mode"] == "auto" and rec["label"] == "8 May 2023", rec
    assert (rec["rows_in"], rec["rows_out"], rec["note"]) == (3, 2, None), rec
    held, rec = W.decide("auto", "What did we talk about on 8 May?", trie, ANCHOR)
    assert held == {3, 4} and rec["label"] == "8 May 2023", rec
    held, rec = W.decide("auto", "and on the 25th of May?", trie, ANCHOR)
    assert held == {0, 1, 2} and rec["rows_in"] == 2, rec
    held, rec = W.decide("auto", "What about 9 May 2023?", trie, ANCHOR)
    assert held is None and rec["note"] == W.EMPTY and rec["rows_in"] == 0, rec
    assert rec["label"] == "9 May 2023", rec
    held, rec = W.decide("auto", "What did we talk about yesterday?", trie, ANCHOR)
    assert held is None and rec["note"] == W.EMPTY, rec
    assert W.decide("auto", "Anything else?", trie, ANCHOR) == (None, None)
    assert W.decide("off", "What did we talk about on 8 May 2023?", trie, ANCHOR) == (None, None)
    pinned = W.parse("25 May 2023", ANCHOR)
    held, rec = W.decide("off", "Anything else?", trie, ANCHOR, pinned=pinned)
    assert held == {0, 1, 2} and rec["mode"] == W.PINNED, rec
    trie.alive[3] = False
    held, rec = W.decide("auto", "on 8 May 2023", trie, ANCHOR)
    assert held == {4}, held
    c = W.census()
    assert tuple(c) == W.CENSUS_KEYS
    for r in (rec, None, {**rec, "note": W.EMPTY}, {**rec, "mode": W.PINNED}):
        W.count(c, r)
    assert c == {"asked": 4, "windowed": 1, "pinned": 1, "plain": 1, "empty": 1}, c
    text = W.census_lines(c)[0]
    assert text.startswith("time census: 4 turns looked at, 1 windowed by the rule, "
                           "1 pinned by hand, 1 naming no time, 1 with no rows"), text
    assert W.lines(rec) == ["time window (auto): 8 May 2023, 3 conversation rows in, "
                            "1 held out"], W.lines(rec)
    assert W.lines({**rec, "note": W.EMPTY}) == [
        "time window (auto): not applied '8 May 2023', no conversation rows in that window"]
    assert W.lines(None) == []
    print("B. rows: a day window holds out the other days' conversation rows "
          "and never a file row or an undated row, a year is taken from the "
          "conversation, an empty time is reported and not applied, the mode "
          "and a pinned window are honored, and the record and census print "
          "as promised")


BGE_MODEL = "BAAI/bge-small-en-v1.5"
MAY8 = datetime(2023, 5, 8, 13, 56).timestamp()
MAY25 = datetime(2023, 5, 25, 13, 14).timestamp()
DAY_ONE = [
    ("Caroline: I went to the LGBTQ support group for the first time.",
     "Melanie: That sounds like the group helped you feel accepted."),
    ("Caroline: I want to study counseling and work in mental health.",
     "Melanie: Counseling suits your empathy and your patience."),
    ("Caroline: The transgender stories at the group were inspiring.",
     "Melanie: Hearing those stories takes real courage."),
]
DAY_TWO = [
    ("Caroline: The camping trip to the lake is booked for July.",
     "Melanie: Bring the good tent and the camp stove."),
    ("Caroline: The kayak rental at the lake costs forty dollars.",
     "Melanie: Forty dollars for a kayak day is fair."),
    ("Caroline: We should pack the bear canister for the lake.",
     "Melanie: The bear canister goes in the tent vestibule."),
]
DOC_NAME = "orchard.txt"
DOC_TEXT = ("The orchard holds forty apple trees and twelve pear trees on "
            "the south slope. Apple trees are pruned in late winter before "
            "the buds swell. Pear trees tolerate wetter ground than apple "
            "trees do. The orchard's drip line runs from the pond pump.")
UNDATED = "Which tent did Melanie recommend?"
DATED = "What did Caroline and Melanie talk about on 8 May 2023?"


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
        yield "Noted, and here is what I make of it."


def chat_session(root, tok, mdl, device, flags):
    from salt.chat import cli
    # a budget that spans the whole session, so what a window holds out
    # is the only reason a row is missing from the block
    args = cli.build_parser().parse_args(
        ["--device", device, "--sync-ingest", "--budget-pct", "0.9", *flags])
    trie = SessionTrie("when_flag", cache_dir=root, model_name=BGE_MODEL,
                       budget_pct_default=args.budget_pct)
    for pairs, t0 in ((DAY_ONE, MAY8), (DAY_TWO, MAY25)):
        for i, (user, assistant) in enumerate(pairs):
            trie.add_turn(user, "user", tokenizer=tok, model=mdl, device=device,
                          save=False, filed_at=t0 + 60 * i)
            trie.add_turn(assistant, "assistant", tokenizer=tok, model=mdl,
                          device=device, save=False, filed_at=t0 + 60 * i + 30)
    trie.add_turn(DOC_TEXT, "user", tokenizer=tok, model=mdl, device=device,
                  source=DOC_NAME, save=False)
    return cli.ChatState(args, tok, mdl, _FakeRunner(tok), trie)


def turn(state, line):
    from salt.chat import cli
    with redirect_stdout(io.StringIO()):
        cli.chat_turn(state, line)
    stats = {k: v for k, v in (state.last_stats or {}).items()
             if not any(t in k for t in ("time", "elapsed", "seconds"))}
    return state.runner.prompts[-1], stats


def block_of(prompt):
    return prompt[-1]["content"]


def ledger_keys(state):
    from salt.chat import cli
    return set(cli.build_stats(state)["kv"]["last_event"] or {})


def stats_text(state):
    from salt.chat import cli
    buf = io.StringIO()
    with redirect_stdout(buf):
        cli.print_stats(state)
    return buf.getvalue().splitlines()


def check_flag(tmp, tok, mdl, device):
    from salt.chat import cli
    off = chat_session(tmp / "c_off", tok, mdl, device, ["--when", "off"])
    on = chat_session(tmp / "c_on", tok, mdl, device, ["--when", "auto"])
    p_off, s_off = turn(off, UNDATED)
    p_on, s_on = turn(on, UNDATED)
    assert p_off == p_on and s_off == s_on
    assert on.last_when is None and off.last_when is None
    assert not any(k.startswith("when_") for k in ledger_keys(on))
    p_off, s_off = turn(off, DATED)
    assert off.last_when is None
    assert "tent" in block_of(p_off) and "support group" in block_of(p_off), (
        "under off the block should span both days")
    n_before = on.trie.n_sentences
    p_on, s_on = turn(on, DATED)
    rec = on.last_when
    assert rec is not None and tuple(rec) == W.RECORD_KEYS, rec
    assert (rec["mode"], rec["label"], rec["note"]) == ("auto", "8 May 2023", None), rec
    # day two's six rows plus whatever the session's own earlier turns
    # filed, all of it outside 8 May 2023
    assert rec["rows_in"] == 6 and rec["rows_out"] == 6 + (n_before - 16), rec
    block = block_of(p_on)
    assert "support group" in block and "counseling" in block, block
    assert "tent" not in block and "kayak" not in block and "bear" not in block, block
    assert "orchard" in block, "the file's rows were held out"
    assert s_on["excluded_sent"] >= 6, s_on
    assert ledger_keys(on) >= {"when_label", "when_rows_out"}, ledger_keys(on)
    p_empty, s_empty = turn(on, "What did they talk about on 9 May 2023?")
    assert on.last_when["note"] == W.EMPTY and on.last_when["label"] == "9 May 2023"
    p_same, _ = turn(off, "What did they talk about on 9 May 2023?")
    assert block_of(p_empty) == block_of(p_same), "an empty window changed the block"
    assert not any(k.startswith("when_") for k in ledger_keys(on))
    assert cli.build_parser().parse_args([]).when == "auto"
    print("C. flag: --when auto is byte-identical to off on an undated line, "
          "a dated line under auto holds the other day out and keeps the "
          "file, records the window and ledgers it, a time with no rows "
          "selects as off and says so, and under off a dated line selects "
          "as before")


def check_command_census(tmp, tok, mdl, device):
    from salt.chat import cli
    st = chat_session(tmp / "d", tok, mdl, device, ["--when", "off"])
    assert cli.when_command(st, []) == "time window: off"
    assert cli.build_stats(st)["when_census"] is None
    assert cli.when_command(st, ["nonsense", "words"]).startswith("Could not read a time")
    assert cli.when_command(st, ["25", "May", "2023"]) == (
        "time window: pinned, 25 May 2023 until the next /when")
    assert cli.when_command(st, []) == "time window: pinned, 25 May 2023"
    p, _ = turn(st, UNDATED)
    assert st.last_when["mode"] == W.PINNED and st.last_when["rows_out"] == 6
    assert "tent" in block_of(p) and "support group" not in block_of(p), block_of(p)
    assert cli.when_command(st, ["auto"]) == "time window: auto"
    assert st.when_pinned is None
    turn(st, UNDATED)
    assert st.last_when is None
    turn(st, DATED)
    turn(st, "and on 9 May 2023?")
    c = st.when_stats
    assert tuple(c) == W.CENSUS_KEYS and c == {"asked": 4, "windowed": 1,
                                               "pinned": 1, "plain": 1,
                                               "empty": 1}, c
    assert cli.build_stats(st)["when_census"] == c
    n_before = st.trie.n_sentences
    turn(st, DATED)
    text = stats_text(st)
    held = st.last_when["rows_out"]
    assert held == 6 + (n_before - 16), (held, n_before)
    assert any(ln.startswith(f"time window (auto): 8 May 2023, 6 conversation rows "
                             f"in, {held} held out") for ln in text), text
    assert any(ln.startswith("time census: 5 turns looked at, 2 windowed by the rule, "
                             "1 pinned by hand, 1 naming no time, 1 with no rows")
               for ln in text), text
    assert cli.when_command(st, ["off"]) == "time window: off"
    turn(st, DATED)
    assert st.last_when is None and st.when_stats["asked"] == 5
    assert cli.build_stats(st)["when_census"] == st.when_stats
    assert "/when" in cli.COMMANDS and "/when" in cli.HELP
    print("D. command and census: /when shows, sets and pins, a pinned window "
          "holds out under off and clears on a mode, an unreadable time is "
          "refused, the census counts what was looked at, and /stats prints "
          "the window and the census")


def main():
    import shutil
    import tempfile
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="salt_when_"))
    try:
        check_parser()
        check_rows(tmp)
        from transformers import AutoModel, AutoTokenizer
        print(f"Loading BGE encoder {BGE_MODEL} on {args.device} ...")
        tok = AutoTokenizer.from_pretrained(BGE_MODEL)
        mdl = AutoModel.from_pretrained(BGE_MODEL, output_attentions=True).eval()
        mdl.to(args.device)
        check_flag(tmp, tok, mdl, args.device)
        check_command_census(tmp, tok, mdl, args.device)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS")


if __name__ == "__main__":
    main()
