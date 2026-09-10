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

No encoder needed; the run takes seconds.

Usage:
    python scripts/chat_when_regression.py
"""

import sys
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


def main():
    import shutil
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="salt_when_"))
    try:
        check_parser()
        check_rows(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS")


if __name__ == "__main__":
    main()
