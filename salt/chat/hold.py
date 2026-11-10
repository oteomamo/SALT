# -*- coding: utf-8 -*-
"""The most a turn reads when memory holds nothing for it.

A fifth of a long session is thousands of words whether or not any of
them bear on the question. Under a hold the compressor measures how much
the rows it may select from bear on the turn, and a turn they hold
nothing for reads a short block in place of the full one. A held turn is
never empty: it still reads the best the selector has, up to the hold.

The measuring is the compressor's. This module says which turns must
never be held, the ones that are about the memory itself, and keeps what
a turn and a session report about it.
"""

import re

from salt.chat.summary import is_summary_ask

HOLD_WORDS = 400
SHORT_WORDS = 25
# what a turn records about its hold, in this order
RECORD_KEYS = ("words", "held", "top3", "match", "claimed")
# the census of a session's turns, one counter per outcome
CENSUS_KEYS = ("asked", "judged", "held", "claimed")
SUMMARY = "asks for a summary"
PAST = "refers to the shared past"
PERSONAL = "speaks in the first person"
WINDOW = "searches a time window"
NAMED = "searches files named by hand"

_PAST = tuple(re.compile(p) for p in (
    r"\b(?:remember|recall|remind)\w*\b",
    r"\b(?:you|we|i) (?:said|told|mentioned|suggested|recommended|"
    r"discussed|talked|decided|agreed|asked)\b",
    r"\b(?:last time|earlier|previously|before|again|so far)\b",
    r"\b(?:last|previous|earlier|first) (?:chat|conversation|talk|session|"
    r"message)\b",
))
_FIRST_PERSON = re.compile(r"\b(?:i|me|my|mine|myself|we|us|our|ours)\b")


def parse(val):
    """The hold in words from `--memory-hold`: None for 'off', HOLD_WORDS
    for 'auto', else a whole number of words above zero."""
    v = str(val).strip().lower()
    if v == "off":
        return None
    if v == "auto":
        return HOLD_WORDS
    n = int(v)
    if n <= 0:
        raise ValueError(f"not a number of words above 0: {val!r}")
    return n


def claimed(line):
    """Why a line is about the memory itself, or None. Only a short line
    can be: a long one carries the material it works on."""
    if len((line or "").split()) > SHORT_WORDS:
        return None
    text = " ".join((line or "").lower().split())
    if is_summary_ask(text):
        return SUMMARY
    if any(p.search(text) for p in _PAST):
        return PAST
    if _FIRST_PERSON.search(text):
        return PERSONAL
    return None


def may_hold(line, summary=False, windowed=False, named=False):
    """Whether this turn may be held when memory holds nothing for it, as
    (ok, why): `why` names what claimed the turn when it may not."""
    if summary:
        return False, SUMMARY
    if windowed:
        return False, WINDOW
    if named:
        return False, NAMED
    why = claimed(line)
    return why is None, why


def record(words, stats, why):
    """What the hold did on one turn, for /stats and the trace, or None
    in a session without one."""
    if words is None:
        return None
    rec = {"words": words, "held": bool(stats.get("word_budget_held")),
           "top3": stats.get("evidence_top3"),
           "match": stats.get("evidence_match"), "claimed": why}
    assert tuple(rec) == RECORD_KEYS
    return rec


def census():
    """A fresh census: turns selected under a hold, turns whose memory
    was measured against them, turns held, and turns that could not be
    held because they are about the memory. Session-lifetime, never
    persisted."""
    return {"asked": 0, "judged": 0, "held": 0, "claimed": 0}


def count(census, rec):
    """One more turn selected under a hold: `rec` is that turn's
    record."""
    if rec is None:
        return
    census["asked"] += 1
    if rec["top3"] is not None:
        census["judged"] += 1
    if rec["held"]:
        census["held"] += 1
    if rec["claimed"]:
        census["claimed"] += 1


def census_lines(census):
    """The /stats line for a session's census; nothing when the session
    has no hold."""
    if census is None:
        return []
    assert tuple(census) == CENSUS_KEYS
    return [f"hold census: {census['asked']} turns, {census['judged']} "
            f"measured against memory, {census['held']} held, "
            f"{census['claimed']} about the memory itself"]


def lines(words, rec):
    """The /stats lines for the hold: the setting, and what it did on
    the last turn; nothing in a session without one."""
    if words is None:
        return []
    if rec is None:
        return [f"memory hold: {words} words"]
    assert tuple(rec) == RECORD_KEYS
    out = [f"memory hold: {words} words"
           + (" - the last block was held to it" if rec["held"] else "")]
    if rec["top3"] is not None:
        out.append(f"  closest rows {rec['top3']:.3f}, question terms found "
                   f"{rec['match']:.2f}")
    if rec["claimed"]:
        out.append(f"  never held: the turn {rec['claimed']}")
    return out
