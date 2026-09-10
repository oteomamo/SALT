# -*- coding: utf-8 -*-
"""Turns that ask for a summary, and what they select under.

The trie's themes are the keywords above a frequency cutoff, and
coverage spreads the budget across those themes, so a minor topic that
never crossed the cutoff has no branch to be covered by. A summary
wants breadth: on a turn recognized as a summary ask, the compressor
profiles themes at a lower cutoff and discounts a filling branch
faster, for that call only. Every other turn selects as before.
"""

import re

SUMMARY_THEMES = 0.6
SUMMARY_LAM = 0.3
MODES = ("off", "auto")
# what a summary turn records, in this order
RECORD_KEYS = ("why", "themes", "lam", "n_themes", "branches")
# the census of a session's summary turns, one counter per outcome
CENSUS_KEYS = ("asked", "summary", "marked", "plain")
MARKED = "marked by /summary"
WORDED = "the question asks for a summary"

_PATTERNS = tuple(re.compile(p) for p in (
    r"\bsummar(?:y|ize|ise|ies|izing|ising|ized|ised)\b",
    r"\brecap\b", r"\boverview\b", r"\bsynopsis\b", r"\brundown\b",
    r"\btl;?dr\b",
    r"\bsum (?:it |this |that |everything |things )?up\b",
    r"\b(?:main|key) (?:points|takeaways|things)\b",
    r"\bcatch me up\b", r"\bbring me up to speed\b",
    r"\bwhat (?:did|have|had) we (?:discuss|talk|cover|decide|agree|go over)",
    r"\bwhat we(?:'ve| have)? (?:discussed|talked about|covered|decided|"
    r"agreed|gone over)\b",
    r"\beverything (?:we|you)(?:'ve| have)? (?:discussed|talked about|"
    r"covered|said|told)\b",
    r"\bso far\b.*\b(?:we|discussed|talked|covered)\b",
    r"\b(?:we|discussed|talked|covered)\b.*\bso far\b",
))


def is_summary_ask(text):
    """Whether a line asks for a summary, read off its wording alone: a
    small lexicon of the ways people ask for one, matched whole-word."""
    line = " ".join((text or "").lower().split())
    return any(p.search(line) for p in _PATTERNS)


def decide(mode, line, marked=False):
    """Whether this turn selects as a summary, as (on, why). A turn
    marked by hand is one whatever the mode; under `auto` the wording
    decides; under `off` nothing else does."""
    if marked:
        return True, MARKED
    if mode == "auto" and is_summary_ask(line):
        return True, WORDED
    return False, None


def record(on, why, stats):
    """What one summary turn selected under, for /stats and the trace,
    or None on an ordinary turn."""
    if not on:
        return None
    rec = {"why": why, "themes": stats.get("theme_percentile_used"),
           "lam": stats.get("lam_used"),
           "n_themes": stats.get("theme_keywords_total"),
           "branches": stats.get("n_branches")}
    assert tuple(rec) == RECORD_KEYS
    return rec


def census():
    """A fresh census: turns looked at, how many selected as summaries,
    how many of those were marked by hand, how many stayed plain.
    Session-lifetime, never persisted."""
    return {"asked": 0, "summary": 0, "marked": 0, "plain": 0}


def count(census, rec):
    """One more turn looked at: `rec` is that turn's record, None when
    it selected as an ordinary turn."""
    census["asked"] += 1
    if rec is None:
        census["plain"] += 1
        return
    census["summary"] += 1
    if rec["why"] == MARKED:
        census["marked"] += 1


def census_lines(census):
    """The /stats lines for a session's census; nothing when the session
    does not look."""
    if census is None:
        return []
    assert tuple(census) == CENSUS_KEYS
    return [f"summary census: {census['asked']} turns looked at, "
            f"{census['summary']} selected as summaries "
            f"({census['marked']} marked by hand), {census['plain']} plain"]


def lines(rec):
    """The /stats lines for one summary turn; nothing on an ordinary
    turn."""
    if rec is None:
        return []
    assert tuple(rec) == RECORD_KEYS
    return [f"summary turn: {rec['why']}",
            f"  themes at the {rec['themes']:g} percentile, lam "
            f"{rec['lam']:g}: {rec['n_themes']} themes, {rec['branches']} "
            f"branches selected"]
