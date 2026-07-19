"""
Short-turn predicates: let terse user decisions ("go with option B") past
the junk filter's length gates at conversation ingest.
"""

import re

from salt.engine.sentence_filter import (
    JUNK_CONTAINS,
    JUNK_PATTERNS,
    MIN_CHAR_LENGTH,
    MIN_WORD_COUNT,
    contains_url,
    is_aggressive_junk,
)
from salt.engine.trie_core import is_content_word

ACK_WORDS = frozenset({
    "yes", "yeah", "yep", "no", "nope", "sure", "ok", "okay", "correct",
    "right", "agreed", "first", "second", "third", "one", "two", "option",
    "that", "this",
})

_WORD_RE = re.compile(r"[a-z]+")


def _junk_ignoring_length(text):
    """The two regex loops of is_junk without its length gates. For a
    sub-threshold unit the JUNK_CONTAINS branch of is_junk always drops it
    (clean_wiki_markup never grows text past MIN_CHAR_LENGTH), so a plain
    True is faithful; the lenient gate needs 8+ words and cannot fire."""
    t = text.strip()
    for pattern in JUNK_PATTERNS:
        if re.match(pattern, t, re.IGNORECASE):
            return True
    for pattern in JUNK_CONTAINS:
        if re.search(pattern, t, re.IGNORECASE):
            return True
    return False


def is_short_user_unit(text):
    """keep= predicate for user-role chat ingest: True only for a unit the
    LENGTH rule alone would kill. keep= exempts a unit from every junk
    test, so the URL, aggressive-junk and junk-regex tests re-apply here."""
    t = text.strip()
    if not t:
        return False
    if len(t) >= MIN_CHAR_LENGTH and len(t.split()) >= MIN_WORD_COUNT:
        return False
    if not any(c.isalpha() for c in t):
        return False
    return not (contains_url(t) or is_aggressive_junk(t)
                or _junk_ignoring_length(t))


def acknowledgement_only(text):
    """True when every content word is a bare acknowledgement ("yes",
    "the second one"), so the unit is meaningless without the question it
    answers."""
    words = _WORD_RE.findall(text.lower())
    return all(w in ACK_WORDS for w in words if is_content_word(w))
