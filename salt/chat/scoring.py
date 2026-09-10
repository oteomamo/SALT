# -*- coding: utf-8 -*-
"""Scoring a scripted run against the reference answers its items carry.

Conservative on purpose: a free-form answer is graded by three rules
in order, and the row says which one fired, so a reader can discount
the loosest. The raw answer is always kept beside the verdict.
"""

import re
from collections import Counter

GOLD_KEYS = ("gold", "answer", "solution", "expected")
MATCHES = ("exact", "final", "contains")

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_MARKUP = re.compile(r"[*_`#>]+")
_LABEL = re.compile(r"^(?:final\s+answer|the\s+answer\s+is|answer\s+is|"
                    r"answer)\s*[:\-]?\s*", re.I)
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")
_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")


def normalize(text):
    """Case, markup, punctuation and articles removed, spaces collapsed.
    A decimal point between digits stays, so 3.14 is not 314, and a
    thousands comma goes, so 1,000 is 1000."""
    text = _MARKUP.sub("", str(text).casefold())
    text = _DECIMAL.sub("\x00", _THOUSANDS.sub("", text))
    text = "".join(ch if ch.isalnum() or ch.isspace() or ch == "\x00"
                   else " " for ch in text)
    text = _ARTICLES.sub(" ", text.replace("\x00", "."))
    return " ".join(text.split())


def final_line(text):
    """The last non-empty line of an answer with a leading label such as
    "Answer:" removed: where a model that reasons puts what it settled
    on."""
    lines = [l.strip() for l in str(text).splitlines() if l.strip()]
    if not lines:
        return ""
    return _LABEL.sub("", _MARKUP.sub("", lines[-1]).strip())


def references(gold):
    """The acceptable answers an item's gold names, as strings: one, or
    any of a list."""
    values = gold if isinstance(gold, (list, tuple)) else [gold]
    return [str(v).strip() for v in values
            if v is not None and str(v).strip()]


def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def _contains(toks, ref_toks):
    n = len(ref_toks)
    return n > 0 and any(toks[i:i + n] == ref_toks
                         for i in range(len(toks) - n + 1))


def verdict(answer, gold):
    """How an answer fares against its reference: `correct`, the `match`
    rule that decided it (exact, final, contains, or None) and the token
    `f1` against the closest reference. A missing answer scores
    nothing."""
    refs = references(gold)
    if answer is None or not refs:
        return {"correct": False, "match": None, "f1": 0.0}
    whole, last = normalize(answer), normalize(final_line(answer))
    toks = whole.split()
    match = None
    for rule in MATCHES:
        for ref in refs:
            norm = normalize(ref)
            if not norm:
                continue
            if ((rule == "exact" and whole == norm)
                    or (rule == "final" and last == norm)
                    or (rule == "contains" and _contains(toks, norm.split()))):
                match = rule
                break
        if match:
            break
    f1 = max(token_f1(answer, ref) for ref in refs)
    return {"correct": match is not None, "match": match, "f1": round(f1, 4)}


def summary(rows):
    """Counts over the rows that carry a verdict, the latest row of each
    item counting once: scored, correct, mean f1, and correct against
    scored per category where the item names one."""
    latest = {}
    for r in rows:
        if "correct" not in r:
            continue
        key = (("id", r["id"]) if r.get("id") is not None
               else ("turn", r.get("turn")))
        latest[key] = r
    scored = list(latest.values())
    cats = {}
    for r in scored:
        cat = (r.get("item") or {}).get("category")
        if cat is not None:
            c = cats.setdefault(str(cat), [0, 0])
            c[0] += bool(r["correct"])
            c[1] += 1
    n = len(scored)
    return {"scored": n,
            "correct": sum(bool(r["correct"]) for r in scored),
            "f1": (round(sum(float(r.get("f1") or 0.0) for r in scored) / n, 4)
                   if n else 0.0),
            "categories": cats}


def summary_line(score):
    """The one line a scored run ends with, or None when nothing was
    scored."""
    if not score["scored"]:
        return None
    parts = [f"scored {score['correct']}/{score['scored']} correct, "
             f"mean F1 {score['f1']:.3f}"]
    if score["categories"]:
        parts.append("by category " + ", ".join(
            f"{k} {c}/{n}" for k, (c, n) in sorted(score["categories"].items())))
    return "[" + ", ".join(parts) + "]"
