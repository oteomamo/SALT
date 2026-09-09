# -*- coding: utf-8 -*-
"""Which branches of the trie a turn searches.

A session with several attached files holds one branch per file beside
the conversation. The rule here keeps the files a question is about and
hands the rest to the compressor as out of scope for that turn, so the
block is drawn from where the evidence is and sized by the words inside
the scope. The conversation is always kept.
"""

SCOPE_CENTROID_MARGIN = 0.02
SCOPE_PEAK_MARGIN = 0.10
MODES = ("off", "auto")
# what a turn records about its scope, in this order
SCOPED_KEYS = ("mode", "kept", "out", "words", "budget", "excluded", "note")
# the census of a session's routing, one counter per outcome
CENSUS_KEYS = ("asked", "scoped", "plain", "guarded", "branches", "kept")


def scope_of(rows, centroid_margin=SCOPE_CENTROID_MARGIN,
             peak_margin=SCOPE_PEAK_MARGIN):
    """The source names a turn may select from, given the branch rows of
    branch_scores(), or None when there is nothing to route: no attached
    branch among the rows. The conversation (None) is always in. An
    attached branch stays in when its centroid is within
    `centroid_margin` of the best attached centroid, its peak within
    `peak_margin` of the best attached peak, or it carries the most
    proper-noun hits of the attached branches, at least one."""
    files = [r for r in rows if r["name"] is not None]
    if not files:
        return None
    best_c = max(r["centroid"] for r in files)
    best_p = max(r["peak"] for r in files)
    best_n = max(r["names"] for r in files)
    keep = {None}
    for r in files:
        if (r["centroid"] >= best_c - centroid_margin
                or r["peak"] >= best_p - peak_margin
                or (best_n > 0 and r["names"] == best_n)):
            keep.add(r["name"])
    return keep


def decide(mode, rows, centroid_margin=SCOPE_CENTROID_MARGIN,
           peak_margin=SCOPE_PEAK_MARGIN):
    """The scope for one turn as (sources, note): `sources` is None when
    the turn is not scoped, and `note` says why when the rule could not
    run. Never raises into a turn."""
    if mode != "auto" or rows is None:
        return None, None
    try:
        return scope_of(rows, centroid_margin, peak_margin), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def record(mode, rows, sources, note, stats):
    """What one turn's scope was, for /stats and the trace, or None on a
    turn that was neither scoped nor stopped by a failing rule."""
    if sources is None and note is None:
        return None
    files = [r["name"] for r in (rows or []) if r["name"] is not None]
    kept = sorted(n for n in files if sources is not None and n in sources)
    out = sorted(n for n in files if sources is None or n not in sources)
    rec = {"mode": mode, "kept": kept, "out": out,
           "words": stats.get("scope_words"),
           "budget": stats.get("word_budget"),
           "excluded": stats.get("scope_excluded", 0), "note": note}
    assert tuple(rec) == SCOPED_KEYS
    return rec


def census():
    """A fresh census: turns the rule was asked about, how each came out,
    how many files each scoped turn kept, and how often each file was
    kept. Session-lifetime, never persisted."""
    return {"asked": 0, "scoped": 0, "plain": 0, "guarded": 0,
            "branches": {}, "kept": {}}


def count(census, rec):
    """One more turn the rule was asked about: `rec` is that turn's
    record, None when there was nothing to route."""
    census["asked"] += 1
    if rec is None:
        census["plain"] += 1
    elif rec["note"]:
        census["guarded"] += 1
    else:
        census["scoped"] += 1
        n = len(rec["kept"])
        census["branches"][n] = census["branches"].get(n, 0) + 1
        for name in rec["kept"]:
            census["kept"][name] = census["kept"].get(name, 0) + 1


def census_lines(census):
    """The /stats lines for a session's census; nothing when the session
    does not route."""
    if census is None:
        return []
    assert tuple(census) == CENSUS_KEYS
    out = [f"scope census: {census['asked']} turns asked, "
           f"{census['scoped']} scoped, {census['plain']} with nothing to "
           f"route, {census['guarded']} not applied"]
    if census["branches"]:
        hist = ", ".join(f"{n} x{c}" for n, c in sorted(census["branches"].items()))
        out.append(f"  files kept per scoped turn: {hist}")
    if census["kept"]:
        kept = ", ".join(f"{k!r} x{c}" for k, c in
                         sorted(census["kept"].items(), key=lambda kv: (-kv[1], kv[0])))
        out.append(f"  kept: {kept}")
    return out


def lines(rec):
    """The /stats lines for one turn's scope; nothing on a turn without
    one."""
    if rec is None:
        return []
    assert tuple(rec) == SCOPED_KEYS
    if rec["note"]:
        return [f"scope ({rec['mode']}): not applied, {rec['note']}"]
    kept = ", ".join(repr(k) for k in rec["kept"]) or "no attached file"
    out = ", ".join(repr(k) for k in rec["out"]) or "nothing"
    return [f"scope ({rec['mode']}): searched the conversation and {kept}, "
            f"kept out {out}",
            f"  {rec['words']} words in scope, budget {rec['budget']} words, "
            f"{rec['excluded']} rows held back"]
