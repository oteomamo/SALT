# -*- coding: utf-8 -*-
"""What a decision about the memory switches is allowed to read.

Two things live here, and they are the read half of the agent seam: the
numbers a session can be judged by, and the switches those numbers are
judged in order to set. Both are closed sets, pinned once, because a
rule that can name any attribute of a session is a rule nobody can
verify and a decision nobody can reproduce.

A snapshot is flat, typed and cheap: every value is a number, a boolean
or None, and None means "this session cannot say", never zero. A
conversation held by the MCP server has no chat model and no verbatim
tail, so the signals that describe those come back as None there, and a
rule that needs them simply does not fire.

Nothing here computes anything the engine already computed. The signals
that come from a compression are read off the last one, so a snapshot
costs a dictionary lookup and never a pass over the trie.
"""

import time
from dataclasses import dataclass

SCHEMA = "salt-snapshot/1"

# the closed set, in the order a reader meets them: what the session
# holds, what its last compression saw, and what is still in flight
KEYS = ("n_sentences", "n_alive", "masked", "alive_ratio", "n_turns",
        "live_words", "n_attachments", "attachment_words",
        "attachment_share", "near_dups", "session_age_s", "budget_pct",
        "orphan_keys", "orphan_mass", "orphan_share", "drift_cos",
        "drift_ema", "topic_shift", "coverage_keys", "tail_occupancy",
        "model_window", "pending_ingest")

# what a rule may name. The same set, deliberately: a signal a session
# reports and a signal a decision may read apart would be two answers to
# one question, and the check below is what keeps them one
RULE_SIGNALS = KEYS


@dataclass(frozen=True)
class Switch:
    """One memory switch: how it is set, what it does, and which number
    reports whether it did anything."""

    name: str
    flag: str
    default: object
    stats_key: str
    what: str


# every switch travels as a per-call kwarg to compress() or add_turn(),
# which is why `name` is the kwarg and not the flag: a decision sets the
# call, never the session
SWITCHES = (
    Switch("coverage_half_life", "--coverage-half-life", None,
           "coverage_half_life",
           "let a surfaced theme's suppression fade over quiet turns"),
    Switch("coverage_decay_docs", "--coverage-decay-docs", False,
           "coverage_half_life", "let attached files decay too"),
    Switch("shift_damping", "--shift-damping", None, "shift_damped",
           "scale down stale suppression when the question pivots"),
    Switch("shift_margin", "--shift-margin", 0.12, "drift_cos",
           "how far a question must pivot to count as a shift"),
    Switch("shift_query_boost", "--shift-query-boost", 1.5, "topic_shift",
           "how much the query weighs on a turn that pivoted"),
    Switch("per_source_themes", "--per-source-themes", False, "theme_scope",
           "profile the conversation and each file separately"),
    Switch("query_identifiers", "--query-identifiers", False,
           "query_identifiers",
           "let the question's dates, versions and numbers match memory"),
    Switch("episode_gap", "--episode-gap", None, "episodes",
           "split memory into time episodes, each its own branch"),
    Switch("assistant_weight", "--assistant-weight", None,
           "down_weighted_rows",
           "count model-authored rows as a fraction in the theme profile"),
    Switch("row_coverage", "--row-coverage", False, "coverage_rows",
           "remember what was shown per sentence, not per tree branch"),
    Switch("stable_coverage_keys", "--stable-coverage-keys", False,
           "coverage_seed_matched",
           "freeze the keyword order so discounts survive growth"),
    Switch("coverage_gc", "--coverage-gc", False, "coverage_gc_dropped",
           "collect coverage keys nothing live matches any more"),
    Switch("coverage_max_keys", "--coverage-max-keys", None,
           "coverage_capped_dropped", "hard limit on the coverage table"),
    Switch("dedup_cos", "--dedup-cos", None, "near_dups",
           "skip a new sentence too close to one the same speaker said"),
    Switch("max_sentences", "--max-sentences", None, "masked",
           "cap how many conversation sentences stay selectable"),
    Switch("tail_exclude", "--no-tail-exclude", True, "excluded_sent",
           "leave out what the model is already reading verbatim"),
)


def _stats(state, stats=None):
    """The compression this session was last given, from the caller or
    from wherever that session keeps it. Empty is a fact, not a fault: a
    conversation nobody has read from has no compression to describe."""
    if stats is not None:
        return stats
    return getattr(state, "last_stats", None) or {}


def _attachment_words(trie):
    return sum(w for w, s, a in zip(trie.n_words, trie.sources, trie.alive)
               if s and a)


def _age(trie):
    """How long this conversation has been going, from the oldest thing
    in it. None for a session written before turn timestamps existed."""
    stamps = [t for t in trie.timestamps if t]
    return round(time.time() - min(stamps), 1) if stamps else None


def _tail_occupancy(state):
    """How full the verbatim tail is, 0 to 1. None where there is no
    tail at all, which is every session the MCP server holds. The tail
    list holds messages and compacts past two per tail_max exchange, so
    two per exchange is the capacity the fill is measured against."""
    tail = getattr(state, "tail", None)
    cap = getattr(state, "tail_max", None)
    if tail is None or not cap:
        return None
    return round(min(1.0, len(tail) / (2.0 * cap)), 3)


def _window(state):
    runner = getattr(state, "runner", None)
    return runner.input_budget() if runner is not None else None


def _share(part, whole):
    """One quantity as a fraction of another, or None where the whole is
    nothing. Proportions are signals rather than something a rule works
    out, because the language a rule is written in compares and does not
    calculate."""
    return round(part / float(whole), 3) if whole else None


def _orphan_share(trie, orphan_mass):
    """How much of the accumulated suppression no longer matches
    anything live, 0 to 1. The mass alone is a count in the thousands
    that grows with the conversation, so a threshold against it means
    nothing across sessions of different sizes; this is the same fact
    as a share of the whole coverage table. Clamped because the mass is
    read off the last compression while the table is read now, and a
    commit that just collected orphans can leave the old mass briefly
    larger than what survived it."""
    if orphan_mass is None:
        return None
    share = _share(orphan_mass, sum(trie.coverage.values()))
    return None if share is None else min(1.0, share)


def snapshot(state, stats=None):
    """This session as the closed set of signals a decision may read.

    One definition with two consumers: the MCP server answers with it,
    and the switch policy decides on it. They must never drift, so
    neither builds its own.
    """
    trie = state.trie
    s = _stats(state, stats)
    ingest = getattr(state, "ingest", None)
    attachment_words = _attachment_words(trie)
    out = {"n_sentences": trie.n_sentences,
           "n_alive": trie.n_alive,
           "masked": trie.n_masked,
           "alive_ratio": _share(trie.n_alive, trie.n_sentences),
           "n_turns": trie.n_turns,
           "live_words": trie.live_words,
           "n_attachments": len(trie.attached_sources),
           "attachment_words": attachment_words,
           "attachment_share": _share(attachment_words, trie.live_words),
           "near_dups": trie.n_near_dups,
           "session_age_s": _age(trie),
           "budget_pct": getattr(state, "budget", None),
           "orphan_keys": s.get("coverage_orphan_keys"),
           "orphan_mass": s.get("coverage_orphan_mass"),
           "orphan_share": _orphan_share(trie,
                                         s.get("coverage_orphan_mass")),
           "drift_cos": s.get("drift_cos"),
           "drift_ema": trie.drift_ema,
           "topic_shift": s.get("topic_shift"),
           "coverage_keys": len(trie.coverage),
           "tail_occupancy": _tail_occupancy(state),
           "model_window": _window(state),
           "pending_ingest": ingest.pending if ingest is not None else None}
    if tuple(out) != KEYS:
        raise AssertionError("the snapshot grew a key nothing pinned")
    if tuple(out) != RULE_SIGNALS:
        raise AssertionError(
            "what a session reports and what a rule may read have drifted")
    return out


def default_switches():
    """Every switch at its shipped value. What a caller that sets none
    of them is running under, which is the MCP server today."""
    return {sw.name: sw.default for sw in SWITCHES}


def switch_values(state):
    """What this session currently passes for each switch. Read off the
    state by the kwarg's own name, so a switch that exists and a switch
    that is reported are the same list by construction."""
    values = default_switches()
    for sw in SWITCHES:
        if hasattr(state, sw.name):
            values[sw.name] = getattr(state, sw.name)
    return values


def switch_inventory(values=None):
    """The switches, what each one is set to, and the number in the
    session's own statistics that says whether it did anything."""
    current = default_switches() if values is None else dict(values)
    return [{"name": sw.name, "flag": sw.flag, "value": current.get(sw.name),
             "default": sw.default, "stats_key": sw.stats_key,
             "what": sw.what, "changed": current.get(sw.name) != sw.default}
            for sw in SWITCHES]
