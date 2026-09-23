# -*- coding: utf-8 -*-
"""Who decides how one turn's memory is selected.

Every memory switch already travels as a keyword on the call that uses
it rather than being baked into the session, which is what makes a
per-turn decision possible at all: something can vary a switch for one
selection and leave the session it belongs to untouched.

A policy is asked once per turn, is given the snapshot and nothing else,
and answers with the switches it wants changed for that call. An empty
answer is the normal answer. The session's own settings are the
starting point every turn, so a decision cannot accumulate: whatever a
policy did last turn is gone by this one unless it decides the same
thing again.

The default policy decides nothing and is not even asked, so a session
that was never given one selects exactly as it did before this file
existed.
"""

import json
import math

from salt.agents.snapshot import SWITCHES

# what a decision may set, and the compress() keyword each one travels
# as. tail_exclude is here with no keyword of its own because it does
# not reach compress: it decides what that call is told to leave out
SELECTION = {"coverage_half_life": "coverage_half_life",
             "coverage_decay_docs": "coverage_decay_docs",
             "shift_damping": "shift_damping",
             "shift_margin": "shift_margin",
             "shift_query_boost": "shift_query_boost",
             "per_source_themes": "per_source_themes",
             "query_identifiers": "query_identifiers",
             "episode_gap": "episode_gap",
             "assistant_weight": "assistant_weight",
             "row_coverage": "row_coverage",
             "stable_coverage_keys": "stable_keys",
             "coverage_gc": "coverage_gc",
             "coverage_max_keys": "coverage_max_keys",
             "tail_exclude": None}
KWARGS = tuple(SELECTION)
# the rest of the inventory acts when a sentence is written down rather
# than when one is chosen, so a decision made per selection has nothing
# to apply them to
INGEST_ONLY = tuple(sw.name for sw in SWITCHES if sw.name not in SELECTION)


def selection_kwargs(value_of):
    """The compress() keywords a selection runs under, each read by its
    switch's name. Every call that selects builds them here, so a turn
    and the pieces it hands out cannot select under different lists."""
    return {kwarg: value_of(name) for name, kwarg in SELECTION.items()
            if kwarg is not None}


def _finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _off_or(test):
    return lambda value: value is None or (_finite(value) and test(value))


_YES_NO = (lambda value: isinstance(value, bool), "true or false")

VALUES = {
    "coverage_half_life": (_off_or(lambda v: v > 0),
                           "null or a number of turns above 0"),
    "coverage_decay_docs": _YES_NO,
    "shift_damping": (_off_or(lambda v: 0 < v < 1),
                      "null or a scale strictly between 0 and 1"),
    "shift_margin": (lambda v: _finite(v) and v >= 0,
                     "a cosine drop of 0 or more"),
    "shift_query_boost": (lambda v: _finite(v) and v >= 1,
                          "a multiplier of 1 or more"),
    "per_source_themes": _YES_NO,
    "query_identifiers": _YES_NO,
    "episode_gap": (_off_or(lambda v: v > 0),
                    "null or a number of hours above 0"),
    "assistant_weight": (_off_or(lambda v: 0 < v < 1),
                         "null or a weight strictly between 0 and 1"),
    "row_coverage": _YES_NO,
    "stable_coverage_keys": _YES_NO,
    "coverage_gc": _YES_NO,
    "coverage_max_keys": (lambda v: v is None or (
        isinstance(v, int) and not isinstance(v, bool)),
        "null or a whole number of keys"),
    "tail_exclude": _YES_NO,
}
if set(VALUES) != set(SELECTION):
    raise AssertionError("a settable switch has no value it is held to")


def value_error(name, value):
    """Why this value cannot be given to this switch, or None if it can.
    The same values a session refuses at launch."""
    ok, wanted = VALUES[name]
    if ok(value):
        return None
    try:
        shown = json.dumps(value)
    except (TypeError, ValueError):
        shown = repr(value)
    return f"{name} takes {wanted}, not {shown}"


class PolicyError(Exception):
    """A policy asked for something a turn cannot give it."""


class SwitchPolicy:
    """The seam. One question per turn, answered with a dict.

    `decide(snapshot)` returns the switches to change for that turn's
    selection, keyed by the switch's own name. An empty dict means the
    session's settings stand, which is what every policy returns most
    of the time.
    """

    name = "policy"
    # whether asking is worth the snapshot it costs. A policy that never
    # decides anything is not asked at all, which is how the default
    # path stays exactly as cheap as it was
    decides = True

    def decide(self, snapshot):
        return {}

    def bind(self, state):
        """The session this policy decides for, handed over once it
        exists. A policy that needs nothing from it ignores this."""
        return self

    def explain(self):
        """Why the last decision came out as it did, one entry per
        reason. Empty from a policy that has no reasons to give, which
        is every policy that is not written down."""
        return ()


class NullPolicy(SwitchPolicy):
    """Nobody decides. The session's own settings, every turn."""

    name = "none"
    decides = False


def check(overrides):
    """What a policy proposed, or a refusal naming what is wrong with it:
    a name no turn can set, or a value that switch does not take."""
    if not isinstance(overrides, dict):
        raise PolicyError(f"a policy answers with a dict of switches to "
                          f"change, and this one answered with "
                          f"{type(overrides).__name__}")
    unknown = [name for name in overrides if name not in SELECTION]
    if unknown:
        ingest = [name for name in unknown if name in INGEST_ONLY]
        why = (f" {ingest} act when a sentence is remembered rather than "
               f"when one is selected, so a per-turn decision cannot set "
               f"them." if ingest else "")
        raise PolicyError(f"a decision named {unknown}, which is not "
                          f"something a turn's selection can set.{why} "
                          f"Allowed: {', '.join(KWARGS)}")
    for name, value in overrides.items():
        why = value_error(name, value)
        if why:
            raise PolicyError(why)
    return dict(overrides)
