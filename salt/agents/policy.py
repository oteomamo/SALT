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
             "stable_coverage_keys": "stable_keys",
             "coverage_gc": "coverage_gc",
             "coverage_max_keys": "coverage_max_keys",
             "tail_exclude": None}
KWARGS = tuple(SELECTION)
# the rest of the inventory acts when a sentence is written down rather
# than when one is chosen, so a decision made per selection has nothing
# to apply them to
INGEST_ONLY = tuple(sw.name for sw in SWITCHES if sw.name not in SELECTION)


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


class NullPolicy(SwitchPolicy):
    """Nobody decides. The session's own settings, every turn."""

    name = "none"
    decides = False


def check(overrides):
    """What a policy proposed, or a refusal naming what is wrong with it.

    Names are checked and nothing else here. What a value may be is a
    question about that particular switch, and the rules that set them
    are checked against their own language when they load.
    """
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
    return dict(overrides)
