# -*- coding: utf-8 -*-
"""Rules a session can be given about which turns are worth planning.

The same language as the switch rules and the same parser reading it:
comparisons over a closed set of signals, no arithmetic, no call, no
attribute, no eval. What differs is only what a rule may read and what
it may set, which is what `rules.Language` exists to carry.

`plan false` is the interesting half. A planned turn is three or more
model calls where a plain turn is one, so the rule worth writing is
usually the one that declines: this ask is small, or nobody is ready, or
the helpers are all described the same way and a fan-out was never going
to happen anyway.

READING A SIGNAL THAT ROUTING ITSELF MOVES IS THE TRAP HERE, and it is
worse than the same mistake in a switch rule. "The last round was slow,
so do not plan" fires once, the next turn is plain, and then the signal
never updates because no round ran, so the rule is answered forever by a
number from an hour ago. `turns_since_round` keeps moving when nothing
is planned, which is what a rule reading the other five has to be
qualified by, and the firing census beside `/stats` is what shows a rule
that has stopped saying anything.
"""

from salt.agents import rules
from salt.agents.route import (CLOSED_LOOP, FIELDS, RouteDecision, RouteError,
                               RoutePolicy, SIGNALS, check)

SCHEMA = "salt-route-rules/1"
# `why` is not settable by a rule: a rule already carries `expected` for
# what its author thought would happen, and the audit trail prints the
# sentence that fired, so a second free-text field would be a second
# answer to one question
SETTABLE = tuple(name for name in FIELDS if name != "why")


def route_values(rule_id, then):
    """What a route rule may set a field to, worded as a rule's author
    would want to read it."""
    try:
        check(then)
    except RouteError as exc:
        raise rules.RuleError(f"{rule_id!r}: {exc}") from None
    return then


LANGUAGE = rules.Language(schema=SCHEMA, signals=SIGNALS, settable=SETTABLE,
                          values=route_values,
                          cannot="a decision about planning a turn cannot set")


def loads(data, allow_examples=False, where="<route rules>"):
    return rules.loads(data, allow_examples=allow_examples, where=where,
                       lang=LANGUAGE)


def load(path, allow_examples=False):
    return rules.load(path, allow_examples=allow_examples, lang=LANGUAGE)


def reads_closed_loop(rule):
    """The feedback signals this rule reads without qualifying them.

    Empty is the safe shape. A rule that reads one of these and also
    constrains `turns_since_round` has said how stale a number it will
    act on, which is the difference between a rule that stops firing and
    one that freezes.
    """
    read = rules.names(rule.node)
    if "turns_since_round" in read:
        return ()
    return tuple(sorted(read & set(CLOSED_LOOP)))


class RouteRulePolicy(RoutePolicy):
    """Decide by rule: every rule whose sentence is true of this turn.

    Later rules win a field two rules both set, so the file order is
    meaningful and a narrow rule can sit under a broad one. What fired
    is kept, and how often each rule has fired is counted, because a
    rule that never fires and a rule that fires every turn are the two
    ways a rules file can be quietly useless.
    """

    name = "route rules"

    def __init__(self, rule_list, path=""):
        self.rules = tuple(rule_list)
        self.path = str(path)
        self.fired = ()
        # the live census: one count per rule, never reset, printed
        # beside what its author expected it to do
        self.fires = {rule.id: 0 for rule in self.rules}
        self.asked = 0

    @property
    def decides(self):
        return bool(self.rules)

    def decide(self, signals):
        self.asked += 1
        proposal, fired = {}, []
        for rule in self.rules:
            if rule.fires(signals):
                proposal.update(rule.then)
                fired.append(rule.id)
                self.fires[rule.id] = self.fires.get(rule.id, 0) + 1
        self.fired = tuple(fired)
        return check(proposal) if proposal else RouteDecision()

    def explain(self):
        by_id = {rule.id: rule for rule in self.rules}
        return tuple({"id": rule_id, "when": by_id[rule_id].when,
                      "then": dict(by_id[rule_id].then)}
                     for rule_id in self.fired if rule_id in by_id)

    def census(self):
        """Every rule, how often it has fired, and what its author said
        it was for. The one-session read that replaces a sweep."""
        return tuple({"id": rule.id, "fired": self.fires.get(rule.id, 0),
                      "asked": self.asked, "expected": rule.expected,
                      "feedback": reads_closed_loop(rule)}
                     for rule in self.rules)
