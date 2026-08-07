# -*- coding: utf-8 -*-
"""Rules a session can be given about how it selects.

A rule is a sentence about the session and a switch to change when that
sentence is true: `n_attachments > 0` means profile the files apart from
the conversation. The whole language is comparison. There is no
arithmetic, no function call, no attribute, no way to name anything but
the signals a session reports about itself, and no eval anywhere near
it: the text is tokenized, parsed into a tree of comparisons by hand,
and walked. A rules file is configuration, and configuration that can
run code is not configuration.

Everything a rule could get wrong is found when the file loads rather
than mid-conversation. A signal nobody reports, a switch a turn cannot
set, an expression that does not parse, two rules with one id, and a set
that could turn on two switches known to cancel each other are all
refused at the door, naming what was wrong and what was allowed.

A signal a session cannot report reads as nothing, and a comparison
against nothing is false. A rule about attachments does not fire for a
conversation that cannot say whether it has any, which is the only
honest thing for it to do.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from salt.agents.policy import KWARGS, SELECTION, SwitchPolicy, check
from salt.agents.snapshot import RULE_SIGNALS

SCHEMA = "salt-switch-rules/1"
RULE_KEYS = ("id", "when", "then", "expected", "evidence", "example")
REQUIRED = ("id", "when", "then")
LITERALS = {"true": True, "false": False, "null": None}
WORDS = ("and", "or", "not")
ORDERED = ("<", "<=", ">", ">=")
OPS = ORDERED + ("==", "!=")
_TOKEN = re.compile(r"""\s*(?:(?P<op><=|>=|==|!=|<|>)
                            |(?P<par>[()])
                            |(?P<num>-?\d+(?:\.\d+)?)
                            |(?P<name>[A-Za-z_][A-Za-z0-9_]*))""", re.X)


class RuleError(Exception):
    """A rules file said something this salt will not act on."""


def tokens(text):
    """The expression as (kind, value, position), or a refusal pointing
    at the first character that is not part of the language."""
    out, pos = [], 0
    while pos < len(text):
        if text[pos].isspace():
            pos += 1
            continue
        match = _TOKEN.match(text, pos)
        if match is None:
            raise RuleError(f"{text!r} has {text[pos]!r} at position {pos}, "
                            f"which is not part of the language")
        kind = match.lastgroup
        out.append((kind, match.group(kind), match.start(kind)))
        pos = match.end()
    return out


class _Parser:
    """One expression, read left to right and never re-read."""

    def __init__(self, text):
        self.text = text
        self.tokens = tokens(text)
        self.at = 0

    def peek(self):
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def take(self):
        token = self.peek()
        if token is None:
            raise RuleError(f"{self.text!r} stops early, with something "
                            f"still expected after it")
        self.at += 1
        return token

    def word(self, name):
        token = self.peek()
        return token is not None and token[0] == "name" and token[1] == name

    def parse(self):
        node = self.disjunction()
        left = self.peek()
        if left is not None:
            raise RuleError(f"{self.text!r} has {left[1]!r} left over at "
                            f"position {left[2]}")
        return node

    def disjunction(self):
        node = self.conjunction()
        while self.word("or"):
            self.take()
            node = ("or", node, self.conjunction())
        return node

    def conjunction(self):
        node = self.negation()
        while self.word("and"):
            self.take()
            node = ("and", node, self.negation())
        return node

    def negation(self):
        if self.word("not"):
            self.take()
            return ("not", self.negation())
        return self.comparison()

    def comparison(self):
        left = self.operand()
        token = self.peek()
        if token is not None and token[0] == "op":
            self.take()
            return ("cmp", token[1], left, self.operand())
        return left

    def operand(self):
        kind, value, where = self.take()
        if kind == "par":
            if value != "(":
                raise RuleError(f"{self.text!r} closes a bracket at "
                                f"{where} that was never opened")
            node = self.disjunction()
            closing = self.take()
            if closing[0] != "par" or closing[1] != ")":
                raise RuleError(f"{self.text!r} opens a bracket it never "
                                f"closes")
            return node
        if kind == "num":
            return ("lit", float(value) if "." in value else int(value))
        if value in LITERALS:
            return ("lit", LITERALS[value])
        if value in WORDS:
            raise RuleError(f"{self.text!r} uses {value!r} at position "
                            f"{where} where a signal or a value belongs")
        return ("name", value)


def parse(text):
    if not isinstance(text, str) or not text.strip():
        raise RuleError("a rule's 'when' is an expression, and this one is "
                        f"{text!r}")
    return _Parser(text).parse()


def names(node):
    """Every signal an expression reads."""
    if node[0] == "name":
        return {node[1]}
    return set().union(*(names(part) for part in node[1:]
                         if isinstance(part, tuple))) if len(node) > 1 \
        else set()


def truth(value):
    """A bare signal read as a yes or a no. Nothing is a no: a session
    that cannot say has not said yes."""
    return bool(value) if value is not None else False


def compare(op, left, right):
    if op in ORDERED and (left is None or right is None):
        # an order between a number and nothing is not false, it is
        # unanswerable, and a rule that cannot be answered does not fire
        return False
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    return left >= right


def evaluate(node, signals):
    """Whether this expression is true of that snapshot."""
    kind = node[0]
    if kind == "lit":
        return node[1]
    if kind == "name":
        return signals.get(node[1])
    if kind == "not":
        return not truth(evaluate(node[1], signals))
    if kind == "and":
        return truth(evaluate(node[1], signals)) and truth(
            evaluate(node[2], signals))
    if kind == "or":
        return truth(evaluate(node[1], signals)) or truth(
            evaluate(node[2], signals))
    return compare(node[1], evaluate(node[2], signals),
                   evaluate(node[3], signals))


@dataclass(frozen=True)
class Rule:
    """One sentence about a session, and what to do when it is true."""

    id: str
    when: str
    then: dict
    node: tuple = ()
    expected: str = ""
    evidence: str = ""
    example: bool = False

    def fires(self, signals):
        return truth(evaluate(self.node, signals))


def read_rule(raw, index):
    """One entry of a rules file, checked against everything it could
    have got wrong before anything is done with it."""
    where = f"rule {index + 1}"
    if not isinstance(raw, dict):
        raise RuleError(f"{where} is a {type(raw).__name__}, and a rule is "
                        f"an object")
    unknown = sorted(set(raw) - set(RULE_KEYS))
    if unknown:
        raise RuleError(f"{where} carries {unknown}, which a rule has no "
                        f"place for. A rule takes: {', '.join(RULE_KEYS)}")
    missing = [key for key in REQUIRED if key not in raw]
    if missing:
        raise RuleError(f"{where} has no {missing}, and a rule needs "
                        f"{', '.join(REQUIRED)}")
    rule_id = raw["id"]
    if not isinstance(rule_id, str) or not rule_id.strip():
        raise RuleError(f"{where} has {rule_id!r} for an id, and an id is "
                        f"a name the audit trail can print")
    node = parse(raw["when"])
    unreadable = sorted(names(node) - set(RULE_SIGNALS))
    if unreadable:
        raise RuleError(
            f"{rule_id!r} reads {unreadable}, which is not something a "
            f"session reports about itself. It may read: "
            f"{', '.join(RULE_SIGNALS)}")
    then = raw["then"]
    if not isinstance(then, dict) or not then:
        raise RuleError(f"{rule_id!r} changes nothing, and a rule that "
                        f"fires has to change something")
    unsettable = sorted(set(then) - set(SELECTION))
    if unsettable:
        raise RuleError(
            f"{rule_id!r} sets {unsettable}, which a turn's selection "
            f"cannot set. It may set: {', '.join(KWARGS)}")
    for name, value in then.items():
        if not isinstance(value, (bool, int, float, type(None))):
            raise RuleError(f"{rule_id!r} sets {name} to "
                            f"{type(value).__name__}, and a switch takes a "
                            f"number, a yes or no, or nothing")
    return Rule(id=rule_id, when=raw["when"], then=dict(then), node=node,
                expected=raw.get("expected", "") or "",
                evidence=raw.get("evidence", "") or "",
                example=bool(raw.get("example", False)))


# switch pairs that cancel each other, and why. Quoted from the measured
# reasoning rather than restated, because a refusal a reader cannot act
# on is a refusal that gets worked around
CONFLICTS = (
    ("coverage_gc", "stable_coverage_keys",
     "a frozen keyword order reconciles orphans as they appear, so the "
     "collector has no grace window left to work in and turning both on "
     "measured identical to turning on only the keys"),
    ("coverage_gc", "coverage_half_life",
     "a decaying seed floor already collects what the collector would "
     "chase, so the two overlap and only one of them can be credited "
     "with the result"),
)


def guard(rules):
    """Refuse a set that could turn on two switches known to cancel.

    Whether two rules can both fire is not knowable from their text, so
    a set that CONTAINS both is treated as a set that could apply both.
    A rule set is written once and read for a long time, and the cost of
    being wrong here is a switch that quietly does nothing.
    """
    setters = {}
    for rule in rules:
        for name, value in rule.then.items():
            if value:
                setters.setdefault(name, []).append(rule.id)
    for one, other, why in CONFLICTS:
        if setters.get(one) and setters.get(other):
            raise RuleError(
                f"{sorted(set(setters[one]))} turns on {one} and "
                f"{sorted(set(setters[other]))} turns on {other}, and "
                f"{why}. Keep one of them.")
    return rules


def loads(data, allow_examples=False, where="<rules>"):
    """A rules document as rules, or a refusal naming what is wrong."""
    if not isinstance(data, dict):
        raise RuleError(f"{where} is a {type(data).__name__}, and a rules "
                        f"file is an object with a version and a list of "
                        f"rules")
    version = data.get("version")
    if version != SCHEMA:
        raise RuleError(f"{where} is written as {version!r}, and this salt "
                        f"reads {SCHEMA}")
    raw = data.get("rules")
    if not isinstance(raw, list):
        raise RuleError(f"{where} has no list of rules")
    rules = [read_rule(entry, i) for i, entry in enumerate(raw)]
    seen = {}
    for rule in rules:
        if rule.id in seen:
            raise RuleError(f"{where} has two rules called {rule.id!r}, and "
                            f"an audit trail that names one of them could "
                            f"mean either")
        seen[rule.id] = rule
    live = [r for r in rules if allow_examples or not r.example]
    return guard(live)


def load(path, allow_examples=False):
    text = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise RuleError(f"{path} is not readable as JSON: {exc}")
    return loads(data, allow_examples=allow_examples, where=str(path))


class RulePolicy(SwitchPolicy):
    """Decide by rule: every rule whose sentence is true of this turn.

    Later rules win a switch two rules both set, which is what makes the
    file order meaningful and lets a narrow rule sit under a broad one.
    What fired is kept so the turn can say why it selected as it did.
    """

    name = "rules"

    def __init__(self, rules, path=""):
        self.rules = tuple(rules)
        self.path = str(path)
        self.fired = ()

    @property
    def decides(self):
        return bool(self.rules)

    def decide(self, signals):
        overrides, fired = {}, []
        for rule in self.rules:
            if rule.fires(signals):
                overrides.update(rule.then)
                fired.append(rule.id)
        self.fired = tuple(fired)
        return check(overrides)

    def explain(self):
        """The rules that fired last, each with the sentence that was
        true and what it changed. What the audit trail is made of."""
        by_id = {rule.id: rule for rule in self.rules}
        return tuple({"id": rule_id, "when": by_id[rule_id].when,
                      "then": dict(by_id[rule_id].then)}
                     for rule_id in self.fired if rule_id in by_id)
