# -*- coding: utf-8 -*-
"""The one thing an orchestrating model is allowed to say.

A model deciding what to delegate has to say so in a form the session
can act on. That form is a single JSON object: either an answer, or a
list of tasks with the worker each one goes to. Nothing else is a
directive, and half of one is not a directive either.

The reading is tolerant at the front and strict after it. A local model
puts its reasoning above the object, or fences it in markdown, or opens
with a sentence about what it is about to do, and none of that is worth
a failed round. What is inside the object is held to the letter: an
unknown key, a missing task, a subtask list past the cap, all refuse
with a reason the caller can act on rather than a message it has to
read.

Refusing is the normal path, not the exception. The caller re-prompts
once with the reason and then gives up on the directive and keeps the
model's own words, so this module never guesses at what was meant.

A worker's answer is never parsed here. Worker output is quoted
material, and text that looks like a directive inside it is text.
"""

import json
import re
from dataclasses import dataclass, field

SCHEMA = "salt-agent-directive/1"
ACTIONS = ("answer", "delegate")
# a plan wider than this is a model that has misunderstood the job, not
# an ambitious plan. Bounded here so nothing downstream has to be
MAX_SUBTASKS = 8
TOP_KEYS = {"version", "action", "answer", "subtasks"}
SUBTASK_KEYS = {"id", "task", "target", "query", "budget_pct", "max_tokens"}
REQUIRED_SUBTASK_KEYS = ("id", "task", "target")
# reasons a caller branches on. The sentence beside each one is for a
# person and for the re-prompt; the code is what the caller reads
REASONS = ("no_json", "bad_json", "not_an_object", "wrong_version",
           "bad_action", "unknown_keys", "no_answer", "no_subtasks",
           "too_many_subtasks", "bad_subtask", "duplicate_id", "bad_number")
_THINK_OPEN = re.compile(r"<think\b[^>]*>", re.I)
_THINK_CLOSE = re.compile(r"</think\s*>", re.I)


class ProtocolError(ValueError):
    """A reply that is not a directive, and the machine-usable reason."""

    def __init__(self, reason, detail=""):
        self.reason = reason if reason in REASONS else "bad_json"
        self.detail = str(detail)
        super().__init__(f"{self.reason}: {self.detail}")


@dataclass(frozen=True)
class Subtask:
    """One piece of work, and who it goes to."""

    id: str
    task: str
    target: str
    query: str = None
    budget_pct: float = None
    max_tokens: int = None

    @property
    def context_query(self):
        """What its context is selected for: the task, unless the plan
        named a better line to search the conversation with."""
        return self.query or self.task


@dataclass(frozen=True)
class Directive:
    """What the orchestrator decided: answer now, or delegate first."""

    action: str
    answer: str = ""
    subtasks: tuple = field(default_factory=tuple)

    @property
    def delegates(self):
        return self.action == "delegate"

    @property
    def targets(self):
        """Every worker this plan needs, once each, in plan order."""
        seen = []
        for sub in self.subtasks:
            if sub.target not in seen:
                seen.append(sub.target)
        return tuple(seen)


def strip_think(text, reasoning_content=None):
    """Reasoning removed, answer kept.

    Counted rather than matched, because a model that opens a second
    think block inside the first would leave the tail of its own
    reasoning behind under a non-greedy match, and that tail then reads
    as the answer. An unclosed block is a reply that ran out of room
    mid-thought, so everything from the opening tag on is thinking too.

    `reasoning_content` is what a server hands back beside the answer
    when it separates the two itself. It is dropped on the floor here:
    what is kept is whatever the model actually said.
    """
    if not text:
        return ""
    out, depth, i = [], 0, 0
    while i < len(text):
        opened = _THINK_OPEN.match(text, i)
        closed = _THINK_CLOSE.match(text, i)
        if opened:
            depth += 1
            i = opened.end()
        elif closed:
            depth = max(0, depth - 1)
            i = closed.end()
        else:
            if depth == 0:
                out.append(text[i])
            i += 1
    if depth > 0:
        # the reply ended inside a thought, so nothing after the last
        # opening tag was ever an answer
        cut = list(_THINK_OPEN.finditer(text))[-1].start()
        return strip_think(text[:cut])
    return "".join(out).strip()


def reply_text(text, reasoning_content=None):
    """What a model said, as opposed to what it thought. One place, so
    a reply on its way into memory and a reply on its way into the
    parser are cut the same."""
    return strip_think(text, reasoning_content)


def find_object(text):
    """The first balanced JSON object in `text`, as a string.

    Scanned rather than matched: braces inside strings are text, and an
    escaped quote does not end one. Returns None when there is nothing
    that could be an object.
    """
    start = text.find("{")
    if start < 0:
        return None
    depth, in_string, escaped = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _text(raw, name, reason="bad_subtask"):
    if not isinstance(raw, str) or not raw.strip():
        raise ProtocolError(reason, f"{name} must be a non-empty string, "
                                    f"got {raw!r}")
    return raw.strip()


def _number(raw, name, kind, low, high=None):
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, kind):
        raise ProtocolError("bad_number",
                            f"{name} must be a number, got {raw!r}")
    if raw <= low or (high is not None and raw > high):
        raise ProtocolError("bad_number",
                            f"{name} is out of range, got {raw!r}")
    return raw


def _subtask(raw, index):
    if not isinstance(raw, dict):
        raise ProtocolError("bad_subtask",
                            f"subtask {index} is not an object, got {raw!r}")
    unknown = sorted(set(raw) - SUBTASK_KEYS)
    if unknown:
        raise ProtocolError("unknown_keys",
                            f"subtask {index} has unknown keys {unknown} "
                            f"(allowed: {sorted(SUBTASK_KEYS)})")
    for key in REQUIRED_SUBTASK_KEYS:
        if key not in raw:
            raise ProtocolError("bad_subtask",
                                f"subtask {index} has no {key}")
    return Subtask(
        id=_text(raw["id"], f"subtask {index} id"),
        task=_text(raw["task"], f"subtask {index} task"),
        target=_text(raw["target"], f"subtask {index} target"),
        query=(None if raw.get("query") is None
               else _text(raw["query"], f"subtask {index} query")),
        budget_pct=_number(raw.get("budget_pct"),
                           f"subtask {index} budget_pct", (int, float), 0, 1),
        max_tokens=_number(raw.get("max_tokens"),
                           f"subtask {index} max_tokens", int, 0))


def parse_directive(text):
    """One model reply read as a directive, or a reason it is not one.

    The think strip runs first, so a model that reasons in the open is
    read by what it decided rather than by what it considered.
    """
    body = find_object(strip_think(text or ""))
    if body is None:
        raise ProtocolError("no_json", "the reply carries no JSON object")
    try:
        raw = json.loads(body)
    except ValueError as exc:
        raise ProtocolError("bad_json", str(exc)) from exc
    if not isinstance(raw, dict):
        raise ProtocolError("not_an_object",
                            f"a directive is an object, got {type(raw).__name__}")
    unknown = sorted(set(raw) - TOP_KEYS)
    if unknown:
        raise ProtocolError("unknown_keys",
                            f"unknown keys {unknown} (allowed: "
                            f"{sorted(TOP_KEYS)})")
    version = raw.get("version", SCHEMA)
    if version != SCHEMA:
        raise ProtocolError("wrong_version",
                            f"this salt reads {SCHEMA!r}, the reply says "
                            f"{version!r}")
    action = raw.get("action")
    if action not in ACTIONS:
        raise ProtocolError("bad_action",
                            f"action must be one of {list(ACTIONS)}, got "
                            f"{action!r}")
    if action == "answer":
        return Directive(action=action,
                         answer=_text(raw.get("answer"), "answer",
                                      "no_answer"))

    subtasks = raw.get("subtasks")
    if not isinstance(subtasks, list) or not subtasks:
        raise ProtocolError("no_subtasks",
                            "a delegating directive needs a non-empty "
                            "subtasks list")
    if len(subtasks) > MAX_SUBTASKS:
        raise ProtocolError("too_many_subtasks",
                            f"{len(subtasks)} subtasks, and at most "
                            f"{MAX_SUBTASKS} are allowed")
    parsed = tuple(_subtask(item, i) for i, item in enumerate(subtasks))
    ids = [sub.id for sub in parsed]
    if len(set(ids)) != len(ids):
        raise ProtocolError("duplicate_id",
                            f"subtask ids must differ, got {ids}")
    return Directive(action=action, subtasks=parsed)


def example_directive(targets=("worker",)):
    """A directive a model can be shown when it cannot be given a
    schema. One example beats a paragraph of prose about JSON."""
    return json.dumps(
        {"version": SCHEMA, "action": "delegate",
         "subtasks": [{"id": "1", "task": "what the worker should do",
                       "target": targets[0]}]},
        indent=2)


@dataclass(frozen=True)
class DirectiveOutcome:
    """One ask for a directive, and what it took to get one.

    `failures` is what a session counts: a round that needed repairing
    cost one, a round that fell back cost two. `fell_back` says the
    directive below was built here rather than returned by the model.
    """

    directive: Directive
    raw: str = ""
    failures: int = 0
    reasons: tuple = field(default_factory=tuple)
    fell_back: bool = False

    @property
    def repaired(self):
        return self.failures == 1 and not self.fell_back


def fallback(text):
    """A model that would not produce a directive still said something,
    and what it said is the answer. Failing closed keeps the turn: the
    session loses the delegation, never the reply.

    Reasoning is not an answer. A reply that is empty once the thinking
    comes out falls back to nothing, and the caller sees a directive
    with no answer in it rather than a private trace presented as one.
    """
    return Directive(action="answer", answer=strip_think(text))


def ask_directive(send, messages, guided=False):
    """Ask for a directive, repair once, then take what was said.

    `send(messages, guided)` returns the model's text. Exactly two
    attempts are possible, which is the point: a model that cannot
    follow the schema must cost a bounded amount before the round goes
    on without it. The repair quotes the actual fault, so the second
    ask corrects something rather than asking harder.
    """
    reasons, raw = [], ""
    for attempt in (1, 2):
        raw = send(messages, guided=guided and attempt == 1) or ""
        try:
            return DirectiveOutcome(directive=parse_directive(raw), raw=raw,
                                    failures=len(reasons),
                                    reasons=tuple(reasons))
        except ProtocolError as exc:
            reasons.append(exc.reason)
            if attempt == 2:
                break
            messages = list(messages) + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": repair_prompt(exc)}]
    return DirectiveOutcome(directive=fallback(raw), raw=raw,
                            failures=len(reasons), reasons=tuple(reasons),
                            fell_back=True)


def repair_prompt(error):
    """What to say to a model whose reply was not a directive. The
    reason is quoted so the model is corrected on the actual fault
    rather than told to try harder."""
    return (f"That reply could not be read as a directive ({error.reason}: "
            f"{error.detail}). Return only the JSON object, with no prose "
            f"around it and nothing after it.")
