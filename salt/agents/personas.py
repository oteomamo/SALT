# -*- coding: utf-8 -*-
"""Personas: roles a helper can be asked to hold.

A roster names weights. A persona names a way of working: the same
model asked to explain, or to write code, or to check a drafted reply
against what the session remembers. One persona is one markdown file -
a few facts between two ``---`` lines, then the system prompt it works
under::

    ---
    name: explainer
    worker: chat
    role: target
    notes: "teaching and explanation: step by step reasoning, analogies,
    the why behind a fact, worked examples"
    ---
    You are the EXPLAINER helper. ...

``worker`` says whose weights the persona rides: a roster entry's name,
or the reserved name ``chat`` for the session's own chat model, which is
what lets a machine with one card - or none - still hold a roster of
roles. ``role`` is ``target`` (the planner may hand it a piece of the
work) or ``verify`` (it is never a target; it checks the round's reply
instead). ``notes`` is what the planner reads to choose a helper, so a
target without notes does not load, and two personas worded alike are a
choice the planner cannot make - the same rule the fitted roster holds
its workers to.

Loading only parses and validates, exactly like the roster: whether the
named worker exists is a fact about a session, so it is checked where a
session binds personas to its roster, not here.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# the session's own chat model, as a worker a persona may name. Reserved:
# no roster entry and no persona may claim it as a name of their own
CHAT_WORKER = "chat"
PERSONA_ROLES = ("target", "verify")

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_KEYS = {"name", "worker", "role", "notes"}
_MARK = "---"


class PersonaError(Exception):
    """User-facing persona failure (bad file, bad field, duplicate)."""


@dataclass(frozen=True)
class Persona:
    name: str
    worker: str
    role: str
    notes: str
    body: str
    path: str

    @property
    def rides_chat(self):
        return self.worker == CHAT_WORKER

    @property
    def is_target(self):
        return self.role == "target"


def _fail(path, msg):
    raise PersonaError(f"{path}: {msg}")


def _facts(path, lines):
    """The key/value lines between the two ``---`` marks, and where the
    prompt starts. Strict on purpose: a key this salt does not read is a
    fact the author believes is doing something, and silently ignoring
    it would let that belief stand."""
    if not lines or lines[0].strip() != _MARK:
        _fail(path, f"a persona opens with a {_MARK!r} line, then its "
                    f"facts, then a second {_MARK!r}, then the prompt")
    if not any(line.strip() == _MARK for line in lines[1:]):
        _fail(path, f"the second {_MARK!r} line never comes, so where the "
                    f"facts end and the prompt begins cannot be told")
    facts = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == _MARK:
            if not facts:
                _fail(path, "the lines between the two "
                            f"{_MARK!r} marks name no facts at all")
            return facts, i + 1
        if not line.strip():
            continue
        key, sep, value = line.partition(":")
        key = key.strip()
        if not sep or not key:
            _fail(path, f"line {i + 1} is not a 'key: value' fact: "
                        f"{line.strip()!r}")
        if key not in _KEYS:
            _fail(path, f"unknown key {key!r} (allowed: {sorted(_KEYS)})")
        if key in facts:
            _fail(path, f"{key!r} is named twice")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not value:
            _fail(path, f"{key!r} has no value")
        facts[key] = value


def load_persona(path):
    """Parse and validate one persona file. Returns a Persona or raises
    PersonaError with the file, the field, and the fix."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise PersonaError(f"Cannot read persona {p}: {exc}") from exc
    facts, at = _facts(p, text.splitlines())
    name = facts.get("name")
    if not name or not _NAME_RE.fullmatch(name):
        _fail(p, f"needs a name of letters, digits, '.', '_', '-', "
                 f"got {name!r}")
    if name == CHAT_WORKER:
        _fail(p, f"{CHAT_WORKER!r} is the reserved name of the session's "
                 f"own chat model, so no persona may take it")
    worker = facts.get("worker")
    if not worker:
        _fail(p, f"names no worker. Say whose weights it rides: a roster "
                 f"entry's name, or {CHAT_WORKER!r} for the session's own "
                 f"chat model")
    role = facts.get("role", "target")
    if role not in PERSONA_ROLES:
        _fail(p, f"role must be one of {PERSONA_ROLES}, got {role!r}")
    notes = facts.get("notes", "")
    if role == "target" and not notes:
        _fail(p, "a target persona carries no notes, so the planner would "
                 "have nothing to choose it by")
    body = "\n".join(text.splitlines()[at:]).strip()
    if not body:
        _fail(p, "the prompt below the facts is empty, so this persona "
              "would work under no instructions at all")
    return Persona(name=name, worker=worker, role=role, notes=notes,
                   body=body, path=str(p))


def load_personas(paths):
    """Every persona the given files and directories hold, file order
    inside a directory, argument order across them. A directory
    contributes its ``*.md`` files. Duplicate names fail across the
    whole set: two roles answering to one name is a plan nobody can
    read back."""
    files = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found = sorted(p.glob("*.md"))
            if not found:
                raise PersonaError(f"{p}: holds no *.md persona files")
            files.extend(found)
        elif p.is_file():
            files.append(p)
        else:
            raise PersonaError(f"{p}: no such file or directory")
    personas, seen = [], {}
    for f in files:
        persona = load_persona(f)
        if persona.name in seen:
            raise PersonaError(
                f"{f}: {persona.name!r} is already the name of the persona "
                f"in {seen[persona.name]}")
        seen[persona.name] = str(f)
        personas.append(persona)
    return tuple(personas)


def sample_dir():
    """The personas that ship with salt, as a directory to copy from."""
    return Path(__file__).resolve().parent / "personas"
