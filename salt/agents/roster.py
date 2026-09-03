# -*- coding: utf-8 -*-
"""Model roster for the saltChat agent layer.

A roster file names the models a session may reach beside the chat model:
worker models that tasks can be handed to, and at most one orchestrator.
An entry either attaches to an already-running ``saltServe`` endpoint
(``server_url``) or describes how to spawn one (``spawn``). Loading only
parses and validates - nothing here opens a connection, spawns a process,
or touches a GPU.

File shape (``salt/agents/roster_sample.json`` is a working example)::

    {"version": "salt-roster/1",
     "models": [{"name": "qwen05", "alias": "qwen05", "role": "worker",
                 "server_url": "http://127.0.0.1:8081"}]}

Every alias must be a registered, downloaded model: the serve client
tokenizes locally, so even a remote worker needs its weights on disk.
"""

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path

ROSTER_SCHEMA = "salt-roster/1"
ROLES = ("worker", "orchestrator")

# the smallest reply length an entry that asks for the working may name.
# A model told to reason writes the working before the answer, and a
# call that has spent three quarters of its reply length reasoning
# without answering is given up on, so a cap under this one is a cap the
# working alone can finish
THINK_FLOOR = 2048

_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_TOP_KEYS = {"version", "models"}
_ENTRY_KEYS = {"name", "alias", "role", "server_url", "spawn", "max_tokens",
               "temperature", "capabilities", "notes", "timeout_s", "think"}
_SPAWN_KEYS = {"port", "gpu", "gpu_mem_util", "max_model_len",
               "ready_timeout", "command"}


class RosterError(Exception):
    """User-facing roster failure (bad file, bad entry, unknown name)."""


@dataclass(frozen=True)
class RosterEntry:
    name: str
    alias: str
    role: str
    server_url: str = None
    spawn: dict = None
    max_tokens: int = None
    temperature: float = None
    capabilities: tuple = ()
    notes: str = ""
    timeout_s: float = None
    # whether this model should be asked to reason on a call, for the
    # models that expose the choice through their chat template. Three
    # states and not two: True asks for the working, False asks for it
    # to be left out, and None - the only default - sends no template
    # setting at all, which is the exact prompt this entry has always
    # sent. A file that says nothing changes no bytes
    think: bool = None
    model: dict = None

    @property
    def attach(self):
        return self.server_url is not None


@dataclass(frozen=True)
class ProbeResult:
    state: str
    served_model: str = None
    max_model_len: int = None
    detail: str = ""

    @property
    def note(self):
        if self.state != "PROBED":
            return self.detail
        window = (f", window {self.max_model_len} tokens"
                  if self.max_model_len else "")
        return f"serving {self.served_model}{window}"


UNPROBED = ProbeResult("UNPROBED")


@dataclass(frozen=True)
class Roster:
    path: str
    entries: tuple

    @property
    def workers(self):
        return tuple(e for e in self.entries if e.role == "worker")

    @property
    def orchestrator(self):
        for e in self.entries:
            if e.role == "orchestrator":
                return e
        return None

    def get(self, name):
        for e in self.entries:
            if e.name == name:
                return e
        known = ", ".join(e.name for e in self.entries) or "none"
        raise RosterError(f"No roster entry named {name!r} (known: {known}).")


def _fail(path, name, msg):
    who = f"entry {name!r}" if name else "roster"
    raise RosterError(f"{path}: {who}: {msg}")


def _check_number(path, name, key, value, kind, low=None, high=None):
    if not isinstance(value, kind) or isinstance(value, bool):
        _fail(path, name, f"{key} must be a number, got {value!r}")
    if low is not None and value < low:
        _fail(path, name, f"{key} must be >= {low}, got {value!r}")
    if high is not None and value > high:
        _fail(path, name, f"{key} must be <= {high}, got {value!r}")
    return value


def _parse_spawn(path, name, raw):
    if not isinstance(raw, dict):
        _fail(path, name, f"spawn must be an object, got {raw!r}")
    unknown = set(raw) - _SPAWN_KEYS
    if unknown:
        _fail(path, name, f"unknown spawn keys {sorted(unknown)} "
                          f"(allowed: {sorted(_SPAWN_KEYS)})")
    spawn = dict(raw)
    port = spawn.get("port", "auto")
    if port != "auto":
        _check_number(path, name, "spawn.port", port, int, 1, 65535)
    spawn["port"] = port
    gpu = spawn.get("gpu")
    if gpu is not None:
        if not isinstance(gpu, str):
            _fail(path, name, f"spawn.gpu must be a string like '0' or "
                              f"'0,1', got {gpu!r}")
        from salt.chat.serve import parse_gpu_list
        try:
            parse_gpu_list(gpu)
        except ValueError as exc:
            _fail(path, name, f"spawn.gpu: {exc}")
    if "gpu_mem_util" in spawn:
        _check_number(path, name, "spawn.gpu_mem_util",
                      spawn["gpu_mem_util"], (int, float), 0.05, 1.0)
    if "max_model_len" in spawn:
        _check_number(path, name, "spawn.max_model_len",
                      spawn["max_model_len"], int, 0)
    if "ready_timeout" in spawn:
        _check_number(path, name, "spawn.ready_timeout",
                      spawn["ready_timeout"], (int, float), 1)
    command = spawn.get("command")
    if command is not None:
        if not isinstance(command, list) or not command or not all(
                isinstance(c, str) and c for c in command):
            _fail(path, name, f"spawn.command must be a non-empty list of "
                              f"strings, got {command!r}")
        spawn["command"] = list(command)
    return spawn


def _parse_entry(path, index, raw, seen_names):
    if not isinstance(raw, dict):
        _fail(path, None, f"models[{index}] must be an object, got {raw!r}")
    unknown = set(raw) - _ENTRY_KEYS
    if unknown:
        _fail(path, raw.get("name"),
              f"unknown keys {sorted(unknown)} (allowed: "
              f"{sorted(_ENTRY_KEYS)})")
    name = raw.get("name")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        _fail(path, None, f"models[{index}] needs a name of letters, "
                          f"digits, '.', '_', '-', got {name!r}")
    if name in seen_names:
        _fail(path, name, "duplicate name")
    seen_names.add(name)
    alias = raw.get("alias", name)
    if not isinstance(alias, str) or not alias:
        _fail(path, name, f"alias must be a non-empty string, got {alias!r}")
    role = raw.get("role", "worker")
    if role not in ROLES:
        _fail(path, name, f"role must be one of {ROLES}, got {role!r}")
    server_url = raw.get("server_url")
    spawn = raw.get("spawn")
    if (server_url is None) == (spawn is None):
        _fail(path, name, "exactly one of server_url (attach to a running "
                          "saltServe) or spawn (launch one) is required")
    if server_url is not None:
        if not isinstance(server_url, str) or not (
                server_url.startswith("http://")
                or server_url.startswith("https://")):
            _fail(path, name, f"server_url must start with http:// or "
                              f"https://, got {server_url!r}")
    else:
        spawn = _parse_spawn(path, name, spawn)
    max_tokens = raw.get("max_tokens")
    if max_tokens is not None:
        _check_number(path, name, "max_tokens", max_tokens, int, 1)
    temperature = raw.get("temperature")
    if temperature is not None:
        _check_number(path, name, "temperature", temperature, (int, float), 0)
    capabilities = raw.get("capabilities", [])
    if not isinstance(capabilities, list) or not all(
            isinstance(c, str) for c in capabilities):
        _fail(path, name, f"capabilities must be a list of strings, "
                          f"got {capabilities!r}")
    notes = raw.get("notes", "")
    if not isinstance(notes, str):
        _fail(path, name, f"notes must be a string, got {notes!r}")
    timeout_s = raw.get("timeout_s")
    if timeout_s is not None:
        _check_number(path, name, "timeout_s", timeout_s, (int, float), 1)
    think = raw.get("think")
    # bool is an int in Python and 1 reads as true, so a number written
    # here would be honoured as a choice its author never made
    if think is not None and not isinstance(think, bool):
        _fail(path, name, f"think must be true or false, or left out to "
                          f"send no thinking setting at all, got {think!r}")
    if think and max_tokens is not None and max_tokens < THINK_FLOOR:
        _fail(path, name, f"think is true but max_tokens is {max_tokens}, "
                          f"and a model writing its working into a reply "
                          f"that short never reaches the answer. Raise "
                          f"max_tokens to {THINK_FLOOR} or more, or leave it "
                          f"out and let the endpoint's own window size the "
                          f"call.")
    return RosterEntry(
        name=name, alias=alias, role=role, server_url=server_url,
        spawn=spawn, max_tokens=max_tokens, temperature=temperature,
        capabilities=tuple(capabilities), notes=notes, timeout_s=timeout_s,
        think=think)


def _resolve(path, entry):
    from salt.chat.registry import RegistryError, resolve_model
    try:
        cfg = resolve_model(entry.alias)
    except RegistryError as exc:
        _fail(path, entry.name, str(exc))
    if not cfg.get("downloaded"):
        _fail(path, entry.name,
              f"model {entry.alias!r} is registered but its weights are "
              f"not downloaded. Fetch them with: saltChat --add "
              f"{cfg.get('hf_id', entry.alias)}")
    return replace(entry, model=cfg)


def load_roster(path):
    """Parse and validate a roster file. Returns a Roster or raises
    RosterError with the file, the entry, and the fix."""
    p = Path(path)
    try:
        data = json.loads(p.read_text())
    except OSError as exc:
        raise RosterError(f"Cannot read roster {p}: {exc}") from exc
    except ValueError as exc:
        raise RosterError(f"Roster {p} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RosterError(
            f"Roster {p} must be a JSON object with a 'models' list.")
    version = data.get("version", ROSTER_SCHEMA)
    if version != ROSTER_SCHEMA:
        raise RosterError(
            f"Roster {p} carries schema {version!r} but this salt reads "
            f"{ROSTER_SCHEMA!r}. Update salt, or rewrite the file.")
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise RosterError(
            f"Roster {p} has unknown top-level keys {sorted(unknown)} "
            f"(allowed: {sorted(_TOP_KEYS)})")
    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise RosterError(f"Roster {p} needs a non-empty 'models' list.")
    seen = set()
    entries = [_parse_entry(p, i, raw, seen) for i, raw in enumerate(models)]
    if sum(1 for e in entries if e.role == "orchestrator") > 1:
        raise RosterError(f"Roster {p} names more than one orchestrator.")
    entries = tuple(_resolve(p, e) for e in entries)
    return Roster(path=str(p), entries=entries)


def _card_window(card):
    v = card.get("max_model_len")
    if isinstance(v, int) and not isinstance(v, bool) and v > 0:
        return v
    return None


def probe(entry, url=None, timeout=5):
    """Ask an endpoint what it is serving, over the same GET /v1/models
    handshake the serve client uses. Never raises: a refused port, a hung
    server and a server holding the wrong model are all answers, returned
    as DEAD with the reason. ``url`` overrides the entry's own endpoint,
    which is how a freshly spawned worker gets probed later."""
    import requests

    url = url or entry.server_url
    if url is None:
        return ProbeResult("UNPROBED",
                           detail="spawn entry, nothing started yet")
    url = url.rstrip("/")
    try:
        resp = requests.get(f"{url}/v1/models", timeout=timeout)
        resp.raise_for_status()
        cards = resp.json().get("data", [])
    except (requests.RequestException, ValueError) as exc:
        return ProbeResult("DEAD", detail=f"no server answering at {url} "
                                          f"({type(exc).__name__}: {exc})")
    cards = [c for c in cards if isinstance(c, dict)] if isinstance(
        cards, list) else []
    cfg = entry.model or {}
    accepted = {cfg.get(k) for k in ("alias", "hf_id", "path")} | {entry.alias}
    card = next((c for c in cards if c.get("id") in accepted), None)
    if card is None:
        served = ", ".join(str(c.get("id")) for c in cards) or "nothing"
        return ProbeResult("DEAD", detail=f"{url} serves {served}, not "
                                          f"{entry.alias!r}")
    return ProbeResult("PROBED", served_model=card["id"],
                       max_model_len=_card_window(card))


# what a capability probe sends: the smallest request that can carry a
# schema at all. One token back is all the answer that is needed
GUIDED_SCHEMA = {"type": "object"}
GUIDED_CAPABLE = "guided"
# the newer spelling of the same capability: vLLM deprecated guided_json
# and newer servers accept only structured_outputs, so which word the
# server took is part of what the probe learns
GUIDED_STRUCTURED = "structured"
GUIDED_PLAIN = "plain"
GUIDED_UNKNOWN = "unknown"
SCHEMA_CAPABLE = (GUIDED_CAPABLE, GUIDED_STRUCTURED)


def schema_body(capability, schema):
    """The request keys that hold this endpoint to a schema, in the
    spelling it was measured to accept. Empty for an endpoint that
    accepted none, so a caller can always splat this into a body."""
    if schema is None:
        return {}
    if capability == GUIDED_CAPABLE:
        return {"guided_json": schema}
    if capability == GUIDED_STRUCTURED:
        return {"structured_outputs": {"json": schema}}
    return {}


def probe_guided(entry, url=None, timeout=10, served_model=None):
    """Whether this endpoint can be made to answer in a schema.

    Asked of the wire, never inferred from a version string: a server
    that says it is new enough and rejects the parameter anyway is the
    case this exists for. A tiny completion carrying vLLM's guided_json
    is sent, and the server's own answer decides. Anything other than
    an accepted request means the plain protocol, since a capability
    that cannot be demonstrated is one to plan without.

    The request goes out under the name the server itself is serving,
    asked for here when the caller does not already know it. A server
    started from a registered alias answers to that alias and not to the
    model's full id, and a probe under the wrong name comes back 404 -
    which is a question about the name, never an answer about schemas.
    """
    import requests

    url = (url or entry.server_url or "").rstrip("/")
    if not url:
        return GUIDED_UNKNOWN, "nothing to probe: no endpoint yet"
    cfg = entry.model or {}
    if served_model is None:
        found = probe(entry, url=url, timeout=timeout)
        served_model = found.served_model
    body = {"model": served_model or cfg.get("hf_id") or entry.alias,
            "prompt": "{", "max_tokens": 1, "temperature": 0,
            "guided_json": GUIDED_SCHEMA}
    try:
        resp = requests.post(f"{url}/v1/completions", json=body,
                             timeout=timeout)
    except requests.RequestException as exc:
        return GUIDED_UNKNOWN, f"{type(exc).__name__}: {exc}"
    if resp.status_code == 200:
        return GUIDED_CAPABLE, "the endpoint accepted a schema"
    detail = (resp.text or "").strip().replace("\n", " ")[:200]
    if resp.status_code == 404:
        # the server does not have this model at all, so it never got as
        # far as the schema. Saying "plain" here would be a guess wearing
        # a measurement's clothes
        return GUIDED_UNKNOWN, (f"the endpoint does not serve "
                                f"{body['model']!r}, so nothing was asked "
                                f"about schemas: {detail}")
    # the old spelling was refused. Newer servers dropped it for
    # structured outputs, so the same question is asked once more in
    # that spelling before the endpoint is written down as plain
    newer = {k: v for k, v in body.items() if k != "guided_json"}
    newer["structured_outputs"] = {"json": GUIDED_SCHEMA}
    try:
        second = requests.post(f"{url}/v1/completions", json=newer,
                               timeout=timeout)
    except requests.RequestException:
        second = None
    if second is not None and second.status_code == 200:
        return GUIDED_STRUCTURED, ("the endpoint accepted a schema as "
                                   "structured outputs")
    return GUIDED_PLAIN, f"the endpoint refused a schema ({resp.status_code}): {detail}"


PLACEMENT_CEILING = 0.95
BGE_CARD_MB = 130
# what a vLLM server actually holds beyond its declared budget: CUDA
# context, NCCL and capture graphs live outside the util * TOTAL the
# server sizes itself to. Measured on 24 GB cards: a model declared at
# 0.62 sat at 0.70 of the card and one at 0.75 sat at 0.83, about two
# GiB either way. The check reads a share as its declaration plus this
PLACEMENT_MARGIN = 0.08


def gpu_free_fractions(timeout=5):
    """FREE over TOTAL per card, asked of nvidia-smi. Empty when there
    is nothing to ask - no tool, no driver, a machine with no cards.

    The ground truth the static arithmetic cannot see: processes no
    roster declares, another session's servers, a desktop. Read at spawn
    time, because vLLM refuses to start unless the card's free memory
    covers its whole declared budget."""
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return {}
    if out.returncode != 0:
        return {}
    free = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        try:
            free[int(parts[0])] = round(float(parts[1]) / float(parts[2]), 3)
        except (IndexError, ValueError, ZeroDivisionError):
            continue
    return free


def entry_cards(entry):
    """The GPU indices a spawn entry asks for, empty when it names none."""
    spawn = entry.spawn or {}
    if not spawn.get("gpu"):
        return ()
    from salt.chat.serve import parse_gpu_list
    return tuple(int(c) for c in parse_gpu_list(spawn["gpu"]) or ())


def check_placement(entry, chat_gpus=(), chat_mem_util=None, bge_gpu=None,
                    running=(), free_fractions=None):
    """Whether this worker may take the cards it asks for. Returns
    (refusal, notes): a refusal string when starting it would fight
    another server for the same memory, else None, plus notes worth
    printing either way.

    ``running`` is (name, cards, gpu_mem_util) per worker already up.
    Two servers CAN share a card, and that is a normal way to run a
    small worker beside a big model. What cannot work is two servers
    whose real footprints together do not fit, because vLLM refuses to
    start unless the card's FREE memory covers util * TOTAL, and a
    server that squeaks past that dies warming up. So sharing needs
    both sides to say in writing how much they take, and the shares
    plus what a server holds beyond its share (PLACEMENT_MARGIN each)
    have to fit inside the card.

    ``free_fractions`` is {card: FREE/TOTAL} measured now, the
    gpu_free_fractions() reading. When it covers a card it is the
    ground truth: it sees what no roster declares. None skips the live
    rule, which is what keeps this callable with no GPU at all."""
    cards = entry_cards(entry)
    notes = []
    if not cards:
        return None, ["it names no gpu, so its server lands wherever vLLM "
                      "defaults to. Give the entry a spawn.gpu to place it."]
    mine = (entry.spawn or {}).get("gpu_mem_util")
    occupants = {}
    for c in chat_gpus:
        occupants.setdefault(int(c), []).append(("the chat model",
                                                 chat_mem_util))
    for name, cards_, util in running:
        for c in cards_:
            occupants.setdefault(int(c), []).append((f"worker {name!r}", util))
    for c in cards:
        for who, util in occupants.get(c, []):
            if util is None or mine is None:
                undeclared = ("this entry" if mine is None
                              else f"{who} does not")
                return (f"GPU {c} already carries {who}, and {undeclared} "
                        f"declares a gpu_mem_util. Two servers claiming "
                        f"their default share of one card run it out of "
                        f"memory at load. Give both an explicit "
                        f"gpu_mem_util, or place this worker on another "
                        f"card.", notes)
    for c in cards:
        # only a declared share can be added up. An undeclared one that got
        # this far is alone on its card, so there is nothing to overrun.
        if mine is None:
            break
        shared = occupants.get(c, [])
        claimed = (mine + PLACEMENT_MARGIN
                   + sum(u + PLACEMENT_MARGIN for _, u in shared if u))
        if shared and claimed > 1.0:
            return (f"GPU {c} would hold {claimed:.2f} of the card once "
                    f"every server's resident overhead (about "
                    f"{PLACEMENT_MARGIN:g} each) is counted beside its "
                    f"declared share. vLLM refuses to start unless the "
                    f"card's free memory covers its whole share, so this "
                    f"one dies at load. Give the big model a smaller "
                    f"gpu_mem_util, or place this worker on another "
                    f"card.", notes)
        if claimed > PLACEMENT_CEILING:
            notes.append(f"GPU {c} would be claimed {claimed:.2f} in total "
                         f"with resident overhead counted, over the "
                         f"{PLACEMENT_CEILING:g} that leaves the card room "
                         f"to work")
        live = (free_fractions or {}).get(int(c))
        if live is not None and mine + PLACEMENT_MARGIN > live:
            return (f"GPU {c} has {live:.2f} of its memory free right now, "
                    f"and this worker needs {mine + PLACEMENT_MARGIN:.2f} "
                    f"({mine:g} declared plus about {PLACEMENT_MARGIN:g} "
                    f"resident overhead). vLLM refuses to start unless the "
                    f"free memory covers the whole share. Free the card, or "
                    f"give this worker a smaller gpu_mem_util it can still "
                    f"serve under.", notes)
    if bge_gpu is not None and int(bge_gpu) in cards:
        notes.append(f"GPU {bge_gpu} also holds this session's BGE encoder "
                     f"(about {BGE_CARD_MB} MB)")
    return None, notes


def probe_roster(roster, timeout=5):
    """Probe every entry in file order. Returns {name: ProbeResult}."""
    return {e.name: probe(e, timeout=timeout) for e in roster.entries}
