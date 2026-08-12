# -*- coding: utf-8 -*-
"""Fitting a roster to the cards a session actually has.

``--roster auto`` reads the model registry, measures what is free on
every card right now, and writes the roster that fits into the session
folder. It starts nothing. A fitted entry is a spawn entry like any
other, and ``--workers-autostart`` stays the only thing that launches a
server.

The arithmetic is the part that matters, because getting it wrong is not
a worse roster, it is a server that dies at load. A vLLM worker sized at
``gpu_mem_util`` has to hold three things inside ``util x TOTAL``: its
weights, the KV cache for the window it serves, and a reserve for
everything else it keeps resident (CUDA context, activations, capture
graphs). That reserve is FLAT rather than proportional, calibrated on
two live points on a 24 GiB card: a 32B model at 0.62 left a 1.24 GiB
residual, and a 135M model at 0.12 died with a residual over 1.1 GiB on
a model that has almost no weights at all.

Two further floors come from what those deaths looked like:

  - Sizing to exactly the need puts a server on the line vLLM refuses
    at. Qwen3-0.6B asked for an 8192 window at 0.15 and was refused with
    0.72 GiB of KV cache against the 0.88 GiB it needed. So the fit adds
    FIT_MARGIN_GIB on top of the need and rounds the dial up to a whole
    UTIL_GRID step.
  - A server costs something that has nothing to do with its model. The
    135M model sized fine at 0.14 and then ran out of memory warming its
    sampler up over 256 dummy requests. So no worker is given less than
    FLOOR_GIB, whatever its weights come to.

Together those put Qwen3-0.6B at an 8192 window on 0.20 and the 135M
model on 0.18, which are the two numbers that were measured to work.

Two rules decide placement, and both have to pass:

  - ``check_placement``, the shipped one, which adds up the declared
    shares on a card and refuses a spawn that would not fit beside what
    is already there.
  - The live ``gpu_free_fractions`` reading, decremented by every worker
    this fit has already placed. Under ``--backend vllm-serve`` the
    session holds no card in this process, so ``check_placement`` is
    told about no chat model and sees a card holding 15 GiB as empty.
    The live reading is the only thing that refuses there, and a fit
    that trusted the static rule alone would put a worker on the card
    the chat model is already holding.

When fewer than MIN_WORKERS fit, the fit refuses. It prints the roster
it would have written and the arithmetic behind every entry it could not
place, because a refusal that shows its work can be argued with and a
bare one cannot.
"""

import json
import math
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from salt.agents.roster import (PLACEMENT_CEILING, PLACEMENT_MARGIN,
                                ROSTER_SCHEMA, RosterEntry, check_placement,
                                gpu_free_fractions)

AUTO = "auto"
ROSTER_FILENAME = "roster.auto.json"

GIB = 1024 ** 3
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")
# a KV cache entry is 16 bit whatever the weights are quantized to
KV_DTYPE_BYTES = 2

RESERVE_GIB = 1.5
FIT_MARGIN_GIB = 0.72
FLOOR_GIB = 4.3
UTIL_GRID = 0.02
# fan-out is the property a fitted roster exists to deliver, and one
# worker cannot fan anything out
MIN_WORKERS = 2


class ProvisionError(Exception):
    """User-facing fitting failure (nothing to fit, nothing to fit it on)."""


@dataclass(frozen=True)
class Job:
    """One described piece of work, and the room it asks for.

    Order matters: jobs are filled in the order they are listed, and the
    one that asks for the most window and the longest reply is filled
    first so the smaller one takes what is left rather than the other
    way round.
    """
    name: str
    window: int
    max_tokens: int


JOBS = (
    Job("writer", 16384, 512),
    Job("finder", 8192, 256),
)


@dataclass(frozen=True)
class Model:
    alias: str
    hf_id: str
    path: str
    weights_gib: float
    kv_bytes_per_token: int
    max_position: int

    def kv_gib(self, window):
        return self.kv_bytes_per_token * int(window) / GIB

    def need_gib(self, window):
        return self.weights_gib + RESERVE_GIB + self.kv_gib(window)


@dataclass(frozen=True)
class Placement:
    job: Job
    model: Model
    card: int
    util: float
    window: int
    total_gib: float
    free_before: float
    shortfall: str = ""

    @property
    def need_gib(self):
        return self.model.need_gib(self.window)

    @property
    def arithmetic(self):
        return (f"{self.model.weights_gib:.2f} W + {RESERVE_GIB:.2f} R + "
                f"{self.model.kv_gib(self.window):.2f} KV = "
                f"{self.need_gib:.2f} GiB of {self.total_gib:.2f}")

    def entry(self):
        return {"name": self.job.name, "alias": self.model.alias,
                "role": "worker",
                "spawn": {"port": "auto", "gpu": str(self.card),
                          "gpu_mem_util": self.util,
                          "max_model_len": self.window},
                "max_tokens": self.job.max_tokens}


@dataclass(frozen=True)
class Fit:
    placements: tuple
    refusals: tuple
    chat_alias: str
    free_fractions: dict
    totals_gib: dict
    wanted: tuple = ()
    backend: str = None
    path: str = None

    @property
    def ok(self):
        return len(self.placements) >= MIN_WORKERS

    def document(self, near_miss=False):
        rows = list(self.placements) + (list(self.wanted) if near_miss else [])
        return {"version": ROSTER_SCHEMA, "models": [p.entry() for p in rows]}

    def text(self, near_miss=False):
        return json.dumps(self.document(near_miss), indent=2) + "\n"

    def with_path(self, path):
        return replace(self, path=str(path))

    def report(self):
        return "\n".join(_report_lines(self))


def _ceil_grid(value):
    return round(min(1.0, math.ceil(value / UTIL_GRID - 1e-9) * UTIL_GRID), 4)


def util_for(need_gib, total_gib):
    """The ``gpu_mem_util`` a server of this size gets on this card."""
    if total_gib <= 0:
        raise ProvisionError("a card with no memory cannot hold a worker")
    return _ceil_grid(max(need_gib + FIT_MARGIN_GIB, FLOOR_GIB) / total_gib)


def weights_gib(path):
    """What this model's weight files come to, the checkpoint only.

    A snapshot directory also carries tokenizers, ONNX exports and any
    other format the repository ships, and counting those would size a
    135M model as though it were ten times itself.
    """
    total = 0
    for f in Path(path).iterdir():
        if f.is_file() and f.suffix in WEIGHT_SUFFIXES:
            total += f.stat().st_size
    return total / GIB


def kv_bytes_per_token(cfg):
    """Layers x KV heads x head dim x 2 (keys and values) x 2 bytes."""
    heads = cfg.get("num_attention_heads") or 0
    head_dim = cfg.get("head_dim")
    if not head_dim:
        hidden = cfg.get("hidden_size") or 0
        head_dim = hidden // heads if heads else 0
    kv_heads = cfg.get("num_key_value_heads") or heads
    layers = cfg.get("num_hidden_layers") or 0
    return int(layers * kv_heads * head_dim * 2 * KV_DTYPE_BYTES)


def model_facts(entry):
    """One registry entry as the numbers a fit needs, or None when its
    snapshot cannot be read."""
    path = Path(entry.get("path") or "")
    try:
        cfg = json.loads((path / "config.json").read_text())
        weights = weights_gib(path)
    except (OSError, ValueError):
        return None
    kv = kv_bytes_per_token(cfg)
    window = cfg.get("max_position_embeddings")
    if not weights or not kv or not window:
        return None
    return Model(alias=entry.get("alias"), hf_id=entry.get("hf_id"),
                 path=str(path), weights_gib=weights, kv_bytes_per_token=kv,
                 max_position=int(window))


def gpu_totals_gib(timeout=5):
    """TOTAL per card in GiB, asked of nvidia-smi. Empty when there is
    nothing to ask, the same way gpu_free_fractions answers."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return {}
    if out.returncode != 0:
        return {}
    totals = {}
    for line in out.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        try:
            totals[int(parts[0])] = float(parts[1]) / 1024.0
        except (IndexError, ValueError):
            continue
    return totals


def chat_alias_for(models, requested=None):
    """Which registered model this session's chat model is."""
    aliases = [m.get("alias") for m in models]
    if requested:
        for m in models:
            if m.get("alias") == requested or (
                    (m.get("hf_id") or "").lower() == requested.lower()):
                return m.get("alias")
        raise ProvisionError(
            f"no registered model matches {requested!r} (known: "
            f"{', '.join(aliases) or 'none registered'})")
    if len(models) == 1:
        return aliases[0]
    raise ProvisionError(
        "name this session's chat model with --model, so the fit knows "
        "which registered model is the planner and which are free to be "
        "workers")


def _pool(models, chat_alias):
    out = []
    for entry in models:
        if entry.get("alias") == chat_alias or not entry.get("downloaded"):
            continue
        facts = model_facts(entry)
        if facts is not None:
            out.append(facts)
    return out


def _ordered(pool):
    """Biggest first. Among the models that fit a card, the largest one
    is the most model the card can be made to hold, and a smaller pick
    would be a worse answer for the same memory."""
    return sorted(pool, key=lambda m: (-m.weights_gib, m.alias))


class _Cards:
    """What is left on each card as the fit places workers onto it."""

    def __init__(self, free_fractions, totals_gib):
        self.free = dict(free_fractions)
        self.totals = dict(totals_gib)
        self.taken = {}

    def order(self):
        cards = [c for c in sorted(self.free) if c in self.totals]
        return sorted(cards, key=lambda c: (c in self.taken, -self.free[c]))

    def place(self, card, name, cards_used, util):
        self.free[card] = round(self.free[card] - util - PLACEMENT_MARGIN, 4)
        self.taken.setdefault(card, []).append(name)
        cards_used.append((name, (card,), util))


def thinking_of(model):
    """Whether this model reasons, measured off its own chat template.

    A model whose template opens a think block it cannot be asked to
    close spends its reply length on the working, so it cannot serve a
    job capped at a chat reply's worth of tokens. Anything that cannot
    be measured answers `unset`, which is the reading that changes
    nothing.
    """
    from salt.agents.thinking import UNSET, template_thinking
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model.path)
        return template_thinking(tok)
    except Exception:
        return UNSET


def _reasons_to_think(job, model, ctx):
    from salt.agents.roster import THINK_FLOOR
    from salt.agents.thinking import ALWAYS
    if job.max_tokens < THINK_FLOOR and ctx["thinking_of"](model) == ALWAYS:
        return (f"{model.alias} always reasons and its template opens the "
                f"block itself, so a reply capped at {job.max_tokens} "
                f"tokens never reaches an answer")
    return None


def _sized(job, model, card, cards):
    window = min(job.window, model.max_position)
    total = cards.totals[card]
    return Placement(job=job, model=model, card=card,
                     util=util_for(model.need_gib(window), total),
                     window=window, total_gib=total,
                     free_before=cards.free[card])


def _shortfall(placement, cards):
    claim = round(placement.util + PLACEMENT_MARGIN, 4)
    if placement.util > PLACEMENT_CEILING:
        return (f"{placement.model.alias} wants {placement.util:.2f} of GPU "
                f"{placement.card} on its own, over the "
                f"{PLACEMENT_CEILING:g} that leaves a card room to work")
    if claim > cards.free[placement.card]:
        return (f"{placement.model.alias} needs {claim:.2f} of GPU "
                f"{placement.card} ({placement.util:.2f} declared plus about "
                f"{PLACEMENT_MARGIN:g} resident overhead) and "
                f"{cards.free[placement.card]:.2f} is free there once every "
                f"worker already fitted is counted")
    return None


def _place(job, model, cards, running, ctx):
    """This model on the best card that holds it, or (None, one reason).

    The reason names the roomiest card it was tried on, because a list
    of every card it did not fit repeats one number per card and says
    nothing the closest call does not already say.
    """
    said = _reasons_to_think(job, model, ctx)
    if said:
        return None, said
    closest = None
    for card in cards.order():
        placement = _sized(job, model, card, cards)
        short = _shortfall(placement, cards)
        if short is None:
            refusal, _ = check_placement(
                _as_entry(placement), chat_gpus=ctx["chat_gpus"],
                chat_mem_util=ctx["chat_mem_util"], bge_gpu=ctx["bge_gpu"],
                running=running, free_fractions=ctx["free_fractions"])
            if not refusal:
                return placement, None
            short = f"GPU {card}: {refusal}"
        if closest is None or cards.free[card] > cards.free[closest[0]]:
            closest = (card, short)
    return None, (closest[1] if closest else
                  f"{model.alias} had no card to be tried on")


def _near_miss(job, pool, used, cards, ctx):
    """The entry this job would have carried, and why it does not fit."""
    for model in _ordered(pool):
        if model.alias in used or _reasons_to_think(job, model, ctx):
            continue
        card = next(iter(cards.order()), None)
        if card is None:
            return None
        placement = _sized(job, model, card, cards)
        return replace(placement, shortfall=_shortfall(placement, cards) or
                       f"{model.alias} was refused on GPU {card}")
    return None


def _as_entry(placement):
    entry = placement.entry()
    return RosterEntry(name=entry["name"], alias=entry["alias"],
                       role="worker", spawn=entry["spawn"],
                       max_tokens=entry["max_tokens"])


def _assign(jobs, pool, used, cards, running, ctx, refusals):
    if not jobs:
        return []
    job, rest = jobs[0], jobs[1:]
    tried = []
    for model in _ordered(pool):
        if model.alias in used:
            continue
        snapshot = (dict(cards.free), dict(cards.taken), list(running))
        placement, reason = _place(job, model, cards, running, ctx)
        if placement is None:
            tried.append(reason)
            continue
        cards.place(placement.card, job.name, running, placement.util)
        tail = _assign(rest, pool, used | {model.alias}, cards, running, ctx,
                       refusals)
        if tail is not None:
            return [placement] + tail
        cards.free, cards.taken = snapshot[0], snapshot[1]
        running[:] = snapshot[2]
        tried.append(f"{model.alias} fitted, but nothing was left for "
                     f"{rest[0].name if rest else 'the next job'}")
    refusals.append((job.name, tried))
    return None


def fit_session(models, chat_alias, *, backend=None, chat_gpus=(),
                chat_mem_util=None, bge_gpu=None, free_fractions=None,
                totals_gib=None, thinks=None):
    """The roster this machine fits right now.

    ``free_fractions`` and ``totals_gib`` default to the live readings.
    Passing them is how the fit and the refusal are pinned without a
    card to run on.
    """
    free = gpu_free_fractions() if free_fractions is None else dict(
        free_fractions)
    totals = gpu_totals_gib() if totals_gib is None else dict(totals_gib)
    if not free or not totals:
        raise ProvisionError(
            "no GPU memory could be read, so there is nothing to fit a "
            "roster to. Write the roster yourself and pass it as "
            "--roster FILE.")
    pool = _pool(models, chat_alias)
    if not pool:
        raise ProvisionError(
            f"the registry holds no downloaded model beside {chat_alias!r} "
            f"to hand work to. Register one with: saltChat --add <hf_id>")
    measured = {}
    reader = thinking_of if thinks is None else thinks
    ctx = {"chat_gpus": tuple(chat_gpus), "chat_mem_util": chat_mem_util,
           "bge_gpu": bge_gpu, "free_fractions": free,
           "thinking_of": lambda m: measured.setdefault(m.alias, reader(m))}
    cards = _Cards(free, totals)
    running = []
    refusals = []
    placements = _assign(list(JOBS), pool, set(), cards, running, ctx,
                         refusals)
    if placements is not None:
        return Fit(placements=tuple(placements), refusals=(),
                   chat_alias=chat_alias, free_fractions=free,
                   totals_gib=totals, backend=backend)
    # nothing fits every job at once, so the report is built from the
    # best partial fit rather than from the search that gave up: a
    # person reading a refusal needs what did fit and what the next
    # entry would have been, not the order the search tried things in
    cards = _Cards(free, totals)
    running, refusals, placements, wanted, used = [], [], [], [], set()
    for job in JOBS:
        found, tried = None, []
        for model in _ordered(pool):
            if model.alias in used:
                continue
            found, reason = _place(job, model, cards, running, ctx)
            if found is not None:
                break
            tried.append(reason)
        if found is None:
            refusals.append((job.name, tried))
            near = _near_miss(job, pool,
                              used | {w.model.alias for w in wanted},
                              cards, ctx)
            if near is not None:
                wanted.append(near)
            continue
        cards.place(found.card, job.name, running, found.util)
        used.add(found.model.alias)
        placements.append(found)
    return Fit(placements=tuple(placements), refusals=tuple(refusals),
               chat_alias=chat_alias, free_fractions=free, totals_gib=totals,
               wanted=tuple(wanted), backend=backend)


def write_roster(fit, session_dir):
    """The fitted roster on disk, beside the session it was fitted for."""
    if not fit.ok:
        raise ProvisionError("a roster that does not fit is never written")
    folder = Path(session_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / ROSTER_FILENAME
    path.write_text(fit.text())
    return path


def _free_line(fit):
    return "free memory read at fit time: " + ", ".join(
        f"GPU {c} {fit.free_fractions[c]:.2f}"
        for c in sorted(fit.free_fractions))


def _row(placement):
    p = placement
    head = (f"  {p.job.name:8} {p.model.alias:24} GPU {p.card}  "
            f"util {p.util:.2f}  window {p.window}  "
            f"max_tokens {p.job.max_tokens}")
    body = f"           {p.arithmetic}"
    if p.shortfall:
        return [head + "   DOES NOT FIT", body, f"           {p.shortfall}"]
    return [head, body + f", and GPU {p.card} had {p.free_before:.2f} free"]


def _report_lines(fit):
    lines = []
    if fit.ok:
        lines.append(f"--roster auto fitted {len(fit.placements)} workers "
                     f"across {len(fit.totals_gib)} card(s).")
        for p in fit.placements:
            lines.extend(_row(p))
        lines.append(f"  the planner is {fit.chat_alias}, this session's own "
                     f"chat model. No orchestrator entry is written, because "
                     f"one would be a second name for the same weights.")
        if fit.backend == "vllm-serve":
            lines.append("  under --backend vllm-serve the chat model holds "
                         "no card in this process, so the static placement "
                         "rule was told about none of it and the live free "
                         "memory below is the only thing that refused a bad "
                         "card.")
        lines.append("  " + _free_line(fit))
        if fit.path:
            lines.append(f"  written to {fit.path}")
        lines.append("  Nothing was started. --workers-autostart is what "
                     "starts a spawn entry.")
        return lines
    fitted = len(fit.placements)
    lines.append(f"--roster auto could not fit {MIN_WORKERS} workers, and a "
                 f"roster with fewer cannot fan out. "
                 f"{fitted} of {MIN_WORKERS} fitted.")
    lines.append("  the roster it would have written, entry by entry:")
    for p in list(fit.placements) + list(fit.wanted):
        lines.extend(_row(p))
    lines.append("  as a file, with the entries that do not fit left in so "
                 "the near miss can be read:")
    lines.extend("  " + ln for ln in fit.text(True).rstrip().splitlines())
    lines.append("  every candidate that was tried and turned down:")
    for name, reasons in fit.refusals:
        lines.append(f"    {name}:")
        for reason in reasons or ["no candidate model was left to try"]:
            lines.append(f"      {reason}")
    lines.append("  " + _free_line(fit))
    lines.append("  Free a card, or write the roster yourself and pass it as "
                 "--roster FILE.")
    lines.append("  Nothing was written and nothing was started.")
    return lines
