# -*- coding: utf-8 -*-
"""saltChat: interactive multi-turn chat backed by SALT's persistent trie.

The session pins three things for its whole lifetime: the BGE encoder and the
chat LLM on GPU, and one ``SessionTrie`` in process RAM (autosaved to
``salt/chat/sessions/<conversation-id>/``). Every turn, the trie compresses
the accumulated conversation (and any ``/doc`` ingested files) into a
query-biased memory block under the token budget; the last ``--tail``
exchanges ride along verbatim. Both sides of each exchange then grow the
trie, so older material keeps flowing through the compressed block instead of
falling out of a context window.

Usage:
    saltChat --add Qwen/Qwen2.5-0.5B-Instruct
    saltChat --model qwen2.5-0.5b-instruct
    saltChat --model qwen2.5-0.5b-instruct --conversation-id demo1 --doc notes.txt

Inside the REPL, ``/help`` lists the slash commands (``/model`` switches
among registered models without touching the session).
"""

import argparse
import gc
import json
import math
import os
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime
from itertools import groupby
from pathlib import Path

import torch

from salt.agents.roster import UNPROBED, RosterError, load_roster
from salt.agents.worker import WorkerHandle
from salt.chat.ingest import IngestWorker
from salt.chat.kvtrace import KVTrace
from salt.chat.pdfio import (PLAIN_SUFFIXES, ExtractionError,
                             is_protected_unit, read_document,
                             split_document_sentences)
from salt.chat.registry import (RegistryError, list_models, register_model,
                                resolve_model)
from salt.chat.runner import make_runner
from salt.chat.serve import default_gpu_mem_util, parse_gpu_list
from salt.chat.shortturn import (acknowledgement_only, fuse_with_question,
                                 is_short_user_unit)
from salt.engine.chat_text import is_protected_chat_unit
from salt.engine.compressor import load_bge
from salt.engine.session_trie import VALID_ROLES, SessionTrie

SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"
FILES_DIR = Path(__file__).resolve().parents[1] / "files"
FILE_SUFFIXES = {".pdf"} | PLAIN_SUFFIXES
BGE_MODEL = "BAAI/bge-small-en-v1.5"

# conversation ids become directory names under SESSIONS_DIR
SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# seam: system-prompt wording is a later tuning knob
SYSTEM_PROMPT = "You are a helpful assistant."

# --memory-cap auto: tokens held back for the reply framing, and the floor
# that keeps the memory block from collapsing when the window is tight.
# The tokens-per-word seed is refined per session by an EMA of the real
# measured ratio, so the cap self-corrects for the active tokenizer.
MEMORY_CAP_RESERVE = 256
MEMORY_CAP_FLOOR_WORDS = 64
TOKENS_PER_WORD_SEED = 1.6
TPW_EMA_ALPHA = 0.3
MEMORY_BLOCK = (
    "SALT memory — compressed excerpts auto-selected for this message "
    "(partial, not full text; each section keeps original order):"
    "\n---\n{body}\n---")
# Conversation section headers. Both spellings are a contract with the
# reading guide in instructions.md — edit the two in lockstep. The
# unlabeled form is what --no-turn-labels restores, byte for byte.
CONVERSATION_LABEL = "[from the earlier conversation]"
CONVERSATION_LABEL_TURN = "[from the earlier conversation — turn {turn}, {role}]"
CONVERSATION_LABEL_AGE = (
    "[from the earlier conversation — turn {turn}, {role}, {age}]")
CONVERSATION_MAP_LABEL = "[map of the conversation so far]"
CONVERSATION_MAP_LABEL_RECENT = (
    "[map of the conversation — most recent {n} of {total} turns]")

# The universal instruction block that teaches the chat model how to read
# SALT's context (memory sections, attachment inventory, citing rules).
# Kept as an editable file so the wording can be tuned without code changes;
# re-read every turn, so edits apply live to a running session.
INSTRUCTIONS_PATH = Path(__file__).resolve().parent / "instructions.md"
FALLBACK_INSTRUCTIONS = (
    'The "SALT memory" block at the top of the newest user message contains '
    "compressed sentence excerpts from earlier conversation and attached "
    "files, grouped by source — it is partial, not full text. Name the file "
    "you draw on; if the excerpts don't cover a question, say so rather "
    "than inventing content.")


def load_instructions():
    # ValueError covers UnicodeDecodeError from a re-saved non-UTF-8 file;
    # a broken instructions file must never take down the chat turn
    try:
        text = INSTRUCTIONS_PATH.read_text(encoding="utf-8").strip()
        return text or FALLBACK_INSTRUCTIONS
    except (OSError, ValueError):
        return FALLBACK_INSTRUCTIONS

HELP = """\
salt@              list attachable files staged in salt/files/
salt@<file>        attach via SALT (.pdf/.txt/.md/.rst): whole text ingested
                   under its own trie branch; compressed into the prompt per turn
attach@<file>      attach IN FULL: the whole text rides in every prompt,
                   uncompressed (the full-context counterpart of salt@)
/help              show this help
/model             list registered models (* = active)
/model <name>      switch chat model (unloads current, loads new; session kept)
/add <hf_id> [alias]  download + register a model by HuggingFace id
/roster [probe]    list the worker models --roster names (probe contacts
                   each one and reports what it is serving)
/worker            show each worker's connection, calls and mean latency
/worker probe <name>  reconnect one worker and report what it is serving
/doc <path>        ingest a text or PDF file into the trie (role=doc)
/budget <pct>      set memory token budget (0.3 or 30 for 30%)
/stats             session, attachments, compression, and GPU memory stats
/new [id]          start (or resume) another conversation
/clear             wipe and restart the current conversation
/exit              leave (also Ctrl-D)"""

# what TAB offers: every command HELP lists, so the two cannot drift
COMMANDS = ["/help", "/model", "/add", "/roster", "/worker", "/doc",
            "/budget", "/stats", "/new", "/clear", "/exit"]


def resolve_gpu_devices(gpus, device, bge_device, gpu_mem_util):
    """Turn a parsed --gpu list into concrete placements. The chat model
    anchors on the first card and the BGE encoder on the last, and PCI bus
    order is pinned for the WHOLE process (returned as the fourth value) so
    an index names the card nvidia-smi calls N - the same numbering the vllm
    worker and saltServe use, so the parent-side BGE and the model never
    disagree on which physical card an index means. Explicit --device /
    --bge-device / --gpu-mem-util win. The caller must export the returned
    CUDA_DEVICE_ORDER before any CUDA init (None = leave it unset)."""
    if device is None:
        device = f"cuda:{gpus[0]}" if gpus else "cuda"
    if bge_device is None and gpus:
        bge_device = f"cuda:{gpus[-1]}"
    if gpu_mem_util is None:
        gpu_mem_util = default_gpu_mem_util(gpus, single=0.85)
    order = "PCI_BUS_ID" if gpus else None
    return device, bge_device, gpu_mem_util, order


def backend_opts(args):
    """Per-backend knobs the shared CLI surface funnels to make_runner."""
    if args.backend == "vllm":
        return {"gpu_memory_utilization": args.gpu_mem_util,
                "max_model_len": args.max_model_len,
                "gpus": parse_gpu_list(args.gpu)}
    if args.backend == "vllm-serve":
        return {"server_url": args.server_url}
    # hf: a --gpu list shards the model across the cards (device_map),
    # capped per card by the same utilization
    return {"gpus": parse_gpu_list(args.gpu),
            "gpu_memory_utilization": args.gpu_mem_util}


class ChatState:
    """Everything a live session pins: models on GPU, trie in RAM."""

    def __init__(self, args, bge_tok, bge_model, runner, trie, roster=None):
        self.device = args.device
        self.bge_device = args.bge_device or args.device
        self.backend = args.backend
        self.backend_opts = backend_opts(args)
        self.bge_tok = bge_tok
        self.bge_model = bge_model
        self.runner = runner
        self.trie = trie
        # None = the agent layer is absent for this session, everywhere
        self.roster = roster
        # name -> WorkerHandle, built on first use and the session's only
        # record of what each worker is doing. A handle costs nothing until
        # it opens a client, so an unused roster entry stays unopened
        self.workers = {}
        self.budget = args.budget_pct
        self.memory_cap = parse_memory_cap(args.memory_cap)
        self.tokens_per_word = TOKENS_PER_WORD_SEED
        # coverage-decay, shift-damping + near-dup knobs live here and
        # travel as per-call kwargs to compress()/add_turn(): SessionTrie
        # .load() overwrites config values from the persisted config.json,
        # so trie config can't carry launch flags
        self.coverage_half_life = args.coverage_half_life
        self.coverage_decay_docs = args.coverage_decay_docs
        self.shift_damping = args.shift_damping
        self.shift_margin = args.shift_margin
        self.shift_query_boost = args.shift_query_boost
        self.per_source_themes = args.per_source_themes
        self.stable_coverage_keys = args.stable_coverage_keys
        self.coverage_gc = args.coverage_gc
        self.coverage_max_keys = args.coverage_max_keys
        self.dedup_cos = args.dedup_cos
        self.max_sentences = args.max_sentences
        self.short_turns = args.short_turns
        self.turn_labels = not args.no_turn_labels
        self.conversation_map = args.conversation_map
        self.tail_exclude = not args.no_tail_exclude
        self.sync_ingest = args.sync_ingest
        # tail policy: grow append-only to tail_max exchanges, then compact
        # to tail_min in ONE stroke. A rolling window (deque) would drop the
        # oldest exchange every turn and break the prompt prefix each time;
        # block compaction keeps the prefix byte-stable between compactions,
        # which is what future KV-cache reuse (vLLM APC) needs.
        self.tail = []
        self.tail_min = args.tail
        self.tail_max = 2 * args.tail
        self.last_stats = None
        self._fixed_tokens_cache = None
        self.full_attachments = {}      # name -> whole text (attach@)
        self.load_full_attachments()
        self.load_tail()
        self.kvtrace = KVTrace(self.trie.cache_dir,
                               self.trie.conversation_id)
        # drained at every dispatch: no reader sees ingest in flight
        self.ingest = IngestWorker(
            journal_path=self.trie.cache_dir / "ingest_failures.jsonl",
            synchronous=args.sync_ingest)

    def worker(self, name):
        """The handle for one roster entry, built on first use."""
        if self.roster is None:
            raise RosterError("No roster loaded - start saltChat with "
                              "--roster FILE.")
        if name not in self.workers:
            self.workers[name] = WorkerHandle(self.roster.get(name))
        return self.workers[name]

    def worker_handles(self):
        """Every roster entry's handle, in file order."""
        if self.roster is None:
            return []
        return [self.worker(e.name) for e in self.roster.entries]

    def compact_tail(self):
        """Cut the tail back to tail_min exchanges once it exceeds tail_max.
        Compaction only bounds the verbatim window: sentences entered the
        trie the moment they were spoken (with --short-turns keep that
        includes terse user turns), though an utterance repeated verbatim
        is stored once per session by the cross-turn dedupe."""
        if len(self.tail) > 2 * self.tail_max:
            del self.tail[: len(self.tail) - 2 * self.tail_min]

    def load_tail(self):
        self.tail = []
        try:
            tail = json.loads((self.trie.cache_dir / "tail.json").read_text())
        except (OSError, ValueError):
            return
        if not (isinstance(tail, list) and len(tail) % 2 == 0):
            return
        roles = ("user", "assistant")
        for i, m in enumerate(tail):
            if not (isinstance(m, dict) and m.get("role") == roles[i % 2]
                    and isinstance(m.get("content"), str)):
                return
        self.tail = tail
        self.compact_tail()

    def save_tail(self, tail=None):
        # mirrored to disk every turn so a resumed session renders the
        # same prompt bytes the server's cache already holds
        tmp = self.trie.cache_dir / "tail.json.tmp"
        tmp.write_text(json.dumps(self.tail if tail is None else tail))
        os.replace(tmp, self.trie.cache_dir / "tail.json")

    def new_trie(self, conversation_id, save_old=True):
        # save first: the ctor below loads this session's own state.pkl
        # when /new is given the same id. /clear passes save_old=False
        # (same id - saving would resurrect the wiped session).
        if save_old and self.trie.dirty:
            try:
                self.trie.save()
            except Exception as exc:
                raise RuntimeError(
                    f"could not save the current session "
                    f"{self.trie.conversation_id!r} before switching: "
                    f"{exc}") from exc
        # build both replacements before touching the live pair, so a
        # failed ctor leaves the current session fully usable
        trie = SessionTrie(conversation_id, cache_dir=SESSIONS_DIR,
                           model_name=BGE_MODEL,
                           budget_pct_default=self.budget)
        worker = IngestWorker(
            journal_path=trie.cache_dir / "ingest_failures.jsonl",
            synchronous=self.sync_ingest)
        self.ingest.close()
        self.trie = trie
        self.ingest = worker
        self.last_stats = None
        self.load_full_attachments()
        self.load_tail()
        self.kvtrace = KVTrace(self.trie.cache_dir,
                               self.trie.conversation_id)
        return self.trie

    # ── attach@ full-context attachments (persisted per session) ─────────
    def attachments_dir(self):
        return self.trie.cache_dir / "attachments"

    def load_full_attachments(self):
        self.full_attachments = {}
        d = self.attachments_dir()
        if not d.is_dir():
            return
        files = {f.name[:-4]: f for f in d.glob("*.txt")}
        # persisted attach order first: the system message must render the
        # same bytes across a restart or a warm server cache misses on it
        for name in self._attach_order():
            f = files.pop(name, None)
            if f is not None:
                self.full_attachments[name] = f.read_text(
                    encoding="utf-8", errors="replace")
        for name in sorted(files):
            self.full_attachments[name] = files[name].read_text(
                encoding="utf-8", errors="replace")

    def _attach_order(self):
        try:
            order = json.loads(
                (self.attachments_dir() / "order.json").read_text())
        except (OSError, ValueError):
            return []
        return [n for n in order if isinstance(n, str)]

    def save_full_attachment(self, name, text):
        # pypdf can emit lone surrogates from broken font CMaps; make the
        # text strictly encodable before it reaches disk or a prompt, and
        # only expose the attachment once the write has succeeded
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        d = self.attachments_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / (name + ".txt")).write_text(text, encoding="utf-8")
        self.full_attachments[name] = text
        # the live dict is the order the prompt renders, so it is the
        # order that must persist - order.json alone can lag it (sessions
        # that predate it, or a torn earlier write)
        tmp = d / "order.json.tmp"
        tmp.write_text(json.dumps(list(self.full_attachments)))
        os.replace(tmp, d / "order.json")

    def count_tokens(self, text):
        if self.runner is None:
            return None
        try:
            return len(self.runner.tokenizer(
                text, add_special_tokens=False).input_ids)
        except Exception:
            return None


def fresh_conversation_id():
    return datetime.now().strftime("chat-%Y%m%d-%H%M%S")


def valid_session_id(cid):
    return bool(SESSION_ID_RE.fullmatch(cid or ""))


def normalize_budget(val):
    """Accept 0.3 or 30 for 30%; None if out of range."""
    if val > 1:
        val /= 100.0
    return val if 0 < val <= 1 else None


def parse_memory_cap(val):
    """'off', 'auto', or a positive token count; None when unparseable."""
    v = str(val or "").strip().lower()
    if v in ("off", "auto"):
        return v
    try:
        n = int(v)
    except ValueError:
        return None
    return n if n > 0 else None


def conversation_sections(trie, idxs, turn_labels=True):
    """The conversation excerpts, cut into one section per contiguous
    (turn, role) run so the model can tell who said what and when instead
    of reading one anonymous block. Selection comes back ascending by
    sentence index and a message's sentences are appended together, so a
    run is exactly one speaker's turn, and higher turn numbers are later.
    A run whose role is unreadable (a session resumed from a build that
    stored something else) keeps the unlabeled header rather than
    inventing provenance."""
    if not turn_labels:
        return [CONVERSATION_LABEL + "\n"
                + " ".join(trie.texts[i] for i in idxs)]
    now = time.time()
    sections = []
    for (turn, role), run in groupby(
            idxs, key=lambda i: (trie.turns[i], trie.roles[i])):
        run = list(run)
        if role in VALID_ROLES and turn is not None:
            # sessions from before ingest time was recorded carry no stamp,
            # so those labels simply drop the age instead of guessing one
            filed_at = trie.timestamps[run[0]]
            head = (CONVERSATION_LABEL_AGE.format(
                        turn=turn, role=role, age=format_age(now - filed_at))
                    if filed_at is not None
                    else CONVERSATION_LABEL_TURN.format(turn=turn, role=role))
        else:
            head = CONVERSATION_LABEL
        sections.append(head + "\n"
                        + " ".join(trie.texts[i] for i in run))
    return sections


def format_age(seconds):
    """How long ago, coarsely: a label only has to place a statement in
    time, not measure it. A clock that went backwards reads as 'just now'
    rather than as a negative age."""
    minutes = seconds / 60.0
    if minutes < 1:
        return "just now"
    if minutes < 60:
        return f"{int(minutes)}m ago"
    hours = minutes / 60.0
    if hours < 24:
        return f"{int(hours)}h ago"
    return f"{int(hours / 24)}d ago"


def conversation_map(trie, n_turns=20, char_cap=600, top_k=3):
    """A compact index of what was discussed when: one line per recent
    conversation turn, `t12 user: retrieval, decay, budget`. Returns those
    lines and how many turns could have produced one, so every caller can
    say plainly when the map shows only the recent part of a longer
    conversation instead of implying it covers everything. Built purely
    from the attention keywords already cached at ingest, so it costs no
    model pass. Attachments are left out - they are inventoried elsewhere
    and one file would otherwise flood the map. Rows a session cap has
    masked are left out too: the map points at material selection can
    still reach, and a line for a retired turn is a pointer to nothing.
    When the character cap bites the OLDEST lines go, since the recent
    end is what orients a reader.

    Cached weights are renormalized per sentence upstream (they sum to 1
    across the words of their own sentence), so a raw weight means "share
    of this sentence's attention" and shrinks as the sentence grows.
    Ranking a turn's sentences against each other therefore needs a
    rescale, or a short aside outranks the long sentence carrying the
    turn's actual topic. The sqrt factor is an empirical flattener rather
    than a derivation: measured over the stored sessions it cuts the
    short-sentence advantage from 5.2x to 1.4x, where the full word count
    overshoots to 2.4x the other way."""
    by_turn = {}
    for i in range(trie.n_sentences):
        if trie.sources[i] is not None or not trie.alive[i]:
            continue
        role, kws = by_turn.setdefault(trie.turns[i], (trie.roles[i], {}))
        scale = math.sqrt(max(trie.n_words[i], 1))
        for word, weight in trie.keyword_weights[i].items():
            kws[word] = max(kws.get(word, 0.0), weight * scale)
    entries = []
    for turn in sorted(by_turn):
        role, kws = by_turn[turn]
        top = sorted(kws, key=lambda w: (-kws[w], w))[:top_k]
        if top:
            entries.append(f"t{turn} {role}: {', '.join(top)}")
    lines = entries[-n_turns:]
    while lines and sum(len(l) + 1 for l in lines) > char_cap:
        lines.pop(0)
    return lines, len(entries)


def format_memory_block(trie, sel_idx, turn_labels=True, conv_map=False):
    """The selected sentences as a labeled memory block: grouped by origin
    (attached files first, then conversation), each section headed with its
    source and per-file selected/total counts so the model knows both where
    an excerpt came from and how partial the selection is. Conversation
    excerpts additionally carry the turn and speaker they came from. The
    labels match the reading guide in instructions.md.

    `conv_map` prepends the conversation map as a first section: an index
    of what was discussed when, so the model can see a topic exists even
    on a turn none of its sentences were selected. It adds pointers only -
    selection is untouched, so the map is a signal and never a gate. The
    memory block is the only correct home for it, since it changes every
    turn and the system prompt has to stay a byte-stable KV prefix."""
    if not sel_idx:
        return ""
    by_src = {}
    for i in sel_idx:
        by_src.setdefault(trie.sources[i], []).append(i)
    totals = Counter(s for s in trie.sources if s)
    sections = []
    if conv_map:
        lines, total = conversation_map(trie)
        if lines:
            # the header states its own coverage: a map that silently drops
            # older turns would read as proof a topic never came up
            head = (CONVERSATION_MAP_LABEL if len(lines) == total
                    else CONVERSATION_MAP_LABEL_RECENT.format(
                        n=len(lines), total=total))
            sections.append(head + "\n" + "\n".join(lines))
    for src in sorted(k for k in by_src if k):
        idxs = by_src[src]
        # explicit quotes, not !r: repr flips to double quotes on names with
        # apostrophes, breaking the label format instructions.md documents
        sections.append(f"[from attached file '{src}' — {len(idxs)} of "
                        f"{totals[src]} indexed sentences]\n"
                        + " ".join(trie.texts[i] for i in idxs))
    if None in by_src:
        sections.extend(conversation_sections(trie, by_src[None], turn_labels))
    return MEMORY_BLOCK.format(body="\n\n".join(sections))


def tail_resident_sent_idx(trie, tail):
    """Row indices whose text the model is already reading verbatim in
    the tail, for compress(exclude_sent_idx=...). Conversation rows only:
    an attached document is never excluded on coincidental overlap.
    Matching is normalized space-bounded substring: a stored sentence is
    a whitespace-normalized piece cut from its message at word
    boundaries, so the exact run reappears space-delimited in the
    normalized message, while the padding keeps a short sentence from
    matching inside a longer word ("yes" in "yesterday"). It
    under-matches when ingest cleaning altered the text (the <url>
    substitution), which fails safe to selecting as before. Messages
    join on newline, which normalized text cannot contain, so a needle
    never matches across two messages. When every living row is
    tail-resident there is nothing else to select, so no exclusion is
    returned."""
    if not tail:
        return set()
    hay = "\n".join(
        " " + " ".join((m.get("content") or "").lower().split()) + " "
        for m in tail)
    idx = set()
    for i in range(trie.n_sentences):
        if not trie.alive[i] or trie.sources[i] is not None:
            continue
        needle = " ".join(trie.texts[i].lower().split())
        if needle and f" {needle} " in hay:
            idx.add(i)
    if len(idx) >= trie.n_alive:
        return set()
    return idx


def attachment_inventory(trie, full_attachments):
    """One system-prompt line per attached file, so the model knows a file
    exists even on turns when none of its sentences were selected."""
    entries = {}
    for src in trie.attached_sources:
        entries[src] = (f"indexed by SALT ({trie.sources.count(src)} "
                        f"sentences); relevant excerpts appear in the "
                        f"'SALT memory' block")
    for name in full_attachments:
        prev = entries.get(name)
        entries[name] = (prev + "; also provided in full below" if prev
                         else "provided in full below")
    if not entries:
        return ""
    return ("Files attached to this conversation:\n"
            + "\n".join(f"- '{n}': {d}" for n, d in sorted(entries.items())))


def build_messages(memory_block, tail, user_msg, attachments=None,
                   inventory="", instructions=""):
    """Cache-shaped prompt: the system message carries only STABLE content
    (instructions, inventory, attach@ full texts), the verbatim tail follows
    append-only, and the per-turn SALT memory rides at the top of the newest
    user message. Everything ahead of the memory block is therefore a
    reusable prefix for KV caching; the volatile 20% selection and the
    question are the only per-turn prefill. The tail stores the clean user
    line, so injected memory never enters history."""
    system = SYSTEM_PROMPT
    if instructions:
        system += "\n\n" + instructions
    if inventory:
        system += "\n\n" + inventory
    for name, text in (attachments or {}).items():
        system += (f"\n\nAttached document '{name}' (full text):"
                   f"\n---\n{text}\n---")
    user = (memory_block + "\n\n" + user_msg) if memory_block else user_msg
    return ([{"role": "system", "content": system}]
            + list(tail)
            + [{"role": "user", "content": user}])


def print_models(active=None):
    models = list_models()
    if not models:
        print("  (none registered - add one with: saltChat --add <hf_id>)")
        return
    width = max(len(m["alias"]) for m in models)
    for m in models:
        mark = "*" if m["alias"] == active else " "
        state = "ready" if m["downloaded"] else "missing weights"
        print(f" {mark} {m['alias']:<{width}}  {m['hf_id']}  "
              f"[{m['dtype']}, {state}]")


def print_roster(roster, probes=None):
    if roster is None:
        print("  (no roster loaded - start saltChat with --roster FILE, "
              "e.g. salt/agents/roster_sample.json)")
        return
    probes = probes or {}
    head = ("NAME", "ROLE", "ALIAS", "MODE", "ENDPOINT", "STATE")
    rows = [(e.name, e.role, e.alias, "attach" if e.attach else "spawn",
             e.server_url if e.attach else f"port {e.spawn['port']}",
             probes.get(e.name, UNPROBED).state) for e in roster.entries]
    width = [max(len(r[i]) for r in (head,) + tuple(rows))
             for i in range(len(head))]

    def render(row):
        return "  ".join(f"{c:<{w}}" for c, w in zip(row, width)).rstrip()

    print(f"  {render(head)}")
    for entry, row in zip(roster.entries, rows):
        print(f"  {render(row)}")
        note = probes.get(entry.name, UNPROBED).note
        if note:
            print(f"      {note}")
    print(f"  from {roster.path}")


def print_workers(handles):
    if not handles:
        print("  (no roster loaded - start saltChat with --roster FILE, "
              "e.g. salt/agents/roster_sample.json)")
        return
    head = ("NAME", "ROLE", "STATE", "CALLS", "MEAN", "ENDPOINT")
    rows = [(h.name, h.role, h.state, str(h.calls),
             f"{h.mean_latency:.1f}s" if h.calls else "-", h.endpoint)
            for h in handles]
    width = [max(len(r[i]) for r in (head,) + tuple(rows))
             for i in range(len(head))]

    def render(row):
        return "  ".join(f"{c:<{w}}" for c, w in zip(row, width)).rstrip()

    print(f"  {render(head)}")
    for handle, row in zip(handles, rows):
        print(f"  {render(row)}")
        if handle.note:
            print(f"      {handle.note}")


def _user_keep(t):
    # add_turn's default protects code/table/link units; user turns must
    # keep that protection alongside the short-turn one
    return is_short_user_unit(t) or is_protected_chat_unit(t)


def warn_load_repair(trie):
    r = getattr(trie, "load_repair", None)
    if not r:
        return
    parts = []
    if r["orphan_rows"]:
        parts.append(f"dropped {r['orphan_rows']} orphan embedding rows")
    if r["dropped_sentences"]:
        parts.append(f"removed {r['dropped_sentences']} sentences whose "
                     f"vectors were lost")
    print(f"Session repaired on load: {' and '.join(parts)} "
          f"(details kept in load_repairs.jsonl).")


def add_to_trie(state, text, role, source=None, sentences=None, keep=None,
                save=True, context=None):
    dedup_cos = state.dedup_cos
    if keep is None and role == "user" and state.short_turns != "off":
        keep = _user_keep
        if (state.short_turns == "fuse" and context
                and acknowledgement_only(text)):
            text = fuse_with_question(text, context)
            # A fused ack quotes the SAME question on both sides, so "yes
            # [in reply to: Q]" and "no [in reply to: Q]" score near
            # identical. The near-dup gate would drop the reversal and keep
            # the opposite decision, so fused acks skip it.
            dedup_cos = None
    return state.trie.add_turn(text, role=role, tokenizer=state.bge_tok,
                               model=state.bge_model, device=state.bge_device,
                               source=source, sentences=sentences, keep=keep,
                               dedup_cos=dedup_cos,
                               max_sentences=state.max_sentences, save=save)


def submit_ingest(state, text, role, save=True, context=None):
    """Queue one side of an exchange for background ingest (inline under
    --sync-ingest, where a failure raises here)."""
    state.ingest.submit(lambda: add_to_trie(state, text, role, save=save,
                                            context=context),
                        label=f"{role}-message ingest", payload=text)


def submit_session_save(state):
    """One coalesced save per turn, queued FIFO behind the turn's
    encodes. A no-op when the trie is not dirty."""
    state.ingest.submit(
        lambda: state.trie.save() if state.trie.dirty else None,
        label="session save")


def submit_tail_save(state):
    """Queued FIFO behind the exchange's ingests, so tail.json can never
    hold a pair the trie has not absorbed."""
    snapshot = list(state.tail)
    state.ingest.submit(lambda: state.save_tail(snapshot),
                        label="tail save")


def report_ingest_failures(failures):
    # the worker never prints - failures surface here, at the next barrier
    for f in failures:
        if f.get("payload") is None:
            note = ("traceback in ingest_failures.jsonl"
                    if f.get("journaled") else "journal write also failed")
        elif f.get("journaled"):
            note = "message text kept in ingest_failures.jsonl"
        else:
            note = "the journal write ALSO failed, text not preserved"
        print(f"[ingest] {f['label']} failed: {f['error']} - {note}")


def close_ingest(state):
    """Drain and stop the worker at exit. Returns True when the queue
    finished cleanly, so the caller may trust `trie.dirty`."""
    try:
        # boundary save: an error turn never queued its coalesced save
        submit_session_save(state)
    except RuntimeError:
        pass                            # already closed (repeat call)
    except KeyboardInterrupt:
        print("\n[ingest] interrupted before the final save was queued")
        return False
    try:
        report_ingest_failures(state.ingest.close())
        return True
    except KeyboardInterrupt:
        print(f"\n[ingest] interrupted - {state.ingest.pending} queued "
              f"job(s) still finishing on exit")
        return False


def ingest_doc(state, path):
    p = Path(path).expanduser()
    if not p.is_file():
        print(f"No such file: {p}")
        return
    try:
        text, n_pages = read_document(p)
    except ExtractionError as exc:
        print(exc)
        return
    merging = p.name in state.trie.attached_sources
    info = add_to_trie(state, text, "doc", source=p.name,
                       sentences=split_document_sentences(text),
                       keep=is_protected_unit)
    if info["added"] == 0:
        if merging:
            print(f"{p.name}: nothing new to add (already attached).")
        else:
            print(f"{p.name}: no ingestible sentences - all "
                  f"{info['filtered']} extracted units were filtered "
                  f"(references/fragments).")
        return
    pages = f"{n_pages} pages, " if n_pages else ""
    branch = (f"merged into the existing {p.name!r} branch" if merging
              else "under its own branch")
    print(f"Attached {p.name}: {pages}{info['added']} sentences "
          f"({info['filtered']} filtered) {branch}; "
          f"{info['n_total']} total in session.")
    if merging:
        print(f"note: a source named {p.name!r} was already attached - "
              f"same-named files share one trie branch.")
    warn_prompt_budget(state)


def staged_files():
    if not FILES_DIR.is_dir():
        return []
    return sorted(p for p in FILES_DIR.iterdir()
                  if p.is_file() and p.suffix.lower() in FILE_SUFFIXES
                  and p.name != "README.md")


def resolve_staged(name):
    """Bare names resolve ONLY in the staging dir (matching what the listing
    and TAB completion show); explicit paths must look like paths."""
    if "/" in name or name.startswith("~"):
        candidate = Path(name).expanduser()
    else:
        candidate = FILES_DIR / name
        if not candidate.is_file() and "." not in name:
            alt = FILES_DIR / (name + ".pdf")
            if alt.is_file():
                candidate = alt
    return candidate if candidate.is_file() else None


def list_staged(state):
    files = staged_files()
    if not files:
        print(f"No attachable files - drop .pdf/.txt/.md/.rst into {FILES_DIR}")
        return
    in_trie = set(state.trie.attached_sources)
    in_full = set(state.full_attachments)
    for f in files:
        marks = (("*" if f.name in in_trie else " ")
                 + ("+" if f.name in in_full else " "))
        print(f" {marks} {f.name}  ({f.stat().st_size / 1024:.0f} KB)")
    print("   (* = in trie via salt@, + = full context via attach@)")


def handle_salt_at(state, line):
    """`salt@` lists the staging dir; `salt@<name>` ingests into the trie."""
    name = line[len("salt@"):].strip()
    if not name:
        list_staged(state)
        return
    candidate = resolve_staged(name)
    if candidate is None:
        print(f"No such file {name!r} - `salt@` lists what's in {FILES_DIR}")
        return
    ingest_doc(state, candidate)


def handle_attach_at(state, line):
    """`attach@<name>` keeps a file's WHOLE text in every prompt - the
    uncompressed, full-context counterpart of salt@."""
    name = line[len("attach@"):].strip()
    if not name:
        list_staged(state)
        return
    candidate = resolve_staged(name)
    if candidate is None:
        print(f"No such file {name!r} - `attach@` lists what's in {FILES_DIR}")
        return
    try:
        text, n_pages = read_document(candidate)
    except ExtractionError as exc:
        print(exc)
        return
    replacing = candidate.name in state.full_attachments
    state.save_full_attachment(candidate.name, text)
    n_tok = state.count_tokens(text)
    pages = f"{n_pages} pages, " if n_pages else ""
    toks = f"~{n_tok} tokens" if n_tok is not None else f"{len(text)} chars"
    print(f"Attached {candidate.name} in full: {pages}{toks} - "
          f"included in every prompt from now on.")
    if replacing:
        print(f"note: replaced the previous full-context attachment named "
              f"{candidate.name!r}.")
    warn_prompt_budget(state)


def prompt_fixed_tokens(state):
    """Tokens the prompt spends before the memory block: the system
    message build_messages assembles (base prompt, instructions,
    inventory, attach@ full texts) plus the verbatim tail. The system
    part is tokenized once per model/attachment/instructions change and
    cached on state; the tail is small and recounted each call. None
    when no runner is loaded or tokenization fails."""
    if state.runner is None:
        return None
    instructions = load_instructions()
    inventory = attachment_inventory(state.trie, state.full_attachments)
    system = SYSTEM_PROMPT
    if instructions:
        system += "\n\n" + instructions
    if inventory:
        system += "\n\n" + inventory
    for name, text in state.full_attachments.items():
        system += (f"\n\nAttached document '{name}' (full text):"
                   f"\n---\n{text}\n---")
    key = (tuple(state.full_attachments), instructions, inventory,
           state.runner.alias)
    if state._fixed_tokens_cache is None or \
            state._fixed_tokens_cache[0] != key:
        n = state.count_tokens(system)
        if n is None:
            return None
        state._fixed_tokens_cache = (key, n)
    fixed = state._fixed_tokens_cache[1]
    if not state.tail:
        return fixed
    tail_n = state.count_tokens("\n".join(m["content"] for m in state.tail))
    return None if tail_n is None else fixed + tail_n


def memory_word_cap(state, user_line=""):
    """Word ceiling for the memory block, or None when the cap is off or
    the inputs are unknowable. 'auto' fits the block to what the window
    has left after the fixed prompt, the user line and a reply reserve;
    an integer cap converts that many tokens directly. Either way the
    token count becomes words via the session's measured tokens-per-word
    ratio, floored so the block never collapses to nothing."""
    cap = state.memory_cap
    if cap in (None, "off"):
        return None
    if cap == "auto":
        if state.runner is None:
            return None
        limit = state.runner.input_budget()
        fixed = prompt_fixed_tokens(state)
        if not limit or fixed is None:
            return None
        line_tokens = state.count_tokens(user_line) or 0
        tokens = int(limit) - fixed - MEMORY_CAP_RESERVE - line_tokens
    else:
        tokens = int(cap)
    words = int(tokens / max(state.tokens_per_word, 0.1))
    return max(words, MEMORY_CAP_FLOOR_WORDS)


def warn_prompt_budget(state):
    """Warn when the whole prompt - the fixed head plus the expected
    memory block - is headed past the model's usable input ceiling (the
    runner tail-truncates, and the head holds the instructions the
    memory-block labels depend on)."""
    if state.runner is None:
        return
    limit = int(state.runner.input_budget() or 0)
    if not limit:
        return
    fixed = prompt_fixed_tokens(state)
    if fixed is None:
        return
    cap_words = memory_word_cap(state)
    if cap_words is not None:
        mem_tokens = int(cap_words * state.tokens_per_word)
    else:
        mem_tokens = int(state.trie.live_words * state.budget
                         * state.tokens_per_word)
    total = fixed + mem_tokens
    if total <= limit:
        return
    dominant = ("the fixed prompt (attachments, instructions, tail)"
                if fixed >= mem_tokens else "the compressed memory block")
    print(f"warning: the prompt is headed for ~{total} tokens ({fixed} "
          f"fixed + ~{mem_tokens} memory block), over the model's usable "
          f"input ceiling ({limit} = context window minus reply headroom); "
          f"{dominant} dominates. Prompts will be tail-truncated and the "
          f"earliest content dropped - consider --memory-cap auto, salt@ "
          f"instead of attach@ for large files, or a longer-context model.")


def switch_model(state, name):
    try:
        cfg = resolve_model(name)
    except RegistryError as exc:
        print(exc)
        return
    if state.runner is not None and cfg["alias"] == state.runner.alias:
        print(f"{cfg['alias']} is already active.")
        return
    if state.backend == "vllm-serve":
        print(f"The vllm-serve backend connects to one server per launch. "
              f"Start saltServe {cfg['alias']} on another port and relaunch "
              f"saltChat with --server-url pointing at it.")
        return
    prev_cfg = state.runner.cfg
    state.runner.unload()  # free before load: never two LLMs on the GPU
    state.runner = None
    try:
        state.runner = make_runner(cfg, device=state.device,
                                   backend=state.backend,
                                   **state.backend_opts)
        warn_prompt_budget(state)  # new model may have a smaller window
    except Exception as exc:
        print(f"Failed to load {cfg['alias']}: {exc}")
        gc.collect()  # drop the failed load's partial allocations first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Reloading previous model {prev_cfg['alias']} ...")
        try:
            state.runner = make_runner(prev_cfg, device=state.device,
                                       backend=state.backend,
                                       **state.backend_opts)
        except Exception as exc2:
            print(f"Also failed to reload {prev_cfg['alias']}: {exc2}")
            print("No model loaded - use /model <name> when ready.")


def handle_command(line, state):
    """Dispatch a slash command. Returns False to exit the REPL."""
    parts = line.split()
    cmd, rest = parts[0].lower(), parts[1:]

    if cmd in ("/exit", "/quit", "/q"):
        return False
    if cmd == "/help":
        print(HELP)
    elif cmd == "/model":
        if not rest:
            print_models(active=state.runner.alias if state.runner else None)
        else:
            switch_model(state, rest[0])
    elif cmd == "/add":
        if not rest:
            print("Usage: /add <hf_id> [alias]")
        else:
            try:
                cfg = register_model(rest[0],
                                     alias=rest[1] if len(rest) > 1 else None)
                print(f"Registered {cfg['hf_id']} as {cfg['alias']!r}.")
            except RegistryError as exc:
                print(exc)
    elif cmd == "/roster":
        if rest and rest[0].lower() != "probe":
            print("Usage: /roster [probe]")
        elif rest and state.roster is None:
            print_roster(None)
        else:
            handles = state.worker_handles()
            if rest:
                print(f"Probing {len(handles)} roster "
                      f"entr{'y' if len(handles) == 1 else 'ies'} ...")
                for handle in handles:
                    handle.probe()
            print_roster(state.roster,
                         {h.name: h.probe_result for h in handles})
    elif cmd == "/worker":
        if rest and (rest[0].lower() != "probe" or len(rest) != 2):
            print("Usage: /worker [probe <name>]")
        elif rest and state.roster is None:
            print_workers([])
        else:
            if rest:
                try:
                    handle = state.worker(rest[1])
                except RosterError as exc:
                    print(exc)
                    return True
                print(f"Probing worker {handle.name!r} at "
                      f"{handle.endpoint} ...")
                handle.probe()
            print_workers(state.worker_handles())
    elif cmd == "/doc":
        if not rest:
            print("Usage: /doc <path>")
        else:
            ingest_doc(state, " ".join(rest))
    elif cmd == "/budget":
        try:
            val = normalize_budget(float(rest[0]))
        except (IndexError, ValueError):
            val = None
        if val is None:
            print("Usage: /budget <pct>   e.g. /budget 0.3 or /budget 30")
            return True
        state.budget = val
        print(f"Memory budget set to {val:.0%}.")
    elif cmd == "/stats":
        t = state.trie
        counted = (f"{t.n_sentences} sentences" if not t.n_masked
                   else f"{t.n_alive} of {t.n_sentences} sentences live")
        print(f"session {t.conversation_id!r}: {counted} over "
              f"{t.n_turns} turns, budget {state.budget:.0%}, "
              f"model {state.runner.alias if state.runner else 'none'}")
        files = t.attached_sources
        if files:
            print(f"trie attachments ({len(files)}): {', '.join(files)}")
        if state.full_attachments:
            parts = []
            for name, text in state.full_attachments.items():
                n_tok = state.count_tokens(text)
                parts.append(f"{name} (~{n_tok} tok)" if n_tok else name)
            print(f"full-context attachments ({len(parts)}): {', '.join(parts)}")
        cmap, cmap_total = conversation_map(t)
        if cmap:
            span = (f"all {cmap_total}" if len(cmap) == cmap_total
                    else f"most recent {len(cmap)} of {cmap_total}")
            print(f"conversation map ({span} turns):")
            for line in cmap:
                print(f"  {line}")
        s = state.last_stats or {}
        if s:
            trie_info = s.get("trie", {})
            print(f"last compression: theme coverage "
                  f"{s.get('theme_coverage_pct', 0):.1%}, "
                  f"{trie_info.get('n_nodes', '?')} nodes / "
                  f"{trie_info.get('n_branches', '?')} branches")
        fixed = prompt_fixed_tokens(state)
        if fixed is not None:
            limit = state.runner.input_budget()
            of = f" of {int(limit)} usable input tokens" if limit else ""
            print(f"prompt fixed cost: ~{fixed} tokens ahead of the "
                  f"memory block{of}")
        # reported from ChatState, not last_stats: the setting is visible
        # even before the first compress of a session
        if state.coverage_half_life:
            keys = s.get("coverage_keys")
            print(f"coverage decay: half-life "
                  f"{state.coverage_half_life:g} turns"
                  + (", attached files included" if state.coverage_decay_docs
                     else "")
                  + (f", {keys} theme keys tracked" if keys is not None
                     else ""))
        if state.tail_exclude:
            n = s.get("excluded_sent")
            print("tail exclusion: on"
                  + (f" - {n} tail-resident sentences left out last turn"
                     if n is not None else ""))
        else:
            print("tail exclusion: off this launch (--no-tail-exclude)")
        if state.shift_damping:
            print(f"shift damping: x{state.shift_damping:g} stale-seed "
                  f"scale on shift turns, query boost "
                  f"x{state.shift_query_boost:g}")
        if s.get("coverage_seed_keys") is not None:
            print(f"coverage seed: {s['coverage_seed_matched']} of "
                  f"{s['coverage_seed_keys']} carried-over keys matched "
                  f"this turn's memory tree, {s['coverage_orphan_keys']} "
                  f"orphaned (mass {s['coverage_orphan_mass']:g})")
        if s.get("coverage_persisted_orphans") is not None:
            print(f"coverage dict: {s['coverage_persisted_live']} live "
                  f"keys, {s['coverage_persisted_orphans']} orphaned "
                  f"({s['coverage_orphan_doc_keys']} from attachments, "
                  f"mass {s['coverage_persisted_orphan_mass']:g})")
        if state.per_source_themes:
            note = ""
            if s.get("theme_scope") == "source":
                note = (f" - last turn profiled {s.get('theme_sources')} "
                        f"sources, {s.get('theme_keywords_conv')} "
                        f"conversation theme keywords")
            print(f"theme scope: per-source (--per-source-themes){note}")
        # count read from the trie, not last_stats: suppression happens at
        # ingest, and a resumed session carries its count even when the
        # gate is off this launch
        if state.dedup_cos:
            print(f"near-dup gate: skip conversation sentences at cos >= "
                  f"{state.dedup_cos:g}; {t.n_near_dups} suppressed so far "
                  f"(near_dups.jsonl has each one)")
        elif t.n_near_dups:
            print(f"near-dup gate: off this launch; {t.n_near_dups} "
                  f"sentences suppressed earlier in this session")
        # same reason as the near-dup counter: masking happens at ingest,
        # so a resumed session reports its state even with the flag off
        if state.max_sentences:
            bounded = (state.coverage_gc or state.coverage_max_keys
                       or state.coverage_half_life
                       or state.stable_coverage_keys)
            note = ("" if bounded else
                    " - no coverage bound is on, so the theme keys those "
                    "sentences left behind stay in the dictionary")
            print(f"session cap: {state.max_sentences} conversation "
                  f"sentences kept, {t.n_masked} masked out of memory so "
                  f"far{note}")
        elif t.n_masked:
            print(f"session cap: off this launch; {t.n_masked} sentences "
                  f"masked earlier in this session")
        ing = state.ingest.stats
        fail_note = (f", {ing['failures']} failed (ingest_failures.jsonl)"
                     if ing["failures"] else "")
        if state.sync_ingest:
            print(f"ingest: synchronous (--sync-ingest) - {ing['jobs']} "
                  f"jobs, {ing['busy_s']:.1f}s inline{fail_note}")
        else:
            print(f"ingest: background - {ing['jobs']} jobs, "
                  f"{ing['busy_s']:.1f}s of encode kept off the prompt "
                  f"path{fail_note}"
                  + (f", {state.ingest.pending} in flight"
                     if state.ingest.pending else ""))
        if s.get("drift_cos") is not None:
            base = s.get("drift_ema")
            base_str = (f"{base:.3f}" if base is not None
                        else "n/a (first measure)")
            mark = ""
            if s.get("topic_shift"):
                mark = " - TOPIC SHIFT"
                if s.get("shift_damped"):
                    mark += (f" ({s.get('shift_damped_keys', 0)} stale keys "
                             f"damped)")
            # margin always shown: detection runs even with damping off,
            # precisely so the margin can be tuned before trusting it
            print(f"topic drift: cos {s['drift_cos']:.3f} vs ema {base_str} "
                  f"(margin {state.shift_margin:g}){mark}")
        kv = state.kvtrace
        if kv.last_event:
            u, tot = kv.last_event["usage"], kv.totals
            print(f"kv ledger: turn {kv.last_event['turn']} - "
                  f"read {u['input_cached_tokens']}, write {u['input']}, "
                  f"output {u['output']} tok | session totals "
                  f"read {tot['input_cached_tokens']}, write {tot['input']}, "
                  f"output {tot['output']}")
        apc = (kv.last_event or {}).get("apc_cached_tokens")
        if apc is not None:
            n = (kv.last_event or {}).get("apc_prompt_tokens") or 0
            print(f"APC: {apc}/{n} prompt tokens served from the engine's "
                  f"prefix cache" + (f" ({apc / n:.0%})" if n else ""))
        elif (kv.last_event or {}).get("engine_backend") == "vllm-serve":
            print("APC: the server reported no cache stats this turn - if "
                  "this persists, start it with "
                  "--enable-prompt-tokens-details (saltServe passes it)")
        if state.roster is not None:
            # reports what is already known: /roster probe and /worker probe
            # are what contact an endpoint, never /stats
            told = [f"{h.name} {h.state}"
                    + (f" ({h.calls} calls, {h.mean_latency:.1f}s mean)"
                       if h.calls else "")
                    for h in state.worker_handles()]
            print(f"workers: {len(told)} in the roster - {', '.join(told)}")
        if torch.cuda.is_available():
            dev = state.device if str(state.device).startswith("cuda") else None
            print(f"GPU memory allocated ({state.device}): "
                  f"{torch.cuda.memory_allocated(dev) / 2**30:.2f} GiB")
    elif cmd == "/new":
        cid = rest[0] if rest else fresh_conversation_id()
        if not valid_session_id(cid):
            print("Session ids may only contain letters, digits, '.', '_', '-'.")
            return True
        try:
            trie = state.new_trie(cid)
        except Exception as exc:
            print(f"Could not open session {cid!r}: {exc}")
            return True
        print(f"{'Resumed' if trie.is_loaded else 'Started'} session {cid!r}.")
        warn_load_repair(trie)
    elif cmd == "/clear":
        cid = state.trie.conversation_id
        target = state.trie.cache_dir.resolve()
        if SESSIONS_DIR.resolve() not in target.parents:
            print(f"Refusing to wipe {target}: not under {SESSIONS_DIR}.")
            return True
        shutil.rmtree(target, ignore_errors=True)
        state.new_trie(cid, save_old=False)
        print(f"Session {cid!r} wiped.")
    else:
        print(f"Unknown command {cmd!r} - /help lists commands.")
    return True


def chat_turn(state, line):
    if state.runner is None:
        print("No chat model loaded - /model <name> to load one.")
        return
    # barrier: this turn's trie reads need the previous turn's ingest done
    report_ingest_failures(state.ingest.drain())
    ts_start = datetime.now().isoformat(timespec="seconds")
    # trie holds turns 1..N-1 here (this message is added after generation):
    # the verbatim tail covers recent turns, the trie covers older ones
    # (selection honors that division while a sentence still rides in
    # the tail, unless --no-tail-exclude)
    memory_block, selected_idx, drift_extra, commit = "", [], None, None
    if state.trie.n_sentences > 0:
        excl = (tail_resident_sent_idx(state.trie, state.tail)
                if state.tail_exclude else None)
        comp = state.trie.compress(query=line, budget_pct=state.budget,
                                   tokenizer=state.bge_tok,
                                   model=state.bge_model,
                                   device=state.bge_device,
                                   coverage_half_life=state.coverage_half_life,
                                   coverage_decay_docs=state.coverage_decay_docs,
                                   shift_damping=state.shift_damping,
                                   shift_margin=state.shift_margin,
                                   shift_query_boost=state.shift_query_boost,
                                   per_source_themes=state.per_source_themes,
                                   max_words=memory_word_cap(state, line),
                                   stable_keys=state.stable_coverage_keys,
                                   coverage_gc=state.coverage_gc,
                                   coverage_max_keys=state.coverage_max_keys,
                                   defer_commit=True,
                                   exclude_sent_idx=excl)
        selected_idx = comp["selected_sent_idx"]
        commit = comp.get("commit")
        state.last_stats = comp["stats"]
        memory_block = format_memory_block(state.trie, selected_idx,
                                           state.turn_labels,
                                           state.conversation_map)
        n_blk = state.count_tokens(memory_block)
        if n_blk:
            tpw = n_blk / max(1, len(memory_block.split()))
            state.tokens_per_word = (TPW_EMA_ALPHA * tpw
                                     + (1 - TPW_EMA_ALPHA)
                                     * state.tokens_per_word)
        # additive ledger fields; only on turns where detection actually ran
        if comp["stats"].get("drift_cos") is not None:
            drift_extra = {"drift_cos": comp["stats"]["drift_cos"],
                           "topic_shift": bool(comp["stats"].get("topic_shift"))}

    inventory = attachment_inventory(state.trie, state.full_attachments)
    # unconditional: gating this on memory/attachments flips the system
    # prompt after turn 1 and invalidates the whole KV prefix
    instructions = load_instructions()
    # this turn's trie reads are done, so the user line encodes while
    # the model generates (sync mode keeps the post-reply position below)
    if not state.sync_ingest:
        submit_ingest(state, line, "user", save=False,
                      context=state.tail[-1]["content"] if state.tail
                      else None)
    messages = build_messages(memory_block, state.tail, line,
                              state.full_attachments, inventory, instructions)
    # cleared per turn so an interrupt before tokenization can't record the
    # previous turn's prompt size for this one
    state.runner.last_prompt_tokens = None
    state.runner.last_engine_stats = None
    print(f"{state.runner.alias}> ", end="", flush=True)
    pieces = []
    interrupted = False
    gen_ok = False
    try:
        for piece in state.runner.stream_chat(messages):
            print(piece, end="", flush=True)
            pieces.append(piece)
        gen_ok = True
    except KeyboardInterrupt:
        # an interrupt produced real partial output that is ingested
        # below, so its bookkeeping commits like a finished turn
        interrupted = True
        gen_ok = True
    print("\n" if not interrupted else "  [interrupted]\n")
    # the turn's coverage/EMA bookkeeping lands only now that the model
    # actually answered - a runner error skips this, so the retry sees
    # the same memory this attempt did. save=False: the coalesced
    # session save below persists it FIFO behind this turn's encodes.
    if commit is not None and gen_ok:
        commit(save=False)

    reply = "".join(pieces).strip()
    # no drain here (it would put a big paste's leftover encode back on
    # the prompt path): record_turn reads only pre-turn rows, and
    # appends never move them
    extra = dict(drift_extra or {})
    extra.update(getattr(state.runner, "last_engine_stats", None) or {})
    try:
        state.kvtrace.record_turn(
            tokenizer=state.runner.tokenizer, trie=state.trie,
            selected_idx=selected_idx, reply_text=reply,
            model_id=state.runner.cfg["hf_id"], ts_start=ts_start,
            ts_end=datetime.now().isoformat(timespec="seconds"),
            prompt_tokens=getattr(state.runner, "last_prompt_tokens", None),
            extra=extra or None)
    except Exception as exc:
        print(f"[kvtrace] recording failed for this turn: {exc}")
    # sync mode ingests both sides here, post-reply, as before the worker
    if state.sync_ingest:
        submit_ingest(state, line, "user",
                      context=state.tail[-1]["content"] if state.tail
                      else None)
    if reply:
        submit_ingest(state, reply, "assistant",
                      save=state.sync_ingest)  # seam: --no-assistant-memory
        # tail only grows in pairs so strict chat templates always see
        # alternating roles (a replyless user turn still reaches the trie)
        state.tail.append({"role": "user", "content": line})
        state.tail.append({"role": "assistant", "content": reply})
        state.compact_tail()
        submit_tail_save(state)
    if not state.sync_ingest:
        submit_session_save(state)
    return reply


_TURN_TEXT_KEYS = ("prompt", "question", "puzzle", "text", "content",
                   "message", "user", "turn")


def _parse_turns_file(raw):
    """A --turns file is a JSON array of items, or JSONL (one JSON value per
    non-blank line). Returns the list of items."""
    raw = raw.strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [json.loads(l) for l in raw.splitlines() if l.strip()]
    return data if isinstance(data, list) else [data]


def _turn_text(item, field, i):
    """The user message for one turn item: a bare string, or a field of an
    object (explicit --turns-field, else a common key, else its lone string
    field)."""
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        raise ValueError(f"turn {i} is neither a string nor an object")
    if field is not None:
        if field not in item:
            raise ValueError(f"turn {i} has no field {field!r}")
        return str(item[field])
    for k in _TURN_TEXT_KEYS:
        if isinstance(item.get(k), str):
            return item[k]
    strings = [k for k, v in item.items() if isinstance(v, str) and k != "id"]
    if len(strings) == 1:
        return item[strings[0]]
    raise ValueError(
        f"turn {i}: no obvious message text - pass --turns-field with the "
        f"key that holds it")


def load_turns(path, field=None):
    """Read a --turns file into an ordered list of (id, text) user turns."""
    items = _parse_turns_file(Path(path).read_text(encoding="utf-8"))
    turns = []
    for i, item in enumerate(items):
        turn_id = item.get("id") if isinstance(item, dict) else None
        turns.append((turn_id, _turn_text(item, field, i)))
    return turns


def run_turns(state, turns, out_path=None):
    """Feed a scripted list of user turns through the chat path one after
    another, so SALT's memory builds across them exactly as in a live
    session. Every backend works, including --backend vllm-serve. With
    out_path, each answer is appended to a JSONL file."""
    out = open(out_path, "w", encoding="utf-8") if out_path else None
    try:
        for i, (turn_id, text) in enumerate(turns):
            label = turn_id if turn_id is not None else i
            print(f"\n=== turn {i + 1}/{len(turns)} [{label}] ===")
            print(f"you> {text}")
            report_ingest_failures(state.ingest.drain())
            reply = None
            try:
                reply = chat_turn(state, text)
            except KeyboardInterrupt:
                print("\n[interrupted - stopping the run]")
                break
            except Exception as exc:
                print(f"[turn {label} failed: {exc}]")
            if out is not None:
                out.write(json.dumps(
                    {"id": turn_id, "turn": i, "question": text,
                     "answer": reply}, ensure_ascii=False) + "\n")
                out.flush()
    finally:
        if out is not None:
            out.close()


def _setup_completion():
    """TAB completes /commands and salt@<staged file> where readline exists."""
    try:
        import readline
    except ImportError:
        return
    def complete(text, i):
        if text.startswith("salt@") or text.startswith("attach@"):
            at = text.index("@") + 1
            prefix = text[at:]
            opts = [text[:at] + f.name for f in staged_files()
                    if f.name.startswith(prefix)]
        elif text.startswith("/"):
            opts = [c for c in COMMANDS if c.startswith(text)]
        else:
            opts = []
        return opts[i] if i < len(opts) else None

    readline.set_completer_delims(" \t\n")
    readline.set_completer(complete)
    readline.parse_and_bind("tab: complete")


def repl(state):
    _setup_completion()
    print("saltChat ready - /help lists commands, salt@ lists attachable "
          "files, /exit leaves.\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        try:
            # barrier: no dispatch sees a trie with ingest in flight
            report_ingest_failures(state.ingest.drain())
            if line.startswith("salt@"):
                handle_salt_at(state, line)
            elif line.startswith("attach@"):
                handle_attach_at(state, line)
            elif line.startswith("/"):
                if not handle_command(line, state):
                    break
            else:
                chat_turn(state, line)
        except KeyboardInterrupt:
            print("\n[interrupted]")
        except Exception as exc:
            # the REPL must survive any command/turn failure
            print(f"[error] {type(exc).__name__}: {exc}")


def build_parser():
    p = argparse.ArgumentParser(
        prog="saltChat",
        description="Chat with a registered model; SALT's persistent trie "
                    "compresses the conversation memory every turn.")
    p.add_argument("--model",
                   help="registered alias or HF id to chat with (default: "
                        "the single registered model, if there is one)")
    p.add_argument("--add", metavar="HF_ID",
                   help="download + register a model by HuggingFace id, then exit")
    p.add_argument("--alias", help="short name for --add (default: repo name, lowercased)")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="dtype recorded by --add (default: bfloat16)")
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing registry entry on --add")
    p.add_argument("--list", action="store_true",
                   help="list registered models, then exit")
    p.add_argument("--conversation-id",
                   help="session id; reuse one to resume its trie "
                        "(default: a fresh timestamped id)")
    p.add_argument("--device", default=None,
                   help="device for the chat model "
                        "(default: cuda, or cuda:<gpu> when --gpu is set)")
    p.add_argument("--gpu", default=None, metavar="LIST",
                   help="CUDA GPU index or comma list in PCI order: --gpu 0 "
                        "or --gpu 0,1. Several cards split the chat model's "
                        "weights across them (tensor-parallel on vllm, "
                        "device_map on hf); the BGE encoder rides the LAST "
                        "card, inside the headroom the memory cap leaves "
                        "free. Default: current CUDA device.")
    p.add_argument("--bge-device", default=None,
                   help="device for the BGE encoder (default: --device)")
    p.add_argument("--backend", default="hf",
                   choices=["hf", "vllm", "vllm-serve"],
                   help="inference backend for the chat model (default: hf; "
                        "vllm reuses the stable prompt prefix from the GPU "
                        "KV cache across turns - needs the optional vLLM "
                        "install, README step 5; vllm-serve connects to a "
                        "persistent server started with saltServe, so the "
                        "cache survives restarts)")
    p.add_argument("--server-url", default="http://127.0.0.1:8000",
                   metavar="URL",
                   help="vLLM server address for --backend vllm-serve "
                        "(default: http://127.0.0.1:8000)")
    p.add_argument("--gpu-mem-util", type=float, default=None,
                   help="fraction of each card's memory SALT may use for the "
                        "model (the vllm backend, and the hf backend when "
                        "--gpu lists several cards; default: 0.80 across "
                        "several cards, else 0.85, leaving room for the BGE "
                        "encoder)")
    p.add_argument("--max-model-len", type=int, default=0, metavar="N",
                   help="cap the vllm backend's context window to N tokens "
                        "(default: 0 = the model's own window; set this "
                        "when the full window's KV cache does not fit GPU "
                        "memory)")
    p.add_argument("--budget-pct", type=float, default=0.20,
                   help="token budget for the compressed memory block")
    p.add_argument("--memory-cap", default="auto", metavar="N|auto|off",
                   help="absolute ceiling on the compressed memory block, "
                        "in tokens. 'auto' fits the block to the space the "
                        "model's window has left after the fixed prompt "
                        "and a reply reserve, a number caps it at that "
                        "many tokens, 'off' restores the old unbounded "
                        "percentage sizing (default: auto)")
    p.add_argument("--doc", action="append", default=[], metavar="PATH",
                   help="text or PDF file to ingest into the trie at startup "
                        "(repeatable)")
    p.add_argument("--tail", type=int, default=4,
                   help="exchanges kept verbatim after tail compaction (the "
                        "window grows append-only to 2x this, then compacts "
                        "back in one stroke)")
    p.add_argument("--coverage-half-life", type=float, default=None,
                   metavar="TURNS",
                   help="halve each theme's cross-turn suppression every "
                        "this-many turns of silence, so topics the "
                        "conversation returns to can resurface (default: "
                        "off - coverage accumulates for the whole session)")
    p.add_argument("--coverage-decay-docs", action="store_true",
                   help="apply --coverage-half-life to attached-file "
                        "branches too (default: files are exempt, so "
                        "selection keeps advancing through a document)")
    p.add_argument("--shift-damping", type=float, default=None,
                   metavar="SCALE",
                   help="on a detected topic shift, scale the STALE part of "
                        "the cross-turn coverage seed by this factor for "
                        "that turn only (e.g. 0.25) and boost the query "
                        "channels, so a pivot back to a long-quiet topic is "
                        "not fought by its accumulated suppression while "
                        "the topic being left stays suppressed (default: "
                        "off; drift detection still runs and /stats "
                        "reports it)")
    p.add_argument("--shift-margin", type=float, default=0.12, metavar="COS",
                   help="cosine drop below the session's own drift baseline "
                        "(EMA) that counts as a topic shift (default: 0.12)")
    p.add_argument("--shift-query-boost", type=float, default=1.5,
                   metavar="X",
                   help="multiplier (>= 1) on the query-mass ratio during a "
                        "shift turn while --shift-damping is active "
                        "(default: 1.5)")
    p.add_argument("--coverage-gc", action="store_true",
                   help="garbage-collect remembered theme keys that no "
                        "longer match any branch of the memory tree, "
                        "after a grace window, so long sessions stop "
                        "carrying dead suppression in every save "
                        "(default: off; /stats counts live vs orphaned "
                        "keys either way)")
    p.add_argument("--coverage-max-keys", type=int, default=None,
                   metavar="N",
                   help="hard cap on remembered theme keys: past N, "
                        "orphaned keys drop first, then the stalest and "
                        "weakest live ones. The only unconditional bound "
                        "when decay is off (default: off)")
    p.add_argument("--max-sentences", type=int, default=None, metavar="N",
                   help="cap the conversation sentences kept in memory: "
                        "past N the oldest are masked out of selection "
                        "rather than deleted, so their text and their "
                        "numbering survive while a long session stops "
                        "growing. Attached files are never masked "
                        "(default: off; pair it with --coverage-gc or "
                        "--coverage-max-keys, which bound the theme keys "
                        "masked sentences leave behind)")
    p.add_argument("--stable-coverage-keys", action="store_true",
                   help="freeze the session's keyword order so cross-turn "
                        "memory discounts survive as the conversation "
                        "grows. New keywords join at the tail instead of "
                        "reshuffling the memory tree, so a theme already "
                        "shown keeps its discount (default: off; /stats "
                        "reports matched and orphaned keys either way)")
    p.add_argument("--per-source-themes", action="store_true",
                   help="profile conversation themes separately from each "
                        "attached file, so a large attachment cannot push "
                        "the conversation's own keywords below the theme "
                        "cutoff (default: off; /stats shows how many "
                        "conversation themes the split recovers)")
    p.add_argument("--dedup-cos", type=float, default=None, metavar="COS",
                   help="skip a new user/assistant sentence whose embedding "
                        "cosine against an earlier conversation sentence of "
                        "the same role reaches this threshold (e.g. 0.92), "
                        "so restatements and re-asked questions stop "
                        "inflating theme statistics (default: off; attached "
                        "files and fused --short-turns acks are never gated; "
                        "/stats counts suppressions)")
    p.add_argument("--short-turns", choices=("off", "keep", "fuse"),
                   default="keep",
                   help="keep short user messages ('go with option B') in "
                        "conversation memory instead of letting the junk "
                        "filter's length gates drop them. URL-only and "
                        "junk-shaped lines still drop. 'fuse' additionally "
                        "stores a bare acknowledgement ('yes', 'the second "
                        "one') together with the question it answers, "
                        "quoted from the previous reply (default: keep; "
                        "'off' restores the old dropping behavior)")
    p.add_argument("--no-turn-labels", action="store_true",
                   help="head each conversation excerpt with the plain "
                        "'[from the earlier conversation]' label instead of "
                        "naming the turn and speaker it came from (default: "
                        "labels carry the turn number and the speaker, so "
                        "the model can tell who said what and which "
                        "statement came later)")
    p.add_argument("--conversation-map", action="store_true",
                   help="open the memory block with a map of the "
                        "conversation: one line per earlier turn listing "
                        "that turn's main keywords, so the model can see a "
                        "topic was discussed even on a turn none of its "
                        "sentences were selected. A long conversation "
                        "shows its recent turns and the header says so "
                        "(default: off; the map is always in /stats)")
    p.add_argument("--tail-exclude", action="store_true",
                   help="accepted for compatibility: tail exclusion is "
                        "the default")
    p.add_argument("--no-tail-exclude", action="store_true",
                   help="let the memory block select sentences that are "
                        "still shown verbatim in the recent messages "
                        "(default: they are excluded, so the budget buys "
                        "older context instead of repeating what is on "
                        "screen; a sentence's themes start counting as "
                        "shown once it leaves the tail)")
    p.add_argument("--sync-ingest", action="store_true",
                   help="run the per-turn keyword/embedding ingest on the "
                        "REPL thread as before, instead of in the "
                        "background: the prompt then waits for it and an "
                        "ingest error raises at the call site (default: "
                        "ingest runs on a background worker and long "
                        "pasted messages never delay the next prompt)")
    p.add_argument("--roster", metavar="FILE",
                   help="load a model roster: the JSON file naming the "
                        "worker models this session may hand work to, and "
                        "at most one orchestrator. An entry either attaches "
                        "to a running saltServe endpoint or describes how "
                        "to spawn one. Loading validates the file only, so "
                        "nothing is contacted or started. See "
                        "salt/agents/roster_sample.json, and /roster in the "
                        "REPL for what was loaded.")
    p.add_argument("--turns", metavar="FILE",
                   help="run a scripted conversation from a JSON array or "
                        "JSONL file instead of the interactive REPL. Each "
                        "item is one user turn, fed in order into the same "
                        "session so SALT's memory builds across them. A "
                        "string item is the message; an object item takes "
                        "its message from --turns-field (default: a common "
                        "key such as question/puzzle/prompt). Works with "
                        "every backend, including --backend vllm-serve.")
    p.add_argument("--turns-field", metavar="KEY", default=None,
                   help="for object items in --turns, the key holding the "
                        "user message (default: auto-detect)")
    p.add_argument("--turns-out", metavar="FILE", default=None,
                   help="append each --turns answer to this JSONL file as "
                        "{id, turn, question, answer}")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if parse_memory_cap(args.memory_cap) is None:
        print("--memory-cap must be 'off', 'auto', or a positive token "
              "count", file=sys.stderr)
        return 1

    try:
        gpus = parse_gpu_list(args.gpu)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    args.device, args.bge_device, args.gpu_mem_util, cuda_order = \
        resolve_gpu_devices(gpus, args.device, args.bge_device,
                            args.gpu_mem_util)
    if cuda_order:
        # export before load_bge (the first CUDA init) so the parent process
        # (BGE) and the vllm worker enumerate cards the same way - otherwise
        # an index could name different physical cards for BGE and the model
        os.environ["CUDA_DEVICE_ORDER"] = cuda_order

    if args.add:
        try:
            cfg = register_model(args.add, alias=args.alias, dtype=args.dtype,
                                 force=args.force)
        except RegistryError as exc:
            print(exc, file=sys.stderr)
            return 1
        print(f"Registered {cfg['hf_id']} as {cfg['alias']!r} -> {cfg['path']}")
        return 0
    if args.list:
        print_models()
        return 0

    if args.conversation_id and not valid_session_id(args.conversation_id):
        print("--conversation-id may only contain letters, digits, '.', '_', '-'.",
              file=sys.stderr)
        return 1
    budget = normalize_budget(args.budget_pct)
    if budget is None:
        print("--budget-pct must be in (0, 1], or 0-100 as a percentage.",
              file=sys.stderr)
        return 1
    args.budget_pct = budget
    # isfinite: argparse's type=float happily parses "nan" and "inf"; nan
    # even passes a <= 0 check and would then show up as enabled in /stats
    if args.coverage_half_life is not None and not (
            math.isfinite(args.coverage_half_life)
            and args.coverage_half_life > 0):
        print("--coverage-half-life must be a positive, finite number of "
              "turns.", file=sys.stderr)
        return 1
    # strictly inside (0, 1): 1.0 means "no damping" (use no flag instead)
    # and 0.0 is a falsy total-amnesia trap the enable-check would skip
    if args.shift_damping is not None and not (
            math.isfinite(args.shift_damping)
            and 0 < args.shift_damping < 1):
        print("--shift-damping must be a scale strictly between 0 and 1 "
              "(e.g. 0.25).", file=sys.stderr)
        return 1
    if not (math.isfinite(args.shift_margin) and args.shift_margin >= 0):
        print("--shift-margin must be a non-negative, finite cosine drop.",
              file=sys.stderr)
        return 1
    if not (math.isfinite(args.shift_query_boost)
            and args.shift_query_boost >= 1):
        print("--shift-query-boost must be a finite multiplier >= 1 "
              "(1 leaves the query mass unchanged).", file=sys.stderr)
        return 1
    # strictly inside (0, 1): 1.0 can never fire on distinct sentences
    # (exact repeats are already hash-deduped) and BGE cosines run hot, so
    # a threshold at or below 0 would suppress every conversation sentence
    if args.dedup_cos is not None and not (
            math.isfinite(args.dedup_cos) and 0 < args.dedup_cos < 1):
        print("--dedup-cos must be a cosine threshold strictly between 0 "
              "and 1 (e.g. 0.92).", file=sys.stderr)
        return 1
    if not (math.isfinite(args.gpu_mem_util) and 0 < args.gpu_mem_util <= 1):
        print("--gpu-mem-util must be a fraction in (0, 1].", file=sys.stderr)
        return 1
    if args.max_model_len < 0:
        print("--max-model-len must be >= 0 (0 = the model's own window).",
              file=sys.stderr)
        return 1
    if not args.server_url.startswith(("http://", "https://")):
        print("--server-url must start with http:// or https://",
              file=sys.stderr)
        return 1

    # load a scripted-turns file up front so a bad path or format fails
    # before the model is loaded
    turns = None
    if args.turns:
        try:
            turns = load_turns(args.turns, args.turns_field)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"--turns: {exc}", file=sys.stderr)
            return 1
        if not turns:
            print(f"--turns: {args.turns} has no turns", file=sys.stderr)
            return 1

    # same reason for the roster: a bad entry, or a worker whose weights
    # are missing, must fail before the chat model is loaded
    roster = None
    if args.roster:
        try:
            roster = load_roster(args.roster)
        except RosterError as exc:
            print(f"--roster: {exc}", file=sys.stderr)
            return 1

    models = list_models()
    if args.model:
        try:
            cfg = resolve_model(args.model)
        except RegistryError as exc:
            print(exc, file=sys.stderr)
            return 1
    elif len(models) == 1:
        cfg = models[0]
    else:
        if models:
            print("Pick a model with --model. Registered models:")
            print_models()
        else:
            print("No models registered yet. Add one with: saltChat --add <hf_id>")
        return 1
    if not cfg["downloaded"]:
        print(f"Weights for {cfg['alias']!r} are missing (broken symlink?). "
              f"Re-register with: saltChat --add {cfg['hf_id']} --force",
              file=sys.stderr)
        return 1

    # pinning order: BGE encoder first (tiny), then the session trie
    # (CPU/RAM, resumes from disk), then the chat LLM - all stay resident
    bge_device = args.bge_device or args.device
    print(f"Loading BGE encoder {BGE_MODEL} on {bge_device}")
    bge_tok, bge_model = load_bge(BGE_MODEL, bge_device)

    conversation_id = args.conversation_id or fresh_conversation_id()
    try:
        trie = SessionTrie(conversation_id, cache_dir=SESSIONS_DIR,
                           model_name=BGE_MODEL,
                           budget_pct_default=args.budget_pct)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if trie.is_loaded:
        print(f"Resumed session {conversation_id!r}: {trie.n_sentences} "
              f"sentences over {trie.n_turns} turns.")
        warn_load_repair(trie)
    else:
        print(f"New session {conversation_id!r}.")

    runner = make_runner(cfg, device=args.device, backend=args.backend,
                         **backend_opts(args))
    state = ChatState(args, bge_tok, bge_model, runner, trie, roster)
    if state.full_attachments:
        print(f"Restored {len(state.full_attachments)} full-context "
              f"attachment(s): {', '.join(state.full_attachments)}")
        warn_prompt_budget(state)

    for doc in args.doc:
        ingest_doc(state, doc)

    try:
        if turns is not None:
            run_turns(state, turns, args.turns_out)
        else:
            repl(state)
    finally:
        # "Session saved" must be true: a failed or interrupted final
        # save gets the warning instead
        if close_ingest(state) and not state.trie.dirty:
            print(f"Session saved under {state.trie.cache_dir}")
        else:
            print(f"warning: the last exchange may not have reached disk "
                  f"under {state.trie.cache_dir} (see messages above)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
