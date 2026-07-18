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
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

from salt.chat.ingest import IngestWorker
from salt.chat.kvtrace import KVTrace
from salt.chat.pdfio import (PLAIN_SUFFIXES, ExtractionError,
                             is_protected_unit, read_document,
                             split_document_sentences)
from salt.chat.registry import (RegistryError, list_models, register_model,
                                resolve_model)
from salt.chat.runner import make_runner
from salt.chat.serve import default_gpu_mem_util, parse_gpu_list
from salt.engine.compressor import load_bge
from salt.engine.session_trie import SessionTrie

SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"
FILES_DIR = Path(__file__).resolve().parents[1] / "files"
FILE_SUFFIXES = {".pdf"} | PLAIN_SUFFIXES
BGE_MODEL = "BAAI/bge-small-en-v1.5"

# conversation ids become directory names under SESSIONS_DIR
SESSION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# seam: system-prompt wording is a later tuning knob
SYSTEM_PROMPT = "You are a helpful assistant."
MEMORY_BLOCK = (
    "SALT memory — compressed excerpts auto-selected for this message "
    "(partial, not full text; each section keeps original order):"
    "\n---\n{body}\n---")

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
/doc <path>        ingest a text or PDF file into the trie (role=doc)
/budget <pct>      set memory token budget (0.3 or 30 for 30%)
/stats             session, attachments, compression, and GPU memory stats
/new [id]          start (or resume) another conversation
/clear             wipe and restart the current conversation
/exit              leave (also Ctrl-D)"""


def backend_opts(args):
    """Per-backend knobs the shared CLI surface funnels to make_runner."""
    if args.backend == "vllm":
        return {"gpu_memory_utilization": args.gpu_mem_util,
                "max_model_len": args.max_model_len,
                "gpus": parse_gpu_list(args.gpu)}
    if args.backend == "vllm-serve":
        return {"server_url": args.server_url}
    return {}


class ChatState:
    """Everything a live session pins: models on GPU, trie in RAM."""

    def __init__(self, args, bge_tok, bge_model, runner, trie):
        self.device = args.device
        self.bge_device = args.bge_device or args.device
        self.backend = args.backend
        self.backend_opts = backend_opts(args)
        self.bge_tok = bge_tok
        self.bge_model = bge_model
        self.runner = runner
        self.trie = trie
        self.budget = args.budget_pct
        # coverage-decay, shift-damping + near-dup knobs live here and
        # travel as per-call kwargs to compress()/add_turn(): SessionTrie
        # .load() overwrites config values from the persisted config.json,
        # so trie config can't carry launch flags
        self.coverage_half_life = args.coverage_half_life
        self.coverage_decay_docs = args.coverage_decay_docs
        self.shift_damping = args.shift_damping
        self.shift_margin = args.shift_margin
        self.shift_query_boost = args.shift_query_boost
        self.dedup_cos = args.dedup_cos
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
        self.full_attachments = {}      # name -> whole text (attach@)
        self.load_full_attachments()
        self.load_tail()
        self.kvtrace = KVTrace(self.trie.cache_dir,
                               self.trie.conversation_id)
        # drained at every dispatch: no reader sees ingest in flight
        self.ingest = IngestWorker(
            journal_path=self.trie.cache_dir / "ingest_failures.jsonl",
            synchronous=args.sync_ingest)

    def compact_tail(self):
        """Cut the tail back to tail_min exchanges once it exceeds tail_max.
        Nothing is lost: every sentence already entered the trie the moment
        it was spoken — compaction only bounds the verbatim window."""
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


def format_memory_block(trie, sel_idx):
    """The selected sentences as a labeled memory block: grouped by origin
    (attached files first, then conversation), each section headed with its
    source and per-file selected/total counts so the model knows both where
    an excerpt came from and how partial the selection is. The labels match
    the reading guide in instructions.md."""
    if not sel_idx:
        return ""
    by_src = {}
    for i in sel_idx:
        by_src.setdefault(trie.sources[i], []).append(i)
    totals = Counter(s for s in trie.sources if s)
    sections = []
    for src in sorted(k for k in by_src if k):
        idxs = by_src[src]
        # explicit quotes, not !r: repr flips to double quotes on names with
        # apostrophes, breaking the label format instructions.md documents
        sections.append(f"[from attached file '{src}' — {len(idxs)} of "
                        f"{totals[src]} indexed sentences]\n"
                        + " ".join(trie.texts[i] for i in idxs))
    if None in by_src:
        sections.append("[from the earlier conversation]\n"
                        + " ".join(trie.texts[i] for i in by_src[None]))
    return MEMORY_BLOCK.format(body="\n\n".join(sections))


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


def add_to_trie(state, text, role, source=None, sentences=None, keep=None,
                save=True):
    return state.trie.add_turn(text, role=role, tokenizer=state.bge_tok,
                               model=state.bge_model, device=state.bge_device,
                               source=source, sentences=sentences, keep=keep,
                               dedup_cos=state.dedup_cos, save=save)


def submit_ingest(state, text, role, save=True):
    """Queue one side of an exchange for background ingest (inline under
    --sync-ingest, where a failure raises here)."""
    state.ingest.submit(lambda: add_to_trie(state, text, role, save=save),
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
    warn_attachment_budget(state)


def warn_attachment_budget(state):
    """Warn when the full attachments plus the fixed system-prompt overhead
    (instructions + inventory) exceed the active model's context window (the
    runner tail-truncates, silently dropping the head)."""
    if state.runner is None or not state.full_attachments:
        return
    limit = int(state.runner.input_budget() or 0)
    if not limit:
        return
    total = sum(state.count_tokens(t) or 0
                for t in state.full_attachments.values())
    total += (state.count_tokens(load_instructions()) or 0)
    total += (state.count_tokens(
        attachment_inventory(state.trie, state.full_attachments)) or 0)
    if total > limit:
        print(f"warning: full-context attachments (+ the system prompt's "
              f"instruction/inventory overhead) total ~{total} tokens, over "
              f"the model's usable input ceiling ({limit} = context window "
              f"minus reply headroom) - prompts will be tail-truncated and "
              f"the earliest content dropped. Prefer salt@ for large files, "
              f"or switch to a longer-context model.")


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
        warn_attachment_budget(state)  # new model may have a smaller window
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
        print(f"session {t.conversation_id!r}: {t.n_sentences} sentences over "
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
        s = state.last_stats or {}
        if s:
            trie_info = s.get("trie", {})
            print(f"last compression: theme coverage "
                  f"{s.get('theme_coverage_pct', 0):.1%}, "
                  f"{trie_info.get('n_nodes', '?')} nodes / "
                  f"{trie_info.get('n_branches', '?')} branches")
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
        if state.shift_damping:
            print(f"shift damping: x{state.shift_damping:g} stale-seed "
                  f"scale on shift turns, query boost "
                  f"x{state.shift_query_boost:g}")
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
    memory_block, selected_idx, drift_extra = "", [], None
    if state.trie.n_sentences > 0:
        comp = state.trie.compress(query=line, budget_pct=state.budget,
                                   tokenizer=state.bge_tok,
                                   model=state.bge_model,
                                   device=state.bge_device,
                                   coverage_half_life=state.coverage_half_life,
                                   coverage_decay_docs=state.coverage_decay_docs,
                                   shift_damping=state.shift_damping,
                                   shift_margin=state.shift_margin,
                                   shift_query_boost=state.shift_query_boost)
        selected_idx = comp["selected_sent_idx"]
        state.last_stats = comp["stats"]
        memory_block = format_memory_block(state.trie, selected_idx)
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
        submit_ingest(state, line, "user", save=False)
    messages = build_messages(memory_block, state.tail, line,
                              state.full_attachments, inventory, instructions)
    # cleared per turn so an interrupt before tokenization can't record the
    # previous turn's prompt size for this one
    state.runner.last_prompt_tokens = None
    state.runner.last_engine_stats = None
    print(f"{state.runner.alias}> ", end="", flush=True)
    pieces = []
    interrupted = False
    try:
        for piece in state.runner.stream_chat(messages):
            print(piece, end="", flush=True)
            pieces.append(piece)
    except KeyboardInterrupt:
        interrupted = True
    print("\n" if not interrupted else "  [interrupted]\n")

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
        submit_ingest(state, line, "user")
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


def _setup_completion():
    """TAB completes /commands and salt@<staged file> where readline exists."""
    try:
        import readline
    except ImportError:
        return
    commands = ["/help", "/model", "/add", "/doc", "/budget", "/stats",
                "/new", "/clear", "/exit"]

    def complete(text, i):
        if text.startswith("salt@") or text.startswith("attach@"):
            at = text.index("@") + 1
            prefix = text[at:]
            opts = [text[:at] + f.name for f in staged_files()
                    if f.name.startswith(prefix)]
        elif text.startswith("/"):
            opts = [c for c in commands if c.startswith(text)]
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
                        "or --gpu 0,1. The chat model loads on the first card "
                        "(the vllm backend tensor-parallels its weights "
                        "across all of them); the BGE encoder rides the LAST "
                        "card, off the cards holding the model. Default: "
                        "current CUDA device.")
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
                   help="fraction of each card's memory the vLLM engine may "
                        "manage (vllm backend only; default: 0.80 across "
                        "several cards, else 0.85, leaving room for the BGE "
                        "encoder)")
    p.add_argument("--max-model-len", type=int, default=0, metavar="N",
                   help="cap the vllm backend's context window to N tokens "
                        "(default: 0 = the model's own window; set this "
                        "when the full window's KV cache does not fit GPU "
                        "memory)")
    p.add_argument("--budget-pct", type=float, default=0.20,
                   help="token budget for the compressed memory block")
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
    p.add_argument("--dedup-cos", type=float, default=None, metavar="COS",
                   help="skip a new user/assistant sentence whose embedding "
                        "cosine against an earlier conversation sentence of "
                        "the same role reaches this threshold (e.g. 0.92), "
                        "so restatements and re-asked questions stop "
                        "inflating theme statistics (default: off; attached "
                        "files are never gated; /stats counts suppressions)")
    p.add_argument("--sync-ingest", action="store_true",
                   help="run the per-turn keyword/embedding ingest on the "
                        "REPL thread as before, instead of in the "
                        "background: the prompt then waits for it and an "
                        "ingest error raises at the call site (default: "
                        "ingest runs on a background worker and long "
                        "pasted messages never delay the next prompt)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    try:
        gpus = parse_gpu_list(args.gpu)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    if args.device is None:
        args.device = f"cuda:{gpus[0]}" if gpus else "cuda"
    if args.bge_device is None and gpus:
        # the BGE encoder rides the LAST card, so the model keeps the
        # earlier ones (with one card that is the same card as before)
        args.bge_device = f"cuda:{gpus[-1]}"
    if args.gpu_mem_util is None:
        args.gpu_mem_util = default_gpu_mem_util(gpus, single=0.85)

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
    else:
        print(f"New session {conversation_id!r}.")

    runner = make_runner(cfg, device=args.device, backend=args.backend,
                         **backend_opts(args))
    state = ChatState(args, bge_tok, bge_model, runner, trie)
    if state.full_attachments:
        print(f"Restored {len(state.full_attachments)} full-context "
              f"attachment(s): {', '.join(state.full_attachments)}")
        warn_attachment_budget(state)

    for doc in args.doc:
        ingest_doc(state, doc)

    try:
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
