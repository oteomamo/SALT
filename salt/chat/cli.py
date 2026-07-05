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
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

from salt.chat.kvtrace import KVTrace
from salt.chat.pdfio import (PLAIN_SUFFIXES, ExtractionError, read_document,
                             split_document_sentences)
from salt.chat.registry import (RegistryError, list_models, register_model,
                                resolve_model)
from salt.chat.runner import ChatRunner
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


class ChatState:
    """Everything a live session pins: models on GPU, trie in RAM."""

    def __init__(self, args, bge_tok, bge_model, runner, trie):
        self.device = args.device
        self.bge_device = args.bge_device or args.device
        self.bge_tok = bge_tok
        self.bge_model = bge_model
        self.runner = runner
        self.trie = trie
        self.budget = args.budget_pct
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
        self.kvtrace = KVTrace(self.trie.cache_dir,
                               self.trie.conversation_id)

    def compact_tail(self):
        """Cut the tail back to tail_min exchanges once it exceeds tail_max.
        Nothing is lost: every sentence already entered the trie the moment
        it was spoken — compaction only bounds the verbatim window."""
        if len(self.tail) > 2 * self.tail_max:
            del self.tail[: len(self.tail) - 2 * self.tail_min]

    def new_trie(self, conversation_id):
        self.trie = SessionTrie(conversation_id, cache_dir=SESSIONS_DIR,
                                model_name=BGE_MODEL,
                                budget_pct_default=self.budget)
        self.tail.clear()
        self.last_stats = None
        self.load_full_attachments()
        self.kvtrace = KVTrace(self.trie.cache_dir,
                               self.trie.conversation_id)
        return self.trie

    # ── attach@ full-context attachments (persisted per session) ─────────
    def attachments_dir(self):
        return self.trie.cache_dir / "attachments"

    def load_full_attachments(self):
        self.full_attachments = {}
        d = self.attachments_dir()
        if d.is_dir():
            for f in sorted(d.glob("*.txt")):
                self.full_attachments[f.name[:-4]] = f.read_text(
                    encoding="utf-8", errors="replace")

    def save_full_attachment(self, name, text):
        # pypdf can emit lone surrogates from broken font CMaps; make the
        # text strictly encodable before it reaches disk or a prompt, and
        # only expose the attachment once the write has succeeded
        text = text.encode("utf-8", errors="replace").decode("utf-8")
        d = self.attachments_dir()
        d.mkdir(parents=True, exist_ok=True)
        (d / (name + ".txt")).write_text(text, encoding="utf-8")
        self.full_attachments[name] = text

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


def add_to_trie(state, text, role, source=None, sentences=None):
    return state.trie.add_turn(text, role=role, tokenizer=state.bge_tok,
                               model=state.bge_model, device=state.bge_device,
                               source=source, sentences=sentences)


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
                       sentences=split_document_sentences(text))
    if info["added"] == 0:
        if merging:
            print(f"{p.name}: nothing new to add (already attached).")
        else:
            print(f"{p.name}: no ingestible sentences - all "
                  f"{info['filtered']} extracted units were filtered "
                  f"(tables/references/fragments).")
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
    (instructions + inventory) exceed the active model's input window (the
    runner tail-truncates, silently dropping the head)."""
    if state.runner is None or not state.full_attachments:
        return
    limit = int(state.runner.cfg.get("max_input_len") or 0)
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
              f"the model's max_input_len ({limit}) - prompts will be "
              f"tail-truncated and the earliest content silently dropped. "
              f"Raise max_input_len in salt/models/{state.runner.alias}/"
              f"config.json, or prefer salt@ for large files.")


def switch_model(state, name):
    try:
        cfg = resolve_model(name)
    except RegistryError as exc:
        print(exc)
        return
    if state.runner is not None and cfg["alias"] == state.runner.alias:
        print(f"{cfg['alias']} is already active.")
        return
    prev_cfg = state.runner.cfg
    state.runner.unload()  # free before load: never two LLMs on the GPU
    state.runner = None
    try:
        state.runner = ChatRunner(cfg, device=state.device)
        warn_attachment_budget(state)  # new model may have a smaller window
    except Exception as exc:
        print(f"Failed to load {cfg['alias']}: {exc}")
        gc.collect()  # drop the failed load's partial allocations first
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Reloading previous model {prev_cfg['alias']} ...")
        try:
            state.runner = ChatRunner(prev_cfg, device=state.device)
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
        kv = state.kvtrace
        if kv.last_event:
            u, tot = kv.last_event["usage"], kv.totals
            print(f"kv ledger: turn {kv.last_event['turn']} - "
                  f"read {u['input_cached_tokens']}, write {u['input']}, "
                  f"output {u['output']} tok | session totals "
                  f"read {tot['input_cached_tokens']}, write {tot['input']}, "
                  f"output {tot['output']}")
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
        state.new_trie(cid)
        print(f"Session {cid!r} wiped.")
    else:
        print(f"Unknown command {cmd!r} - /help lists commands.")
    return True


def chat_turn(state, line):
    if state.runner is None:
        print("No chat model loaded - /model <name> to load one.")
        return
    ts_start = datetime.now().isoformat(timespec="seconds")
    # trie holds turns 1..N-1 here (this message is added after generation):
    # the verbatim tail covers recent turns, the trie covers older ones
    memory_block, selected_idx = "", []
    if state.trie.n_sentences > 0:
        comp = state.trie.compress(query=line, budget_pct=state.budget,
                                   tokenizer=state.bge_tok,
                                   model=state.bge_model,
                                   device=state.bge_device)
        selected_idx = comp["selected_sent_idx"]
        state.last_stats = comp["stats"]
        memory_block = format_memory_block(state.trie, selected_idx)

    inventory = attachment_inventory(state.trie, state.full_attachments)
    instructions = (load_instructions()
                    if (memory_block or inventory or state.full_attachments)
                    else "")
    messages = build_messages(memory_block, state.tail, line,
                              state.full_attachments, inventory, instructions)
    # cleared per turn so an interrupt before tokenization can't record the
    # previous turn's prompt size for this one
    state.runner.last_prompt_tokens = None
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
    try:
        state.kvtrace.record_turn(
            tokenizer=state.runner.tokenizer, trie=state.trie,
            selected_idx=selected_idx, reply_text=reply,
            model_id=state.runner.cfg["hf_id"], ts_start=ts_start,
            ts_end=datetime.now().isoformat(timespec="seconds"),
            prompt_tokens=getattr(state.runner, "last_prompt_tokens", None))
    except Exception as exc:
        print(f"[kvtrace] recording failed for this turn: {exc}")
    add_to_trie(state, line, "user")
    if reply:
        add_to_trie(state, reply, "assistant")  # seam: --no-assistant-memory
        # tail only grows in pairs so strict chat templates always see
        # alternating roles (a replyless user turn still reaches the trie)
        state.tail.append({"role": "user", "content": line})
        state.tail.append({"role": "assistant", "content": reply})
        state.compact_tail()


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
    print(f"Session saved under {state.trie.cache_dir}")


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
    p.add_argument("--gpu", type=int, default=None, metavar="N",
                   help="CUDA GPU index for this chat (chat model + BGE); "
                        "shorthand for --device cuda:N. Default: current CUDA "
                        "device. Use different indices to run chats on "
                        "separate GPUs.")
    p.add_argument("--bge-device", default=None,
                   help="device for the BGE encoder (default: --device)")
    p.add_argument("--budget-pct", type=float, default=0.20,
                   help="token budget for the compressed memory block")
    p.add_argument("--doc", action="append", default=[], metavar="PATH",
                   help="text or PDF file to ingest into the trie at startup "
                        "(repeatable)")
    p.add_argument("--tail", type=int, default=4,
                   help="exchanges kept verbatim after tail compaction (the "
                        "window grows append-only to 2x this, then compacts "
                        "back in one stroke)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.device is None:
        args.device = f"cuda:{args.gpu}" if args.gpu is not None else "cuda"

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

    runner = ChatRunner(cfg, device=args.device)
    state = ChatState(args, bge_tok, bge_model, runner, trie)
    if state.full_attachments:
        print(f"Restored {len(state.full_attachments)} full-context "
              f"attachment(s): {', '.join(state.full_attachments)}")
        warn_attachment_budget(state)

    for doc in args.doc:
        ingest_doc(state, doc)

    repl(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
