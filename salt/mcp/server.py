# -*- coding: utf-8 -*-
"""The salt-mcp entry point: SALT compression over MCP.

A client execs this, speaks JSON-RPC over stdio, and calls the tools
below. Imports stay light at module level on purpose: this is what a
client starts, and a server that loads torch before it has been asked
for anything is a server that looks broken while it starts. The encoder
loads on the first call that needs it and stays resident after.

The compression itself is assembled here from the engine's own building
blocks, the same ones salt/compress.py wires for a one-shot run. The
defaults come from that command's parser, so a tool call and a command
line compress a text the same way.

HOW THIS SURFACE IS ALLOWED TO GROW. A client discovers the tools at
runtime and compiles nothing in, which is what makes adding safe and
renaming dangerous:

1. Tool names are forever. Adding tools is free, renaming one is a
   breaking change every client feels silently.
2. Schemas evolve additively: a new argument is optional and has a
   default, a new response field is added beside the others. An
   existing field is never repurposed.

The version in the handshake is the package version, so a client log
always records which contract it spoke to, and salt_contract states
the tool contract's own number beside the tool list it covers.
"""

import argparse
import atexit
import os
import signal
import sys
from typing import Any

from salt.mcp.errors import (DEFAULT_MAX_CHARS, ToolError, guarded,
                             need_budget, need_text)

SERVER_NAME = "salt"
# the tool contract's own number, bumped only when a tool is renamed or
# removed, which is to say never on purpose. Additive growth leaves it
# exactly where it is
TOOLS_CONTRACT = 1
# the whole surface, in the order a client is offered it. Written down
# here rather than left to whatever the registry happens to hold, so a
# rename or a removal is a failure at startup instead of a silence a
# client discovers in the field
TOOL_NAMES = ("salt_compress", "roster_list", "salt_contract",
              "salt_switches", "salt_delegate", "session_create",
              "session_resume", "session_list", "session_add_turn",
              "session_memory", "salt_ingest_document", "session_stats")
DEFAULT_BUDGET_PCT = 0.20
# every refusal a read-only server makes starts with this, so a client
# can tell "this server will not" apart from "this call was wrong"
READ_ONLY_PREFIX = "read-only server:"


def refuse_write(tool, what):
    """The one shape a read-only refusal takes. Stable wording: clients
    match on it, so it is part of the contract rather than a message."""
    raise ToolError("read_only", f"{tool} would {what}, and this server "
                                 f"was started with --read-only")


def salt_version():
    """The version this server carries.

    A checkout's pyproject wins over the installed metadata, because an
    editable install keeps reporting whatever version it was installed
    at while the tree moves on, and the handshake should say what is
    actually running.
    """
    import re
    from pathlib import Path
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    try:
        match = re.search(r'^version = "([^"]+)"',
                          pyproject.read_text(encoding="utf-8"), re.M)
        if match:
            return match.group(1)
    except OSError:
        pass
    try:
        from importlib.metadata import version
        return version("salt")
    except Exception:
        return "unknown"


def resolve_device(args):
    """Where the encoder goes, from the same three flags saltChat takes.

    An explicit device wins, then the last card of --gpu, then whatever
    torch has. A server with no CUDA falls back to the CPU rather than
    failing at the first call: an MCP client is often an editor on a
    laptop.
    """
    device = args.bge_device or args.device
    if device is None and args.gpu:
        device = f"cuda:{args.gpu.split(',')[-1].strip()}"
    if device is None:
        device = "cuda"
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            return "cpu"
    return device


class Engine:
    """The resident encoder and the settings every call compresses under.

    Built once per server, on the first call that needs it, because a
    client may connect and never compress anything.
    """

    def __init__(self, device):
        self.device = device
        self.tokenizer = None
        self.model = None
        self.args = None

    def ready(self):
        if self.model is not None:
            return self
        from salt.compress import build_parser
        from salt.engine import dataset_modes
        from salt.engine.compressor import load_bge
        args = build_parser().parse_args(["--output", "-"])
        dataset_modes.resolve_mode_defaults(args)
        args.device = self.device
        self.args = args
        self.tokenizer, self.model = load_bge(args.model, self.device)
        return self


def compress_text(engine, text, budget_pct=None, query=None,
                  max_chars=DEFAULT_MAX_CHARS):
    """One text compressed under a token budget, optionally biased to a
    query. Returns {"compressed": str, "stats": dict}."""
    from salt.engine import compressor
    from salt.engine.celf import coverage_select

    text = need_text("salt_compress", text, max_chars)
    budget = need_budget(budget_pct, DEFAULT_BUDGET_PCT)
    engine = engine.ready()
    args, tok, model, device = (engine.args, engine.tokenizer, engine.model,
                                engine.device)

    sentences, all_texts, orig_words, word_budget, _ = \
        compressor.prep_prose_sentences(text, "", tok, budget)
    if not sentences:
        # the same key set as the selected path, so the degenerate input
        # (a text of pure symbols passes the guard and yields nothing)
        # cannot surprise a client that read the ordinary shape
        return {"compressed": "",
                "stats": {"orig_words": orig_words, "kept_words": 0,
                          "n_sentences": 0, "n_sentences_raw": len(all_texts),
                          "n_selected": 0,
                          "word_budget": word_budget,
                          "actual_tokens": 0,
                          "theme_coverage_pct": 0,
                          "compression_ratio": 0.0,
                          "budget_pct": budget,
                          "queried": bool(query)}}

    sent_data, kw_df, theme_keywords = compressor.build_prose_sent_data(
        sentences, tok, model, device, args.max_keywords_ratio,
        args.theme_percentile)
    query_kw, query_emb, query_pns, qtype = compressor.enrich_query(
        query, tok, model, device, args.extended_stopwords, args.bge_prefix)

    clean_words = sum(sd["n_words"] for sd in sent_data)
    if clean_words <= word_budget:
        compressed, used_words, sel_stats = compressor.fits_whole(
            sent_data, theme_keywords, " ",
            query_uncapped=args.query_uncapped,
            neighbor_window=args.neighbor_window, qtype=qtype,
            query_proper_nouns=query_pns)
    else:
        selected, sel_stats = coverage_select(
            sent_data, kw_df, theme_keywords, word_budget,
            query_keywords=query_kw, query_embedding=query_emb,
            query_proper_nouns=query_pns, lam=args.lam,
            query_mass_ratio=args.query_mass)
        sel_stats["qtype"] = qtype
        # no structural anchors in a bare text: the same merge the command
        # line runs, with nothing to merge in
        compressed, used_words = compressor.merge_anchored(
            selected, [], theme_keywords, query_kw, set(), 0, " ", sel_stats)

    return {"compressed": compressed,
            "stats": {"orig_words": orig_words,
                      "kept_words": used_words,
                      "n_sentences": len(sentences),
                      "n_sentences_raw": len(all_texts),
                      "n_selected": sel_stats.get("n_selected", 0),
                      "word_budget": word_budget,
                      "actual_tokens": compressor.count_tokens(compressed),
                      "theme_coverage_pct": sel_stats.get(
                          "theme_coverage_pct", 0),
                      "compression_ratio": round(
                          used_words / max(orig_words, 1), 4),
                      "budget_pct": budget,
                      "queried": bool(query)}}


def session_payload(pool, conversation_id, created=False):
    """What a client is told about a session it just opened.

    A new one is written to disk before it is answered for, so that
    creating a conversation is a fact a later call can see rather than
    something only this process knows.
    """
    session = pool.get(conversation_id)
    trie = session.trie
    if created:
        trie.save()
    out = {"conversation_id": trie.conversation_id,
           "created": created,
           "n_turns": trie.n_turns,
           "n_sentences": trie.n_sentences,
           "path": str(trie.cache_dir)}
    if session.warnings:
        out["warnings"] = list(session.warnings)
    return out


def known(pool, conversation_id):
    """The conversation exists, on disk or in this server's hands. The
    id itself is checked first, so a malformed one is a bad argument
    rather than a conversation nobody made."""
    if not pool.exists(conversation_id) and (
            conversation_id not in pool.open):
        raise ToolError("not_found",
                        f"no session named {conversation_id!r} - "
                        f"session_list shows what is there")
    return conversation_id


def create_session(pool, conversation_id=""):
    from salt.chat.cli import fresh_conversation_id
    if pool.read_only:
        refuse_write("session_create", "make a conversation on disk")
    cid = conversation_id or fresh_conversation_id()
    # in this server's hands counts as existing too: an open session's
    # save may still be queued, and "created" must never be claimed for
    # a conversation that already holds turns
    if pool.exists(cid) or cid in pool.open:
        raise ToolError("invalid_argument",
                        f"session {cid!r} already exists - resume it with "
                        f"session_resume")
    return session_payload(pool, cid, created=True)


def resume_session(pool, conversation_id):
    return session_payload(pool, known(pool, conversation_id))


def list_payload(pool):
    from salt.chat.cli import list_sessions
    found = list_sessions(pool.cache_dir)
    return {"sessions": found, "n": len(found), "open": sorted(pool.open)}


def session_stats_payload(pool, conversation_id):
    """The session's own numbers, drained first so a turn submitted a
    moment ago is counted rather than missed.

    The snapshot rides along as its own block: the same closed set of
    signals the switch policy decides on, so an agent outside this
    process reads exactly what one inside it would.
    """
    from salt.agents.snapshot import SCHEMA, snapshot
    session = pool.get(known(pool, conversation_id))
    session.drain()
    trie = session.trie
    return {"conversation_id": trie.conversation_id,
            "warnings": list(session.warnings),
            "snapshot": snapshot(session, session.last_stats),
            "snapshot_schema": SCHEMA,
            "n_turns": trie.n_turns,
            "n_sentences": trie.n_sentences,
            "n_alive": trie.n_alive,
            "n_masked": trie.n_masked,
            "n_near_dups": trie.n_near_dups,
            "attachments": list(trie.attached_sources),
            "budget_pct": trie.config.get("budget_pct_default"),
            "open_sessions": len(pool.open),
            "path": str(trie.cache_dir)}


TURN_ROLES = ("user", "assistant")


def add_turns(engine, pool, conversation_id, exchange, sync=False,
              max_chars=DEFAULT_MAX_CHARS):
    """Add one side of a conversation, or a whole exchange at once.

    The rows go through the session's ingest worker, the same FIFO the
    REPL uses, so the encode order is the order they were said in. With
    `sync` the call waits for the queue to finish - the worker thread
    still does the work, which is what keeps the trie single-submitter -
    and a failure is reported here rather than journaled.
    """
    if pool.read_only:
        refuse_write("session_add_turn", "add to a conversation's memory")
    rows = []
    for i, item in enumerate(exchange):
        if not isinstance(item, dict):
            raise ToolError("invalid_argument",
                            f"turn {i} is not an object with a role and "
                            f"a text")
        role = item.get("role", "user")
        if role not in TURN_ROLES:
            raise ToolError("invalid_argument",
                            f"turn {i}: role must be one of "
                            f"{list(TURN_ROLES)}, got {role!r}")
        rows.append((role, need_text(f"turn {i}", item.get("text", ""),
                                     max_chars)))
    if not rows:
        raise ToolError("invalid_argument",
                        "session_add_turn needs a text or an exchange")

    engine = engine.ready()
    session = pool.get(known(pool, conversation_id))
    before = session.trie.n_sentences
    for role, text in rows:
        session.ingest.submit(
            lambda role=role, text=text: session.trie.add_turn(
                text, role=role, tokenizer=engine.tokenizer,
                model=engine.model, device=engine.device, save=False),
            label=f"{role}-message ingest", payload=text)
    session.ingest.submit(
        lambda: session.trie.save() if session.trie.dirty else None,
        label="session save")
    failures = session.drain() if sync else []
    return {"conversation_id": session.conversation_id,
            "added": len(rows),
            "n_turns": session.trie.n_turns,
            "n_sentences": session.trie.n_sentences,
            "new_sentences": session.trie.n_sentences - before,
            "pending": session.ingest.pending,
            "failures": len(failures),
            "sync": bool(sync)}


def session_memory(engine, pool, conversation_id, query,
                   budget_pct=None, max_chars=DEFAULT_MAX_CHARS):
    """What this conversation remembers about a question.

    The block is the labeled one a saltChat turn is given, headed by
    where each excerpt came from. Reading is a turn here: the selection
    is committed, so the next call sees the coverage this one moved,
    exactly as a chat turn would.
    """
    from salt.chat.cli import format_memory_block
    query = need_text("session_memory's query", query, max_chars)
    budget_pct = need_budget(budget_pct)
    known(pool, conversation_id)
    engine = engine.ready()
    session = pool.get(conversation_id)
    # the drain is the barrier: a turn submitted a moment ago has to be
    # in the trie before this reads it, or the answer silently misses it
    session.drain()
    trie = session.trie
    if trie.n_sentences == 0:
        # the same key set as the answered path, so a client reading a
        # field on the empty conversation gets an empty value, never a
        # missing one
        return {"conversation_id": trie.conversation_id, "memory": "",
                "stats": {"n_selected": 0, "n_sentences": 0,
                          "n_turns": trie.n_turns,
                          "theme_coverage_pct": None,
                          "budget_pct": (
                              trie.config.get("budget_pct_default")
                              if budget_pct is None else budget_pct),
                          "committed": False,
                          "query": query}}
    comp = trie.compress(query=query, budget_pct=budget_pct,
                         tokenizer=engine.tokenizer, model=engine.model,
                         device=engine.device, defer_commit=True)
    selected = comp["selected_sent_idx"]
    block = format_memory_block(trie, selected)
    # a read is a turn, and commits like one. A read-only server drops
    # the commit instead, which is the same thing a delegation does:
    # the selection happened, the session did not move
    commit = comp.get("commit")
    if commit is not None and not pool.read_only:
        commit(save=True)
    stats = session.last_stats = comp["stats"]
    return {"conversation_id": trie.conversation_id,
            "memory": block,
            "stats": {"n_selected": len(selected),
                      "n_sentences": trie.n_sentences,
                      "n_turns": trie.n_turns,
                      "theme_coverage_pct": stats.get("theme_coverage_pct"),
                      "budget_pct": (
                          trie.config.get("budget_pct_default")
                          if budget_pct is None else budget_pct),
                      "committed": not pool.read_only,
                      "query": query}}


def safe_source_name(name, fallback="document"):
    """The name an excerpt is filed under. Only the last component of
    whatever was passed, so a source name can never point at a place on
    disk or climb out of the session's own branch."""
    from pathlib import Path
    cleaned = Path(str(name or "")).name.strip()
    return cleaned or fallback


def ingest_document(engine, pool, conversation_id, path=None, text=None,
                    source_name="", max_chars=DEFAULT_MAX_CHARS,
                    doc_root=None):
    """Put a document into a conversation's memory, from a file or from
    text that was already read.

    A path is resolved and read here, so a file the server cannot read
    is refused with the reason rather than half ingested. The text is
    split with the document splitter and filed under its own branch of
    the session, which is what keeps a long file from crowding out the
    conversation at selection time.

    A server started with --doc-root reads paths only from under that
    folder. The check runs on the resolved path, so a symlink pointing
    out of the root is outside it, and it runs before the file is looked
    for, so a refusal never reports whether a file beyond the root
    exists.
    """
    from pathlib import Path
    from salt.chat.pdfio import (ExtractionError, is_protected_unit,
                                 read_document, split_document_sentences)
    if pool.read_only:
        refuse_write("salt_ingest_document",
                     "put a document into a conversation's memory")
    if bool(path) == bool(text):
        raise ToolError("invalid_argument",
                        "salt_ingest_document takes exactly one of path "
                        "or text")
    n_pages = None
    if path:
        target = Path(str(path)).expanduser().resolve()
        if doc_root is not None and not target.is_relative_to(doc_root):
            raise ToolError("invalid_argument",
                            f"this server reads documents only from under "
                            f"{doc_root} (--doc-root), and that path is "
                            f"outside it")
        if not target.is_file():
            raise ToolError("not_found", f"no such file: {target}")
        try:
            body, n_pages = read_document(target)
        except (ExtractionError, OSError) as exc:
            raise ToolError("invalid_argument", exc) from exc
        if max_chars and len(body) > max_chars:
            # the same bound the text form meets, applied to what the
            # file turned out to hold - a path is just a text this
            # server read itself
            raise ToolError("too_large",
                            f"{target.name} is {len(body)} characters of "
                            f"text and this server accepts {max_chars} "
                            f"(--max-ingest-chars)")
        name = safe_source_name(source_name or target.name, target.name)
    else:
        body = need_text("salt_ingest_document's text", text, max_chars)
        name = safe_source_name(source_name)
    if not body or not body.strip():
        raise ToolError("invalid_argument",
                        "there is no text in that document to remember")

    engine = engine.ready()
    session = pool.get(known(pool, conversation_id))
    session.drain()
    trie = session.trie
    merging = name in trie.attached_sources
    before = trie.n_sentences
    info = trie.add_turn(body, role="doc", tokenizer=engine.tokenizer,
                         model=engine.model, device=engine.device,
                         source=name,
                         sentences=split_document_sentences(body),
                         keep=is_protected_unit, save=True)
    return {"conversation_id": trie.conversation_id,
            "source": name,
            "merged_into_existing": merging,
            "pages": n_pages,
            "added": info["added"],
            "filtered": info["filtered"],
            "n_sentences": trie.n_sentences,
            "new_sentences": trie.n_sentences - before,
            "attachments": list(trie.attached_sources)}


def stamped(server, expected):
    """Fix this server's tool list, and refuse to start with a surface
    that has drifted from the one written down. A tool name is a
    contract with every client, so a rename is caught here, once, rather
    than in somebody else's editor."""
    names = tuple(server._tool_manager._tools)
    if names != tuple(expected):
        raise RuntimeError(
            f"the MCP tool surface has drifted: {list(names)} were "
            f"registered where {list(expected)} were declared. A tool name "
            f"is forever, so change TOOL_NAMES only to ADD.")
    server.tool_names = names
    return server


def build_server(engine, pool=None, roster=None,
                 max_chars=DEFAULT_MAX_CHARS, doc_root=None):
    """The MCP server and its tools, with the engine they compress on."""
    from mcp.server import MCPServer
    from salt.mcp.agents import AgentRuntime, roster_payload, run_delegation

    read_only = bool(pool is not None and pool.read_only)
    runtime = AgentRuntime(engine, pool=pool, roster=roster)
    server = MCPServer(name=SERVER_NAME, version=salt_version(),
                       instructions="SALT compresses long text down to the "
                                    "part that answers a question."
                                    + (" This server is read-only: it "
                                       "answers about conversations and "
                                       "changes none of them."
                                       if read_only else ""))

    @server.tool(name="salt_compress",
                 description="Compress a text to a fraction of its words, "
                             "keeping what covers its themes and, when a "
                             "query is given, what answers it.",
                 structured_output=True)
    def salt_compress(text: str, budget_pct: float = DEFAULT_BUDGET_PCT,
                      query: str = "") -> dict[str, Any]:
        return guarded(compress_text, engine, text, budget_pct,
                       query or None, max_chars=max_chars)

    # the handles the delegation tools open outlive the calls that opened
    # them, so shutdown needs a way back to them
    server.runtime = runtime

    @server.tool(name="roster_list",
                 description="The helper models this server can reach, "
                             "with what each one is and where it lives. "
                             "Probe to find out which are answering.",
                 structured_output=True)
    def roster_list(probe: bool = False) -> dict[str, Any]:
        return guarded(roster_payload, runtime, probe=probe)

    @server.tool(name="salt_contract",
                 description="Which version of this tool contract the "
                             "server speaks, and every tool it offers.",
                 structured_output=True)
    def salt_contract() -> dict[str, Any]:
        return {"salt_mcp_tools": TOOLS_CONTRACT,
                "salt_version": salt_version(),
                "tools": list(server.tool_names),
                "read_only": read_only,
                "doc_root": str(doc_root) if doc_root else None,
                "growth": ["tool names are forever",
                           "arguments and response fields are added, "
                           "never repurposed"]}

    @server.tool(name="salt_switches",
                 description="The memory switches, what this server has "
                             "each one set to, and which measured number "
                             "reports whether it did anything.",
                 structured_output=True)
    def salt_switches() -> dict[str, Any]:
        from salt.agents.snapshot import KEYS, SCHEMA, switch_inventory
        # the server sets none of them, so the shipped values are what it
        # is running under, and a client is told that rather than left to
        # infer it
        return {"switches": switch_inventory(),
                "writable": False,
                "snapshot_schema": SCHEMA,
                "snapshot_keys": list(KEYS)}

    @server.tool(name="salt_delegate",
                 description="Hand one task to a helper model. With a "
                             "conversation, that conversation's memory "
                             "is selected for the task and sent with it.",
                 structured_output=True)
    def salt_delegate(task: str, conversation_id: str = "",
                      target: str = "", context_query: str = "",
                      budget_pct: float = None,
                      ingest: bool = False) -> dict[str, Any]:
        return guarded(run_delegation, runtime, task,
                       conversation_id=conversation_id,
                       target=target or None,
                       context_query=context_query or None,
                       budget_pct=budget_pct, ingest=ingest,
                       max_chars=max_chars)

    if pool is None:
        # a server with no conversations offers the tools that need none
        return stamped(server, TOOL_NAMES[:5])

    @server.tool(name="session_create",
                 description="Start a conversation whose memory this "
                             "server keeps. Without an id, one is made "
                             "from the date and time.",
                 structured_output=True)
    def session_create(conversation_id: str = "") -> dict[str, Any]:
        return guarded(create_session, pool, conversation_id)

    @server.tool(name="session_resume",
                 description="Open a conversation that already exists, "
                             "with the memory it had when it was last "
                             "written.",
                 structured_output=True)
    def session_resume(conversation_id: str) -> dict[str, Any]:
        return guarded(resume_session, pool, conversation_id)

    @server.tool(name="session_list",
                 description="Every conversation on disk, most recently "
                             "written first.",
                 structured_output=True)
    def session_list() -> dict[str, Any]:
        return guarded(list_payload, pool)

    @server.tool(name="session_add_turn",
                 description="Remember something said in a conversation: "
                             "one message, or a whole exchange at once.",
                 structured_output=True)
    def session_add_turn(conversation_id: str, text: str = "",
                         role: str = "user",
                         exchange: list[dict[str, str]] = None,
                         sync: bool = False) -> dict[str, Any]:
        rows = list(exchange) if exchange else (
            [{"role": role, "text": text}] if text else [])
        return guarded(add_turns, engine, pool, conversation_id, rows,
                       sync=sync, max_chars=max_chars)

    @server.tool(name="session_memory",
                 description="What this conversation remembers about a "
                             "question, as the labeled memory block a "
                             "chat turn would be given.",
                 structured_output=True)
    def session_memory_tool(conversation_id: str, query: str,
                            budget_pct: float = None) -> dict[str, Any]:
        return guarded(session_memory, engine, pool, conversation_id,
                       query, budget_pct, max_chars=max_chars)

    @server.tool(name="salt_ingest_document",
                 description="Read a document into a conversation's "
                             "memory, from a path or from text, filed "
                             "under its own source name.",
                 structured_output=True)
    def salt_ingest_document(conversation_id: str, path: str = "",
                             text: str = "",
                             source_name: str = "") -> dict[str, Any]:
        return guarded(ingest_document, engine, pool, conversation_id,
                       path=path or None, text=text or None,
                       source_name=source_name, max_chars=max_chars,
                       doc_root=doc_root)

    @server.tool(name="session_stats",
                 description="What one conversation holds: turns, "
                             "sentences, attachments and its budget.",
                 structured_output=True)
    def session_stats(conversation_id: str) -> dict[str, Any]:
        return guarded(session_stats_payload, pool, conversation_id)

    return stamped(server, TOOL_NAMES)


def load_roster(path):
    """The roster this server delegates to. A roster that will not load
    stops the server here, where the reason can be read, rather than at
    the first delegation where it looks like the task's fault."""
    from salt.agents.roster import RosterError
    from salt.agents.roster import load_roster as read_roster
    try:
        return read_roster(path)
    except (RosterError, OSError) as exc:
        print(f"salt-mcp: {exc}", file=sys.stderr)
        raise SystemExit(2)


def document_root(path):
    """The one folder a path may be read from, resolved once at startup.

    A root that is not a folder stops the server here, where the reason
    can be read, rather than turning every document call into a refusal
    nobody can account for.
    """
    if not path:
        return None
    from pathlib import Path
    root = Path(str(path)).expanduser().resolve()
    if not root.is_dir():
        print(f"salt-mcp: --doc-root is not a folder: {root}",
              file=sys.stderr)
        raise SystemExit(2)
    return root


def build_parser():
    p = argparse.ArgumentParser(
        prog="salt-mcp", description=__doc__.splitlines()[0])
    p.add_argument("--version", action="store_true",
                   help="print the salt version this server carries")
    p.add_argument("--device", default=None,
                   help="device for the encoder (default: cuda when there "
                        "is one, else cpu)")
    p.add_argument("--gpu", default=None, metavar="LIST",
                   help="CUDA index or comma list, with the encoder on "
                        "the last card")
    p.add_argument("--bge-device", default=None,
                   help="device for the encoder, winning over --device")
    p.add_argument("--sessions-dir", default=None,
                   help="where conversations live (default: the same "
                        "folder saltChat uses)")
    p.add_argument("--max-open-sessions", type=int, default=8,
                   help="how many conversations stay open at once "
                        "(default: 8)")
    p.add_argument("--roster", default=None, metavar="FILE",
                   help="roster of helper models this server may delegate "
                        "to (e.g. salt/agents/roster_sample.json)")
    p.add_argument("--max-ingest-chars", type=int, default=DEFAULT_MAX_CHARS,
                   metavar="N",
                   help=f"longest text one call may carry "
                        f"(default: {DEFAULT_MAX_CHARS})")
    p.add_argument("--doc-root", default=None, metavar="DIR",
                   help="read documents by path only from under this "
                        "folder (default: any file this server can read)")
    p.add_argument("--read-only", action="store_true",
                   help="answer reads and refuse every write, leaving "
                        "every conversation exactly as it was found")
    return p


def shutdown_once(pool, runtime):
    """One way to end, however the end arrives.

    A client hanging up, a Ctrl-C and a kill all mean the same thing:
    every open conversation drained and written, every worker connection
    let go. Idempotent because more than one of those paths can fire,
    and a second close must not undo the first one's work.
    """
    state = {"done": False}

    def close(*_args):
        if state["done"]:
            return
        state["done"] = True
        # a tool call may still be running on the SDK's thread. Waiting a
        # bounded moment for it keeps the save off a trie mid-mutation;
        # past the bound the close goes ahead, because a kill that never
        # closes is worse than a save that races one call
        from salt.mcp.errors import SERIAL
        held = SERIAL.acquire(timeout=10)
        try:
            pool.close_all()
        finally:
            try:
                runtime.close()
            finally:
                if held:
                    SERIAL.release()

    return close


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.version:
        print(salt_version())
        return 0
    from salt.chat.cli import sessions_root
    from salt.mcp.pool import SessionPool
    pool = SessionPool(args.sessions_dir or sessions_root(),
                       capacity=args.max_open_sessions,
                       read_only=args.read_only)
    roster = load_roster(args.roster) if args.roster else None
    server = build_server(Engine(resolve_device(args)), pool, roster,
                          max_chars=args.max_ingest_chars,
                          doc_root=document_root(args.doc_root))
    # the last word a session gets is written when this server ends, so
    # every way of ending has to reach the same close
    closing = shutdown_once(pool, server.runtime)
    atexit.register(closing)

    def on_signal(*_args):
        # the close is the part that matters and it has already happened
        # by the time this exits. The stop is hard on purpose: the
        # transport reads stdin on a thread of its own that will not
        # return until the client hangs up, and waiting for it would
        # turn a kill into a hang. The exit rides a finally, because a
        # close that failed on one session must still end the process
        try:
            closing()
        finally:
            os._exit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, on_signal)
    try:
        server.run("stdio")
    finally:
        closing()
    return 0


if __name__ == "__main__":
    sys.exit(main())
