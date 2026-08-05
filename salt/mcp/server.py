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
"""

import argparse
import sys
from typing import Any

SERVER_NAME = "salt"
DEFAULT_BUDGET_PCT = 0.20
# every refusal a read-only server makes starts with this, so a client
# can tell "this server will not" apart from "this call was wrong"
READ_ONLY_PREFIX = "read-only server:"


def refuse_write(tool, what):
    """The one shape a read-only refusal takes. Stable wording: clients
    match on it, so it is part of the contract rather than a message."""
    raise ValueError(f"{READ_ONLY_PREFIX} {tool} would {what}, and this "
                     f"server was started with --read-only")


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


def compress_text(engine, text, budget_pct=None, query=None):
    """One text compressed under a token budget, optionally biased to a
    query. Returns {"compressed": str, "stats": dict}."""
    from salt.engine import compressor
    from salt.engine.celf import coverage_select

    if not isinstance(text, str) or not text.strip():
        raise ValueError("salt_compress needs a non-empty text")
    budget = DEFAULT_BUDGET_PCT if budget_pct is None else float(budget_pct)
    if not 0 < budget <= 1:
        raise ValueError(f"budget_pct must be over 0 and at most 1, "
                         f"got {budget_pct!r}")
    engine = engine.ready()
    args, tok, model, device = (engine.args, engine.tokenizer, engine.model,
                                engine.device)

    sentences, all_texts, orig_words, word_budget, _ = \
        compressor.prep_prose_sentences(text, "", tok, budget)
    if not sentences:
        return {"compressed": "",
                "stats": {"orig_words": orig_words, "kept_words": 0,
                          "n_sentences": 0, "n_selected": 0,
                          "word_budget": word_budget,
                          "compression_ratio": 0.0,
                          "budget_pct": budget}}

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
    if session.warning:
        out["warning"] = session.warning
    return out


def session_stats_payload(pool, conversation_id):
    """The session's own numbers, drained first so a turn submitted a
    moment ago is counted rather than missed."""
    session = pool.get(conversation_id)
    session.drain()
    trie = session.trie
    return {"conversation_id": trie.conversation_id,
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


def add_turns(engine, pool, conversation_id, exchange, sync=False):
    """Add one side of a conversation, or a whole exchange at once.

    The rows go through the session's ingest worker, the same FIFO the
    REPL uses, so the encode order is the order they were said in. With
    `sync` the work happens inline and a failure is reported here rather
    than journaled.
    """
    if pool.read_only:
        refuse_write("session_add_turn", "add to a conversation's memory")
    rows = []
    for i, item in enumerate(exchange):
        if not isinstance(item, dict):
            raise ValueError(f"turn {i} is not an object with a role and "
                             f"a text")
        role = item.get("role", "user")
        text = item.get("text", "")
        if role not in TURN_ROLES:
            raise ValueError(f"turn {i}: role must be one of "
                             f"{list(TURN_ROLES)}, got {role!r}")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"turn {i}: no text to remember")
        rows.append((role, text))
    if not rows:
        raise ValueError("session_add_turn needs a text or an exchange")

    engine = engine.ready()
    session = pool.get(conversation_id)
    if sync:
        session.ingest.synchronous = True
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
                   budget_pct=None):
    """What this conversation remembers about a question.

    The block is the labeled one a saltChat turn is given, headed by
    where each excerpt came from. Reading is a turn here: the selection
    is committed, so the next call sees the coverage this one moved,
    exactly as a chat turn would.
    """
    from salt.chat.cli import format_memory_block
    if not isinstance(query, str) or not query.strip():
        raise ValueError("session_memory needs a query to select for")
    engine = engine.ready()
    session = pool.get(conversation_id)
    # the drain is the barrier: a turn submitted a moment ago has to be
    # in the trie before this reads it, or the answer silently misses it
    session.drain()
    trie = session.trie
    if trie.n_sentences == 0:
        return {"conversation_id": trie.conversation_id, "memory": "",
                "stats": {"n_selected": 0, "n_sentences": 0,
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
    stats = comp["stats"]
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
                    source_name=""):
    """Put a document into a conversation's memory, from a file or from
    text that was already read.

    A path is resolved and read here, so a file the server cannot read
    is refused with the reason rather than half ingested. The text is
    split with the document splitter and filed under its own branch of
    the session, which is what keeps a long file from crowding out the
    conversation at selection time.
    """
    from pathlib import Path
    from salt.chat.pdfio import (ExtractionError, is_protected_unit,
                                 read_document, split_document_sentences)
    if pool.read_only:
        refuse_write("salt_ingest_document",
                     "put a document into a conversation's memory")
    if bool(path) == bool(text):
        raise ValueError("salt_ingest_document takes exactly one of path "
                         "or text")
    n_pages = None
    if path:
        target = Path(str(path)).expanduser().resolve()
        if not target.is_file():
            raise ValueError(f"no such file: {target}")
        try:
            body, n_pages = read_document(target)
        except (ExtractionError, OSError) as exc:
            raise ValueError(str(exc)) from exc
        name = safe_source_name(source_name or target.name, target.name)
    else:
        body = text
        name = safe_source_name(source_name)
    if not body or not body.strip():
        raise ValueError("there is no text in that document to remember")

    engine = engine.ready()
    session = pool.get(conversation_id)
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


def build_server(engine, pool=None):
    """The MCP server and its tools, with the engine they compress on."""
    from mcp.server import MCPServer

    read_only = bool(pool is not None and pool.read_only)
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
        return compress_text(engine, text, budget_pct, query or None)

    if pool is None:
        return server

    @server.tool(name="session_create",
                 description="Start a conversation whose memory this "
                             "server keeps. Without an id, one is made "
                             "from the date and time.",
                 structured_output=True)
    def session_create(conversation_id: str = "") -> dict[str, Any]:
        from salt.chat.cli import fresh_conversation_id
        if pool.read_only:
            refuse_write("session_create", "make a conversation on disk")
        cid = conversation_id or fresh_conversation_id()
        if pool.exists(cid):
            raise ValueError(f"session {cid!r} already exists - resume it "
                             f"with session_resume")
        return session_payload(pool, cid, created=True)

    @server.tool(name="session_resume",
                 description="Open a conversation that already exists, "
                             "with the memory it had when it was last "
                             "written.",
                 structured_output=True)
    def session_resume(conversation_id: str) -> dict[str, Any]:
        if not pool.exists(conversation_id):
            raise ValueError(f"no session named {conversation_id!r} - "
                             f"session_list shows what is there")
        return session_payload(pool, conversation_id)

    @server.tool(name="session_list",
                 description="Every conversation on disk, most recently "
                             "written first.",
                 structured_output=True)
    def session_list() -> dict[str, Any]:
        from salt.chat.cli import list_sessions
        found = list_sessions(pool.cache_dir)
        return {"sessions": found, "n": len(found),
                "open": sorted(pool.open)}

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
        return add_turns(engine, pool, conversation_id, rows, sync=sync)

    @server.tool(name="session_memory",
                 description="What this conversation remembers about a "
                             "question, as the labeled memory block a "
                             "chat turn would be given.",
                 structured_output=True)
    def session_memory_tool(conversation_id: str, query: str,
                            budget_pct: float = None) -> dict[str, Any]:
        return session_memory(engine, pool, conversation_id, query,
                              budget_pct)

    @server.tool(name="salt_ingest_document",
                 description="Read a document into a conversation's "
                             "memory, from a path or from text, filed "
                             "under its own source name.",
                 structured_output=True)
    def salt_ingest_document(conversation_id: str, path: str = "",
                             text: str = "",
                             source_name: str = "") -> dict[str, Any]:
        return ingest_document(engine, pool, conversation_id,
                               path=path or None, text=text or None,
                               source_name=source_name)

    @server.tool(name="session_stats",
                 description="What one conversation holds: turns, "
                             "sentences, attachments and its budget.",
                 structured_output=True)
    def session_stats(conversation_id: str) -> dict[str, Any]:
        if not pool.exists(conversation_id) and (
                conversation_id not in pool.open):
            raise ValueError(f"no session named {conversation_id!r}")
        return session_stats_payload(pool, conversation_id)

    return server


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
    p.add_argument("--read-only", action="store_true",
                   help="answer reads and refuse every write, leaving "
                        "every conversation exactly as it was found")
    return p


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
    try:
        build_server(Engine(resolve_device(args)), pool).run("stdio")
    finally:
        # the client hanging up is how this server ends, so the last
        # word a session gets is written here or not at all
        pool.close_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
