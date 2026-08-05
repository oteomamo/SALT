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


def build_server(engine, pool=None):
    """The MCP server and its tools, with the engine they compress on."""
    from mcp.server import MCPServer

    server = MCPServer(name=SERVER_NAME, version=salt_version(),
                       instructions="SALT compresses long text down to the "
                                    "part that answers a question.")

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
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.version:
        print(salt_version())
        return 0
    from salt.chat.cli import sessions_root
    from salt.mcp.pool import SessionPool
    pool = SessionPool(args.sessions_dir or sessions_root(),
                       capacity=args.max_open_sessions)
    try:
        build_server(Engine(resolve_device(args)), pool).run("stdio")
    finally:
        # the client hanging up is how this server ends, so the last
        # word a session gets is written here or not at all
        pool.close_all()
    return 0


if __name__ == "__main__":
    sys.exit(main())
