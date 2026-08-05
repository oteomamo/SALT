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


def build_server(engine):
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
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.version:
        print(salt_version())
        return 0
    engine = Engine(resolve_device(args))
    build_server(engine).run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
