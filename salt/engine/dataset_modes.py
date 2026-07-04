# -*- coding: utf-8 -*-
"""
SALT dataset-mode adapters: the `--synthetic` and `--code` modes, plus the
per-mode hyperparameter defaults.

Each mode changes the *representation* fed to the selector, not the selector
itself. The `--synthetic` query sub-policy takes an injected `select_fn`
(`retrieval.trie_select` or `celf.coverage_select` via `salt/compress.py`), so it
stays selector-agnostic; the `--code` helpers build code sent_data / anchors
that the connector feeds to whichever selector is active. This module imports no
selector directly; it depends only on `trie_core` primitives and `compressor`
helpers.

  * `--synthetic` (passage_count, passage_retrieval_en): `Paragraph N:` units
    become the records fed to the injected selector; query-less samples render
    deterministic per-unit prefixes (no selector at all).
  * `--code` (lcc, repobench-p): non-blank physical lines are the records, with
    identifier keywords and trie paths rooted at file/class/function structure,
    force-keeping the completion-site tail plus import/signature lines.
  * `MODE_DEFAULTS` / `resolve_mode_defaults`: per-mode tuned knobs (lam, query
    mass, theme percentile, tail fraction) frozen from dev50 sweeps.
"""
import re

from salt.engine.trie_core import (
    is_content_word, profile_themes, clean_text_words,
    get_bge_sentence_embeddings, extract_query_keywords,
    extract_proper_nouns_in_query, embed_query,
)
from salt.engine.compressor import count_tokens, select_anchors


# =============================================================================
# Mode grammar / constants
# =============================================================================
# Unit grammar for --synthetic: deliberately the same "Paragraph N" surface
# form as compressor.DATASET_ANCHORS so the adapter and the anchor regex cannot
# drift.
PARA_UNIT_RE = re.compile(r"^\s*Paragraph\s+(\d+):\s*(.*)$", re.IGNORECASE)

# =============================================================================
# --synthetic: paragraph-unit adapters (structural units as records)
# =============================================================================
def parse_paragraph_units(context):
    """
    Split an enumerated-paragraph context into structural units.
    """
    units = []
    for raw in context.split("\n"):
        if not raw.strip():
            continue
        m = PARA_UNIT_RE.match(raw)
        if m:
            units.append({"unit_idx": len(units),
                          "label_num": int(m.group(1)),
                          "body": m.group(2).strip()})
        elif units:
            units[-1]["body"] += " " + raw.strip()
    for u in units:
        u["n_words"] = len(u["body"].split())
    return units


def render_count_prefixes(units, word_budget, min_prefix=8):
    """
    Query-less synthetic adapter (passage_count): deterministic even-split
    prefix per unit. EVERY unit renders as 'Paragraph N: <first K words>' so
    exact-duplicate units stay textually identical (the LLM does the
    dedup-and-count); selection-based picking would desynchronize duplicates
    via diminishing returns. Pure I/O adapter: no model calls, no selector.
    """
    n_units = len(units)
    k = max(min_prefix, word_budget // max(n_units, 1) - 2)
    rendered, used = [], 0
    for u in units:
        words = u["body"].split()[:k]
        rendered.append(f"Paragraph {u['label_num']}: " + " ".join(words))
        used += 2 + len(words)
    return "\n\n".join(rendered), used, n_units, k


def select_retrieval_units(units, word_budget, tokenizer, model, device,
                           query_text, theme_percentile, select_fn,
                           sig_words=12, bge_prefix=True,
                           extended_stopwords=True):
    """
    Query-mode synthetic adapter (passage_retrieval_en): units become the
    records fed to the injected selector `select_fn`.
    """
    sig_text, sig_cost, full_text = {}, {}, {}
    total_sig = 0
    for u in units:
        body_words = u["body"].split()
        kept = body_words[:sig_words]
        txt = f"Paragraph {u['label_num']}: " + " ".join(kept)
        if len(body_words) > len(kept):
            txt += " ..."
        sig_text[u["unit_idx"]] = txt
        sig_cost[u["unit_idx"]] = 2 + len(kept)
        full_text[u["unit_idx"]] = f"Paragraph {u['label_num']}: {u['body']}"
        total_sig += sig_cost[u["unit_idx"]]
    effective_budget = max(0, word_budget - total_sig)

    unit_texts = [full_text[u["unit_idx"]] for u in units]
    unit_embs = get_bge_sentence_embeddings(unit_texts, tokenizer, model, device)

    sent_data = []
    for u, emb in zip(units, unit_embs):
        inc_cost = max(1, (2 + u["n_words"]) - sig_cost[u["unit_idx"]])
        kw = {w: 1.0 for w in clean_text_words(u["body"]) if is_content_word(w)}
        sent_data.append({"sent_idx": u["unit_idx"],
                          "text": full_text[u["unit_idx"]],
                          "n_words": inc_cost,
                          "keyword_weights": kw,
                          "embedding_l2": emb})

    kw_df, theme_keywords = profile_themes(sent_data,
                                           theme_percentile=theme_percentile)
    query_keywords = extract_query_keywords(query_text,
                                            use_extended=extended_stopwords)
    query_pns = extract_proper_nouns_in_query(query_text)
    query_emb = embed_query(query_text, tokenizer, model, device,
                            retrieval=bge_prefix)

    selected, sel_stats = select_fn(
        sent_data, kw_df, theme_keywords, effective_budget,
        query_keywords=query_keywords, query_embedding=query_emb,
        query_proper_nouns=query_pns)

    chosen = {sr.sent_idx for sr in selected}
    parts = [full_text[u["unit_idx"]] if u["unit_idx"] in chosen
             else sig_text[u["unit_idx"]] for u in units]
    used_words = total_sig + sel_stats["words_used"]

    sel_stats["words_used"] = used_words
    sel_stats["n_units"] = len(units)
    sel_stats["n_full_units"] = len(chosen)
    sel_stats["sig_words"] = sig_words
    sel_stats["sig_words_total"] = total_sig
    sel_stats["n_selected"] = len(units)
    sel_stats["avg_words_per_sent"] = round(used_words / max(len(units), 1), 1)
    return "\n\n".join(parts), used_words, sel_stats


def compress_synthetic(sample, sample_id, args, tokenizer, model,
                       dataset_name, all_classes, context,
                       orig_words, word_budget, select_fn):
    units = parse_paragraph_units(context)
    if not units:
        return None

    if not (sample.get("input", "").strip()):
        compressed_context, used_words, n_units, prefix_k = \
            render_count_prefixes(units, word_budget,
                                  min_prefix=args.count_min_prefix)
        method = "synthetic_prefix_count"
        sel_stats = {"n_selected": n_units, "pass1_selected": n_units,
                     "pass2_selected": 0, "pass3_query_anchored": 0,
                     "n_branches": 0, "words_used": used_words,
                     "avg_words_per_sent": round(used_words / max(n_units, 1), 1),
                     "theme_keywords_total": 0, "theme_keywords_covered": 0,
                     "theme_coverage_pct": 0, "n_precision_skips": 0,
                     "n_redundancy_skips": 0, "trie": {},
                     "n_units": n_units, "prefix_words": prefix_k,
                     "selected_details": []}
    else:
        compressed_context, used_words, sel_stats = select_retrieval_units(
            units, word_budget, tokenizer, model, args.device,
            sample.get("input", "").strip(), theme_percentile=args.theme_percentile,
            select_fn=select_fn, sig_words=args.sig_words,
            bge_prefix=args.bge_prefix,
            extended_stopwords=args.extended_stopwords)
        method = f"synthetic_unit_{args.selector}"

    sel_stats["mode"] = "synthetic"
    sel_stats["n_anchored"] = 0
    sel_stats["anchored_words"] = 0
    actual_tokens = count_tokens(compressed_context)
    sel_stats["actual_tokens"] = actual_tokens
    final_pct = used_words / max(orig_words, 1)

    metadata = {"_id": sample_id, "mode": "synthetic",
                "orig_words": orig_words,
                "word_budget": word_budget,
                "kept_words": used_words,
                "actual_tokens": actual_tokens,
                "budget_utilization": round(used_words / max(word_budget, 1), 4),
                "compression_ratio": round(final_pct, 4),
                "n_sents_clean": len(units),
                "n_selected": sel_stats["n_selected"],
                "pass1": sel_stats.get("pass1_selected", 0),
                "pass2": sel_stats.get("pass2_selected", 0),
                "n_units": len(units),
                "n_full_units": sel_stats.get("n_full_units"),
                "prefix_words": sel_stats.get("prefix_words"),
                "avg_words_per_sent": sel_stats.get("avg_words_per_sent", 0),
                "theme_coverage_pct": sel_stats.get("theme_coverage_pct", 0),
                "n_precision_skips": 0, "n_redundancy_skips": 0,
                "n_anchored": 0,
                "selected_details": []}
    result = {"_id": sample_id, "input": sample.get("input", ""),
              "context": compressed_context,
              "context_original_chars": len(context),
              "context_compressed_chars": len(compressed_context),
              "answers": sample.get("answers", []),
              "length": sample.get("length", 0),
              "dataset": dataset_name or "gov_report",
              "language": sample.get("language", "en"),
              "all_classes": all_classes,
              "compression_stats": {
                  "method": method,
                  "mode": "synthetic",
                  "selector": args.selector,
                  "selection_stats": sel_stats,
                  "n_units": len(units),
                  "orig_words": orig_words,
                  "kept_words": used_words,
                  "actual_tokens": actual_tokens,
                  "word_budget": word_budget,
                  "compression_ratio": round(final_pct, 4),
                  "token_budget_pct": args.token_budget_pct}}
    info = {"method": method, "n_units": len(units), "orig_words": orig_words,
            "used_words": used_words, "final_pct": final_pct,
            "n_full_units": sel_stats.get("n_full_units"),
            "prefix_words": sel_stats.get("prefix_words")}
    return result, metadata, info


# =============================================================================
# Per-mode hyperparameter defaults
# =============================================================================
# --code structure grammar. Path lines delimit cross-file snippets
# (repobench-p prefixes each snippet with a bare/commented file path);
# signature lines open block units; import-ish lines are API surface.
CODE_PATH_LINE_RE = re.compile(
    r"^\s*(?://|#|/\*)?\s*[\w.\-]+(?:[/\\][\w.\-]+)+\.[A-Za-z]{1,5}"
    r"\s*(?:\*/)?\s*$")
CODE_SIG_RE = re.compile(
    r"^\s{0,8}(?:"
    r"(?:(?:public|private|protected|internal|static|final|abstract|sealed"
    r"|async|export|default)\s+)*"
    r"(?:def|class|function|interface|enum|struct|trait|impl)\s+\w"
    r"|(?:public|private|protected|internal|static|final|abstract|async"
    r"|override|virtual|export)\b[^=;]*\()")
CODE_IMPORT_RE = re.compile(
    r"^\s*(?:import\s|from\s+\S+\s+import\s|using\s+[\w.]+\s*;"
    r"|#\s*include\s*[<\"]|require\s*\()")
CODE_ANCHOR_RE = re.compile(
    "(?:%s)|(?:%s)|(?:%s)" % (CODE_PATH_LINE_RE.pattern,
                              CODE_SIG_RE.pattern,
                              CODE_IMPORT_RE.pattern))

# Language keywords / ubiquitous builtins excluded from identifier keywords.
CODE_STOPWORDS = frozenset({
    # python
    "def", "class", "return", "import", "from", "as", "if", "elif", "else",
    "for", "while", "try", "except", "finally", "with", "lambda", "yield",
    "pass", "break", "continue", "raise", "assert", "del", "global",
    "nonlocal", "in", "is", "not", "and", "or", "none", "true", "false",
    "self", "cls", "init", "print", "args", "kwargs",
    # java / c# / js
    "public", "private", "protected", "internal", "static", "final",
    "abstract", "void", "new", "this", "super", "extends", "implements",
    "interface", "enum", "struct", "package", "throws", "throw", "catch",
    "switch", "case", "default", "do", "instanceof", "native",
    "synchronized", "transient", "volatile", "const", "var", "let",
    "function", "export", "async", "await", "using", "namespace",
    "override", "virtual", "readonly", "sealed", "out", "ref", "null",
    "get", "set",
    # common types / builtins
    "int", "long", "short", "byte", "float", "double", "boolean", "bool",
    "char", "string", "str", "list", "dict", "map", "array", "object",
    "len", "range", "value", "values", "key", "keys", "item", "items",
})

_CODE_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

# Per-mode tuned hyperparameter defaults for the dataset-mode adapters
# (--synthetic / --code). Explicit CLI values always win; flags below are
# None-sentinels resolved by resolve_mode_defaults(). Values frozen from
# dev50 slice sweeps.
MODE_DEFAULTS = {
    None:        dict(lam=0.5, query_mass=1.0, theme_percentile=0.9),
    "synthetic": dict(lam=0.5, query_mass=1.0, theme_percentile=0.75,
                      sig_words=12, count_min_prefix=8),
    "code":      dict(lam=0.5, query_mass=1.0, theme_percentile=0.5,
                      code_struct_frac=0.10),  # code_tail_frac: per-sample auto
}


def resolve_mode_defaults(args, parser=None):
    """Resolve None-sentinel knobs: explicit CLI > mode default > global
    default. Sets args.mode to 'synthetic' | 'code' | None."""
    mode = "synthetic" if args.synthetic else ("code" if args.code else None)
    args.mode = mode
    merged = dict(MODE_DEFAULTS[None])
    merged.update(MODE_DEFAULTS.get(mode) or {})
    for k, v in merged.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return mode


# =============================================================================
# --code: line records, identifier keywords, structure-rooted trie paths
# =============================================================================
def split_code_lines(context):
    """Records for code = non-blank physical lines, original order. Replaces
    prose cleaning + sentence splitting + filter_texts (whose is_junk drops
    ~76% of code lines: '}', 'return x;', short declarations...)."""
    return [ln.rstrip() for ln in context.split("\n") if ln.strip()]


def extract_code_identifiers(text):
    """Identifier tokens [A-Za-z_][A-Za-z0-9_]* plus camelCase/snake_case
    subtokens, lowercased, minus language keywords. Replaces the prose
    word/stem tokenizer for code on both the query and sentence sides
    (clean_text_words mangles 'm_Participants[i];')."""
    idents = set()
    for tok in _CODE_IDENT_RE.findall(text):
        low = tok.lower()
        if len(low) >= 2 and low not in CODE_STOPWORDS:
            idents.add(low)
        parts = _CAMEL_PART_RE.findall(tok)
        if len(parts) > 1:
            for p in parts:
                pl = p.lower()
                if len(pl) >= 3 and pl.isalpha() and pl not in CODE_STOPWORDS:
                    idents.add(pl)
    return idents


def code_query_terms(text, tail_words=200):
    """Query terms for code = identifiers from the informative TAIL (the
    region adjacent to the completion point)."""
    words = text.split()
    return extract_code_identifiers(" ".join(words[-tail_words:]))


def detect_code_units(lines):
    """Block/file unit id per line: a new unit opens at a file-path line
    (cross-file snippet boundary) or a def/class/method signature at
    indent <= 8. Lines before the first boundary share unit 0."""
    unit_ids, uid = [], 0
    for ln in lines:
        if CODE_PATH_LINE_RE.match(ln) or CODE_SIG_RE.match(ln):
            uid += 1
        unit_ids.append(uid)
    return unit_ids


def build_code_sent_data(lines, theme_percentile):
    """sent_data for code: keyword_weights = line identifiers plus one
    pseudo-keyword per enclosing structure unit. The pseudo-keyword is forced
    into theme_keywords with df = max_real_df + 1, so the df-sorted trie roots
    every line's path at its structure node: the trie's depth-1 level IS
    document structure, identifiers hang below, and the existing multi-anchor
    mass conservation splits identifier mass across unit contexts. No encoder
    passes (embedding_l2=None: the semantic channel stays off for code).

    Returns (sent_data, kw_df, theme_keywords, n_units).
    """
    unit_ids = detect_code_units(lines)
    sent_data = []
    for i, (ln, uid) in enumerate(zip(lines, unit_ids)):
        kw = {ident: 1.0 for ident in extract_code_identifiers(ln)}
        kw["\x00unit%d" % uid] = 1.0
        sent_data.append({"sent_idx": i, "text": ln,
                          "n_words": max(1, len(ln.split())),
                          "keyword_weights": kw,
                          "embedding_l2": None})
    kw_df, theme_keywords = profile_themes(sent_data,
                                           theme_percentile=theme_percentile)
    real_dfs = [df for kw, df in kw_df.items() if not kw.startswith("\x00")]
    max_real_df = max(real_dfs) if real_dfs else 1
    for kw in list(kw_df):
        if kw.startswith("\x00"):
            kw_df[kw] = max_real_df + 1
            theme_keywords.add(kw)
    n_units = len(set(unit_ids))
    return sent_data, kw_df, theme_keywords, n_units


def select_tail_anchors(sent_data, word_budget, tail_frac):
    """Positional generalization of select_anchors: force-include the
    contiguous suffix of records (the completion site) within
    tail_frac * word_budget words. Same return contract as select_anchors."""
    cap = int(word_budget * tail_frac)
    out, indices, used = [], set(), 0
    for sd in reversed(sent_data):
        if used + sd["n_words"] > cap:
            break
        out.append(sd)
        indices.add(sd["sent_idx"])
        used += sd["n_words"]
    out.reverse()
    return out, indices, used


def prep_code_sentences(context, token_budget_pct):
    """Line records for code mode. Returns (sentences, all_texts, orig_words,
    word_budget)."""
    sentences = split_code_lines(context)
    all_texts = sentences
    orig_words = sum(len(s.split()) for s in all_texts)
    word_budget = int(orig_words * token_budget_pct)
    return sentences, all_texts, orig_words, word_budget


def select_code_anchors(sent_data, word_budget, args, query_text):
    """Tail + structural anchors for code mode. Returns (anchored_sd,
    anchored_indices, anchored_words, anchor_re, extra) where `extra` carries
    the per-run stats the code sel_stats needs."""
    # Tail anchor: the contiguous completion-site suffix. Auto default expresses
    # "locality lives in the input when one exists": 0.10 (repobench-style: the
    # in-file prefix is appended uncompressed at prompt time, so cross-file
    # coverage matters more) vs 0.90 (lcc-style: the completion site is the
    # context tail itself; dev50 sweep peaked at 0.90). Frozen from dev50.
    tail_frac = (args.code_tail_frac if args.code_tail_frac is not None
                 else (0.10 if query_text else 0.90))
    tail_sd, tail_idx, tail_words_used = select_tail_anchors(
        sent_data, word_budget, tail_frac)
    non_tail_sd = [sd for sd in sent_data if sd["sent_idx"] not in tail_idx]
    anchor_re = CODE_ANCHOR_RE
    # Struct anchors live in the budget left after the tail, so
    # tail_frac + struct_frac can never exceed the word budget.
    struct_cap = max(0.0, min(
        args.code_struct_frac,
        (word_budget - tail_words_used) / max(word_budget, 1)))
    struct_sd, struct_idx, struct_words = select_anchors(
        non_tail_sd, anchor_re, word_budget, max_anchor_frac=struct_cap)
    anchored_sd = sorted(tail_sd + struct_sd, key=lambda sd: sd["sent_idx"])
    anchored_indices = tail_idx | struct_idx
    anchored_words = tail_words_used + struct_words
    extra = {"tail_frac": tail_frac, "n_tail_lines": len(tail_idx),
             "tail_words": tail_words_used, "n_struct_anchored": len(struct_idx)}
    return anchored_sd, anchored_indices, anchored_words, anchor_re, extra
