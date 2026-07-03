# -*- coding: utf-8 -*-
"""
SALT dataset-mode adapter: the `--synthetic` mode.

This changes the *representation* fed to the selector, not the selector itself,
so it is selector-agnostic: the query sub-policy of `--synthetic` takes an
injected `select_fn` (`retrieval.trie_select` via `compress.py`). It imports no
selector; it depends only on `trie_core` primitives and `compressor` helpers.

  * `--synthetic` (passage_count, passage_retrieval_en): `Paragraph N:` units
    become the records fed to the injected selector; query-less samples render
    deterministic per-unit prefixes (no selector at all).
"""
import re

from salt.engine.trie_core import (
    is_content_word, profile_themes, clean_text_words,
    get_bge_sentence_embeddings, extract_query_keywords,
    extract_proper_nouns_in_query, embed_query,
)
from salt.engine.compressor import count_tokens


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
