# -*- coding: utf-8 -*-
"""
SALT shared compression runner: the prose pipeline the `compress.py` connector
builds on. It owns:

  * model / sample loading and JSONL output,
  * the prose sentence pipeline (clean -> split -> filter -> dense-attention
    keywords -> BGE embeddings -> theme profiling),
  * query-side enrichment, structural anchor selection + merge-back,
  * few-shot bypass handling, and
  * the canonical result / metadata record assembly.

The selector is injected by the connector (`trie_select`), so this module never
imports a selector directly.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from salt.engine.embedder import split_sentences as embed_split_sentences
from salt.engine.sentence_filter import filter_texts, clean_text_for_embedding
from salt.engine.fewshot import detect as detect_fewshot, select_blocks
from salt.engine.trie_core import (
    is_content_word, run_dense_attention, get_bge_sentence_embeddings,
    profile_themes, extract_query_keywords, extract_proper_nouns_in_query,
    detect_question_type, embed_query, SentenceRecord, clean_text_words,
)

# Optional tiktoken for actual-token budget verification (token-aware logging).
try:
    import tiktoken
    _TIKTOKEN = tiktoken.encoding_for_model("gpt-4")
except Exception:
    _TIKTOKEN = None


# =============================================================================
# Shared dataset-aware constants (prose path)
# =============================================================================
CODE_DATASETS = frozenset({"lcc", "repobench-p"})

# Structural anchor regexes. Sentences matching these are force-selected
# (bounded by a per-dataset cap) regardless of normal scoring.
import re as _re
DATASET_ANCHORS = {
    "passage_retrieval_en": _re.compile(r"^\s*Paragraph\s+\d+", _re.IGNORECASE),
    "passage_count":        _re.compile(r"^\s*Paragraph\s+\d+", _re.IGNORECASE),
}

# Per-dataset join delimiter when reassembling selected sentences.
DATASET_JOIN = {
    "lcc":                  "\n",
    "repobench-p":          "\n",
    "passage_retrieval_en": "\n\n",
    "passage_count":        "\n\n",
}


def count_tokens(text):
    """GPT-4 BPE token count for actual-budget verification, or None."""
    return len(_TIKTOKEN.encode(text)) if _TIKTOKEN and text else None


# =============================================================================
# Model / IO
# =============================================================================
def load_samples(path, max_samples=None):
    with open(path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if max_samples:
        samples = samples[:max_samples]
    return samples


def load_bge(model_name, device):
    """Load the BGE encoder with attention output enabled."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    if device != "cpu":
        model = model.to(device)
    return tokenizer, model


def write_outputs(output_path, results, metadata):
    """Write results + metadata JSONL. Returns the metadata path."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta_path = output_path.replace(".jsonl", "_metadata.jsonl")
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    return meta_path


def select_anchors(sent_data, anchor_re, word_budget, max_anchor_frac=0.6):
    """Returns (anchored_records, anchored_indices, anchored_words).
    Force-selects sentences whose text matches `anchor_re`, capped at
    `max_anchor_frac * word_budget` total words to prevent anchors from
    starving the rest of the budget.

    Anchors are taken in original document order so structural enumeration
    (Paragraph 1, 2, 3...) is preserved consistently rather than truncated
    arbitrarily.
    """
    if anchor_re is None or not sent_data:
        return [], set(), 0
    cap = int(word_budget * max_anchor_frac)
    out, indices, used = [], set(), 0
    for sd in sent_data:
        if anchor_re.search(sd["text"]):
            if used + sd["n_words"] > cap:
                continue
            out.append(sd)
            indices.add(sd["sent_idx"])
            used += sd["n_words"]
    return out, indices, used



def prep_prose_sentences(context, dataset_name, tokenizer, token_budget_pct):
    """Clean -> split -> filter prose into selection sentences.

    Returns (sentences, all_texts, orig_words, word_budget, is_code). `is_code`
    (dataset in CODE_DATASETS) disables prose cleaning / aggressive filtering so
    code datasets survive even in the standard prose path.
    """
    is_code = dataset_name in CODE_DATASETS
    context_clean = context if is_code else clean_text_for_embedding(context)
    raw_sentences = embed_split_sentences(context_clean, max_tokens=400,
                                          tokenizer=tokenizer)
    all_texts = [t for t, s, e in raw_sentences]
    sentences, n_junk, n_url, n_aggr, n_dedup = filter_texts(
        all_texts, aggressive=not is_code,
        remove_urls=not is_code, deduplicate=not is_code)
    orig_words = sum(len(s.split()) for s in all_texts)
    word_budget = int(orig_words * token_budget_pct)
    return sentences, all_texts, orig_words, word_budget, is_code


def build_prose_sent_data(sentences, tokenizer, model, device,
                          max_keywords_ratio, theme_percentile):
    """Dense-attention keywords + aligned BGE embeddings + theme profiling.

    Returns (sent_data, kw_df, theme_keywords).
    """
    attn_results = run_dense_attention(
        sentences, tokenizer, model,
        max_keywords_ratio=max_keywords_ratio, overlap_sents=2)

    # True [CLS] embeddings aligned with the query space.
    bge_embeddings = get_bge_sentence_embeddings(sentences, tokenizer, model, device)

    sent_data = []
    for i, sent in enumerate(sentences):
        ar = attn_results[i]
        emb_l2 = bge_embeddings[i]
        kw_weights = {}
        for j, w in enumerate(ar["word_labels"]):
            if w in ar["important_words"] and is_content_word(w):
                if j < len(ar["word_attns"]):
                    kw_weights[w] = max(kw_weights.get(w, 0.0),
                                        float(ar["word_attns"][j]))
        sent_data.append({"sent_idx": i, "text": sent,
                          "n_words": len(sent.split()),
                          "keyword_weights": kw_weights,
                          "embedding_l2": emb_l2})

    kw_df, theme_keywords = profile_themes(
        sent_data, theme_percentile=theme_percentile)
    return sent_data, kw_df, theme_keywords


def enrich_query(query_text, tokenizer, model, device,
                 extended_stopwords, bge_prefix):
    """Query-side enrichment for prose: keywords, proper nouns, question type,
    BGE query embedding. Returns (query_keywords, query_emb, query_proper_nouns,
    qtype); all None when there is no query.
    """
    if not query_text:
        return None, None, None, None
    query_keywords = extract_query_keywords(query_text, use_extended=extended_stopwords)
    query_proper_nouns = extract_proper_nouns_in_query(query_text)
    qtype = detect_question_type(query_text)
    query_emb = embed_query(query_text, tokenizer, model, device,
                            retrieval=bge_prefix)
    return query_keywords, query_emb, query_proper_nouns, qtype


def merge_anchored(selected_records, anchored_sd, theme_keywords, query_keywords,
                   anchored_indices, anchored_words, delim, sel_stats):
    """Inject anchored sentences as full SentenceRecords, sort into document
    order, join into the compressed context, and patch the selection stats to
    the post-merge state (mutates `sel_stats`). Returns (compressed_context,
    used_words).
    """
    effective_themes = theme_keywords | (query_keywords or set())
    for sd in anchored_sd:
        kw_set = set(sd["keyword_weights"].keys())
        theme_kws = kw_set & effective_themes
        sr = SentenceRecord(sent_idx=sd["sent_idx"], text=sd["text"],
                            n_words=sd["n_words"],
                            keyword_weights=sd["keyword_weights"],
                            theme_keywords=theme_kws,
                            embedding_l2=sd.get("embedding_l2"))
        selected_records.append(sr)
    selected_records.sort(key=lambda sr: sr.sent_idx)

    compressed_context = delim.join(sr.text for sr in selected_records)
    used_words = sel_stats["words_used"] + anchored_words
    sel_stats["words_used"] = used_words
    sel_stats["n_selected"] = len(selected_records)
    sel_stats["n_anchored"] = len(anchored_sd)
    sel_stats["anchored_words"] = anchored_words
    if anchored_sd:
        sel_stats["selected_details"] = [
            {"sent_idx": sr.sent_idx, "n_words": sr.n_words,
             "n_theme_kw": len(sr.theme_keywords),
             "theme_precision": round(sr.theme_precision, 3),
             "anchored": sr.sent_idx in anchored_indices}
            for sr in selected_records]
    return compressed_context, used_words


def fits_whole(sent_data, theme_keywords, delim, *, query_uncapped,
               neighbor_window, qtype, query_proper_nouns):
    """The 'everything fits under budget' short-circuit: keep all sentences, no
    selection. Returns (compressed_context, used_words, sel_stats).
    """
    clean_words = sum(sd["n_words"] for sd in sent_data)
    sel_stats = {"n_selected": len(sent_data), "pass1_selected": 0,
                 "pass2_selected": 0, "pass3_query_anchored": 0,
                 "words_used": clean_words,
                 "avg_words_per_sent": round(clean_words / max(len(sent_data), 1), 1),
                 "theme_keywords_total": len(theme_keywords),
                 "theme_keywords_covered": len(theme_keywords),
                 "theme_coverage_pct": 1.0, "trie": {},
                 "query_budget_uncapped": query_uncapped,
                 "neighbor_window": neighbor_window,
                 "qtype": qtype,
                 "n_query_proper_nouns": len(query_proper_nouns or set()),
                 "n_anchored": 0, "anchored_words": 0,
                 "selected_details": []}
    compressed_context = delim.join(sd["text"] for sd in sent_data)
    return compressed_context, clean_words, sel_stats


def empty_result_record(sample, sample_id, dataset_name, all_classes, orig_words):
    """Record for a sample whose context yielded no usable sentences. Matches
    the original quirk of always tagging method 'keyword_trie' and emitting no
    metadata row (caller must NOT append metadata for this case)."""
    return {"_id": sample_id, "input": sample.get("input", "").strip(),
            "context": "",
            "answers": sample.get("answers", []),
            "length": sample.get("length", 0),
            "dataset": dataset_name or "gov_report",
            "language": sample.get("language", "en"),
            "all_classes": all_classes,
            "compression_stats": {"method": "keyword_trie",
                                  "orig_words": orig_words,
                                  "kept_words": 0,
                                  "actual_tokens": 0,
                                  "compression_ratio": 0.0}}


def build_standard_records(sample, sample_id, args, *, dataset_name, all_classes,
                           context, compressed_context, sel_stats, all_texts,
                           sentences, theme_keywords, kw_df, orig_words,
                           clean_words, used_words, word_budget, delim, anchor_re,
                           actual_tokens, method_name, selector_name):
    """Assemble the (result, metadata) pair for the standard selection path.
    `sel_stats` should already carry actual_tokens / any mode-specific keys."""
    final_pct = used_words / max(orig_words, 1)
    neighbor_window = args.neighbor_window
    query_uncapped = args.query_uncapped

    metadata = {"_id": sample_id, "orig_words": orig_words,
                "word_budget": word_budget, "kept_words": used_words,
                "actual_tokens": actual_tokens,
                "budget_utilization": round(used_words / max(word_budget, 1), 4),
                "compression_ratio": round(final_pct, 4),
                "n_sents_clean": len(sentences),
                "n_selected": sel_stats["n_selected"],
                "pass1": sel_stats.get("pass1_selected", 0),
                "pass2": sel_stats.get("pass2_selected", 0),
                "pass3": sel_stats.get("pass3_query_anchored", 0),
                "n_anchored": sel_stats.get("n_anchored", 0),
                "anchored_words": sel_stats.get("anchored_words", 0),
                "avg_words_per_sent": sel_stats.get("avg_words_per_sent", 0),
                "theme_coverage_pct": sel_stats.get("theme_coverage_pct", 0),
                "n_precision_skips": sel_stats.get("n_precision_skips", 0),
                "n_redundancy_skips": sel_stats.get("n_redundancy_skips", 0),
                "neighbor_window": neighbor_window,
                "query_uncapped": query_uncapped,
                "qtype": sel_stats.get("qtype"),
                "n_query_proper_nouns": sel_stats.get("n_query_proper_nouns", 0),
                "selected_details": sel_stats.get("selected_details", [])}

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
                  "method": method_name,
                  "selector": selector_name,
                  "theme_percentile": args.theme_percentile,
                  "min_keywords": args.min_keywords,
                  "selection_stats": sel_stats,
                  "n_sentences_raw": len(all_texts),
                  "n_sentences_clean": len(sentences),
                  "n_sentences_kept": sel_stats["n_selected"],
                  "n_theme_keywords": len(theme_keywords),
                  "n_total_keywords": len(kw_df),
                  "orig_words": orig_words,
                  "clean_words": clean_words,
                  "kept_words": used_words,
                  "actual_tokens": actual_tokens,
                  "word_budget": word_budget,
                  "compression_ratio": round(final_pct, 4),
                  "token_budget_pct": args.token_budget_pct,
                  "model": args.model,
                  "neighbor_window": neighbor_window,
                  "query_uncapped": query_uncapped,
                  "bge_prefix": args.bge_prefix,
                  "extended_stopwords": args.extended_stopwords,
                  "join_delimiter": delim,
                  "anchor_pattern": (anchor_re.pattern if anchor_re else None)}}

    if getattr(args, "mode", None):
        metadata["mode"] = args.mode
        result["compression_stats"]["mode"] = args.mode
    return result, metadata


# =============================================================================
# Few-shot bypass (selector-agnostic)
# =============================================================================
def compress_fewshot(fs, sample, sample_id, args, *, dataset_name, all_classes,
                     context, orig_words, word_budget):
    """Few-shot block-selection bypass. Returns (result, metadata, info) where
    `info` carries fields for the caller's progress log."""
    compressed_context, used_words, n_sel, n_total = select_blocks(
        fs.blocks, word_budget)
    final_pct = used_words / max(orig_words, 1)
    actual_tokens = count_tokens(compressed_context)

    sel_stats = {"n_selected": n_sel, "pass1_selected": n_sel,
                 "pass2_selected": 0, "n_branches": 0,
                 "words_used": used_words,
                 "avg_words_per_sent": round(used_words / max(n_sel, 1), 1),
                 "theme_keywords_total": 0, "theme_keywords_covered": 0,
                 "theme_coverage_pct": 0, "n_precision_skips": 0,
                 "n_redundancy_skips": 0, "trie": {},
                 "fewshot_pattern": fs.pattern_name,
                 "fewshot_blocks_total": n_total,
                 "fewshot_blocks_kept": n_sel,
                 "n_anchored": 0, "anchored_words": 0,
                 "actual_tokens": actual_tokens,
                 "selected_details": []}
    metadata = {"_id": sample_id, "orig_words": orig_words,
                "word_budget": word_budget, "kept_words": used_words,
                "actual_tokens": actual_tokens,
                "budget_utilization": round(used_words / max(word_budget, 1), 4),
                "compression_ratio": round(final_pct, 4),
                "n_sents_clean": n_total, "n_selected": n_sel,
                "pass1": n_sel, "pass2": 0,
                "avg_words_per_sent": round(used_words / max(n_sel, 1), 1),
                "theme_coverage_pct": 0, "n_precision_skips": 0,
                "n_redundancy_skips": 0,
                "fewshot_pattern": fs.pattern_name,
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
                  "method": "fewshot_block_select",
                  "fewshot_pattern": fs.pattern_name,
                  "fewshot_blocks_total": n_total,
                  "fewshot_blocks_kept": n_sel,
                  "selection_stats": sel_stats,
                  "orig_words": orig_words,
                  "kept_words": used_words,
                  "actual_tokens": actual_tokens,
                  "word_budget": word_budget,
                  "compression_ratio": round(final_pct, 4),
                  "token_budget_pct": args.token_budget_pct}}
    info = {"n_sel": n_sel, "n_total": n_total, "used_words": used_words,
            "final_pct": final_pct, "actual_tokens": actual_tokens,
            "pattern_name": fs.pattern_name}
    return result, metadata, info


def compress_fewshot_units(fs, sample, sample_id, args, *, dataset_name,
                           all_classes, context, orig_words, word_budget,
                           tokenizer, model, device, select_fn):
    """Trie-driven few-shot mode (--fewshot): each exemplar block becomes a unit
    record and the injected `select_fn` picks a diverse subset of WHOLE
    exemplars under budget -- vs `compress_fewshot`'s uniform-spacing bypass.
    Exemplar integrity is preserved (blocks are never split), but the selection
    is now trie-driven, so these datasets actually exercise the main mechanism.
    Returns (result, metadata, info)."""
    blocks = fs.blocks
    n_total = len(blocks)
    query_text = sample.get("input", "").strip()
    total_words = sum(len(b.split()) for b in blocks)

    if total_words <= word_budget:
        # Everything fits: keep all exemplars (matches select_blocks).
        chosen = list(range(n_total))
        used_words = total_words
        sel_stats = {"n_selected": n_total, "pass1_selected": n_total,
                     "pass2_selected": 0, "n_branches": 0, "words_used": used_words,
                     "avg_words_per_sent": round(used_words / max(n_total, 1), 1),
                     "theme_keywords_total": 0, "theme_keywords_covered": 0,
                     "theme_coverage_pct": 0, "n_precision_skips": 0,
                     "n_redundancy_skips": 0, "trie": {},
                     "selector": args.selector, "selected_details": []}
    else:
        block_embs = get_bge_sentence_embeddings(blocks, tokenizer, model, device)
        sent_data = []
        for i, (blk, emb) in enumerate(zip(blocks, block_embs)):
            kw = {w: 1.0 for w in clean_text_words(blk) if is_content_word(w)}
            sent_data.append({"sent_idx": i, "text": blk,
                              "n_words": max(1, len(blk.split())),
                              "keyword_weights": kw, "embedding_l2": emb})
        kw_df, theme_keywords = profile_themes(
            sent_data, theme_percentile=args.theme_percentile)
        if query_text:
            qkw = extract_query_keywords(query_text, use_extended=args.extended_stopwords)
            qpn = extract_proper_nouns_in_query(query_text)
            qemb = embed_query(query_text, tokenizer, model, device,
                               retrieval=args.bge_prefix)
        else:
            qkw = qpn = qemb = None
        selected, sel_stats = select_fn(
            sent_data, kw_df, theme_keywords, word_budget,
            query_keywords=qkw, query_embedding=qemb, query_proper_nouns=qpn)
        chosen = sorted(sr.sent_idx for sr in selected)
        used_words = sel_stats["words_used"]

    kept = [blocks[i] for i in sorted(chosen)]
    compressed_context = "\n".join(kept)
    n_sel = len(kept)
    actual_tokens = count_tokens(compressed_context)
    final_pct = used_words / max(orig_words, 1)

    sel_stats["fewshot_pattern"] = fs.pattern_name
    sel_stats["fewshot_blocks_total"] = n_total
    sel_stats["fewshot_blocks_kept"] = n_sel
    sel_stats["mode"] = "fewshot"
    sel_stats["actual_tokens"] = actual_tokens
    sel_stats["n_anchored"] = 0
    sel_stats["anchored_words"] = 0

    metadata = {"_id": sample_id, "orig_words": orig_words,
                "word_budget": word_budget, "kept_words": used_words,
                "actual_tokens": actual_tokens,
                "budget_utilization": round(used_words / max(word_budget, 1), 4),
                "compression_ratio": round(final_pct, 4),
                "n_sents_clean": n_total, "n_selected": n_sel,
                "pass1": sel_stats.get("pass1_selected", 0),
                "pass2": sel_stats.get("pass2_selected", 0),
                "avg_words_per_sent": sel_stats.get("avg_words_per_sent", 0),
                "theme_coverage_pct": sel_stats.get("theme_coverage_pct", 0),
                "n_precision_skips": 0, "n_redundancy_skips": 0,
                "fewshot_pattern": fs.pattern_name, "mode": "fewshot",
                "n_anchored": 0, "selected_details": []}
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
                  "method": f"fewshot_unit_{args.selector}",
                  "mode": "fewshot",
                  "selector": args.selector,
                  "fewshot_pattern": fs.pattern_name,
                  "fewshot_blocks_total": n_total,
                  "fewshot_blocks_kept": n_sel,
                  "selection_stats": sel_stats,
                  "orig_words": orig_words,
                  "kept_words": used_words,
                  "actual_tokens": actual_tokens,
                  "word_budget": word_budget,
                  "compression_ratio": round(final_pct, 4),
                  "token_budget_pct": args.token_budget_pct}}
    info = {"n_sel": n_sel, "n_total": n_total, "used_words": used_words,
            "final_pct": final_pct, "actual_tokens": actual_tokens,
            "pattern_name": fs.pattern_name}
    return result, metadata, info


# =============================================================================
# Summary
# =============================================================================
def print_summary(results, metadata, args, total_raw, total_kept,
                  output_path, meta_path):
    fr = total_kept / max(total_raw, 1)
    avg_tc = np.mean([r["compression_stats"].get("selection_stats", {}).get("theme_coverage_pct", 0)
                      for r in results
                      if r["compression_stats"].get("selection_stats")]) if results else 0
    n_fewshot = sum(1 for r in results
                    if r["compression_stats"].get("method") == "fewshot_block_select")
    n_anchored_samples = sum(1 for m in metadata if m.get("n_anchored", 0) > 0)
    print(f"\n{'='*60}")
    print(f"COMPRESSION SUMMARY ({len(results)} samples)")
    print(f"{'='*60}")
    print(f"  Method: keyword_trie | Budget: {args.token_budget_pct:.0%}")
    if n_fewshot:
        print(f"  Few-shot bypass: {n_fewshot}/{len(results)} samples")
    if n_anchored_samples:
        print(f"  Anchor-boosted: {n_anchored_samples}/{len(results)} samples")
    print(f"  Words: {total_raw} -> {total_kept} ({fr:.1%})")
    print(f"  Avg theme coverage: {avg_tc:.1%}")
    if metadata:
        avg_util = np.mean([m["budget_utilization"] for m in metadata])
        avg_prec_skip = np.mean([m["n_precision_skips"] for m in metadata])
        avg_red_skip = np.mean([m["n_redundancy_skips"] for m in metadata])
        print(f"  Avg budget utilization: {avg_util:.1%}")
        print(f"  Avg precision skips: {avg_prec_skip:.0f}, redundancy skips: {avg_red_skip:.0f}")
        if _TIKTOKEN:
            toks = [m.get("actual_tokens") for m in metadata if m.get("actual_tokens")]
            if toks:
                tk_per_word = np.mean([t / max(m["kept_words"], 1) for t, m
                                       in zip(toks, metadata) if t])
                print(f"  Avg actual tokens (GPT-4 BPE): {np.mean(toks):.0f}, "
                      f"tokens/word: {tk_per_word:.2f}")
    print(f"  Saved: {output_path}")
    print(f"  Metadata: {meta_path}")
