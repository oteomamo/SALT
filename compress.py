# -*- coding: utf-8 -*-
"""
SALT compressor: keyword-trie guided-traversal selection.

The one-shot compressor. It runs the prose pipeline (few-shot bypass ->
dense-attention keywords -> theme profiling -> structural anchors) and selects
sentences with the `trie_select` heuristic (`salt.engine.retrieval`). The
`--synthetic` paragraph-unit adapter is also available (driving `trie_select`
via `salt.engine.dataset_modes`).

Evaluate a compressed prompt with `eval.py`.

Usage:
    python compress.py --data DATA.jsonl --output OUT.jsonl \
        --device cuda --token-budget-pct 0.20 [--synthetic]
"""
import argparse
import time

from salt.engine import compressor, dataset_modes
from salt.engine.compressor import DATASET_JOIN, DATASET_ANCHORS
from salt.engine.retrieval import trie_select
from salt.engine.fewshot import detect as detect_fewshot


def build_parser():
    p = argparse.ArgumentParser(
        description="SALT compressor: keyword trie + guided traversal (retrieval).")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--token-budget-pct", type=float, default=0.20)
    p.add_argument("--max-keywords-ratio", type=float, default=0.4)
    p.add_argument("--theme-percentile", type=float, default=None,
                   help="df percentile for theme keywords "
                        "(default 0.9; 0.75 under --synthetic).")
    p.add_argument("--min-keywords", type=int, default=2)
    p.add_argument("--pass1-budget", type=float, default=0.5)
    p.add_argument("--theme-prec-min", type=float, default=0.0)
    p.add_argument("--redundancy-thresh", type=float, default=1.0)
    p.add_argument("--intro-pct", type=float, default=0.03)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--neighbor-window", type=int, default=0)
    p.add_argument("--query-uncapped", action="store_true")
    p.add_argument("--query-budget-pct", type=float, default=0.75)
    p.add_argument("--branch-floor", type=int, default=0)
    p.add_argument("--no-bge-prefix", action="store_false", dest="bge_prefix")
    p.add_argument("--no-extended-stopwords", action="store_false",
                   dest="extended_stopwords")
    p.add_argument("--anchor-max-frac", type=float, default=0.6,
                   help="Cap on fraction of word budget consumed by structural "
                        "anchors (default 0.6).")
    p.add_argument("--synthetic", action="store_true",
                   help="Paragraph-unit adapter for enumerated synthetic tasks "
                        "(passage_count, passage_retrieval_en): units replace "
                        "sentences as the records fed to trie_select; query-less "
                        "samples render deterministic per-unit prefixes.")
    p.add_argument("--sig-words", type=int, default=None,
                   help="[--synthetic] words of body kept in every unit's "
                        "mandatory signature (default 12).")
    p.add_argument("--count-min-prefix", type=int, default=None,
                   help="[--synthetic] floor on the per-unit prefix length for "
                        "query-less rendering (default 8).")
    p.add_argument("--fewshot", action="store_true",
                   help="Trie-driven few-shot mode: for detected exemplar "
                        "contexts (trec/samsum/triviaqa/...), pick a diverse "
                        "subset of WHOLE exemplars with trie_select "
                        "instead of the uniform-spacing bypass.")
    p.set_defaults(bge_prefix=True, extended_stopwords=True)
    p.add_argument("--verbose", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    args.mode = "synthetic" if args.synthetic else None
    args.selector = "legacy"
    # Mode-resolved defaults for the knobs the retrieval path uses.
    if args.theme_percentile is None:
        args.theme_percentile = 0.75 if args.mode == "synthetic" else 0.9
    if args.sig_words is None:
        args.sig_words = 12
    if args.count_min_prefix is None:
        args.count_min_prefix = 8

    # Legacy selector wrapped for the synthetic query sub-policy; injected into
    # dataset_modes so the adapter itself stays selector-agnostic.
    def legacy_fn(sent_data, kw_df, theme_keywords, budget, *,
                  query_keywords, query_embedding, query_proper_nouns):
        return trie_select(
            sent_data, kw_df, theme_keywords, budget,
            query_keywords=query_keywords, query_embedding=query_embedding,
            query_proper_nouns=query_proper_nouns, qtype=None,
            min_keywords=args.min_keywords, pass1_budget_pct=args.pass1_budget,
            theme_prec_min=args.theme_prec_min,
            redundancy_thresh=args.redundancy_thresh, intro_pct=args.intro_pct,
            neighbor_window=args.neighbor_window,
            query_budget_uncapped=args.query_uncapped,
            query_budget_pct=args.query_budget_pct, branch_floor=args.branch_floor)

    print(f"Loading: {args.data}")
    samples = compressor.load_samples(args.data, args.max_samples)
    print(f"Loaded {len(samples)} samples")

    print(f"Loading {args.model}...")
    tokenizer, model = compressor.load_bge(args.model, args.device)
    print(f"Ready. Budget={args.token_budget_pct:.0%}, "
          f"theme_pctl={args.theme_percentile}, min_kw={args.min_keywords}, "
          f"bge_prefix={args.bge_prefix}, ext_stopwords={args.extended_stopwords}, "
          f"tiktoken={'on' if compressor._TIKTOKEN else 'off'}\n")

    results, metadata = [], []
    total_raw, total_kept = 0, 0

    for si, sample in enumerate(samples):
        t0 = time.time()
        context = sample.get("context", "")
        sample_id = sample.get("_id", f"sample_{si}")
        query_text = sample.get("input", "").strip()
        dataset_name = sample.get("dataset", "")
        all_classes = sample.get("all_classes", None)
        orig_words = len(context.split())
        word_budget = int(orig_words * args.token_budget_pct)
        delim = DATASET_JOIN.get(dataset_name, " ")

        # --- Dataset mode: --synthetic (paragraph-unit adapter). Must come
        # before few-shot detection, which would otherwise capture these sets. ---
        if args.mode == "synthetic":
            out = dataset_modes.compress_synthetic(
                sample, sample_id, args, tokenizer, model, dataset_name,
                all_classes, context, orig_words, word_budget, legacy_fn)
            if out is not None:
                result, meta_rec, info = out
                results.append(result); metadata.append(meta_rec)
                total_raw += orig_words; total_kept += info["used_words"]
                detail = (f"full={info['n_full_units']}/{info['n_units']}"
                          if info['n_full_units'] is not None
                          else f"K={info['prefix_words']}")
                print(f"  [{si+1}/{len(samples)}] {sample_id[:20]}..."
                      f" [SYNTH:{info['method'].split('_', 1)[1]}] | {detail} "
                      f"| {orig_words}->{info['used_words']} ({info['final_pct']:.1%}) "
                      f"| {time.time()-t0:.1f}s")
                continue
            print(f"  [{si+1}/{len(samples)}] {sample_id[:20]}..."
                  f" [SYNTH:no-units] falling through to standard pipeline")

        # --- Few-shot exemplars (TREC/SAMSum/TriviaQA-style blocks): uniform
        # bypass, or trie-driven exemplar selection under --fewshot. ---
        fs = detect_fewshot(context)
        if fs is not None and fs.detected and fs.strategy == "bypass":
            if args.fewshot:
                result, meta_rec, info = compressor.compress_fewshot_units(
                    fs, sample, sample_id, args, dataset_name=dataset_name,
                    all_classes=all_classes, context=context,
                    orig_words=orig_words, word_budget=word_budget,
                    tokenizer=tokenizer, model=model, device=args.device,
                    select_fn=legacy_fn)
            else:
                result, meta_rec, info = compressor.compress_fewshot(
                    fs, sample, sample_id, args, dataset_name=dataset_name,
                    all_classes=all_classes, context=context,
                    orig_words=orig_words, word_budget=word_budget)
            results.append(result); metadata.append(meta_rec)
            total_raw += orig_words; total_kept += info["used_words"]
            tag = "FEWSHOT-UNIT" if args.fewshot else "FEWSHOT"
            print(f"  [{si+1}/{len(samples)}] {sample_id[:20]}..."
                  f" [{tag}:{info['pattern_name']}] | {info['n_sel']}/{info['n_total']} blocks "
                  f"| {orig_words}->{info['used_words']} ({info['final_pct']:.1%}) | {time.time()-t0:.1f}s")
            continue

        # --- Standard prose pipeline ---
        sentences, all_texts, orig_words, word_budget, is_code = \
            compressor.prep_prose_sentences(context, dataset_name, tokenizer,
                                            args.token_budget_pct)
        if not sentences:
            results.append(compressor.empty_result_record(
                sample, sample_id, dataset_name, all_classes, orig_words))
            continue
        total_raw += orig_words

        sent_data, kw_df, theme_keywords = compressor.build_prose_sent_data(
            sentences, tokenizer, model, args.device,
            args.max_keywords_ratio, args.theme_percentile)
        query_keywords, query_emb, query_pns, qtype = compressor.enrich_query(
            query_text, tokenizer, model, args.device,
            args.extended_stopwords, args.bge_prefix)

        anchor_re = DATASET_ANCHORS.get(dataset_name)
        anchored_sd, anchored_indices, anchored_words = compressor.select_anchors(
            sent_data, anchor_re, word_budget, max_anchor_frac=args.anchor_max_frac)

        clean_words = sum(sd["n_words"] for sd in sent_data)
        if clean_words <= word_budget:
            compressed_context, used_words, sel_stats = compressor.fits_whole(
                sent_data, theme_keywords, delim,
                query_uncapped=args.query_uncapped,
                neighbor_window=args.neighbor_window, qtype=qtype,
                query_proper_nouns=query_pns)
        else:
            non_anchored_sd = [sd for sd in sent_data
                               if sd["sent_idx"] not in anchored_indices]
            effective_budget = max(0, word_budget - anchored_words)
            selected_records, sel_stats = trie_select(
                non_anchored_sd, kw_df, theme_keywords, effective_budget,
                query_keywords=query_keywords, query_embedding=query_emb,
                query_proper_nouns=query_pns, qtype=qtype,
                min_keywords=args.min_keywords,
                pass1_budget_pct=args.pass1_budget,
                theme_prec_min=args.theme_prec_min,
                redundancy_thresh=args.redundancy_thresh,
                intro_pct=args.intro_pct, neighbor_window=args.neighbor_window,
                query_budget_uncapped=args.query_uncapped,
                query_budget_pct=args.query_budget_pct,
                branch_floor=args.branch_floor)
            compressed_context, used_words = compressor.merge_anchored(
                selected_records, anchored_sd, theme_keywords, query_keywords,
                anchored_indices, anchored_words, delim, sel_stats)

        actual_tokens = compressor.count_tokens(compressed_context)
        sel_stats["actual_tokens"] = actual_tokens
        total_kept += used_words

        result, meta_rec = compressor.build_standard_records(
            sample, sample_id, args, dataset_name=dataset_name,
            all_classes=all_classes, context=context,
            compressed_context=compressed_context, sel_stats=sel_stats,
            all_texts=all_texts, sentences=sentences,
            theme_keywords=theme_keywords, kw_df=kw_df, orig_words=orig_words,
            clean_words=clean_words, used_words=used_words,
            word_budget=word_budget, delim=delim, anchor_re=anchor_re,
            actual_tokens=actual_tokens, method_name="keyword_trie",
            selector_name="legacy")
        results.append(result); metadata.append(meta_rec)

        nb = sel_stats.get("n_branches", 0)
        p1 = sel_stats.get("pass1_selected", 0)
        p2 = sel_stats.get("pass2_selected", 0)
        p3 = sel_stats.get("pass3_query_anchored", 0)
        na = sel_stats.get("n_anchored", 0)
        avg_w = sel_stats.get("avg_words_per_sent", 0)
        util = used_words / max(word_budget, 1)
        final_pct = used_words / max(orig_words, 1)
        qt = " [Q]" if query_text else ""
        tk_str = f" tok={actual_tokens}" if actual_tokens is not None else ""
        print(f"  [{si+1}/{len(samples)}] {sample_id[:20]}...{qt} "
              f"| {nb} branches | p1={p1} p2={p2} p3={p3}"
              f"{f' anc={na}' if na else ''} (avg {avg_w}w) "
              f"| util={util:.0%} | {orig_words}->{used_words}{tk_str} "
              f"({final_pct:.1%}) | {time.time()-t0:.1f}s")

        if args.verbose:
            top_themes = sorted([(kw, kw_df[kw]) for kw in theme_keywords if kw in kw_df],
                                key=lambda x: -x[1])[:15]
            print(f"    Themes ({len(theme_keywords)}): "
                  f"{', '.join(f'{k}(df={v})' for k, v in top_themes)}")

    meta_path = compressor.write_outputs(args.output, results, metadata)
    compressor.print_summary(results, metadata, args, total_raw, total_kept,
                             args.output, meta_path)


if __name__ == "__main__":
    main()
