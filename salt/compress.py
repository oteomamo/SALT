# -*- coding: utf-8 -*-
"""
SALT compressor: keyword-trie sentence selection under a token budget.

One connector, two selectors (`--selector`, default `coverage`):

  coverage (default)  submodular trie-coverage via CELF lazy greedy
                      (`salt.engine.celf.coverage_select`). Single objective;
                      `--lam` / `--query-mass` are its dials.
  legacy              the multi-phase heuristic selector
                      (`salt.engine.retrieval.trie_select`): query anchoring,
                      branch budgeting, thematic fill.

Runs the shared prose pipeline (few-shot bypass -> dense-attention keywords ->
theme profiling -> structural anchors) and the dataset-mode adapters
(`salt.engine.dataset_modes`):

  --synthetic  paragraph-unit tasks (passage_count, passage_retrieval_en)
  --code       code-completion tasks (lcc, repobench-p) -- coverage only

Per-mode tuned hyperparameters (lam, query mass, theme percentile, tail
fraction) are frozen in dataset_modes.MODE_DEFAULTS; explicit flags override
them. Evaluate a compressed prompt with `eval.py`.

Usage (installed as the `salt` console command, also runnable as
`python -m salt.compress`):
    salt --data DATA.jsonl --output OUT.jsonl \
        [--device cpu] [--token-budget-pct 0.20] [--synthetic | --code] \
        [--selector legacy]
"""
import argparse
import time

from salt.engine import compressor, dataset_modes
from salt.engine.compressor import DATASET_JOIN, DATASET_ANCHORS
from salt.engine.celf import coverage_select
from salt.engine.retrieval import trie_select
from salt.engine.fewshot import detect as detect_fewshot


def build_parser():
    p = argparse.ArgumentParser(
        prog="salt",
        description="SALT compressor: keyword-trie selection "
                    "(coverage/CELF by default, --selector legacy available).")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--selector", choices=["coverage", "legacy"],
                   default="coverage",
                   help="coverage: submodular CELF (default). "
                        "legacy: the multi-phase trie_select heuristic.")
    p.add_argument("--model", type=str, default="BAAI/bge-small-en-v1.5")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--token-budget-pct", type=float, default=0.20)
    p.add_argument("--max-keywords-ratio", type=float, default=0.4)
    p.add_argument("--theme-percentile", type=float, default=None,
                   help="df percentile for theme keywords (default 0.9; "
                        "mode-resolved, see dataset_modes.MODE_DEFAULTS).")
    p.add_argument("--min-keywords", type=int, default=2)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--neighbor-window", type=int, default=0)
    p.add_argument("--query-uncapped", action="store_true")

    # --- coverage (CELF) knobs ---
    p.add_argument("--lam", type=float, default=None,
                   help="[coverage] discount in (0,1). lam->1: scalar ranking; "
                        "lam->0: hard set cover. Default 0.5 (mode-resolved).")
    p.add_argument("--query-mass", type=float, default=None,
                   help="[coverage] query mass as a ratio of document trie mass "
                        "(relevance odds). Default 1.0 (mode-resolved).")

    # --- legacy (trie_select) knobs ---
    p.add_argument("--pass1-budget", type=float, default=0.5,
                   help="[legacy] fraction of budget for the query-anchored pass.")
    p.add_argument("--theme-prec-min", type=float, default=0.0,
                   help="[legacy] min theme precision to admit a sentence.")
    p.add_argument("--redundancy-thresh", type=float, default=1.0,
                   help="[legacy] redundancy cutoff for pass-2 admission.")
    p.add_argument("--intro-pct", type=float, default=0.03,
                   help="[legacy] leading fraction kept as document intro.")
    p.add_argument("--query-budget-pct", type=float, default=0.75,
                   help="[legacy] cap on budget the query pass may consume.")
    p.add_argument("--branch-floor", type=int, default=0,
                   help="[legacy] minimum sentences guaranteed per theme branch.")

    # --- dataset modes ---
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument("--synthetic", action="store_true",
                            help="Paragraph-unit adapter for enumerated "
                                 "synthetic tasks (passage_count, "
                                 "passage_retrieval_en).")
    mode_group.add_argument("--code", action="store_true",
                            help="Code adapter (lcc, repobench-p): line "
                                 "records, identifier keywords, tail + "
                                 "import/signature anchors. Requires "
                                 "--selector coverage.")
    p.add_argument("--sig-words", type=int, default=None,
                   help="[--synthetic] words of body kept in every unit's "
                        "mandatory signature (default 12).")
    p.add_argument("--count-min-prefix", type=int, default=None,
                   help="[--synthetic] floor on the per-unit prefix length "
                        "for query-less rendering (default 8).")
    p.add_argument("--code-tail-frac", type=float, default=None,
                   help="[--code] budget fraction reserved for the contiguous "
                        "tail of the context (default: 0.10 if a query exists "
                        "else 0.90; frozen from dev50 sweeps).")
    p.add_argument("--code-struct-frac", type=float, default=None,
                   help="[--code] budget cap for import/signature/path "
                        "structural anchors (default 0.10).")
    p.add_argument("--fewshot", action="store_true",
                   help="Trie-driven few-shot mode: for detected exemplar "
                        "contexts (trec/samsum/triviaqa/...), pick a diverse "
                        "subset of WHOLE exemplars with the active selector "
                        "instead of the uniform-spacing bypass.")
    p.add_argument("--no-bge-prefix", action="store_false", dest="bge_prefix")
    p.add_argument("--no-extended-stopwords", action="store_false",
                   dest="extended_stopwords")
    p.add_argument("--anchor-max-frac", type=float, default=0.6,
                   help="Cap on fraction of word budget consumed by structural "
                        "anchors (default 0.6).")
    p.set_defaults(bge_prefix=True, extended_stopwords=True)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.code and args.selector != "coverage":
        parser.error("--code is coverage-only; drop --selector legacy "
                     "(or drop --code).")
    dataset_modes.resolve_mode_defaults(args)   # sets args.mode + mode defaults

    print(f"Loading: {args.data}")
    samples = compressor.load_samples(args.data, args.max_samples)
    print(f"Loaded {len(samples)} samples")

    print(f"Loading {args.model}...")
    tokenizer, model = compressor.load_bge(args.model, args.device)
    knobs = (f"lam={args.lam}, query_mass={args.query_mass}"
             if args.selector == "coverage"
             else f"min_kw={args.min_keywords}, pass1={args.pass1_budget}")
    print(f"Ready. selector={args.selector}, budget={args.token_budget_pct:.0%}, "
          f"mode={args.mode}, theme_pctl={args.theme_percentile}, {knobs}, "
          f"tiktoken={'on' if compressor._TIKTOKEN else 'off'}\n")

    # Selector wrapped for the synthetic query sub-policy; injected into
    # dataset_modes so the adapter itself stays selector-agnostic.
    def coverage_fn(sent_data, kw_df, theme_keywords, budget, *,
                    query_keywords, query_embedding, query_proper_nouns):
        return coverage_select(
            sent_data, kw_df, theme_keywords, budget,
            query_keywords=query_keywords, query_embedding=query_embedding,
            query_proper_nouns=query_proper_nouns,
            lam=args.lam, query_mass_ratio=args.query_mass)

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

    select_fn = coverage_fn if args.selector == "coverage" else legacy_fn
    if args.selector == "coverage":
        method_name, selector_name = "trie_coverage_celf", "coverage"
    else:
        method_name, selector_name = "keyword_trie", "legacy"

    # Standard-path selection call, branched on the active selector.
    def run_selector(sent_data, kw_df, theme_keywords, budget,
                     query_keywords, query_emb, query_pns, qtype):
        if args.selector == "coverage":
            selected, sel_stats = coverage_select(
                sent_data, kw_df, theme_keywords, budget,
                query_keywords=query_keywords, query_embedding=query_emb,
                query_proper_nouns=query_pns, lam=args.lam,
                query_mass_ratio=args.query_mass,
                token_fn=(dataset_modes.extract_code_identifiers
                          if args.mode == "code" else None))
            sel_stats["qtype"] = qtype
        else:
            selected, sel_stats = trie_select(
                sent_data, kw_df, theme_keywords, budget,
                query_keywords=query_keywords, query_embedding=query_emb,
                query_proper_nouns=query_pns, qtype=qtype,
                min_keywords=args.min_keywords, pass1_budget_pct=args.pass1_budget,
                theme_prec_min=args.theme_prec_min,
                redundancy_thresh=args.redundancy_thresh, intro_pct=args.intro_pct,
                neighbor_window=args.neighbor_window,
                query_budget_uncapped=args.query_uncapped,
                query_budget_pct=args.query_budget_pct, branch_floor=args.branch_floor)
        return selected, sel_stats

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
        if args.mode == "code":
            delim = "\n"   # line records must reassemble as lines

        # --- Dataset mode: --synthetic (paragraph-unit adapters). Must come
        # before few-shot detection: the numbered-paragraphs pattern otherwise
        # captures every enumerated-paragraph sample.
        if args.mode == "synthetic":
            out = dataset_modes.compress_synthetic(
                sample, sample_id, args, tokenizer, model, dataset_name,
                all_classes, context, orig_words, word_budget, select_fn)
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

        # --- Few-shot exemplars (skipped for code mode): uniform bypass, or
        # trie-driven exemplar selection under --fewshot. ---
        fs = None if args.mode == "code" else detect_fewshot(context)
        if fs is not None and fs.detected and fs.strategy == "bypass":
            if args.fewshot:
                result, meta_rec, info = compressor.compress_fewshot_units(
                    fs, sample, sample_id, args, dataset_name=dataset_name,
                    all_classes=all_classes, context=context,
                    orig_words=orig_words, word_budget=word_budget,
                    tokenizer=tokenizer, model=model, device=args.device,
                    select_fn=select_fn)
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

        # --- Standard path: prose or code sentence building ---
        if args.mode == "code":
            sentences, all_texts, orig_words, word_budget = \
                dataset_modes.prep_code_sentences(context, args.token_budget_pct)
        else:
            sentences, all_texts, orig_words, word_budget, is_code = \
                compressor.prep_prose_sentences(context, dataset_name, tokenizer,
                                                args.token_budget_pct)
        if not sentences:
            results.append(compressor.empty_result_record(
                sample, sample_id, dataset_name, all_classes, orig_words))
            continue
        total_raw += orig_words

        n_code_units = None
        code_extra = None
        if args.mode == "code":
            sent_data, kw_df, theme_keywords, n_code_units = \
                dataset_modes.build_code_sent_data(sentences, args.theme_percentile)
            q_src = query_text if query_text else context
            query_keywords = dataset_modes.code_query_terms(q_src, tail_words=200)
            query_emb = None; query_pns = None; qtype = None
        else:
            sent_data, kw_df, theme_keywords = compressor.build_prose_sent_data(
                sentences, tokenizer, model, args.device,
                args.max_keywords_ratio, args.theme_percentile)
            query_keywords, query_emb, query_pns, qtype = compressor.enrich_query(
                query_text, tokenizer, model, args.device,
                args.extended_stopwords, args.bge_prefix)

        # --- Structural anchors ---
        if args.mode == "code":
            (anchored_sd, anchored_indices, anchored_words,
             anchor_re, code_extra) = dataset_modes.select_code_anchors(
                sent_data, word_budget, args, query_text)
        else:
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
            selected_records, sel_stats = run_selector(
                non_anchored_sd, kw_df, theme_keywords, effective_budget,
                query_keywords, query_emb, query_pns, qtype)
            compressed_context, used_words = compressor.merge_anchored(
                selected_records, anchored_sd, theme_keywords, query_keywords,
                anchored_indices, anchored_words, delim, sel_stats)

        actual_tokens = compressor.count_tokens(compressed_context)
        sel_stats["actual_tokens"] = actual_tokens
        if args.mode:
            sel_stats["mode"] = args.mode
        if args.mode == "code":
            sel_stats["tail_frac"] = code_extra["tail_frac"]
            sel_stats["n_tail_lines"] = code_extra["n_tail_lines"]
            sel_stats["tail_words"] = code_extra["tail_words"]
            sel_stats["n_struct_anchored"] = code_extra["n_struct_anchored"]
            sel_stats["n_code_units"] = n_code_units
        total_kept += used_words

        result, meta_rec = compressor.build_standard_records(
            sample, sample_id, args, dataset_name=dataset_name,
            all_classes=all_classes, context=context,
            compressed_context=compressed_context, sel_stats=sel_stats,
            all_texts=all_texts, sentences=sentences,
            theme_keywords=theme_keywords, kw_df=kw_df, orig_words=orig_words,
            clean_words=clean_words, used_words=used_words,
            word_budget=word_budget, delim=delim, anchor_re=anchor_re,
            actual_tokens=actual_tokens, method_name=method_name,
            selector_name=selector_name)
        results.append(result); metadata.append(meta_rec)

        nb = sel_stats.get("n_branches", 0)
        p1 = sel_stats.get("pass1_selected", 0)
        p2 = sel_stats.get("pass2_selected", 0)
        na = sel_stats.get("n_anchored", 0)
        avg_w = sel_stats.get("avg_words_per_sent", 0)
        util = used_words / max(word_budget, 1)
        final_pct = used_words / max(orig_words, 1)
        qt = " [Q]" if query_text else ""
        md = f" [{args.mode}]" if args.mode else ""
        tk_str = f" tok={actual_tokens}" if actual_tokens is not None else ""
        print(f"  [{si+1}/{len(samples)}] {sample_id[:20]}...{qt}{md} "
              f"| {nb} branches | p1={p1} p2={p2}"
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
