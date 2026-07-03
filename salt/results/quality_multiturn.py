# --- salt path bootstrap: make 'salt' importable regardless of CWD (portable) ---
import os as _os, sys as _sys
_p = _os.path.dirname(_os.path.abspath(__file__))
while _p != _os.path.dirname(_p) and not _os.path.isdir(_os.path.join(_p, "salt", "engine")):
    _p = _os.path.dirname(_p)
if _p not in _sys.path:
    _sys.path.insert(0, _p)
# --- end salt path bootstrap ---
# -*- coding: utf-8 -*-
"""
Multi-turn SALT experiment on QuALITY (MCQ) / LooGLE (free-form).

The premise: a long document is read ONCE and then asked MANY questions. SALT
indexes the article a single time (split -> filter -> dense-attention keywords
-> BGE embeddings -> theme profile), then every question is a cheap "turn": embed
the query and run the keyword-trie guided-traversal selector (`trie_select`) to
pull a small, query-relevant context for the LLM. This amortizes the expensive
encode over all turns — the headline compares one-time index cost against the
mean per-turn cost.

Format is auto-detected from each article's `format` field:
  mcq       QuALITY   -> argmax over " A"/" B"/" C"/" D" logits at the answer pos.
  freeform  LooGLE    -> greedy generation scored with SQuAD-style token-F1.

With `--llm` set, each turn also measures TTFT (prefill), decode TPOT, peak GPU
memory, and accuracy. `--no-compression` feeds the raw article as the baseline.
A warmup article runs first to absorb CUDA cold-start cost.

Build the data with `python -m salt.datasets.download_quality` (see
salt/datasets/README.md). This runner uses the one-shot `trie_select` selector
per turn; cross-turn coverage memory is a separate concern (see
salt.engine.session_trie).

Usage:
    python salt/results/quality_multiturn.py \
        --data salt/datasets/quality/quality_subset_50.json \
        --llm meta-llama/Llama-3.1-8B-Instruct \
        --compress-pct 0.20 --device cuda:0 --output runs/quality.jsonl
"""
import argparse
import json
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from salt.engine import compressor
from salt.engine.compressor import (
    prep_prose_sentences, build_prose_sent_data, enrich_query,
)
from salt.engine.retrieval import trie_select

DEFAULT_DATA = (Path(__file__).resolve().parent.parent
                / "datasets" / "quality" / "quality_subset_50.json")

# Per-turn selector knobs (mirror the query-mode defaults exercised in the paper).
TURN_SELECT_KW = dict(
    min_keywords=2, pass1_budget_pct=0.5, theme_prec_min=0.0,
    redundancy_thresh=1.0, intro_pct=0.03, neighbor_window=1,
    query_budget_uncapped=False, query_budget_pct=0.75, branch_floor=0,
)


# ---------- GPU helpers ----------
def _reset_peak(device):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_mb(device):
    # NB: pass `device` — max_memory_allocated() defaults to the current device,
    # which is not necessarily the one the model runs on (e.g. cuda:1).
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


# ---------- Token-F1 (SQuAD style) ----------
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def _normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES_RE.sub(" ", s)
    return " ".join(s.split())


def token_f1(pred, gold):
    pred_toks = _normalize_answer(pred).split()
    gold_toks = _normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pred_toks)
    r = num_same / len(gold_toks)
    return 2 * p * r / (p + r)


# ---------- INDEX-ONCE ----------
def build_article_index(article_text, embed_tok, embed_model, device,
                        compress_pct):
    """Split -> filter -> dense-attention keywords -> BGE embeddings -> themes.
    Done a single time per article; every turn reuses the returned index."""
    timings, mem = {}, {}
    _reset_peak(device)
    t_total = time.perf_counter()

    t = time.perf_counter()
    sentences, all_texts, orig_words, _wb, _is_code = prep_prose_sentences(
        article_text, dataset_name="", tokenizer=embed_tok,
        token_budget_pct=compress_pct)
    timings["preprocess_s"] = time.perf_counter() - t

    t = time.perf_counter()
    sent_data, kw_df, theme_keywords = build_prose_sent_data(
        sentences, embed_tok, embed_model, device,
        max_keywords_ratio=0.4, theme_percentile=0.9)
    timings["encode_themes_s"] = time.perf_counter() - t

    timings["index_total_s"] = time.perf_counter() - t_total
    timings["orig_words"] = orig_words
    timings["n_sentences"] = len(sentences)
    timings["n_theme_keywords"] = len(theme_keywords)
    mem["index_peak_mb"] = _peak_mb(device)

    return {"sent_data": sent_data, "kw_df": kw_df,
            "theme_keywords": theme_keywords, "orig_words": orig_words,
            "index_timings": timings, "index_mem": mem}


# ---------- PER-TURN (SALT) ----------
def run_turn(index, query_text, embed_tok, embed_model, device, compress_pct):
    """One question turn: embed the query, then trie_select a query-relevant
    context from the already-indexed article."""
    timings, mem = {}, {}
    _reset_peak(device)
    t_total = time.perf_counter()

    t = time.perf_counter()
    q_kws, q_emb, q_pns, qtype = enrich_query(
        query_text, embed_tok, embed_model, device,
        extended_stopwords=True, bge_prefix=True)
    timings["query_embed_s"] = time.perf_counter() - t

    t = time.perf_counter()
    word_budget = int(index["orig_words"] * compress_pct)
    clean_words = sum(sd["n_words"] for sd in index["sent_data"])
    if clean_words <= word_budget:
        compressed = " ".join(sd["text"] for sd in index["sent_data"])
        n_selected, theme_cov = len(index["sent_data"]), 1.0
    else:
        selected, sel_stats = trie_select(
            index["sent_data"], dict(index["kw_df"]), index["theme_keywords"],
            word_budget, query_keywords=q_kws, query_embedding=q_emb,
            query_proper_nouns=q_pns, qtype=qtype, **TURN_SELECT_KW)
        compressed = " ".join(sr.text for sr in selected)
        n_selected = sel_stats.get("n_selected", len(selected))
        theme_cov = sel_stats.get("theme_coverage_pct", 0.0)
    timings["retrieval_s"] = time.perf_counter() - t

    timings["turn_total_s"] = time.perf_counter() - t_total
    timings["kept_words"] = len(compressed.split())
    timings["compression_ratio"] = timings["kept_words"] / max(index["orig_words"], 1)
    mem["turn_peak_mb"] = _peak_mb(device)

    return {"compressed_text": compressed, "n_selected": n_selected,
            "theme_coverage_pct": theme_cov,
            "turn_timings": timings, "turn_mem": mem}


# ---------- LLM: MCQ branch ----------
def measure_llm_mcq(context_text, question, options, gold_label,
                    llm_tok, llm_model, device, total_tokens):
    letters = ["A", "B", "C", "D"]
    opt_block = "\n".join(f"{L}. {o}" for L, o in zip(letters, options))
    prompt = f"{context_text}\n\nQuestion: {question}\n{opt_block}\n\nAnswer:"
    enc = llm_tok(prompt, return_tensors="pt").to(device)
    n_in = int(enc.input_ids.shape[1])

    _reset_peak(device)
    if device.startswith("cuda"): torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = llm_model(input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask, use_cache=True)
    if device.startswith("cuda"): torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    ttft_mb = _peak_mb(device)

    logits = out.logits[0, -1, :]
    letter_ids = [llm_tok.encode(f" {L}", add_special_tokens=False)[0]
                  for L in letters]
    option_logits = logits[letter_ids]
    pred_label = int(torch.argmax(option_logits).item()) + 1
    correct = (gold_label is not None) and (pred_label == gold_label)

    past = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    n_dec = max(total_tokens - 1, 0)
    dec_total, dec_mb = 0.0, 0.0
    if n_dec > 0:
        _reset_peak(device)
        if device.startswith("cuda"): torch.cuda.synchronize()
        td = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_dec):
                out = llm_model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        if device.startswith("cuda"): torch.cuda.synchronize()
        dec_total = time.perf_counter() - td
        dec_mb = _peak_mb(device)

    return {"format": "mcq",
            "n_input_tokens": n_in, "ttft_s": ttft, "ttft_peak_mb": ttft_mb,
            "n_decode_tokens": n_dec, "decode_total_s": dec_total,
            "decode_per_token_s": dec_total / n_dec if n_dec > 0 else 0.0,
            "decode_peak_mb": dec_mb,
            "pred_label": pred_label, "gold_label": gold_label,
            "correct": bool(correct), "f1": None, "pred_text": None,
            "option_logits": [float(x) for x in option_logits.tolist()]}


# ---------- LLM: free-form branch ----------
def measure_llm_freeform(context_text, question, gold_answer,
                         llm_tok, llm_model, device, max_new_tokens):
    """Greedy generate; score token-F1 vs gold_answer."""
    prompt = (f"{context_text}\n\n"
              f"Answer the question concisely based on the passage.\n"
              f"Question: {question}\nAnswer:")
    enc = llm_tok(prompt, return_tensors="pt").to(device)
    n_in = int(enc.input_ids.shape[1])

    _reset_peak(device)
    if device.startswith("cuda"): torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = llm_model(input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask, use_cache=True)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    if device.startswith("cuda"): torch.cuda.synchronize()
    ttft = time.perf_counter() - t0
    ttft_mb = _peak_mb(device)

    gen_ids = [int(nxt.item())]
    eos_id = llm_tok.eos_token_id
    n_dec = max(max_new_tokens - 1, 0)
    dec_total, dec_mb = 0.0, 0.0
    if n_dec > 0:
        _reset_peak(device)
        if device.startswith("cuda"): torch.cuda.synchronize()
        td = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_dec):
                out = llm_model(input_ids=nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tid = int(nxt.item())
                gen_ids.append(tid)
                if eos_id is not None and tid == eos_id:
                    break
        if device.startswith("cuda"): torch.cuda.synchronize()
        dec_total = time.perf_counter() - td
        dec_mb = _peak_mb(device)

    pred_text = llm_tok.decode(gen_ids, skip_special_tokens=True)
    pred_for_score = pred_text.split("\n")[0].strip()   # ignore trailing commentary
    f1 = token_f1(pred_for_score, gold_answer) if gold_answer else None

    return {"format": "freeform",
            "n_input_tokens": n_in, "ttft_s": ttft, "ttft_peak_mb": ttft_mb,
            "n_decode_tokens": len(gen_ids) - 1, "decode_total_s": dec_total,
            "decode_per_token_s": dec_total / max(len(gen_ids) - 1, 1),
            "decode_peak_mb": dec_mb,
            "pred_label": None, "gold_label": None, "correct": None,
            "pred_text": pred_text, "f1": f1, "option_logits": None}


# ---------- Main ----------
def build_parser():
    ap = argparse.ArgumentParser(
        description="Multi-turn SALT benchmark on QuALITY (MCQ) / LooGLE (free-form).")
    ap.add_argument("--data", default=str(DEFAULT_DATA),
                    help="Subset JSON built by download_quality.py "
                         "(QuALITY or LooGLE; format auto-detected).")
    ap.add_argument("--output", default="runs/quality_multiturn.jsonl")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--llm", default=None,
                    help="Optional causal LM for accuracy + TTFT measurement.")
    ap.add_argument("--decode", type=int, default=10,
                    help="MCQ: total tokens (1 = TTFT only). "
                         "Free-form: max new tokens (32 is sensible).")
    ap.add_argument("--compress-pct", type=float, default=0.20)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-articles", type=int, default=None)
    ap.add_argument("--max-questions-per-article", type=int, default=None)
    ap.add_argument("--no-compression", action="store_true",
                    help="Baseline: feed the raw article every turn.")
    return ap


def main():
    args = build_parser().parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.data) as f:
        articles = json.load(f)
    if args.max_articles:
        articles = articles[:args.max_articles]

    fmt = articles[0].get("format", "mcq") if articles else "mcq"
    src = articles[0].get("source_dataset", "?") if articles else "?"
    mode = "NO-COMPRESSION (baseline)" if args.no_compression else "SALT"
    print(f"Loaded {len(articles)} articles | format={fmt} | dataset={src}")
    print(f"Mode: {mode}")

    embed_tok = embed_model = None
    if not args.no_compression:
        print(f"Loading embed model: {args.embed_model}")
        embed_tok, embed_model = compressor.load_bge(args.embed_model, args.device)

    llm_tok = llm_model = None
    if args.llm:
        from transformers import AutoTokenizer
        print(f"Loading LLM: {args.llm}")
        llm_tok = AutoTokenizer.from_pretrained(args.llm)
        llm_model = AutoModelForCausalLM.from_pretrained(
            args.llm, torch_dtype=torch.bfloat16).to(args.device)
        llm_model.eval()

    # Warmup (absorb CUDA cold-start on one throwaway article/question).
    print("Warming up GPU...")
    if articles and articles[0]["questions"]:
        wa, wq = articles[0], articles[0]["questions"][0]
        if not args.no_compression:
            build_article_index(wa["article"], embed_tok, embed_model,
                                args.device, args.compress_pct)
        if args.llm:
            if fmt == "mcq":
                measure_llm_mcq("warmup " * 200, wq["question"], wq["options"],
                                None, llm_tok, llm_model, args.device,
                                total_tokens=max(args.decode, 1))
            else:
                measure_llm_freeform("warmup " * 200, wq["question"], "",
                                     llm_tok, llm_model, args.device,
                                     max_new_tokens=max(args.decode, 1))
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    print("Warmup done.\n")

    fout = open(args.output, "w", encoding="utf-8")
    rows = []

    for ai, art in enumerate(articles):
        aid = art["article_id"]
        qs = art["questions"]
        if args.max_questions_per_article:
            qs = qs[:args.max_questions_per_article]
        print(f"[{ai+1}/{len(articles)}] article {str(aid)[:30]} "
              f"({art['word_count']}w, {len(qs)} q)")

        if args.no_compression:
            raw_context = art["article"]
            raw_words = len(raw_context.split())
            index = None
            fout.write(json.dumps({
                "kind": "index", "article_id": aid, "word_count": art["word_count"],
                "n_questions": len(qs), "mode": "no_compression",
                "index_timings": {"index_total_s": 0.0, "orig_words": raw_words,
                                  "n_sentences": 0, "n_theme_keywords": 0},
                "index_mem": {"index_peak_mb": 0.0}}) + "\n")
            print(f"  baseline: raw {raw_words} words")
        else:
            index = build_article_index(art["article"], embed_tok, embed_model,
                                        args.device, args.compress_pct)
            it = index["index_timings"]
            print(f"  index: pre={it['preprocess_s']:.2f}s "
                  f"encode={it['encode_themes_s']:.2f}s "
                  f"TOTAL={it['index_total_s']:.2f}s "
                  f"peak={index['index_mem']['index_peak_mb']:.0f}MB "
                  f"n_sents={it['n_sentences']} n_themes={it['n_theme_keywords']}")
            fout.write(json.dumps({
                "kind": "index", "article_id": aid, "word_count": art["word_count"],
                "n_questions": len(qs), "index_timings": it,
                "index_mem": index["index_mem"], "mode": "salt"}) + "\n")

        for ti, q in enumerate(qs):
            qtext = q["question"]
            if args.no_compression:
                tt = {"query_embed_s": 0.0, "retrieval_s": 0.0,
                      "turn_total_s": 0.0, "kept_words": raw_words,
                      "compression_ratio": 1.0}
                tm = {"turn_peak_mb": 0.0}
                context_for_llm, n_selected, theme_cov = raw_context, 0, 1.0
            else:
                turn = run_turn(index, qtext, embed_tok, embed_model,
                                args.device, args.compress_pct)
                tt, tm = turn["turn_timings"], turn["turn_mem"]
                context_for_llm = turn["compressed_text"]
                n_selected, theme_cov = turn["n_selected"], turn["theme_coverage_pct"]

            rec = {"kind": "turn", "article_id": aid, "turn_idx": ti,
                   "question_unique_id": q["question_unique_id"],
                   "difficult": q.get("difficult", 0),
                   "gold_label": q.get("gold_label"),
                   "gold_answer": q.get("answer"),
                   "n_selected": n_selected, "theme_coverage_pct": theme_cov,
                   "turn_timings": tt, "turn_mem": tm,
                   "mode": "no_compression" if args.no_compression else "salt",
                   "compressed_text_preview": context_for_llm[:300]}
            if args.llm:
                if fmt == "mcq":
                    m = measure_llm_mcq(context_for_llm, qtext, q["options"],
                                        q.get("gold_label"), llm_tok, llm_model,
                                        args.device, total_tokens=max(args.decode, 1))
                else:
                    m = measure_llm_freeform(context_for_llm, qtext,
                                             q.get("answer", ""), llm_tok, llm_model,
                                             args.device, max_new_tokens=max(args.decode, 1))
                rec["llm"] = m
            fout.write(json.dumps(rec) + "\n")

            llm_str = ""
            if args.llm:
                if fmt == "mcq":
                    mark = "OK" if rec["llm"]["correct"] else "X"
                    llm_str = (f" | ttft={rec['llm']['ttft_s']:.3f}s "
                               f"in={rec['llm']['n_input_tokens']} "
                               f"pred={rec['llm']['pred_label']} "
                               f"gold={rec['llm']['gold_label']} {mark}")
                else:
                    llm_str = (f" | ttft={rec['llm']['ttft_s']:.3f}s "
                               f"in={rec['llm']['n_input_tokens']} "
                               f"f1={rec['llm']['f1']:.3f}")
            if args.no_compression:
                print(f"    turn {ti+1:>2}/{len(qs)}: raw={raw_words}w{llm_str}")
            else:
                print(f"    turn {ti+1:>2}/{len(qs)}: "
                      f"q_emb={tt['query_embed_s']:.3f}s retrv={tt['retrieval_s']:.3f}s "
                      f"total={tt['turn_total_s']:.3f}s kept={tt['kept_words']}w "
                      f"({tt['compression_ratio']:.1%}) cov={theme_cov:.1%}{llm_str}")

            rows.append({
                "article_id": aid, "turn_idx": ti,
                "mode": "no_compression" if args.no_compression else "salt",
                "format": fmt,
                "index_total_s": (0.0 if args.no_compression
                                  else index["index_timings"]["index_total_s"]),
                "turn_total_s": tt["turn_total_s"],
                "query_embed_s": tt["query_embed_s"],
                "retrieval_s": tt["retrieval_s"],
                "kept_words": tt["kept_words"], "theme_coverage_pct": theme_cov,
                "ttft_s": rec["llm"]["ttft_s"] if args.llm else None,
                "decode_per_token_s": rec["llm"]["decode_per_token_s"] if args.llm else None,
                "n_input_tokens": rec["llm"]["n_input_tokens"] if args.llm else None,
                "correct": rec["llm"]["correct"] if args.llm else None,
                "f1": rec["llm"]["f1"] if args.llm else None,
                "pred_label": rec["llm"]["pred_label"] if args.llm else None,
                "gold_label": q.get("gold_label"), "difficult": q.get("difficult", 0)})

    fout.close()
    write_summary_tsv(args.output, rows)
    print_headline(mode, fmt, rows, args)
    print(f"\nSaved: {args.output}")


def write_summary_tsv(output_path, rows):
    tsv = output_path.replace(".jsonl", "_summary.tsv")
    cols = ["article_id", "turn_idx", "mode", "format", "index_total_s",
            "turn_total_s", "query_embed_s", "retrieval_s", "kept_words",
            "theme_coverage_pct", "n_input_tokens", "ttft_s",
            "decode_per_token_s", "correct", "f1", "pred_label", "gold_label",
            "difficult"]
    with open(tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            def cell(k):
                v = r[k]
                if v is None:
                    return ""
                if k == "correct":
                    return str(int(v))
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)
            f.write("\t".join(cell(c) for c in cols) + "\n")
    print(f"       {tsv}")


def print_headline(mode, fmt, rows, args):
    print(f"\n=== HEADLINE ({mode}, format={fmt}) ===")
    print(f"  articles: {len({r['article_id'] for r in rows})}")
    print(f"  total turns: {len(rows)}")
    if not args.no_compression and rows:
        idx_times = sorted({r["index_total_s"] for r in rows})
        tts = [r["turn_total_s"] for r in rows]
        print(f"  index time/article: mean={np.mean(idx_times):.3f}s "
              f"median={np.median(idx_times):.3f}s")
        print(f"  turn time:          mean={np.mean(tts):.4f}s "
              f"median={np.median(tts):.4f}s p95={np.percentile(tts, 95):.4f}s")
        print(f"  ratio (index / mean turn): "
              f"{np.mean(idx_times)/max(np.mean(tts), 1e-9):.1f}x")
    if args.llm:
        ttfts = [r["ttft_s"] for r in rows if r["ttft_s"] is not None]
        ins = [r["n_input_tokens"] for r in rows if r["n_input_tokens"] is not None]
        if fmt == "mcq":
            corr = [r["correct"] for r in rows if r["correct"] is not None]
            hard = [r["correct"] for r in rows if r["correct"] is not None and r["difficult"]]
            easy = [r["correct"] for r in rows if r["correct"] is not None and not r["difficult"]]
            if corr: print(f"  accuracy (all):  {np.mean(corr):.3f} ({sum(corr)}/{len(corr)})")
            if easy: print(f"  accuracy (easy): {np.mean(easy):.3f} ({sum(easy)}/{len(easy)})")
            if hard: print(f"  accuracy (HARD): {np.mean(hard):.3f} ({sum(hard)}/{len(hard)})")
        else:
            f1s = [r["f1"] for r in rows if r["f1"] is not None]
            if f1s:
                print(f"  token-F1: mean={np.mean(f1s):.3f} median={np.median(f1s):.3f} n={len(f1s)}")
        if ins:   print(f"  n_input_tokens: mean={np.mean(ins):.0f} median={np.median(ins):.0f}")
        if ttfts: print(f"  ttft_s: mean={np.mean(ttfts):.4f} median={np.median(ttfts):.4f} "
                        f"p95={np.percentile(ttfts, 95):.4f}")


if __name__ == "__main__":
    main()
