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
Needle-in-a-haystack TTFT / memory benchmark for SALT.

Measures how prefill latency (TTFT), decode throughput, and peak GPU memory
scale with context length, and whether SALT compression preserves the answer.
Each sample is a long PG-19 passage with a "magic number" needle inserted at a
random depth plus the question that asks for it (built by
`salt/datasets/download_niah.py`).

Two conditions:
  --raw (baseline)   feed the full-length prompt straight to the LLM.
  SALT (default)     compress the haystack with the needle-question as the query
                     (`trie_select`), then feed compressed_context + question.

Per sample it runs a clean two-phase greedy loop — one prefill forward pass, then
`max_new_tokens-1` single-token KV-cache steps — and reports:
  prefill_s, decode_s, latency_s, tpot_s, peak_mem_mb, ctx_tokens, needle_correct.
Results are aggregated per target length (mean / median / p95 / max) alongside
needle-retrieval accuracy and the achieved compression ratio.

Usage:
    python salt/results/niah_ttft.py \
        --llm meta-llama/Llama-3.1-8B-Instruct \
        --lengths 32000 64000 128000 --compress-pct 0.20 \
        --device cuda:0 --output runs/niah.jsonl
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from salt.engine import compressor
from salt.engine.compressor import prep_prose_sentences, build_prose_sent_data, enrich_query
from salt.engine.retrieval import trie_select
from salt.datasets.download_niah import QUESTION_TPL

DEFAULT_DATA_DIR = (Path(__file__).resolve().parent.parent / "datasets" / "niah")

SELECT_KW = dict(
    min_keywords=2, pass1_budget_pct=0.5, theme_prec_min=0.0,
    redundancy_thresh=1.0, intro_pct=0.03, neighbor_window=1,
    query_budget_uncapped=False, query_budget_pct=0.75, branch_floor=0,
)


# ---------- GPU helpers ----------
def gpu_reset_peak(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)


def gpu_peak_mb(device):
    # NB: pass `device` — max_memory_allocated() defaults to the current device,
    # which is not necessarily the one the model runs on (e.g. cuda:1).
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


def _bench_stats(values):
    """mean / median / p95 / max for a list of floats (skip NaNs)."""
    clean = [v for v in values if v == v]
    if not clean:
        return {"mean": float("nan"), "median": float("nan"),
                "p95": float("nan"), "max": float("nan")}
    s = sorted(clean)
    p95_idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return {"mean": statistics.fmean(s), "median": statistics.median(s),
            "p95": s[p95_idx], "max": s[-1]}


# ---------- Two-phase prefill + decode ----------
def prefill_decode_bench(model, inputs, max_new_tokens, device):
    """Phase 1 (prefill): one forward pass on the full prompt -> first token.
    Phase 2 (decode): max_new_tokens-1 single-token steps reusing the KV cache.
    No generate(), no streamer, no chat-template wrapping."""
    input_ids = inputs["input_ids"]
    context_length = input_ids.shape[-1]
    is_cuda = str(device).startswith("cuda")

    gpu_reset_peak(device)
    if is_cuda: torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
    if is_cuda: torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    past_kv = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    generated = [next_tok]

    n_decode_steps = max(max_new_tokens - 1, 0)
    if is_cuda: torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_decode_steps):
            out = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated.append(next_tok)
    if is_cuda: torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0

    new_ids = torch.cat(generated, dim=-1)
    return {"new_ids": new_ids,
            "prefill_s": prefill_s, "decode_s": decode_s,
            "latency_s": prefill_s + decode_s,
            "tpot_s": decode_s / max(n_decode_steps, 1),
            "n_new": max_new_tokens, "ctx_tokens": context_length,
            "peak_mem_mb": gpu_peak_mb(device)}


# ---------- SALT compression of one needle prompt ----------
def salt_compress(rec, embed_tok, embed_model, device, compress_pct):
    """Split the needle-question off the prompt, compress the haystack with that
    question as the query, and re-attach it. Returns (final_prompt, info)."""
    prompt = rec["prompt"]
    city = rec.get("needle_city")
    question = QUESTION_TPL.format(city=city) if city else ""
    context = prompt[:-len(question)] if question and prompt.endswith(question) else prompt

    t = time.perf_counter()
    sentences, _all, orig_words, _wb, _is_code = prep_prose_sentences(
        context, dataset_name="", tokenizer=embed_tok, token_budget_pct=compress_pct)
    if not sentences:
        return prompt, {"compress_s": time.perf_counter() - t,
                        "orig_words": orig_words, "kept_words": orig_words}
    sent_data, kw_df, theme_keywords = build_prose_sent_data(
        sentences, embed_tok, embed_model, device,
        max_keywords_ratio=0.4, theme_percentile=0.9)

    word_budget = int(orig_words * compress_pct)
    clean_words = sum(sd["n_words"] for sd in sent_data)
    if clean_words <= word_budget:
        compressed = " ".join(sd["text"] for sd in sent_data)
    else:
        q_kws, q_emb, q_pns, qtype = enrich_query(
            question, embed_tok, embed_model, device,
            extended_stopwords=True, bge_prefix=True)
        selected, _stats = trie_select(
            sent_data, dict(kw_df), theme_keywords, word_budget,
            query_keywords=q_kws, query_embedding=q_emb,
            query_proper_nouns=q_pns, qtype=qtype, **SELECT_KW)
        compressed = " ".join(sr.text for sr in selected)

    final = compressed + question
    return final, {"compress_s": time.perf_counter() - t,
                   "orig_words": orig_words, "kept_words": len(compressed.split())}


# ---------- IO ----------
def load_samples(data, data_dir, prefix, lengths, max_samples):
    if data:
        files = [Path(data)]
    else:
        files = sorted(Path(data_dir).glob(f"{prefix}_*.jsonl"))
    if not files:
        raise SystemExit(
            f"No samples found (looked for {prefix}_*.jsonl in {data_dir}). "
            f"Build them with: python -m salt.datasets.download_niah --tokenizer <LLM>")
    samples = []
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    if lengths:
        keep = set(lengths)
        samples = [s for s in samples if s.get("target_length") in keep]
    if max_samples:
        samples = samples[:max_samples]
    return samples


def main():
    ap = argparse.ArgumentParser(
        description="Needle-in-a-haystack TTFT / memory benchmark for SALT.")
    ap.add_argument("--llm", required=True, help="Causal LM to benchmark.")
    ap.add_argument("--data", default=None,
                    help="Single JSONL of samples (overrides --data-dir).")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help=f"Dir of longctx_*.jsonl files (default: {DEFAULT_DATA_DIR}).")
    ap.add_argument("--prefix", default="longctx")
    ap.add_argument("--lengths", nargs="+", type=int, default=None,
                    help="Filter to these target token lengths (default: all found).")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--compress-pct", type=float, default=0.20)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--raw", action="store_true",
                    help="Baseline: feed the uncompressed full-length prompt.")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--output", default="runs/niah_ttft.jsonl")
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    samples = load_samples(args.data, args.data_dir, args.prefix,
                           args.lengths, args.max_samples)
    mode = "RAW (baseline)" if args.raw else "SALT"
    lens = sorted({s.get("target_length") for s in samples})
    print(f"Loaded {len(samples)} samples | lengths={lens} | mode={mode}")

    embed_tok = embed_model = None
    if not args.raw:
        print(f"Loading embed model: {args.embed_model}")
        embed_tok, embed_model = compressor.load_bge(args.embed_model, args.device)

    print(f"Loading LLM: {args.llm}")
    llm_tok = AutoTokenizer.from_pretrained(args.llm)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.llm, torch_dtype=torch.bfloat16).to(args.device)
    llm_model.eval()

    # Warmup: absorb CUDA init / autotune on a short prompt.
    print("Warming up GPU...")
    warm = llm_tok("warmup " * 64, return_tensors="pt").to(args.device)
    prefill_decode_bench(llm_model, warm, max_new_tokens=4, device=args.device)
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    print("Warmup done.\n")

    fout = open(args.output, "w", encoding="utf-8")
    rows = []
    for si, rec in enumerate(samples):
        L = rec.get("target_length")
        needle = rec.get("needle_number")

        if args.raw:
            final_prompt, cinfo = rec["prompt"], {"compress_s": 0.0,
                                                  "orig_words": len(rec["prompt"].split()),
                                                  "kept_words": len(rec["prompt"].split())}
        else:
            final_prompt, cinfo = salt_compress(
                rec, embed_tok, embed_model, args.device, args.compress_pct)

        enc = llm_tok(final_prompt, return_tensors="pt").to(args.device)
        bench = prefill_decode_bench(llm_model, enc, args.max_new_tokens, args.device)
        pred = llm_tok.decode(bench["new_ids"][0], skip_special_tokens=True)
        correct = (str(needle) in pred) if needle is not None else None
        ratio = cinfo["kept_words"] / max(cinfo["orig_words"], 1)

        row = {"kind": "sample", "sample_idx": si, "target_length": L,
               "sample_id": rec.get("sample_id"), "mode": "raw" if args.raw else "salt",
               "needle_number": needle, "needle_depth": rec.get("needle_depth"),
               "needle_correct": correct, "pred_preview": pred[:120],
               "compress_s": cinfo["compress_s"], "compression_ratio": ratio,
               "ctx_tokens": bench["ctx_tokens"],
               "prefill_s": bench["prefill_s"], "decode_s": bench["decode_s"],
               "latency_s": bench["latency_s"], "tpot_s": bench["tpot_s"],
               "peak_mem_mb": bench["peak_mem_mb"]}
        fout.write(json.dumps(row) + "\n")
        rows.append(row)

        mark = "" if correct is None else (" needle=OK" if correct else " needle=X")
        print(f"  [{si+1}/{len(samples)}] L={L} ctx={bench['ctx_tokens']} "
              f"({ratio:.1%}) | prefill={bench['prefill_s']*1000:.0f}ms "
              f"decode={bench['decode_s']*1000:.0f}ms peak={bench['peak_mem_mb']:.0f}MB{mark}")

    fout.close()
    summarize(rows, args.output)
    print(f"\nSaved: {args.output}")


def summarize(rows, output_path):
    by_len = {}
    for r in rows:
        by_len.setdefault(r["target_length"], []).append(r)

    print(f"\n=== HEADLINE (per target length) ===")
    header = (f"{'length':>8} {'n':>4} {'ctx_tok':>8} {'ratio':>6} "
              f"{'prefill_ms':>11} {'decode_ms':>10} {'peak_mb':>9} {'needle_acc':>10}")
    print(header)
    tsv = output_path.replace(".jsonl", "_summary.tsv")
    with open(tsv, "w") as f:
        f.write("length\tn\tctx_tokens_mean\tcompression_ratio_mean\t"
                "prefill_ms_mean\tprefill_ms_p95\tdecode_ms_mean\t"
                "peak_mb_mean\tpeak_mb_max\tneedle_acc\n")
        for L in sorted(by_len):
            g = by_len[L]
            pre = _bench_stats([r["prefill_s"] for r in g])
            dec = _bench_stats([r["decode_s"] for r in g])
            peak = _bench_stats([r["peak_mem_mb"] for r in g])
            ctx = statistics.fmean([r["ctx_tokens"] for r in g])
            ratio = statistics.fmean([r["compression_ratio"] for r in g])
            corr = [r["needle_correct"] for r in g if r["needle_correct"] is not None]
            acc = statistics.fmean([float(c) for c in corr]) if corr else float("nan")
            acc_str = "  n/a" if acc != acc else f"{acc:.3f}"
            print(f"{L:>8} {len(g):>4} {ctx:>8.0f} {ratio:>6.1%} "
                  f"{pre['mean']*1000:>11.1f} {dec['mean']*1000:>10.1f} "
                  f"{peak['mean']:>9.0f} {acc_str:>10}")
            f.write(f"{L}\t{len(g)}\t{ctx:.0f}\t{ratio:.4f}\t"
                    f"{pre['mean']*1000:.1f}\t{pre['p95']*1000:.1f}\t"
                    f"{dec['mean']*1000:.1f}\t{peak['mean']:.1f}\t{peak['max']:.1f}\t"
                    f"{'' if acc != acc else f'{acc:.4f}'}\n")
    print(f"       {tsv}")


if __name__ == "__main__":
    main()
