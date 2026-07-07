# Results / Experiments

Two experiment runners that reproduce the SALT efficiency + accuracy studies.
Both build on the tracked engine (`salt.engine.compressor` + the legacy
`trie_select` selector; the mainline default is the coverage/CELF selector —
these runners keep `trie_select` so their numbers stay comparable with the
original studies) and write a per-turn/per-sample JSONL plus a
`*_summary.tsv` next to it.

Prepare the datasets first (see [`../datasets/README.md`](../datasets/README.md)):

```bash
python -m salt.datasets.download_quality
python -m salt.datasets.download_niah --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

## quality_multiturn.py

Multi-turn reading comprehension: index a long document **once**, then answer
**many** questions as cheap query-mode turns. Amortizes the expensive encode over
all turns and reports one-time index cost vs. mean per-turn cost. Format is
auto-detected — QuALITY (MCQ, `" A"/" B"/" C"/" D"` logit scoring) or LooGLE
(free-form, SQuAD token-F1). With `--llm` it also measures TTFT, decode TPOT, and
accuracy; `--no-compression` is the raw baseline.

```bash
python salt/results/quality_multiturn.py \
    --data salt/datasets/quality/quality_subset_50.json \
    --llm meta-llama/Llama-3.1-8B-Instruct \
    --compress-pct 0.20 --device cuda:0 --output runs/quality.jsonl
```

## niah_ttft.py

Needle-in-a-haystack scaling: how prefill (TTFT), decode throughput, and peak GPU
memory grow with context length, and whether SALT keeps the needle. Runs a clean
two-phase greedy loop (one prefill pass + single-token KV-cache decode steps) and
aggregates per target length (mean / median / p95 / max) with needle-retrieval
accuracy. `--raw` benchmarks the uncompressed baseline.

```bash
python salt/results/niah_ttft.py \
    --llm meta-llama/Llama-3.1-8B-Instruct \
    --lengths 32000 64000 128000 \
    --compress-pct 0.20 --device cuda:0 --output runs/niah.jsonl
```

Outputs land in `runs/` (git-ignored).
