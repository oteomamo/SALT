# 💬 Usage

## Compress a dataset

Compress a document collection with the `salt` command (installed with the
package):

```bash
salt \
  --data salt/datasets/longbench/data/hotpotqa.jsonl \
  --output out/hotpotqa.jsonl \
  --verbose
```

Defaults: 20% token budget (`--token-budget-pct`), GPU compression
(`--device cpu` runs without one), `BAAI/bge-small-en-v1.5` as the
compression model (`--model`).

## Compress a single file

`--doc` compresses one `.pdf`, `.txt`, or `.md` instead of a dataset. The file
goes through the same PDF pipeline the chatbot uses (tables and pseudocode
grouped under their captions, sentences re-joined across figure interruptions,
headings and equations preserved), then through the same selector. `--query`
biases the selection:

```bash
salt --doc paper.pdf --query "average accuracy on LongBench?" --output out/paper.jsonl
```

When a sample carries a query, its keywords and proper nouns are matched against
the document as surprisal-weighted lexical terms (rare, discriminative terms get
more mass) alongside a semantic term scored by BGE query-sentence cosine. Both
re-weight the keyword trie, and the coverage selector picks the sentences that
best cover it under the budget.

## Special dataset modes

`--synthetic` treats `Paragraph N:` units as the selection records so every
paragraph label survives compression. `--code` treats physical lines as records
with identifier keywords and file/function structure, keeping the completion-site
tail. Both modes are chosen automatically by `scripts/run_datasets.sh`.

## Batch runs

Compress all LongBench tasks present at the data path and evaluate the outputs
in one command:

```bash
bash scripts/run_datasets.sh
```

Smoke test (5 samples per task, compression only):

```bash
MAX_SAMPLES=5 RUN_EVAL=0 bash scripts/run_datasets.sh
```

The script routes each dataset to the right mode automatically
(`--synthetic` for `passage_count`/`passage_retrieval_en`, `--code` for
`lcc`/`repobench-p`, few-shot bypass for `trec`/`triviaqa`/`samsum`, standard
prose for the rest), then scores the results. Knobs via env vars: `BUDGET`
(default `0.20`), `GPU` (default `0`), `DATA_DIR`, `OUT_DIR`, `EVAL_BACKEND`
(`vllm`|`hf`), `MAX_INPUT_LEN`. Outputs land in `runs/run_<timestamp>/`,
scores in `eval_all.json`.

## Evaluate

vLLM by default (install it per
[Installation step 5](installation.md)). Pass `--backend hf` for a portable
HF-transformers run that needs no vLLM install:

```bash
python eval.py \
  --data-dir out \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu 0 --max-input-len 14000
```
