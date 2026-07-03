# SALT

<p align="center">
  <img src="salt/assets/banner.png" width="100%">
</p>

## Salience-Aware Lexical Trie for Long-Context Compression

SALT shrinks a long document down to a fixed size before it is sent to a language
model, keeping the sentences that carry the most information. It works with any
model, produces a shorter plain-text prompt, and cuts the compute, memory, and
wait time that long inputs cost.

**The problem.** When a prompt is too long, existing compressors give each
sentence a single relevance score and keep the top-scoring ones until the budget
runs out. Under a tight budget this lets the document's main topic swallow the
whole budget, so smaller but still important points get dropped — a failure called
 *theme collapse* (in multi-hop questions, for example, it can keep
passages about the main entity yet lose the one sentence that links it to a
second).

**The solution.** SALT first maps the document's recurring themes by organizing
each sentence's keywords into a trie, a small keyword tree ordered by how often
those keywords recur, then spreads the budget across those theme branches
before choosing sentences, so minor themes keep their share instead of being
crowded out. Because the theme map is built once, it can be reused across the
turns of a conversation without re-reading the document.


## 📑 Table of contents

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Datasets](#-datasets)
- [Quick start](#-quick-start)
- [Usage examples](#-usage-examples)
- [Results](#-results)
- [License](#-license)

## 🧩 Architecture

SALT runs in two phases. **Indexing** reads the document once and turns it into
a keyword trie — a small, reusable map of its recurring themes. **Selection**
traverses that trie under a token budget and returns a sentence subset in
original document order, either unconditionally (*summary mode*) or biased
toward a query (*query mode*).

```text
 INDEXING — run once per document, reused across turns and budgets
 ──────────────────────────────────────────────────────────────────
 document
    │  sentence split + junk filter
    ▼
 per-sentence keywords     BGE-small [CLS] attention ranks each sentence's
    │                      words; knee detection keeps the top-k per sentence
    ▼
 salience set              sentence frequency SF(w) = #sentences keeping w;
    │                      top-quantile keywords = the recurring themes
    ▼
 keyword trie              each sentence's keywords, sorted by SF, form a
                           root-to-leaf path; leaves store sentence ids

 SELECTION — per budget, with or without a query
 ──────────────────────────────────────────────────────────────────
 (query mode) multi-anchor activation: query keywords activate trie
    │         nodes at any depth; top anchors + neighbors admitted first
    ▼
 branch allocation         remaining budget spread across depth-1 theme
    │                      branches (floor + share ∝ uncovered mass^α)
    ▼
 in-branch selection       greedy pick by marginal theme-coverage gain,
    │                      moderated by length and position priors
    ▼
 global fill               unused budget goes to best remaining sentences
    ▼
 compressed prompt         selected sentences, original order, ≤ budget
```

Where each stage lives:

| Stage | Code |
|---|---|
| Sentence split + junk filter | `salt/engine/embedder.py`, `salt/engine/sentence_filter.py` |
| Keyword extraction, BGE embedding, theme profiling | `salt/engine/trie_core.py` |
| Trie-guided selection (`trie_select`) | `salt/engine/retrieval.py` |
| Prose pipeline runner (load → pipeline → JSONL out) | `salt/engine/compressor.py` |
| Few-shot bypass (`trec`, `triviaqa`, `samsum`) | `salt/engine/fewshot.py` |
| `--synthetic` paragraph-unit adapter | `salt/engine/dataset_modes.py` |
| CLI entry points | `compress.py`, `eval.py` |

## 📦 Installation

Requires Python 3.10 and a CUDA GPU (CPU works for compression, just slower).
If you are in cluster, use
```bash
module load cuda/version
```
**1. Clone the repository**

```bash
git clone https://github.com/oteomamo/SALT.git
cd SALT
```

**2. Create the environment**

With conda:

```bash
conda env create -f environment.yml
conda activate salt
```

Or with venv:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Install SALT in editable mode**

```bash
pip install -e .
```

**4. Authenticate with Hugging Face** — the eval model
(`meta-llama/Llama-3.1-8B-Instruct`) is gated:

```bash
hf auth login
```

Or skip the CLI and export the token directly: `export HF_TOKEN=hf_...`.

**5. (Optional) vLLM eval backend.** `eval.py` defaults to vLLM. Install it into
the same `salt` env — there is only one environment:

```bash
pip install vllm==0.11.0
```

Skip this and run `eval.py --backend hf` for a portable run that needs no vLLM.

> `bash scripts/setup_env.sh` does steps 2–3 in one shot (add `WITH_VLLM=1` to
> include vLLM).

## 📚 Datasets

SALT evaluates on the 16 English tasks of
[LongBench](https://huggingface.co/datasets/THUDM/LongBench). If the data is not
already present, fetch and normalize it with:

```bash
python salt/datasets/download_datasets.py
```

Existing files are skipped (`--force` rebuilds, `--list` shows status). The
canonical JSONL schema and options are documented in
[`salt/datasets/README.md`](salt/datasets/README.md).

## 🚀 Quick start

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
(`--synthetic` for `passage_count`/`passage_retrieval_en`, few-shot bypass for
`trec`/`triviaqa`/`samsum`, standard prose for the rest), then scores the
results. Knobs via env vars: `BUDGET` (default `0.20`), `GPU` (default `1`),
`DATA_DIR`, `OUT_DIR`, `EVAL_BACKEND` (`vllm`|`hf`), `MAX_INPUT_LEN`. Outputs
land in `runs/run_<timestamp>/`, scores in `eval_all.json`.

## 💬 Usage examples

Compress a document to 20% of its token budget:

```bash
python compress.py \
  --data salt/datasets/longbench/data/hotpotqa.jsonl \
  --output out/hotpotqa.jsonl \
  --device cuda \
  --token-budget-pct 0.20 \
  --model BAAI/bge-small-en-v1.5 \
  --verbose
```

When a sample carries a query, its keywords and proper nouns are matched against
the document as surprisal-weighted lexical terms (rare, discriminative terms get
more mass) alongside a semantic term scored by BGE query–sentence cosine. Both
re-weight the keyword trie, and `trie_select` (query anchoring + neighbor
expansion, frequency-weighted branch budgeting, thematic fill) picks the
sentences that fit the budget.

`--synthetic` treats `Paragraph N:` units as the selection records so every
paragraph label survives compression — full text for query-relevant paragraphs,
deterministic prefixes when there is no query (e.g. passage_count).

### Evaluate
(change prefer GPU number 0->#)

vLLM by default (install it per step 5); pass `--backend hf` for a portable
HF-transformers run that needs no vLLM install:

```bash
python eval.py \
  --data-dir out \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu 0 --max-input-len 14000
```

## 🔬 Results

LongBench accuracy with Llama-3.1-8B-Instruct at a 20% compression budget:

| Method | Single-Doc QA | Multi-Doc QA | Summarization | Few-Shot | Synthetic | Code | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-context | 43.58 | 44.65 | 29.22 | 69.48 | 54.21 | 60.01 | 50.19 |
| **KV cache methods (20%)** | | | | | | | |
| SnapKV | 43.29 | 43.92 | 26.59 | 67.95 | 53.75 | 58.74 | 49.04 |
| FastKV | 43.31 | 44.10 | 26.61 | 68.36 | 53.72 | 59.26 | 49.23 |
| SentenceKV | 39.25 | 43.82 | 28.18 | 69.26 | 53.24 | 47.33 | 46.85 |
| DuoAttention | 34.71 | 36.67 | 24.30 | 58.02 | 50.81 | 54.33 | 43.14 |
| **Preprocessing methods (20%)** | | | | | | | |
| EXIT | 31.50 | 23.77 | 24.94 | 58.89 | 13.5 | 37.45 | 31.68 |
| RECOMP | 35.88 | 41.09 | 24.16 | 52.06 | 50.77 | 37.37 | 40.22 |
| CPC | 38.91 | 39.42 | 24.97 | 51.67 | 51.50 | 21.84 | 38.05 |
| Sentinel | 39.85 | 41.66 | 26.15 | 38.78 | 51.55 | 38.83 | 39.47 |
| **SALT** | **40.05** | **41.42** | **26.95** | **62.21** | **53.37** | **37.06** | **43.51** |

Task → category mapping: **Single-Doc QA** `narrativeqa`, `qasper`,
`multifieldqa_en` · **Multi-Doc QA** `hotpotqa`, `2wikimqa`, `musique` ·
**Summarization** `gov_report`, `qmsum`, `multi_news` · **Few-Shot** `trec`,
`triviaqa`, `samsum` · **Synthetic** `passage_count`, `passage_retrieval_en` ·
**Code** `lcc`, `repobench-p`.

## 📄 License

To be announced.
