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
whole budget, so smaller but still important points get dropped - a failure called
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
- [Chatbot mode](#-chatbot-mode)
- [Results](#-results)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

## 🧩 Architecture

Two phases. **Indexing** reads the document once and builds a keyword trie - a
reusable map of its recurring themes. **Selection** then picks a sentence subset
under a token budget, returned in original document order, query-biased or not.

```text
 INDEXING  ── once per document, reused across turns and budgets
   document → split + junk filter
           → per-sentence keywords   (BGE-small [CLS] attention + knee cutoff)
           → theme salience          (SF = #sentences keeping a word; top quantile)
           → keyword trie            (each sentence's themes, SF-ordered, form a
                                      root-to-leaf path; leaves hold sentence ids)

 SELECTION ── per budget, with or without a query
   maximize theme coverage with CELF lazy-greedy: a pick's value shrinks as its
   theme branches fill, so budget spreads across themes instead of collapsing
   onto the dominant one. A query re-weights the trie (lexical + BGE-semantic)
   without rebuilding it. → compressed prompt, original order, ≤ budget
```

The whole system as a blueprint - this map is kept current as SALT grows,
so it is the fastest way to find where a change belongs:

```text
┌──────────────────────────────────────────────────────────────────────────┐
│                                   SALT                                   │
│                                                                          │
│ ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐   │
│ │      Indexing      │  │    Keyword Trie    │  │     Selection      │   │
│ │                    │  │                    │  │                    │   │
│ │ BGE-small encoder  │  │ SF-ordered paths   │  │ coverage (CELF)    │   │
│ │ attention keywords │  │ theme branches     │  │ branch discounting │   │
│ │ knee cutoff        │  │ §file: doc branches│  │ multi-anchor query │   │
│ │ junk filter        │  │ rebuilt cheaply    │  │ ≤ word budget      │   │
│ └────────────────────┘  └────────────────────┘  └────────────────────┘   │
│                                                                          │
│ ┌────────────────────┐  ┌────────────────────┐  ┌────────────────────┐   │
│ │    Session Trie    │  │  Prompt Assembly   │  │    Chat Runner     │   │
│ │                    │  │                    │  │                    │   │
│ │ per-conversation   │  │ stable prefix first│  │ HF streaming       │   │
│ │ lives in DRAM      │  │ append-only tail   │  │ vLLM + APC (opt-in)│   │
│ │ grows every turn   │  │ memory + question  │  │ model registry     │   │
│ │ cross-turn coverage│  │ instructions.md    │  │ GPU-pinned models  │   │
│ │ + half-life decay  │  │                    │  │                    │   │
│ │ + near-dup gate    │  │                    │  │                    │   │
│ └────────────────────┘  └────────────────────┘  └────────────────────┘   │
│                                                                          │
│ ┌────────────────────────────────────────────────────────────────────┐   │
│ │              Document ingest (salt@ files, salt --doc)             │   │
│ │ pypdf extract · furniture scrub · paragraphs rejoined across floats│   │
│ │ tables + pseudocode grouped under captions · footnotes isolated    │   │
│ │ headings, panel labels and equations kept · reference list dropped │   │
│ └────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│ ┌────────────────────────────────────────────────────────────────────┐   │
│ │            Trie shape - the root binds the conversation            │   │
│ │                               ● root - the conversation bind       │   │
│ │         ┌─────────────────────┼─────────────────────┐              │   │
│ │  §file:paper.pdf       §file:notes.txt        conversation         │   │
│ │         │                     │              ┌──────┴──────┐       │   │
│ │   keyword paths         keyword paths     theme A       theme B    │   │
│ │         │                     │              │             │       │   │
│ │     sentences             sentences      sentences     sentences   │   │
│ │                                                                    │   │
│ │ each turn: ≤ budget spread across branches (CELF discounting)      │   │
│ │ the untrie - the verbatim tail - sits OUTSIDE the trie, as the     │   │
│ │ prompt's stable recent-history window                              │   │
│ └────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│ ┌────────────────────────────────────────────────────────────────────┐   │
│ │                  Prompt layout (KV-cache shaped)                   │   │
│ │ [system: instructions · file inventory · attach@ full documents]   │   │
│ │ → [tail: recent exchanges - append-only, block-wise compaction]    │   │
│ │ → [newest user message: SALT memory (≈20% selection) + question]   │   │
│ │ stable prefix = reusable KV ──── fresh suffix = per-turn prefill   │   │
│ └────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│ ┌────────────────────────────────────────────────────────────────────┐   │
│ │                    kvtrace - per-turn KV ledger                    │   │
│ │ read (reused) / write (fresh) / output · events.jsonl + tokens.npy │   │
│ │ usage keys: input (write) · input_cached_tokens (read) · output    │   │
│ │ apc fields: engine-measured prefix-cache reuse (--backend vllm)    │   │
│ └────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│ ┌────────────────────────────────────────────────────────────────────┐   │
│ │                            Entry points                            │   │
│ │ salt (one-shot: --data / --doc) · saltChat (chat REPL) · eval.py   │   │
│ │ salt@ trie attach · attach@ full text · /doc /model /budget /stats │   │
│ └────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

Where each stage lives:

| Stage | Code |
|---|---|
| Split + junk filter | `salt/engine/embedder.py`, `salt/engine/sentence_filter.py` |
| Keywords, BGE embedding, theme profiling | `salt/engine/trie_core.py` |
| Coverage selection (default) | `salt/engine/celf.py` |
| Prose pipeline runner | `salt/engine/compressor.py` |
| Few-shot bypass (`trec`, `triviaqa`, `samsum`) | `salt/engine/fewshot.py` |
| Dataset adapters (`--synthetic`, `--code`) | `salt/engine/dataset_modes.py` |
| Multi-turn session store | `salt/engine/session_trie.py` |
| Document ingest (PDF/text cleanup, `salt@`, `--doc`) | `salt/chat/pdfio.py` |
| Chat REPL + model registry | `salt/chat/`, `salt/models/` |
| CLI entry points | `salt` (`salt/compress.py`), `eval.py`, `saltChat` |

## 📦 Installation

Requires Python 3.10 and a CUDA GPU (CPU works for compression, just slower).

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

This also installs the two console commands: `salt` (one-shot compression,
see [Usage examples](#-usage-examples)) and `saltChat` (interactive chat, see
[Chatbot mode](#-chatbot-mode)).

**4. Authenticate with Hugging Face** - the eval model
(`meta-llama/Llama-3.1-8B-Instruct`) is gated:

```bash
hf auth login
```

Or skip the CLI and export the token directly: `export HF_TOKEN=hf_...`.

**5. (Optional) vLLM backend.** `eval.py` defaults to vLLM, and
`saltChat --backend vllm` uses it for prefix caching. Install it into the
same `salt` env - there is only one environment:

```bash
pip install vllm==0.11.0
```

Skip this and run `eval.py --backend hf` for a portable run that needs no
vLLM; `saltChat` already defaults to its HF backend.

> `bash scripts/setup_env.sh` does steps 2–3 in one shot (add `WITH_VLLM=1` to
> include vLLM).

## 📚 Datasets

SALT evaluates on the 16 English tasks of
[LongBench](https://huggingface.co/datasets/THUDM/LongBench). If the data is not
already present, fetch and normalize it with:

```bash
python salt/datasets/download_longbench.py
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
(`--synthetic` for `passage_count`/`passage_retrieval_en`, `--code` for
`lcc`/`repobench-p`, few-shot bypass for `trec`/`triviaqa`/`samsum`, standard
prose for the rest), then scores the results. Knobs via env vars: `BUDGET`
(default `0.20`), `GPU` (default `0`),
`DATA_DIR`, `OUT_DIR`, `EVAL_BACKEND` (`vllm`|`hf`), `MAX_INPUT_LEN`. Outputs
land in `runs/run_<timestamp>/`, scores in `eval_all.json`.

## 💬 Usage examples

Compress a document with the `salt` command (installed with the package):

```bash
salt \
  --data salt/datasets/longbench/data/hotpotqa.jsonl \
  --output out/hotpotqa.jsonl \
  --verbose
```

Defaults: 20% token budget (`--token-budget-pct`), GPU compression
(`--device cpu` runs without one), `BAAI/bge-small-en-v1.5` as the
compression model (`--model`).

Compress a single file instead of a dataset with `--doc` - a `.pdf`,
`.txt`, or `.md` goes through the same PDF pipeline the chatbot uses
(tables and pseudocode grouped under their captions, sentences re-joined
across figure interruptions, headings and equations preserved), then
through the same selector; `--query` biases the selection:

```bash
salt --doc paper.pdf --query "average accuracy on LongBench?" --output out/paper.jsonl
```

When a sample carries a query, its keywords and proper nouns are matched against
the document as surprisal-weighted lexical terms (rare, discriminative terms get
more mass) alongside a semantic term scored by BGE query-sentence cosine. Both
re-weight the keyword trie, and the coverage selector picks the sentences that
best cover it under the budget.

`--synthetic` treats `Paragraph N:` units as the selection records so every
paragraph label survives compression; `--code` treats physical lines as records
with identifier keywords and file/function structure, keeping the completion-site
tail. Both modes are chosen automatically by `scripts/run_datasets.sh`.

### Evaluate

vLLM by default (install it per step 5); pass `--backend hf` for a portable
HF-transformers run that needs no vLLM install:

```bash
python eval.py \
  --data-dir out \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu 0 --max-input-len 14000
```

## 🤖 Chatbot mode

`saltChat` is an interactive chat REPL where SALT is the conversation memory:
one persistent trie per conversation grows with every exchange (and any
attached documents), and each turn it compresses the accumulated history into
a query-biased context block under the token budget. The BGE encoder and the
chat model stay resident on the GPU for the whole session; the trie lives in
process memory and autosaves to disk, so any conversation can be resumed
later by its id.

Register a model by its HuggingFace name - weights are symlinked from your HF
cache into `salt/models/`, never copied (see
[`salt/models/README.md`](salt/models/README.md)):

```bash
saltChat --add Qwen/Qwen2.5-0.5B-Instruct --alias qwen05
```

Chat, optionally seeding the trie with a document:

```bash
saltChat --model qwen05 --conversation-id demo1 --doc report.txt
```

The default backend runs the model through HF transformers and works
anywhere. `--backend vllm` serves the same registered weights through an
in-process vLLM engine with automatic prefix caching, so the stable prompt
head and tail are reused from the GPU KV cache instead of being re-prefilled
every turn (install vLLM first - step 5 above):

```bash
saltChat --model qwen05 --backend vllm --gpu 1
```

`--gpu-mem-util` caps the engine's share of GPU memory (default `0.85`,
leaving room for the BGE encoder on the same GPU); `--max-model-len` caps
the context window when the model's full window would not fit in the KV
cache. `/model` switching works on both backends.

Inside the REPL:

| Command | Effect |
|---|---|
| `salt@` | list attachable files staged in `salt/files/` |
| `salt@<file>` | attach a `.pdf`/`.txt`/`.md`/`.rst`: whole text, own trie branch |
| `attach@<file>` | attach in full: uncompressed text rides in every prompt |
| `/model` | list registered models; `/model <name>` switches (session kept) |
| `/add <hf_id> [alias]` | download and register another model |
| `/doc <path>` | ingest a text or PDF file into the trie |
| `/budget <pct>` | set the memory budget (`0.3` or `30`) |
| `/stats` | session, attachments, compression, and GPU-memory stats |
| `/new [id]`, `/clear` | start another conversation, wipe this one |
| `/exit` | leave; the session is saved and resumable by id |

Every turn is recorded in a per-conversation KV ledger under
`salt/chat/sessions/<id>/kvtrace/`: an append-only `events.jsonl` whose usage
keys follow the cached-token convention (`input` = freshly prefilled
sentences, `input_cached_tokens` = context re-selected from the previous
turn, `output` = generated tokens) plus a per-token `tokens.npy` matrix;
`/stats` shows the running totals. On `--backend vllm` every event also
records the engine's measured prefix-cache reuse (`apc_cached_tokens` /
`apc_prompt_tokens`) - the positional ground truth next to the ledger's
content-overlap split.

Attached PDFs are read whole (images ignored) and cleaned into proper
sentences before they reach the trie: repeated headers/footers, page numbers,
and ACL/NeurIPS-style margin line numbers are stripped, ligatures and
hyphenation repaired, wrapped lines reflowed into paragraphs, and reference
lists filtered - sentence boundaries never break inside citations.
Paragraphs interrupted mid-sentence by a figure caption, table, or footnote
are re-joined across the float instead of being severed. Tables and
algorithm pseudocode are kept, grouped under their captions as
`|`-separated rows so numbers stay readable against their column names;
section headings, panel labels, and equations survive ingestion (big
operators pypdf flattens to Latin look-alikes are restored where
unambiguous), and a sentence mentioning a URL keeps its prose with the
link as `<url>`. A `salt@` file becomes **its own branch** of the session trie,
hanging off the conversation's root - so multiple attachments never crowd
each other out, and the per-turn budget (default 20%) spreads across files
and conversation themes. An `attach@` file skips the trie entirely: its full
text rides uncompressed in every prompt. TAB completes `/commands`,
`salt@<file>`, and `attach@<file>` names (see
[`salt/files/README.md`](salt/files/README.md)).

The chat model is told what it is looking at: the system prompt carries a
reading guide from [`salt/chat/instructions.md`](salt/chat/instructions.md)
(edit it to tune the wording - it is re-read every turn, even mid-session)
plus an inventory of every attached file, and the compressed memory arrives
at the top of the newest user message, grouped by origin - `[from attached
file 'paper.pdf' - 42 of 358 indexed sentences]` versus `[from the earlier
conversation]` - so answers can cite their source file and the model knows
the excerpts are partial.

The prompt is deliberately **KV-cache shaped**: everything stable (system
prompt, `attach@` full texts, the verbatim tail) comes first, and the only
per-turn content - the SALT memory selection and the question - comes last.
The tail grows append-only and compacts **in blocks** (back to `--tail`
exchanges once it hits twice that) instead of rolling every turn, so the
prompt prefix stays byte-identical between compactions; nothing is lost,
since every sentence already entered the trie the moment it was spoken.
With the default HF backend each turn still prefills the whole prompt;
`--backend vllm` cashes the layout in. The engine's automatic prefix caching
serves the stable prefix straight from the GPU KV cache and prefills only
the fresh suffix - in practice ~95% of prompt tokens on quiet turns. Each
turn's real hit count is recorded in the kvtrace ledger (`apc_cached_tokens`,
next to the selection-overlap split, which measures a different thing), and
`/stats` prints it live.

Cross-turn memory also has a **forgetting** knob. By default a theme that
has been surfaced stays discounted for the whole session, so the memory
block keeps favoring new material - even when the conversation circles back.
`--coverage-half-life 8` makes that suppression fade instead, halving every
8 turns a theme goes unmentioned, so a topic you return to after a long
detour resurfaces gradually rather than staying buried. Attached files are
exempt (selection keeps advancing through a document) unless
`--coverage-decay-docs` is set; `/stats` reports the decay state.

And a **pivot** knob. Every query turn, the question's similarity to the
recent conversation is measured against the session's own running baseline
(`/stats` shows the reading, so the margin can be tuned before trusting
it). With `--shift-damping 0.25`, a turn that drops `--shift-margin` below
that baseline - a topic shift - hands the selector a seed whose **stale**
suppression (themes untouched for a few turns) is scaled down for that
turn only, and the query channels get a `--shift-query-boost` bump: a
pivot back to a long-quiet topic stops being fought by the discount it
accumulated, while the topic being left stays suppressed. Nothing is
forgotten - the persisted counts are rebuilt from genuine increments, so
the damping never reaches disk.

And a **repetition** gate. `--dedup-cos 0.92` skips a new conversation
sentence whose embedding similarity to an earlier sentence of the same
speaker reaches the threshold, so restatements and re-asked questions
stop inflating the theme statistics - the original phrasing stays in
memory, and the repeat still rides the verbatim tail. Only like
suppresses like (an assistant restatement can never displace the user's
own words), and attached files are never gated. `/stats` counts the
suppressions; each one is logged with its similarity score so the
threshold can be tuned before it is trusted.

## 🔬 Results

SALT (coverage/CELF selector) on LongBench with Llama-3.1-8B-Instruct at a 20%
token budget. More datasets coming soon.

| Category | Dataset | Metric | SALT |
|---|---|---|---:|
| Single-Doc QA | `narrativeqa` | qa_f1 | 25.89 |
| | `qasper` | qa_f1 | 42.61 |
| | `multifieldqa_en` | qa_f1 | 51.03 |
| | **average** | | **39.84** |
| Multi-Doc QA | `hotpotqa` | qa_f1 | 56.09 |
| | `2wikimqa` | qa_f1 | 44.26 |
| | `musique` | qa_f1 | 31.76 |
| | **average** | | **44.04** |
| Summarization | `gov_report` | rouge | 31.59 |
| | `qmsum` | rouge | 23.89 |
| | `multi_news` | rouge | 23.78 |
| | **average** | | **26.42** |
| Few-Shot | `trec` | classification | 61.00 |
| | `triviaqa` | qa_f1 | 81.83 |
| | `samsum` | rouge | 42.94 |
| | **average** | | **61.92** |
| Synthetic | `passage_count` | count | 10.00 |
| | `passage_retrieval_en` | retrieval | 97.00 |
| | **average** | | **53.50** |
| Code | `lcc` | code_sim | 48.50 |
| | `repobench-p` | code_sim | 41.38 |
| | **average** | | **44.94** |
| **Overall** | | | **44.60** |

## 🔭 Roadmap

Active goals and next steps:

- **Summarization coverage** - extend the theme-coverage objective to better
  serve summarization, where recall across many minor themes matters most.
- **Provenance-aware memory** - turn, role, and time labels on conversation
  excerpts plus a compact conversation map, so answers can cite who said
  what and when.
- **Background ingestion** - move the per-turn keyword and embedding passes
  off the REPL's critical path, so long pasted messages never delay the
  next prompt.
- **Bounded long sessions** - mask-based (never-delete) eviction and
  growth-stable theme bookkeeping, so long-running sessions stay fast and
  exact as conversations and attachments accumulate.
- **Persistent serving** - a `vllm serve` chat backend, so the KV cache
  survives restarts and sessions resume warm.

## 🤝 Contributing

PRs welcome - see [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

SALT is released under the [MIT License](LICENSE).
