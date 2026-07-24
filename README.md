<div align="center">

# SALT

<p align="center">
  <img src="salt/assets/banner.png" width="100%">
</p>

## Salience-Aware Lexical Trie for Long-Context Compression

[![][docs-shield]][docs-link]
[![][version-shield]][release-link]
[![][license-shield]][license-link]
[![][arxiv-shield]][arxiv-link]

</div>

> [!NOTE]
> **What is next**
>
> - **MCP server** - a `salt-mcp` entry point so AI clients can use SALT as
>   their conversation memory without the REPL.
> - **Scripted conversation runs** - richer tooling around `--turns` for
>   driving and scoring long canned conversations.

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

> The legacy selector described in the paper is tagged
> [`v1.0.0`](https://github.com/oteomamo/SALT/releases/tag/v1.0.0). `main` now
> defaults to the coverage/CELF selector described below.

## 📑 Table of contents

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick start](#-quick-start)
- [Chatbot mode](#-chatbot-mode)
- [Results](#-results)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Reference](#-reference)
- [License](#-license)

## 🧩 Architecture

Two phases. **Indexing** reads the document once and builds a keyword trie - a
reusable map of its recurring themes. **Selection** then picks a sentence subset
under a token budget, returned in original document order, query-biased or not.

```text
 INDEXING  ── once per document, reused across turns and budgets
   document → split + junk filter
           → per-sentence keywords   (BGE-small [CLS] attention + knee cutoff)
           → theme salience          (SF = #sentences keeping a word, top quantile)
           → keyword trie            (each sentence's themes, SF-ordered, form a
                                      root-to-leaf path, leaves hold sentence ids)

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
│ │ grows every turn   │  │ memory + question  │  │ vllm-serve client  │   │
│ │ cross-turn coverage│  │ instructions.md    │  │ model registry     │   │
│ │ + half-life decay  │  │                    │  │ GPU-pinned models  │   │
│ │ + near-dup gate    │  │                    │  │                    │   │
│ │ + background ingest│  │                    │  │                    │   │
│ │ + tail-aware select│  │                    │  │                    │   │
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
│ │ apc fields: engine-measured prefix-cache reuse (vllm + vllm-serve) │   │
│ └────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│ ┌────────────────────────────────────────────────────────────────────┐   │
│ │                            Entry points                            │   │
│ │ salt (one-shot: --data / --doc) · saltChat · saltServe · eval.py   │   │
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
| Chat text handling (verbatim storage, short turns) | `salt/engine/chat_text.py`, `salt/chat/shortturn.py` |
| Background ingest worker (chat) | `salt/chat/ingest.py` |
| Document ingest (PDF/text cleanup, `salt@`, `--doc`) | `salt/chat/pdfio.py` |
| Chat REPL + model registry | `salt/chat/`, `salt/models/` |
| Persistent serving (`saltServe`, serve client) | `salt/chat/serve.py`, `salt/chat/runner_serve.py` |
| Multi-GPU placement (`--gpu` list) | `salt/chat/runner.py`, `salt/chat/serve.py` |
| CLI entry points | `salt` (`salt/compress.py`), `eval.py`, `saltChat`, `saltServe` |

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
see [Usage](https://oteomamo.github.io/SALT/usage/)) and `saltChat` (interactive chat, see
[Chatbot mode](#-chatbot-mode)).

**4. Authenticate with Hugging Face** - the eval model
(`meta-llama/Llama-3.1-8B-Instruct`) is gated:

```bash
hf auth login
```

Or skip the CLI and export the token directly: `export HF_TOKEN=hf_...`.

**5. (Optional) vLLM backend.** `eval.py` defaults to vLLM,
`saltChat --backend vllm` uses it for prefix caching, and `saltServe`
launches a persistent model server with it. Install it into the `salt`
env:

```bash
pip install "vllm==0.11.0" "prometheus-fastapi-instrumentator>=8.0.1"
```

The second pin keeps the server's routes healthy next to newer fastapi
releases. Skip this and run `eval.py --backend hf` for a portable run
that needs no vLLM. `saltChat` already defaults to its HF backend.
`saltServe` can also run a vLLM installed in a separate environment
through `--vllm-bin`.

> `bash scripts/setup_env.sh` does steps 2–3 in one shot (add `WITH_VLLM=1` to
> include vLLM).

## 🚀 Quick start

Fetch the LongBench data, then compress every task present and evaluate the
outputs in one command:

```bash
python salt/datasets/download_longbench.py
bash scripts/run_datasets.sh
```

Smoke test (5 samples per task, compression only):

```bash
MAX_SAMPLES=5 RUN_EVAL=0 bash scripts/run_datasets.sh
```

The script routes each dataset to the right mode automatically, then scores
the results. Every command, flag, and knob is documented on the
[Usage](https://oteomamo.github.io/SALT/usage/) and
[Datasets](https://oteomamo.github.io/SALT/datasets/) pages.

## 🤖 Chatbot mode

`saltChat` is an interactive chat REPL where SALT is the conversation memory:
one persistent trie per conversation grows with every exchange (and any
attached documents), and each turn it compresses the accumulated history into
a query-biased context block under the token budget.

```bash
saltChat --model qwen05 --conversation-id demo1 --doc report.txt
```

A persistent server started with `saltServe` keeps the model loaded and
its cache warm between chats, so a resumed conversation picks up without
re-reading its documents. The
[Chatbot mode guide](https://oteomamo.github.io/SALT/chatbot/) covers
the concepts, the [Serving](https://oteomamo.github.io/SALT/serving/)
page covers the server, and the
[Options](https://oteomamo.github.io/SALT/options/) page lists every
flag in one line each, including the off-by-default switches that make
long sessions better.

## 🔬 Results

SALT (coverage/CELF selector) reaches an overall **44.60** LongBench average
with Llama-3.1-8B-Instruct at a 20% token budget. The full per-dataset table
is on the [Results](https://oteomamo.github.io/SALT/results/) page.

## 🔭 Roadmap

In progress:

- **MCP server** - a `salt-mcp` entry point exposing compression and session
  memory as tools, so AI clients (Claude Code, Claude Desktop, Cursor) can use
  SALT as their conversation memory without the REPL.
- **Scripted conversation runs** - richer tooling around `--turns`, so
  canned conversations can drive long sessions and be scored afterward.

Next:

- **Summarization coverage** - extend the theme-coverage objective to better
  serve summarization, where recall across many minor themes matters most.

## 🤝 Contributing

PRs welcome - see [CONTRIBUTING.md](CONTRIBUTING.md).

## 📝 Reference

If you find this project useful for your research, please consider citing our
paper:

```bibtex
@misc{mamo2026saltsalienceawarelexicaltrie,
      title={SALT: Salience-Aware Lexical Trie for Long-Context Compression},
      author={Oteo Mamo and Hyunjin Yi and Joydhriti Choudhury and Shangqian Gao and Weikuan Yu},
      year={2026},
      eprint={2607.17486},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2607.17486}
}
```

## 📄 License

SALT is released under the [MIT License](LICENSE).

[docs-shield]: https://img.shields.io/badge/docs-oteomamo.github.io%2FSALT-blue
[docs-link]: https://oteomamo.github.io/SALT/
[arxiv-shield]: https://img.shields.io/badge/arXiv-2607.17486-b31b1b.svg
[arxiv-link]: https://arxiv.org/abs/2607.17486
[version-shield]: https://img.shields.io/github/v/tag/oteomamo/SALT?label=version&sort=semver
[release-link]: https://github.com/oteomamo/SALT/tags
[license-shield]: https://img.shields.io/badge/license-MIT-green
[license-link]: LICENSE
