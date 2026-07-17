# 🧩 Architecture

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
| Background ingest worker (chat) | `salt/chat/ingest.py` |
| Document ingest (PDF/text cleanup, `salt@`, `--doc`) | `salt/chat/pdfio.py` |
| Chat REPL + model registry | `salt/chat/`, `salt/models/` |
| CLI entry points | `salt` (`salt/compress.py`), `eval.py`, `saltChat` |
