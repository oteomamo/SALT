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

## Persistent serving

The chat model can run as its own long-lived server instead of inside
saltChat. `saltServe` resolves a registered model and starts a
`vllm serve` process that owns the GPU, and `saltChat --backend
vllm-serve` connects to it as a thin client. The client renders and
tokenizes every prompt itself with the same shared helpers as the
in-process backends and sends token ids over the wire, so the text the
server caches is byte-identical to the text the kv ledger counts.

The design leans on the KV-cache-shaped prompt layout above. The stable
head (instructions, file inventory, `attach@` full texts) and the
verbatim tail are cached by the server's automatic prefix caching, and
both persist with the session: attachments reload in attach order and
the tail reloads verbatim, so a resumed conversation renders the same
prompt bytes the server already holds. Exiting saltChat costs neither
the cache nor the model load. The server reports its measured reuse per
turn through the additive apc fields in the ledger, and the client needs
no vLLM install of its own, so the server can run whichever vLLM release
fits the GPU, even from a separate environment.

## Multi-GPU

A model too big for one card, or a box whose first card also drives the
display, can spread across several cards with a `--gpu` list. `--gpu 0,1`
splits the chat model's weights over GPUs 0 and 1: the vllm backends
(in-process and `saltServe`) tensor-parallel it, and the hf backend loads
it with a balanced `device_map`. The BGE encoder that scores salience
rides the last card in the list, inside the memory the per-card cap leaves
free.

Each card in the group is capped at a fraction of its memory (`0.80` by
default across several cards, `0.90` for a lone server card), which leaves
headroom for activations, the KV cache, and the encoder. The list also
pins PCI bus order, so a `--gpu` index names the card `nvidia-smi` shows,
and the model and the encoder always agree on which physical card an index
means. One card, or no `--gpu`, keeps the original single-card path
unchanged.

## Provenance-aware memory

Compressed conversation memory used to arrive anonymous. Every excerpt sat
under one shared header, so the model could not tell the user's words from
its own, could not tell an early statement from the later one that revised
it, and had no idea whether something was said a minute ago or last week.
The excerpts were accurate and unattributable at the same time.

Each excerpt now carries its origin. The conversation part of the memory
block is cut into one section per turn, headed with that turn's number, who
was speaking, and how long ago it was said, and the reading guide the model
receives tells it that higher turn numbers are later. The turn and the
speaker were already in the session store and simply went unused. The time
is new: every ingest records when it happened, one stamp per message, saved
alongside the sentences. Sessions written before that carry no stamp, and
their labels leave the age out rather than inventing one. `--no-turn-labels`
returns to the single anonymous header.

Sections cost a header of roughly a dozen tokens per turn that wins budget,
which is why they are grouped by turn rather than attached per sentence.
Because selection returns sentences in the order they were spoken, and a
message's sentences enter the store together, a section is always exactly
one speaker's turn and the sections read in chronological order.

Alongside the excerpts sits a map of the conversation: one line per earlier
turn giving that turn's strongest keywords. It is built from the keywords
SALT already extracted at ingest, so it adds no model work at all. `/stats`
always prints it, and `--conversation-map` places it at the top of the
memory block. There it answers a question the excerpts cannot: whether a
topic came up at all. A subject discussed on turn 5 is invisible on any
turn where none of turn 5's sentences win budget, and the map makes it
visible as a pointer the model can ask about.

The map is a signal and never a gate. It changes nothing about which
sentences are selected, and the reading guide tells the model to treat it
as an index rather than as something anyone said. A long conversation
shows only its recent turns, and the header states that coverage, because
a map that quietly dropped older turns would read as proof a topic never
came up.

## Stable coverage keys

The cross-turn memory remembers what it has already shown as counts on
branches of the keyword tree, keyed by each branch's keywords. That
tree is rebuilt from document frequencies every turn, so as a
conversation grows a branch can come back under a different keyword
order, and the remembered counts stop matching anything. The
suppression quietly stops applying, and the same material can be
selected again as if it were never shown.

`--stable-coverage-keys` freezes the keyword order per session. A new
keyword joins at the tail of a persisted order rather than at its
current frequency rank, so every existing branch keeps its identity and
the remembered counts keep applying. Append-only ordering is the whole
mechanism: a keyword added at the tail of the order lands at the tail
of every branch that contains it, so every existing branch prefix, and
with it every remembered count, survives untouched.

Two companions complete it. Theme membership is sticky under the flag,
because a keyword falling below the theme cutoff would otherwise remove
its branches wholesale no matter how the order is frozen. A keyword
keeps its place while its remembered counts are alive and lets go once
forgetting clears them. And keys with no matching branch left, from
before the flag was on, are dropped once and counted, instead of being
carried forever.

The cost is ordering quality: a topic that becomes dominant late sits
deeper in the tree than frequency ordering would place it. The flag is
off by default, and `/stats` reports how many remembered keys matched
or orphaned each turn so the trade can be judged on real sessions.

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
| Persistent serving (`saltServe`, serve client) | `salt/chat/serve.py`, `salt/chat/runner_serve.py` |
| Multi-GPU placement (`--gpu` list) | `salt/chat/runner.py`, `salt/chat/serve.py` |
| CLI entry points | `salt` (`salt/compress.py`), `eval.py`, `saltChat` |
