# 🧩 Architecture

**Why SALT exists.** When a prompt is too long, most compressors give
each sentence one relevance score and keep the top scorers until the
budget runs out. Under a tight budget the document's main topic
swallows everything, a failure called *theme collapse*: in a multi-hop
question, the passages about the main entity survive while the one
sentence linking it to the second entity is dropped. SALT's answer is
to map the document's themes first and make every pick cheaper for a
theme the selection has already served, so minor themes keep their
share.

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
│ │              Agents - saltChat asking other models                 │   │
│ │ roster: worker + orchestrator endpoints, each its own server       │   │
│ │ /offload one task · @NAME one turn · /agent plan, hand out, write  │   │
│ │ every piece gets the trie selected for IT, committing nothing      │   │
│ │ switch agent: rules over the session's own numbers set the switches│   │
│ └────────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│ ┌────────────────────────────────────────────────────────────────────┐   │
│ │                            Entry points                            │   │
│ │ salt (one-shot: --data / --doc) · saltChat · saltServe · eval.py   │   │
│ │ salt-mcp - SALT memory over MCP, for an editor or an agent runtime │   │
│ │ salt@ trie attach · attach@ full text · /doc /model /budget /stats │   │
│ └────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

## Persistent serving

How to run it is on the [Serving](serving.md) page. The design: the
chat model can run as its own long-lived server instead of inside
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

## Bounded long sessions

A conversation that never ends used to grow in the one place that
costs something every turn. Selection read the whole session store,
and the memory block's size is a share of the words stored, so the
per-turn work and the memory block both swelled with the full history.
`--max-sentences` caps that: past the cap, the oldest conversation
sentences are masked out of selection, oldest first, and `/stats`
reports how many are still live.

Masked is not deleted. A masked sentence keeps its text, its embedding
and its row number, so the saved session stays complete, the kv
ledger's earlier references still resolve, and a resumed session sees
the store it saved. Attached files are never masked and never count
against the cap. An attachment is a bounded cost the user chose, while
conversation is the part that grows without asking. Masking a sentence
also forgets its verbatim-dedupe hash, so re-sending it word for word
stores it again rather than dropping it against a row no longer in
memory. The near-duplicate gate likewise compares against living
sentences only, so restating what the cap masked away stores the
restatement instead of suppressing the only living copy, and the
conversation map draws from living turns only, because a line pointing
at a masked turn would point at nothing.

The cap bounds each turn, not the record. Selection reads at most a
cap's worth of conversation plus the attachments, and the memory
block's base stops climbing with age. The stored session itself still
grows with everything ever said, about 1.7 KB per sentence kept,
because keeping everything is what makes the record and its references
permanent. Bookkeeping follows the content: remembered coverage counts
whose sentences are all masked stop matching any branch, so a capped
session usually wants a coverage bound as well, and `/stats` says so
when nothing is set to collect them. Under `--stable-coverage-keys`
the frozen keyword order also keeps absorbing new theme keywords,
measured in kilobytes even at a hundred thousand sentences, and left
append-only on purpose: reordering it is the one thing the stable keys
above cannot survive.

## Incremental compression

Selection reads the whole living session every turn, and much of what
it read it had already worked out before. Every stored sentence was
cleaned and stemmed again to score the question's words against it,
and the keyword counts that decide which themes exist were tallied
again across every sentence. Neither answer can change once a sentence
is stored, so a long conversation kept paying for the same derivation.

Both are now worked out once and carried forward. A sentence's lexical
tokens are derived as it is ingested, where the keyword and embedding
passes already run, so the cost lands in the turn that added the text
rather than in every turn after it. The keyword counts are kept
current as sentences arrive and as the session cap masks old ones out,
which is the same arithmetic a full recount performs.

What is carried is derived state, never the record. None of it is
saved to disk, and anything the session cannot account for is worked
out again rather than trusted. A session reopened from disk, one
repaired after an interrupted save, or a corpus assembled some other
way all fall back to the full derivation and reach the identical
answer. That is what makes it safe to carry anything at all: a miss
costs time and nothing else, so what memory returns is exactly what it
returned before.

What still runs every turn is the part that depends on the question.
The trie is rebuilt and the selection pass runs across the living
sentences each time, because both move with the question asked and
with what memory has already surfaced. Per-file theme profiling
(`--per-source-themes`) also keeps its own full recount, since its
buckets shift as sentences are masked.

## MCP server

Conversation memory was reachable two ways, at the prompt and from a
script, and both of them meant running SALT's own chat. Anything else
holding a conversation, an editor or an agent runtime, had no way in.

`salt-mcp` puts the memory behind the Model Context Protocol. A client
starts it, speaks JSON-RPC over stdio, and discovers the tools at
runtime: compress one text, or open a conversation and add turns to it,
read what it remembers about a question, put a document into it, and
ask what it holds. They are the same conversations the REPL keeps, in
the same folder, so one started in an editor can be resumed at the
prompt and the other way round.

The process is the design. The encoder is loaded once, on the first
call that needs it, and stays resident, because a client that connects
and asks nothing should pay nothing and a client that asks twice should
pay once. Conversations open on demand and stay warm behind a cap, and
the one used longest ago is closed when the cap is reached: closing
finishes what it was still encoding, writes it if it changed, then lets
go. Every way of ending, a client hanging up, a Ctrl-C, a kill, runs
that same close.

Reading is a turn. The memory block a tool call returns is the block a
chat turn would be given, labeled the same way, and the selection is
committed the same way, so what a conversation has already surfaced
shapes what it surfaces next no matter which side asked. A server
started `--read-only` drops that commit and refuses every write, which
is what an automated client can safely be pointed at.

Two more tools belong to the agent layer rather than to memory:
`roster_list` and `salt_delegate` reach the helper models a roster
names, handing one task over with the conversation's memory selected
for it and committing nothing. `session_stats` also carries a flat
snapshot of the signals a decision about memory would be made on, and
`salt_switches` lists the switches those signals correspond to. The
switches are read-only from outside: a client can see how memory is
being selected and measure the result, and the decision to change it
stays where the session is run.

The surface is meant to outlive its clients. Tool names are forever and
schemas grow additively, the tool list is declared in the server so a
rename fails at startup, and `salt_contract` reports which version of
the contract a client has reached.

## Agents

saltChat talks to one model and remembers a conversation. A roster lets
it reach others without giving either of those up. This is not a layer
beside the chat: it is the same session, the same trie and the same
turn, with the question of who answers opened up.

Four things arrived in order, and each one is the previous one asked a
harder question.

**The roster** names models a session may reach, one server each, and
nothing is contacted until something asks. A worker is a model tasks can
be handed to. An orchestrator is a model that decides what the tasks
are. Entries either attach to a server already running or are started by
the session itself, and two servers sharing a card have to say in
writing how much of it each takes.

**Delegation** hands one task to one worker together with this
conversation's memory, selected for that task the way a chat turn
selects for its question. The selection commits nothing: no coverage
moves, the verbatim tail is untouched, and the session is the same after
a delegation as before it. That property is what makes the rest
possible. A round can select memory five times for five different pieces
of work without any of them changing what the next one sees.

**The orchestrator** turns one question into several. It is shown the
conversation's memory and the question, and answers with either the
answer or a list of pieces and the helper each piece goes to. Every
piece then goes out with the memory selected for it alone, so a helper
never sees the plan or the other pieces and each task has to stand on
its own. What comes back goes to the orchestrator once more, with the
original question under it, and what it writes is the reply. Pieces for
different helpers run at the same time. The thread that owns the session
does every trie read and every write, and the threads do HTTP only.

The answer becomes an ordinary turn. Same memory selected for it, same
pair in the verbatim tail, same record kept. That is the whole design
constraint: an agent turn has to be indistinguishable from a chat turn
everywhere downstream, or every reader of a conversation would need to
know which kind it was looking at.

**The switch agent** turns the question inward. Every memory switch
already travels as a keyword on the call that uses it rather than being
baked into the session, so something can vary one for a single selection
and leave the session untouched. A policy is asked once per turn, given
a closed set of signals the session reports about itself, and answers
with the switches to change for that call. Rules are written down as
sentences about the session, read by a parser that only compares and
never runs anything. A model can propose instead, and meets exactly the
refusals a written rule meets.

Every step of this is off until asked for. A session with no roster
consults nobody, describes itself to nobody and costs one call a turn,
byte for byte what it cost before any of this existed.

## Tail-aware memory selection

The prompt already carries the last exchanges verbatim, and selection
used to treat those same sentences as fair candidates. They match the
current question better than anything older, so the memory block kept
spending part of its budget re-showing text sitting a few messages
below, and on a real stored session that duplication reached about a
quarter of the budget. Each re-show also counted as coverage, so a
sentence's themes were most discounted at exactly the moment the tail
stopped carrying it and memory became its only home.

Selection now skips sentences that are still visible word for word in
the recent messages. The skip narrows candidacy and nothing else: the
theme map, the keyword order and every remembered discount still
describe the full living session, so the memory tree keeps the exact
shape it would have without the skip, and a skipped sentence's themes
start counting as shown only once it leaves the recent window. The
freed budget goes to older material instead, and `/stats` reports how
many sentences were left out each turn.

Two guards keep the skip safe. Matching asks for the whole sentence
between word boundaries, so a short reply never matches inside a
longer word, and a sentence altered at ingest simply stays selectable.
And when everything alive is still on screen, early in a session, the
skip stands down rather than hand the model an empty memory block.
`--no-tail-exclude` restores the old overlapping selection.

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
| MCP server (`salt-mcp`) | `salt/mcp/server.py`, `salt/mcp/pool.py`, `salt/mcp/agents.py` |
| Agents (roster, delegation, orchestrator, switch policy) | `salt/agents/` |
| Multi-GPU placement (`--gpu` list) | `salt/chat/runner.py`, `salt/chat/serve.py` |
| CLI entry points | `salt` (`salt/compress.py`), `eval.py`, `saltChat`, `saltServe`, `salt-mcp` |
