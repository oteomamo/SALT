# 🤖 Chatbot mode

`saltChat` is an interactive chat REPL where SALT is the conversation memory:
one persistent trie per conversation grows with every exchange (and any
attached documents), and each turn it compresses the accumulated history into
a query-biased context block under the token budget. The BGE encoder and the
chat model stay resident on the GPU for the whole session. The trie lives in
process memory and autosaves to disk, so any conversation can be resumed
later by its id, recent exchanges included.

## Starting a session

Register a model by its HuggingFace name - weights are symlinked from your HF
cache into `salt/models/`, never copied (see
[`salt/models/README.md`](https://github.com/oteomamo/SALT/blob/main/salt/models/README.md)):

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
every turn (install vLLM first, see
[Installation step 5](installation.md)):

```bash
saltChat --model qwen05 --backend vllm --gpu 1
```

`--gpu-mem-util` caps the engine's share of GPU memory (default `0.85`,
leaving room for the BGE encoder on the same GPU). `--max-model-len` caps
the context window when the model's full window would not fit in the KV
cache. `/model` switching works on both in-process backends.

## Persistent serving

Both backends above load the model inside the saltChat process, so the
model and its cache disappear when you exit. `saltServe` starts the model
as its own long-lived server instead:

```bash
saltServe qwen05 --gpu 1
```

The command resolves the same registry entry, prints the full `vllm serve`
invocation it runs, and serves an OpenAI compatible API on
`http://127.0.0.1:8000` (change it with `--port`). The server keeps the
model loaded and its prefix cache warm across saltChat runs. Stop it with
Ctrl-C when you want the GPU back.

`--gpu-mem-util` sets the share of GPU memory the server claims (default
`0.90`, the BGE encoder no longer shares its GPU). On cards older than
Ampere the launcher serves `bfloat16` models in `float16`, which those
GPUs require. `--vllm-bin` points at another environment's vllm, so the
server can run whichever vLLM release fits your hardware while the SALT
install stays unchanged. Anything after `--` is passed to `vllm serve`
unchanged.

Connect saltChat to the running server:

```bash
saltChat --model qwen05 --backend vllm-serve
```

`--server-url` points at a server on another port or machine (default
`http://127.0.0.1:8000`). The prompt is rendered and tokenized in
saltChat itself, so the text the server caches is exactly the text the
kv ledger counts. Exit, come back later, resume the conversation by its
id: the model is still loaded and the stable prompt head is still
cached, so the first turn only prefills what changed. One server serves
one model, so switching with `/model` needs a second server on another
port. `/stats` shows the measured reuse every turn.

## REPL commands

| Command | Effect |
|---|---|
| `salt@` | list attachable files staged in `salt/files/` |
| `salt@<file>` | attach a `.pdf`/`.txt`/`.md`/`.rst`: whole text, own trie branch |
| `attach@<file>` | attach in full: uncompressed text rides in every prompt |
| `/model` | list registered models, `/model <name>` switches (session kept) |
| `/add <hf_id> [alias]` | download and register another model |
| `/doc <path>` | ingest a text or PDF file into the trie |
| `/budget <pct>` | set the memory budget (`0.3` or `30`) |
| `/stats` | session, attachments, compression, and GPU-memory stats |
| `/new [id]`, `/clear` | start another conversation, wipe this one |
| `/exit` | leave (the session is saved and resumable by id) |

TAB completes `/commands`, `salt@<file>`, and `attach@<file>` names (see
[`salt/files/README.md`](https://github.com/oteomamo/SALT/blob/main/salt/files/README.md)).

## Attachments

A `salt@` file becomes **its own branch** of the session trie, hanging off the
conversation's root - so multiple attachments never crowd each other out, and
the per-turn budget (default 20%) spreads across files and conversation
themes. An `attach@` file skips the trie entirely: its full text rides
uncompressed in every prompt.

## How PDFs are cleaned

Attached PDFs are read whole (images ignored) and cleaned into proper
sentences before they reach the trie: repeated headers/footers, page numbers,
and ACL/NeurIPS-style margin line numbers are stripped, ligatures and
hyphenation repaired, wrapped lines reflowed into paragraphs, and reference
lists filtered - sentence boundaries never break inside citations.
Paragraphs interrupted mid-sentence by a figure caption, table, or footnote
are re-joined across the float instead of being severed. Tables and
algorithm pseudocode are kept, grouped under their captions as
`|`-separated rows so numbers stay readable against their column names.
Section headings, panel labels, and equations survive ingestion (big
operators pypdf flattens to Latin look-alikes are restored where
unambiguous), and a sentence mentioning a URL keeps its prose with the
link as `<url>`.

## What the model sees

The chat model is told what it is looking at: the system prompt carries a
reading guide from
[`salt/chat/instructions.md`](https://github.com/oteomamo/SALT/blob/main/salt/chat/instructions.md)
(edit it to tune the wording - it is re-read every turn, even mid-session)
plus an inventory of every attached file, and the compressed memory arrives
at the top of the newest user message, grouped by origin - `[from attached
file 'paper.pdf' - 42 of 358 indexed sentences]` versus `[from the earlier
conversation]` - so answers can cite their source file and the model knows
the excerpts are partial.

## The prompt layout

The prompt is deliberately **KV-cache shaped**: everything stable (system
prompt, `attach@` full texts, the verbatim tail) comes first, and the only
per-turn content - the SALT memory selection and the question - comes last.
The tail grows append-only and compacts **in blocks** (back to `--tail`
exchanges once it hits twice that) instead of rolling every turn, so the
prompt prefix stays byte-identical between compactions. Nothing is lost,
since every sentence already entered the trie the moment it was spoken.
Attachments render in the order they were attached, and that order is
saved with the session, so a resumed conversation rebuilds the same
prompt bytes and a persistent server can serve them from its warm cache.
The tail is saved with the session too, so resuming restores the recent
exchanges verbatim and the whole stable prefix can stay warm on a
persistent server.
With the default HF backend each turn still prefills the whole prompt.
`--backend vllm` cashes the layout in. The engine's automatic prefix caching
serves the stable prefix straight from the GPU KV cache and prefills only
the fresh suffix - in practice ~95% of prompt tokens on quiet turns. Each
turn's real hit count is recorded in the kvtrace ledger (`apc_cached_tokens`,
next to the selection-overlap split, which measures a different thing), and
`/stats` prints it live.

## Memory knobs

Cross-turn memory has a **forgetting** knob. By default a theme that
has been surfaced stays discounted for the whole session, so the memory
block keeps favoring new material - even when the conversation circles back.
`--coverage-half-life 8` makes that suppression fade instead, halving every
8 turns a theme goes unmentioned, so a topic you return to after a long
detour resurfaces gradually rather than staying buried. Attached files are
exempt (selection keeps advancing through a document) unless
`--coverage-decay-docs` is set. `/stats` reports the decay state.

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
sentence too similar to an earlier one from the same speaker, so
restatements and re-asked questions stop inflating the theme statistics.
Attached files are never gated. `/stats` counts the suppressions.

## Background ingestion

Ingestion runs **in the background**. After every message SALT has
to index what was said (the keyword and embedding passes) so it can be
recalled later. That work used to run right before the next prompt, so
a long pasted message made the next `you>` slow to appear. It now runs
on a worker thread: your message is indexed while the model writes its
reply, and the reply is indexed while you read it. The next prompt
appears immediately, however long the message was. Answers do not
change, because the REPL always waits for the worker to finish before
it reads the conversation memory. If indexing ever fails, the message
text is kept in `ingest_failures.jsonl` in the session folder and the
error is shown at the next prompt, so nothing is lost silently. A nice
side effect: even when a reply fails halfway, your message has already
reached the memory. `/stats` shows how much work stayed off the prompt
path, and `--sync-ingest` restores the old inline behavior.

## The kv ledger

Every turn is recorded in a per-conversation KV ledger under
`salt/chat/sessions/<id>/kvtrace/`: an append-only `events.jsonl` whose usage
keys follow the cached-token convention (`input` = freshly prefilled
sentences, `input_cached_tokens` = context re-selected from the previous
turn, `output` = generated tokens) plus a per-token `tokens.npy` matrix.
`/stats` shows the running totals. On `--backend vllm` every event also
records the engine's measured prefix-cache reuse (`apc_cached_tokens` /
`apc_prompt_tokens`) - the positional ground truth next to the ledger's
content-overlap split.
