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

A model too big for one card splits across several with a `--gpu` list,
on either backend, with the encoder riding the last card. Memory
shares, window caps and device flags are on the [Options](options.md)
page.

## Persistent serving

The backends above load the model inside the saltChat process, so the
model and its cache disappear on exit. `saltServe` keeps them alive as
a separate long-lived server that chats connect to and resume warm. It
has its own page: [Serving](serving.md).

## Other models beside this one

A chat talks to one model, and switching with `/model` unloads the one
it has. A roster names other models a session can reach without giving
up its own, each one a server of its own that stays loaded. `--roster`
declares them and the `/roster` and `/worker` commands below inspect
them. It has its own page: [Agents](agents.md).

## Conversations without the REPL

These conversations are files, and the prompt is not the only way to
reach them. `salt-mcp` serves the same folder over the Model Context
Protocol, so an editor or an agent runtime can add turns to a
conversation and read what it remembers, and a session started there
can be resumed here. The [MCP server](mcp.md) page has the setup.

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
| `/roster` | list the models `--roster` names, `/roster probe` contacts them |
| `/worker` | show each worker's connection, calls and mean latency |
| `/worker probe <name>` | reconnect one worker and report what it serves |
| `/worker start <name>` | launch a spawn entry's server, `start --all` does them all |
| `/worker stop <name>` | stop a server this session started |
| `/offload <task>` | hand a task to a worker, `/offload @NAME <task>` picks one |
| `/offload! @NAME` | put the last delegated task to another worker as well |
| `@NAME <question>` | let that worker answer this turn instead |
| `/agent <task>` | answer this turn by planning it out and handing the pieces to workers (see [agents](agents.md#a-turn-planned-out)) |
| `/new [id]`, `/clear` | start another conversation, wipe this one |
| `/exit` | leave (the session is saved and resumable by id) |

TAB completes `/commands`, `@worker` names, `salt@<file>`, and
`attach@<file>` names (see
[`salt/files/README.md`](https://github.com/oteomamo/SALT/blob/main/salt/files/README.md)).

## Letting another model answer

A line that starts with `@NAME` gives this one turn to a worker from the
[roster](agents.md). The prompt is the turn's own, memory and recent
messages and question together, so the worker answers exactly what the
chat model would have been asked. What comes back is this session's
reply: it joins the recent messages, it is remembered, and the next turn
goes back to the chat model as though nothing had changed hands. The
record of the turn names the model that actually spoke.

```
you> @qwen05 which of those two options is cheaper to install?
qwen05> The second one, because it reuses the existing inverter ...
```

That makes it possible to change model in the middle of a conversation
without ending it, including with `--backend vllm-serve`, where the
server holds one model and `/model` cannot switch it.

## Scripted turns

Instead of typing, `--turns FILE` runs a whole conversation from a file,
one turn after the next, into the same session so the memory builds across
them exactly as it would live. The file is a JSON array or a JSONL file
(one JSON value per line). A string item is a user message. An object item
takes its message from a common key it finds (`question`, `puzzle`,
`prompt`, and a few others), or from `--turns-field KEY` when you name the
key yourself.

```bash
saltChat --model QwQ-32B --backend vllm-serve --turns puzzles.json
```

Every backend works, so the same file can drive a persistent server. Each
turn prints its id and the model's reply. Add `--turns-out results.jsonl`
to also append `{id, turn, question, answer}` per turn, so the run can be
reviewed or scored afterward.

Launched with `--agent`, a scripted run plans every plain line out over
the roster's helpers, the way it would in the REPL, and each row gains a
`planned` field saying whether that turn went through a round. See
[agents](agents.md#planning-every-turn).

An item can also hand its work to a worker instead of the chat model, which
puts a delegation in the middle of a scripted conversation:

```json
[
  "What did we decide about the inverter?",
  {"offload": {"task": "Size the battery bank from that decision",
               "target": "qwen05", "ingest": true}}
]
```

A line can also attach a document before anything else runs, or hand a
turn to the orchestrator to plan out:

```json
[
  {"doc": "salt/files/SALT.pdf"},
  {"agent": "What does that paper claim, and does our sizing agree?"}
]
```

`doc` ingests a file exactly as `/doc` does, and `agent` runs the turn
exactly as `/agent` does, so a whole session is scriptable rather than
only its conversation. Each line does one thing, and a line asking for
several is refused when the file loads.

The task goes to that worker with this session's memory as its context, the
same way `/offload` does at the prompt. `target` picks the worker and can be
left out when the roster has only one, and `ingest` decides whether the
answer is remembered, defaulting to whatever the session was launched with.
Rows in `--turns-out` carry `kind` whenever the line was not an ordinary
turn, and a delegated row carries `status` and `worker` too, so a run can
be told apart from the turns the chat model answered. The
[Agents](agents.md) page covers what a delegation is and what it leaves
behind.

## Attachments

A `salt@` file becomes **its own branch** of the session trie, hanging off the
conversation's root - so multiple attachments never crowd each other out, and
the per-turn budget (default 20%) spreads across files and conversation
themes. The percentage is bounded by what actually fits the model's
window: `--memory-cap auto` (the default) sizes the block to the space
left after the fixed prompt, and `--memory-cap off` restores the old
unbounded sizing. An `attach@` file skips the trie entirely: its full text
rides uncompressed in every prompt.

## How PDFs are cleaned

Attached PDFs are read whole and cleaned into proper sentences before
they reach the trie: headers, footers and reference lists go, broken
lines and hyphenation are repaired, and paragraphs interrupted by a
figure or footnote are re-joined. Tables and pseudocode are kept as
rows grouped under their captions, and headings, panel labels and
equations survive even though they are short. A sentence mentioning a
URL keeps its prose with the link stored as `<url>`.

## What conversation text keeps

Messages are stored the way you typed them. Code keeps its generics,
tables and pipelines keep their pipes, and short decisive turns like
"go with option B" are kept rather than filtered as fragments. Only
whitespace is normalized, a link becomes `<url>`, and lines that are
mostly URL or pure junk still drop. Sessions from before this behavior
keep their previously stored text as is.

## What the model sees

The system prompt carries a reading guide
([`salt/chat/instructions.md`](https://github.com/oteomamo/SALT/blob/main/salt/chat/instructions.md))
plus an inventory of attached files, and the compressed memory arrives
at the top of the newest user message, grouped by origin: `[from
attached file 'paper.pdf' - 42 of 358 indexed sentences]` versus
`[from the earlier conversation - turn 12, user, 2h ago]`.

Conversation excerpts carry their **provenance**: the turn, the
speaker, and how long ago. The model can tell your words from its own,
see which of two conflicting statements came later, and answer "what
did I decide this morning". `--no-turn-labels` restores the plain
anonymous header.

A worker's answer kept with `--offload-ingest` is headed `[from
delegated work - turn 14, qwen05, 2h ago]` instead, naming the worker in
place of a speaker. The model is told to quote it as that worker's
report rather than as something said in this conversation, and the
conversation map credits the same turn to `worker(qwen05)`.

`/stats` prints a **conversation map**, one line per recent turn with
that turn's strongest keywords:

```
conversation map (all 4 turns):
  t10 user: eviction, indices, mask
  t11 assistant: contract, kvtrace, ledger
  t12 user: budget, decay, retrieval
  t13 assistant: coverage, half-life, themes
```

`--conversation-map` puts the map into the prompt as the first section
of the memory block, so the model can see a topic came up even on a
turn none of its sentences were selected. The map is a signal, never a
gate: it changes nothing about which sentences are picked.

## The prompt layout

The prompt is **KV-cache shaped**: everything stable comes first
(system prompt, `attach@` texts, the verbatim tail, compacting in
blocks so it stays byte-identical between compactions), and the only
per-turn content, the memory selection and the question, comes last.
Attachment order and the tail are saved with the session, so a resumed
conversation rebuilds the same prompt bytes and a warm cache still
matches. On the vllm backends the engine serves that stable prefix
straight from the GPU KV cache, in practice ~95% of prompt tokens on
quiet turns, and `/stats` shows the measured reuse.

## Memory knobs

Cross-turn memory has a set of switches. Each is reported in `/stats`
so it can be judged on a real session, and the [Options](options.md)
page lists them all with when to reach for each. In concept:

- **Forgetting.** A surfaced theme stays discounted for the whole
  session by default. `--coverage-half-life` lets that discount fade
  over turns of silence, so returning topics resurface.
- **Pivots.** `--shift-damping` lifts stale discounts for a detected
  topic pivot, that turn only, so coming back to an old topic is not
  fought by its own accumulated suppression.
- **Repetition.** `--dedup-cos` skips near restatements at ingest, so
  rephrasing does not inflate what counts as a theme. It compares
  meaning, so a short reversal built from the same words as the answer
  it corrects ("no, use PostgreSQL" against "yes, use PostgreSQL") can
  read as a restatement and be skipped. Fused acknowledgements
  (`--short-turns fuse`) are exempt, and the gate stays off by default.
- **Recent messages.** The last exchanges ride in the prompt verbatim,
  and selection skips those same sentences while they are visible, so
  the budget buys older context instead of repeating what is on
  screen. Their themes start counting as shown once they leave the
  recent window. `--no-tail-exclude` restores the old overlapping
  selection.
- **Key stability.** Remembered discounts are keyed to branches of a
  tree that is rebuilt every turn. `--stable-coverage-keys` freezes the
  session's keyword order so those keys keep matching as it grows.
- **Bookkeeping bounds.** `--coverage-gc` collects remembered keys that
  match nothing anymore, and `--coverage-max-keys` puts a hard limit
  on the remembered dictionary.
- **Session bounds.** `--max-sentences` caps how many conversation
  sentences stay in memory. Past the cap the oldest are masked out of
  selection rather than deleted, so their text and their numbering
  survive while a long session stops growing. Attached files are never
  masked. Reach for one of the bookkeeping bounds above alongside it,
  which collect the theme keys the masked sentences leave behind.
- **Theme scope.** `--per-source-themes` profiles the conversation and
  each attached file separately, so one large file cannot crowd the
  conversation out of memory.
- **Question identifiers.** `--query-identifiers` lets the question's
  identifier shaped tokens, like dates, versions and numbers, match
  memory directly. The letters only keyword gate drops them by
  default, even though memory indexes them, so a question that hinges
  on a version number can miss the turn that named it.
- **Ceiling.** `--memory-cap auto` (the default) fits the memory block
  to the space the model's window actually has left, instead of a
  percentage that grows without bound.
- **Short turns.** Terse decisions like "go with option B" stay in
  memory by default, and `--short-turns fuse` stores a bare "yes"
  together with the question it answers, so the decision is findable
  by the question's own words.

## Background ingestion

Indexing what was said (the keyword and embedding passes) runs on a
worker thread: your message is indexed while the model writes its
reply, so the next prompt appears immediately however long the paste
was. Answers do not change, because the REPL waits for the worker
before reading memory. A failed indexing keeps the message text in
`ingest_failures.jsonl` and reports at the next prompt, so nothing is
lost silently. `--sync-ingest` restores the old inline behavior.

Memory bookkeeping is failure safe in the other direction too. The
per-turn theme discounts, freshness stamps and topic baseline are
applied only once the model has actually answered. A turn that fails
partway leaves them exactly as they were, so a retry works from the
same memory the failed attempt saw instead of a worse one.

## Interrupted saves

A session's memory is saved as separate files, and a crash or kill can
land between them. When that happens the session is detected and rolled
back to the last complete state the next time it opens, with a one line
notice printed and a record appended to `load_repairs.jsonl` in the
session folder. If any sentence text has to be removed because its
vector was lost, the text itself is kept in that record, so nothing
disappears without a trace.

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

A turn that follows one or more delegations also carries
`agent_delegations`, `agent_delegated_tokens` and `agent_workers`. Those
sit beside the usage keys and never inside them, so anything already
reading the ledger's token accounting sees exactly what it saw before.
