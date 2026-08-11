# 📄 Changelog

What each version added. Versions match the git tags on the
[repository](https://github.com/oteomamo/SALT).

## 2.11.0 - 2.11.27

The MCP milestone. SALT is a working memory layer for any MCP client:
`salt-mcp` serves compression, conversations and delegation over the
Model Context Protocol, and behind it saltChat can plan a turn with a
reasoning model, fan the pieces out to helper models in parallel, and
remember what a helper answered under that helper's own name. Every
agent behavior ships off by default and is bounded and recorded when
switched on. The [MCP server](mcp.md) and [agents](agents.md) pages
cover the tools, the roster and the limits.

- **2.11.1 - 2.11.3** `orphan_share`. The conversation snapshot reports
  orphaned suppression as a share of the whole coverage table beside
  the raw mass, so a rule about stale coverage means the same thing in
  a short conversation and a long one, and the sample rules file's
  orphan example now reads that share.
- **2.11.5 - 2.11.6** `--doc-root`. The MCP server can be started with
  one folder it reads documents from, and `salt_ingest_document` refuses
  a path outside it. Off by default, so a server started as before still
  reads any file it can. Worth setting when the client driving the
  server is a model rather than a person.
- **2.11.8** `--agent-rounds` is checked at launch. A number of rounds
  no turn can run is refused before the model loads, rather than ending
  the first planned turn that reaches it.
- **2.11.9** `--agent` reaches a scripted run. A plain line of a
  `--turns` file is planned out over the roster's helpers the way a
  typed line is, and its `--turns-out` row says whether that turn went
  through a round or was answered plainly because no worker was ready.
- **2.11.10** A turn's memory decision reaches the pieces it hands
  out. Under `--switch-agent`, the subtasks of a planned turn select
  their context the way that turn selected its own, instead of under
  the settings the session was launched with. A delegation that
  belongs to no turn is unchanged.
- **2.11.11** A planning call gets room to reason. A model that thinks
  out loud used to plan under a chat reply's length, spend it on the
  working and be cut off before the answer, which reached the round as
  a reply that was not a plan at all. The allowance is bounded by a
  quarter of the model's window, and a roster entry that names its own
  reply length still keeps it.
- **2.11.12** A round's record says what writing it up cost, or says
  nothing. The figure used to be read off the session's own model
  whatever had happened, so it was empty on every round that worked
  and filled in only on the ones that gave up on their helpers and
  answered directly.
- **2.11.13** `--log-signals`. A session can write down the numbers it
  reports about itself, one line per turn in `signals.jsonl`, which is
  the same closed set a memory decision is allowed to read. Off by
  default. Worth turning on for a session or two before trusting a
  rule whose threshold you have not watched fire.
- **2.11.14** Chat template settings can be set. A few models expose a
  choice through their chat template rather than through sampling, and
  `chat_template_kwargs` now reaches the template on every backend. It
  is rendered locally, so it never reaches the server as a request
  field, and setting nothing renders the prompt it always did.
- **2.11.16** A roster entry can ask a model to reason, or not to, with
  `"think": true` or `"think": false`. Leaving the key out is the
  default and sends no thinking setting at all. The key is a request
  rather than a promise: a model whose template has no such setting is
  unaffected, and one that always reasons keeps reasoning.
- **2.11.18** `--agent-think`. A round is three kinds of call, and this
  says which of them should reason out loud on the models that offer the
  choice. `plan` asks for reasoning where the turn is decided and for
  none where its time goes, `on` and `off` ask for all three or none,
  and the default asks for nothing. A roster entry's own `think` setting
  still wins.
- **2.11.19** A model that never stops reasoning ends its own call. One
  that has spent three quarters of its reply length on the working
  without answering is given up on there, keeping what arrived and
  leaving the worker usable, rather than generating to its cap and
  reaching the round as a reply that was never an answer.
- **2.11.20** The session's own model can be held to the plan's shape.
  Under `--backend vllm-serve` the chat model is reached over HTTP like
  any helper, so its server is asked once whether it will hold a model
  to a schema and is handed one when it will, which is the difference
  between a plan that parses first time and one that needs a repair. A
  model loaded in the session itself is unchanged: it has no request
  body for a schema to ride on and is shown the worked example.
- **2.11.21 - 2.11.25** `--route-agent`. Under `--agent` every line was
  planned out over the helpers, so a turn that said thanks cost the
  same three model calls a turn that needed the work did. A rules file
  can now decide, turn by turn, whether to plan at all and what the
  plan may spend, written in the same language the switch rules are
  over a wider set of signals: the conversation, the ask, who is ready
  to help, and what the last round did. A decision may only spend less
  than the flags already allow. `/stats` prints how often each rule
  actually fired beside what its author expected, since a threshold
  that never fires is the way a decision layer quietly does nothing.
  Off by default, and every rule in the shipped sample is an example.
- **2.11.27** `/stats` counts a switch rule over the whole session, not
  only the last turn. Every rule is listed with how many of the turns it
  was asked about it fired on and the note its author left, so a rule
  that never fires and one that fires on nearly every turn are both
  visible after one conversation.

## 2.10.0 - 2.10.123

The agent line. saltChat is growing an agent layer: a session can name
smaller helper models beside the chat model and, over the 2.10.z
releases, hand parts of its work to them with SALT memory in the loop.

Patch releases, grouped where several versions shipped one thing:

- **2.10.0** Model roster. A `salt.agents` package that reads a
  validated roster file naming helper models and how to reach them,
  with a sample at `salt/agents/roster_sample.json`. Groundwork only,
  nothing in the chat changes yet.
- **2.10.1 - 2.10.9** Workers in a session. `--roster FILE` loads the
  roster at launch, `/roster` and `/worker` list what is there and probe
  it for the model each endpoint is actually serving, and `/stats` names
  the workers. A session with a roster loaded chats exactly as one
  without.
- **2.10.10 - 2.10.17** Starting and stopping workers. `/worker start`
  launches a `saltServe` worker as a child of the session, `/worker
  stop` takes it down, and `--workers-autostart` starts the roster with
  the session. A worker is refused onto a card another server already
  claims unless both sides declare their share. A worker that goes quiet
  mid-reply is given up on, a refused connection is retried once, and
  one that fails twice in a row is left alone until a probe revives it.
- **2.10.18 - 2.10.26** Offload. `/offload <task>` hands one task to a
  worker together with this conversation's memory, selected for the task
  the way a chat turn selects it and committing nothing, so the session
  is the same after a delegation as before it. Every delegation is
  recorded in `delegations.jsonl`. With `--offload-ingest` the answer is
  remembered as a turn of its own, headed with the worker it came from
  rather than shown as something you or the model said.
- **2.10.27 - 2.10.34** Bounds, resume and scripted delegations.
  `--offload-context-cap`, `--offload-budget-pct` and
  `--offload-timeout` bound what a delegation is handed, selects under
  and waits for. `/stats` reports per-worker totals, and a reopened
  session carries its numbering and totals on while retiring a worker
  record no live process backs. A `--turns` file can carry an
  `{"offload": ...}` item, so a scripted conversation delegates where it
  needs to.
- **2.10.35 - 2.10.41** Delegating with fewer keystrokes. TAB completes
  `@NAME` from the roster, `/offload! @NAME` puts the last task to a
  second worker so two models answer one question from the same memory,
  and asking with no roster loaded prints the recipe for having one. A
  mixed chat and delegation conversation ships as
  `salt/agents/demo_turns.json`, and the sample roster now shows both an
  attach and a spawn entry. A delegation runs on the session's own
  thread, and interrupting one twice still leaves its record behind.
- **2.10.42** A worker can answer a turn. A line typed as `@NAME
  question` sends that turn's own prompt to a worker and keeps the
  answer as the session's own, remembered and recorded like any other,
  stamped with the model that gave it. This is changing model in the
  middle of a conversation without ending it, including under
  `--backend vllm-serve`, where the server holds one model and
  `/model` cannot switch it.
- **2.10.44 - 2.10.52** MCP server. `pip install "salt[mcp]"` adds a
  `salt-mcp` command that speaks the Model Context Protocol over
  stdio, so an editor or an agent runtime reaches SALT directly.
  `salt_compress` compresses one text under a budget, optionally
  biased to a query. Conversations are the other half:
  `session_create`, `session_resume`, `session_list` and
  `session_stats` open and report the same sessions saltChat keeps,
  `session_add_turn` remembers a message or a whole exchange in one
  call, `session_memory` returns the labeled memory block a chat turn
  would be given, and `salt_ingest_document` reads a file or a text
  into a conversation under its own source name. Open sessions are
  held warm up to a cap, and the one used longest ago is written to
  disk before it is closed. A server started with `--read-only`
  answers every read and refuses every write, leaving the
  conversations it was pointed at exactly as it found them. The
  [MCP server](mcp.md) page covers the install, the client entry and
  every tool.
- **2.10.54** Helper models over MCP. Started with `--roster FILE`, the
  server can reach the smaller models that file names: `roster_list`
  returns what is in the roster and, on request, which endpoints are
  actually answering, and `salt_delegate` hands one task over with a
  conversation's memory selected for it. Selecting it commits nothing,
  so a conversation is the same after a delegation as before it, and
  every delegation is filed beside the conversation it ran under.
- **2.10.55** A conversation as numbers. `session_stats` now carries a
  snapshot block describing the conversation in one flat set of
  signals, from how much of it is still selectable to what its last
  read measured, and `salt_switches` lists the memory switches, what
  the server has each one set to, and which measured number says
  whether it did anything.
- **2.10.56** Refusals a client can act on. Every refused call now opens
  with a fixed phrase naming what kind of refusal it is, from a bad
  argument to a conversation nobody made, and an unexpected fault is
  reported as one rather than as a traceback over the wire.
  `--max-ingest-chars` bounds how long a text one call may carry.
- **2.10.57** Closing down safely. A conversation closed to make room
  finishes what it was still encoding before it is let go, a
  conversation whose files disagree after a crash opens repaired and
  says so in a `warnings` list on the reply, and a server told to stop
  writes every open conversation before it goes.
- **2.10.59** A contract a client can check. `salt_contract` reports
  which version of the tool surface the server speaks, alongside the
  SALT version and every tool in order. The surface is written down in
  the server itself, so a renamed or dropped tool stops it at startup
  instead of surfacing as a broken client.
- **2.10.64 - 2.10.67** Planning in a form a session can act on. A model
  asked to decide what to delegate answers with one JSON object, read
  through the prose, fences and reasoning a local model wraps it in and
  held strictly to its shape inside. A reply that is not one is
  repaired once, quoting the actual fault, and a second failure keeps
  the model's own words as the answer instead of ending the round. A
  worker is asked once whether its server accepts a schema at all,
  since a version string does not answer that. Reasoning between
  `<think>` tags is cut before an answer is remembered, and
  `--agent-keep-think` keeps it.
- **2.10.68 - 2.10.71** Knowing what a helper can be asked for.
  `/roster probe --deep NAME` asks one helper to return three small
  objects exactly as given and reports whether it is schema-native,
  plain or flaky, remembering the answer beside the session. A planning
  model is then given instructions matched to that: fill the schema, or
  copy this object. The [agents](agents.md) page describes how a
  planning model talks back.
- **2.10.74 - 2.10.80** A turn planned out. `/agent <task>` answers one
  turn by planning it first: the chat model splits the task and names
  the helper each piece goes to, every piece is sent with the
  conversation's memory selected for it alone, and the chat model writes
  the reply from what came back. A piece that never answered is shown to
  it as a gap rather than left out, and a helper's words are quoted on
  the way in, as material to use rather than instructions to follow. The
  turn itself is an ordinary turn, kept and remembered like any other.
  `--agent-max-delegations` and `--agent-max-wall` bound what one round
  may cost, and every planned turn leaves a line in `agent_trace.jsonl`
  that `/stats` reads back.
- **2.10.82 - 2.10.89** A session that decides its own switches.
  `--switch-agent` with `--switch-rules FILE` lets a rules file decide
  the memory switches per turn. A rule is a sentence about the session
  and the switch it changes while that sentence is true, read by a
  parser that only compares the numbers a session reports about itself
  and never runs anything. A file that names a signal nobody reports, a
  switch a turn cannot set, or two switches known to cancel each other
  is refused before the session starts. Every decision lasts one turn
  and is written into nothing, and a turn a rule changed says which
  rule, what was true and what it set, in `/stats` and in that turn's
  own record. A sample ships as
  `salt/agents/switch_rules_sample.json` with one rule there is a
  reason for and two marked as examples that stay unloaded unless
  `--switch-rules-allow-examples` asks for them.
  `--switch-policy model` is the same seam with the chat model
  proposing instead, experimental, and held to the same refusals.
- **2.10.90 - 2.10.101** A reasoning model in the loop. A roster can
  name a model for the planning job with `"role": "orchestrator"`, and
  a turn then plans with it, under that entry's own settings and under
  a schema when its server accepts one, with the reply stamped with the
  model that wrote it. One that is not running costs nothing, since the
  session's own model plans instead. Pieces bound for different helpers
  now go out together and come back in the plan's order however they
  arrived. `--agent` plans every turn instead of one at a time and
  marks the reply with a line `--agent-quiet` drops. `--agent-rounds 2`
  lets the orchestrator ask once for one more thing under what is left
  of the turn's limits. A round where nothing came back answers the
  turn from the conversation instead of writing up nothing, and one
  where some pieces failed is told how many before it reads any of
  them. `--offload-ingest-cap` bounds what a helper's answer may add to
  memory. Two fixes for reasoning models: a reply whose opening think
  tag was never in it is cut correctly, and a turn now keeps what a
  model said rather than what it thought wherever that model sits.
- **2.10.102 - 2.10.107** The whole scenario, run end to end and
  pinned. Turns files gained `doc` and `agent` lines, so a scripted
  conversation can attach a file and plan a turn the way a person
  would, and `salt/agents/demo_turns.json` replays all four kinds of
  line deterministically. A session with no roster is pinned to
  consult nobody, file nothing and cost one call a turn, down to each
  turn's own record. The architecture pages carry the agent layer.
- **2.10.108 - 2.10.119** Hardening across the layer. A parallel round
  now stops for its shared token budget the way it stops for its
  clock, hands out nothing when a second round has nothing left, and
  an interrupt comes back as recorded pieces that keep what arrived
  instead of a frozen session. A request's timeout and its usage
  numbers travel under the worker's own lock, one delegation to a dead
  endpoint no longer condemns the worker, and a server still running
  from an earlier session is protected from being spawned over.
  Placement now checks the memory a server really takes, refusing a
  shared card the declared shares plus resident overhead cannot fit
  and reading the card's live free memory when `nvidia-smi` can be
  asked. A reply of runaway think tags and JSON `NaN` or `Infinity`
  are refused rather than crashed on, a switch decision cannot combine
  with the session's own settings into two switches known to cancel,
  and the tail-occupancy signal counts two messages per exchange. The
  MCP server takes one call at a time however many arrive, its writes
  refuse a conversation nobody made, `sync` waits the ingest queue out,
  and every text one call carries meets the same size bound.
- **2.10.120 - 2.10.123** The sample rules speak the signals' real
  language. `switch_rules_sample.json`'s examples now key on the
  verbatim window running light in a long conversation and on orphan
  mass as the weight of words it is, both able to fire on a real
  session, and the suite pins that they do.

## 2.9.0 - 2.9.123

Background ingestion for `saltChat`. The per-turn keyword and embedding
passes moved off the REPL's critical path onto a worker thread, so long
pasted messages never delay the next prompt. Failed ingests keep the message
text in `ingest_failures.jsonl`. The session is saved once per turn instead
of once per message. `--sync-ingest` restores the old inline behavior.

Patch releases, grouped where several versions shipped one thing:

- **2.9.4 - 2.9.9** Persistent serving. `saltServe` launches a
  long-lived `vllm serve` process, `saltChat --backend vllm-serve`
  connects to it, and a resumed session renders the same prompt bytes
  (attach order and recent exchanges saved), so it picks up warm.
  Server errors surface instead of passing off truncated replies.
- **2.9.12 - 2.9.16** Multi-GPU. `--gpu 0,1` splits a model across
  cards on `saltServe` and both chat backends, the encoder rides the
  last card, and card order is pinned so an index always means the
  same physical card.
- **2.9.19** Scripted turns. `--turns FILE` replays a JSON or JSONL
  conversation into one session and `--turns-out` records each answer.
- **2.9.21 - 2.9.27** Provenance-aware memory. Every excerpt is labeled
  with its turn, speaker and age, and a conversation map (one line per
  turn, `--conversation-map` to put it in the prompt) shows what a long
  session has covered. Refinements keep short sentences from dominating
  the map and make a partial map say so.
- **2.9.31 - 2.9.32** Short turns. Terse decisions like "go with
  option B" stay in conversation memory, and `--short-turns fuse`
  stores a bare "yes" together with the question it answers.
- **2.9.34 - 2.9.36** Verbatim conversation text. Pasted code keeps its
  generics, tables and pipelines keep their pipes, link sentences keep
  their prose with the URL as `<url>`. Older sessions keep their stored
  text as is.
- **2.9.37 - 2.9.38** Interrupted saves. A session whose files disagree
  after a crash rolls back to the last complete state on the next open,
  with a notice and a record in `load_repairs.jsonl`.
- **2.9.41** Theme scope. `--per-source-themes` profiles the
  conversation and each attached file separately, so one large
  attachment cannot crowd the conversation out of memory.
- **2.9.43 - 2.9.47** Memory ceiling. `--memory-cap auto` (the default)
  fits the memory block to the model's window instead of a percentage
  that grows without bound, and the overflow warning counts the whole
  prompt.
- **2.9.49 - 2.9.53** Stable coverage keys. `/stats` counts remembered
  keys that matched or orphaned each turn, and the opt-in
  `--stable-coverage-keys` freezes the keyword order so remembered
  discounts survive as the session grows.
- **2.9.55 - 2.9.58** Bounded bookkeeping. `/stats` splits remembered
  keys into live and orphaned, `--coverage-gc` collects the orphans,
  `--coverage-max-keys` puts a hard limit on the dictionary, and the
  code stops claiming a bound the defaults never delivered.
- **2.9.76 - 2.9.78** Failure-safe bookkeeping. The per-turn theme
  discounts, freshness stamps and topic baseline now commit only after
  the model answers. A turn that errors out leaves memory as it was,
  so the retry is not fighting a discount from the failed attempt.
- **2.9.82 - 2.9.87** Bounded long sessions. `--max-sentences` caps how
  many conversation sentences stay in memory, masking the oldest out of
  selection instead of deleting them, so their text, their numbering
  and the saved record all survive. Attached files are never masked.
  `/stats` reports how many sentences are still live.
- **2.9.94 - 2.9.96** Incremental compression. Each sentence's lexical
  tokens and the session's keyword profile are worked out once, when
  the text arrives, and carried forward instead of being rebuilt from
  the whole conversation every turn. What memory selects is unchanged,
  only the work behind it is smaller.
- **2.9.102** Repeating a capped sentence. With `--max-sentences` on,
  saying something again after the cap masked the old copy away stores
  it again, instead of dropping it as a copy of a sentence no longer in
  memory.
- **2.9.104** Failure-safe frozen keys. With `--stable-coverage-keys`
  on, a turn whose model call fails leaves the frozen keyword order and
  its sticky theme set exactly as they were, so a retry does not build
  on a discarded attempt.
- **2.9.106** Corrections in memory. With `--dedup-cos` on, a fused
  acknowledgement (`--short-turns fuse`) is exempt from the gate, so a
  bare "no" is not dropped as a near-duplicate of the "yes" it corrects.
  A short reversal built from the same words on the default path stays a
  documented limit of the cosine gate.
- **2.9.108 - 2.9.111** Tail-aware memory selection. The memory block
  no longer spends its budget on sentences the model is already reading
  verbatim in the recent messages, so the budget buys older context
  instead of repeating what is on screen. A sentence's themes start
  counting as shown once it leaves the recent messages. On by default,
  `--no-tail-exclude` restores the old overlapping selection, and
  `/stats` reports how many sentences were left out.

## 2.8.0

Near-duplicate memory gate. Opt-in `--dedup-cos` skips a new conversation
sentence too similar to an earlier one from the same speaker, so
restatements stop inflating theme statistics. Each suppression is logged to
`near_dups.jsonl` and counted in `/stats`.

## 2.7.0

PDF tables reach the model: table and pseudocode rows are grouped under
their captions as caption-prefixed units, so numbers stay readable against
their column names. Sturdier PDF extraction overall.

## 2.6.0 - 2.6.7

vLLM chat backend. `saltChat --backend vllm` serves registered weights
through an in-process vLLM engine with automatic prefix caching, streaming
tokens through the same REPL. Real prefix-cache hits are recorded in the
kvtrace ledger next to the selection split. `--gpu-mem-util` and
`--max-model-len` control the engine. Includes a regression harness.

## 2.5.0

Topic-shift damping. Opt-in `--shift-damping` detects when a question pivots
away from the recent conversation and scales down stale theme suppression
for that turn only, with `--shift-margin` and `--shift-query-boost` as
tuning knobs. Drift readings always show in `/stats`.

## 2.4.0

Coverage decay. Opt-in `--coverage-half-life` lets a surfaced theme's
suppression fade over turns of silence, so topics the conversation returns
to can resurface. `--coverage-decay-docs` opts attached files in.

## 2.3.0 - 2.3.1

kvtrace: a per-conversation KV read/write ledger (`events.jsonl` plus a
per-token matrix) recording reused, fresh, and output tokens every turn.

- **2.3.1** PDF-to-sentence pipeline (headers, ligatures, reflow,
  reference filtering), source-labeled memory blocks, and the
  KV-cache-shaped prompt layout with block-wise tail compaction.

## 2.2.0 - 2.2.1

`salt@` attachments: staged files ingest into their own trie branch so
multiple attachments never crowd each other out.

- **2.2.1** `attach@` full-context attachments: a file's whole text
  rides uncompressed in every prompt.

## 2.1.0

Chatbot mode. The `saltChat` REPL with a persistent per-conversation trie,
plus the `salt` console command.

## 2.0.0

CELF coverage selection became the default selector.

## 1.0.0

First release: LongBench compression and evaluation.
