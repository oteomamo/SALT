# 📄 Changelog

What each version added. Versions match the git tags on the
[repository](https://github.com/oteomamo/SALT).

## 2.9.0 - 2.9.104

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
