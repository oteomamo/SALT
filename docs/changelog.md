# 📄 Changelog

What each version added. Versions match the git tags on the
[repository](https://github.com/oteomamo/SALT).

## 2.9.0 (2026-07-16)

Background ingestion for `saltChat`. The per-turn keyword and embedding
passes moved off the REPL's critical path onto a worker thread, so long
pasted messages never delay the next prompt. Failed ingests keep the message
text in `ingest_failures.jsonl`. The session is saved once per turn instead
of once per message. `--sync-ingest` restores the old inline behavior.

Patch releases: 2.9.4 adds `saltServe`, a command that launches a
persistent `vllm serve` process from registered weights, so the model and
its cache outlive individual chat sessions. 2.9.5 adds
`saltChat --backend vllm-serve` with `--server-url`, the client side of
persistent serving, so a chat can connect to the running server and
`/stats` reports the server's measured prefix-cache reuse. 2.9.6 keeps
attachments in their attach order when a session resumes, so the prompt
renders the same bytes across restarts and stays warm in the server
cache. 2.9.7 saves the recent exchanges with the session, so a resumed
conversation remembers its last turns verbatim instead of starting from
compressed memory alone. 2.9.9 surfaces server errors that arrive
mid-reply instead of presenting a truncated answer as complete, streams
any unicode safely, keeps long prompts within the server's window, and
hardens the launcher's GPU detection. 2.9.12 lets `saltServe --gpu 0,1`
split a model's weights across several cards (tensor parallel), so a model
too big for one card still serves. Every card in the group is capped at
0.80 of its memory by default. 2.9.13 extends `--gpu` on saltChat to a
card list too. The `--backend vllm` engine tensor-parallels the model
across the cards and the BGE encoder rides the last one, which the 0.80
memory cap keeps room for. 2.9.14 pins the same PCI card order for the
encoder and the model, so a `--gpu` index means the same physical card
for both. 2.9.16 extends the `--gpu` list to the hf backend, which shards
the model across the cards with a balanced device_map (the BGE encoder
still rides the last card). 2.9.19 adds `--turns FILE`, which runs a
scripted conversation from a JSON or JSONL file one turn after the next
into the same session, so a canned set of questions builds SALT's memory
just like a live chat. `--turns-out` records each answer as JSONL. 2.9.21
labels every conversation excerpt in the memory block with the turn it was
said on and who said it, so the model can tell your words from its own and
can see which of two conflicting statements came later.
`--no-turn-labels` restores the plain unlabeled header. 2.9.22 adds a
conversation map to `/stats`, one line per recent turn with that turn's
strongest keywords, so a long session's coverage is visible at a glance.
2.9.23 adds `--conversation-map`, which puts that map at the top of the
memory block so the model can see a topic came up on a given turn even
when none of that turn's sentences were selected. 2.9.24 adds how long ago
to those labels, so the model can answer questions about when something
was said instead of only about what and by whom. 2.9.26 and 2.9.27 refine
the map. Keyword ranking no longer favors short sentences over the long
ones that carry a turn's actual topic, and the map header states how many
turns it covers, so a long conversation showing only its recent turns
never reads as proof that an older topic was never discussed. 2.9.31
stops the junk filter from dropping short user messages, so terse
decisions like "go with option B" stay in conversation memory for the
whole session. `--short-turns off` restores the old dropping behavior.
2.9.32 adds `--short-turns fuse`, which stores a bare acknowledgement
like "the second one" together with the question it answers, so the
decision can be found again later by the question's own words. 2.9.34
stops chat ingest from scrubbing messages like benchmark documents, so
pasted code keeps its generics, tables and pipelines keep their pipes,
and a sentence with a link keeps its prose with the URL stored as
`<url>`. 2.9.35 protects table rows, pipelines, link sentences and
code-shaped lines from the short-fragment filter, so they reach memory
even when brief.

## 2.8.0 (2026-07-15)

Near-duplicate memory gate. Opt-in `--dedup-cos` skips a new conversation
sentence too similar to an earlier one from the same speaker, so
restatements stop inflating theme statistics. Each suppression is logged to
`near_dups.jsonl` and counted in `/stats`.

## 2.7.0 (2026-07-14)

PDF tables reach the model: table and pseudocode rows are grouped under
their captions as caption-prefixed units, so numbers stay readable against
their column names. Sturdier PDF extraction overall.

## 2.6.0 - 2.6.7 (2026-07-13)

vLLM chat backend. `saltChat --backend vllm` serves registered weights
through an in-process vLLM engine with automatic prefix caching, streaming
tokens through the same REPL. Real prefix-cache hits are recorded in the
kvtrace ledger next to the selection split. `--gpu-mem-util` and
`--max-model-len` control the engine. Includes a regression harness.

## 2.5.0 (2026-07-09)

Topic-shift damping. Opt-in `--shift-damping` detects when a question pivots
away from the recent conversation and scales down stale theme suppression
for that turn only, with `--shift-margin` and `--shift-query-boost` as
tuning knobs. Drift readings always show in `/stats`.

## 2.4.0 (2026-07-06)

Coverage decay. Opt-in `--coverage-half-life` lets a surfaced theme's
suppression fade over turns of silence, so topics the conversation returns
to can resurface. `--coverage-decay-docs` opts attached files in.

## 2.3.1 (2026-07-05)

PDF-to-sentence pipeline (headers, ligatures, reflow, reference filtering),
source-labeled memory blocks, and the KV-cache-shaped prompt layout with
block-wise tail compaction.

## 2.3.0 (2026-07-04)

kvtrace: a per-conversation KV read/write ledger (`events.jsonl` plus a
per-token matrix) recording reused, fresh, and output tokens every turn.

## 2.2.1 (2026-07-04)

`attach@` full-context attachments: a file's whole text rides uncompressed
in every prompt.

## 2.2.0 (2026-07-04)

`salt@` attachments: staged files ingest into their own trie branch so
multiple attachments never crowd each other out.

## 2.1.0 (2026-07-03)

Chatbot mode. The `saltChat` REPL with a persistent per-conversation trie,
plus the `salt` console command.

## 2.0.0 (2026-07-03)

CELF coverage selection became the default selector.

## 1.2.0 (2026-07-03)

QuALITY and NIAH experiments and downloaders.

## 1.0.0 (2026-07-03)

First release: LongBench compression and evaluation.
