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
`/stats` reports the server's measured prefix-cache reuse.

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
