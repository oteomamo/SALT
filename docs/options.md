# 🎛 Options

Every flag of the three commands, one line each. The concepts live on
the [Chatbot mode](chatbot.md), [Serving](serving.md),
[Agents](agents.md) and [Architecture](architecture.md) pages, and the
full detail lives in the code. Defaults shown are what a plain run
uses.

## saltChat quality switches

Off by default. Each one changes what the conversation memory keeps or
resurfaces, and `/stats` reports what it did, so it can be judged on a
real session before it is trusted.

| Flag | What it does | Turn it on when |
|---|---|---|
| `--conversation-map` | opens the memory block with a one line per turn index of the conversation | the model should see a topic came up even when none of its sentences were selected |
| `--dedup-cos 0.92` | skips a new sentence too similar to an earlier one from the same speaker | restatements are inflating what counts as a theme |
| `--coverage-half-life 8` | fades a shown theme's discount over turns of silence | topics the conversation returns to should resurface |
| `--coverage-decay-docs` | applies that fading to attached files too | selection keeps circling a document's head instead of advancing |
| `--shift-damping 0.25` | on a detected topic pivot, lifts stale discounts for that turn only | pivots back to an old topic are fought by its accumulated discount |
| `--per-source-themes` | profiles the conversation and each attached file separately | a large attachment crowds the conversation out of memory |
| `--stable-coverage-keys` | freezes the session's keyword order so remembered discounts keep matching their branches | long sessions re-show material because themes reshuffled |
| `--coverage-gc` | collects remembered keys that no longer match any branch of the memory tree | long sessions carry dead suppression in every save |
| `--coverage-max-keys 500` | hard limit on remembered theme keys | you want a strict bound no matter what else is on |
| `--max-sentences 400` | keeps the most recent conversation sentences in memory and masks older ones instead of deleting them | a long session keeps slowing down as its memory grows |
| `--short-turns fuse` | stores a bare "yes" together with the question it answers | terse decisions should be findable by the question's own words |

## saltChat reference

Models:

| Flag | Default | What it does |
|---|---|---|
| `--model NAME` | none | load a registered model and start the REPL |
| `--add HF_ID` | none | download and register a model |
| `--alias NAME` | derived | short name for `--add` |
| `--dtype T` | `bfloat16` | weight dtype at registration |
| `--force` | off | re-register over an existing alias |
| `--list` | off | list registered models and exit |

Session:

| Flag | Default | What it does |
|---|---|---|
| `--conversation-id ID` | fresh id | resume or name the session |
| `--doc PATH` | none | ingest a text or PDF into the trie at startup, repeatable |
| `--tail N` | `4` | exchanges kept verbatim after compaction |
| `--turns FILE` | none | replay a JSON or JSONL file of user turns into one session |
| `--turns-field KEY` | auto | which key holds the message in `--turns` items |
| `--turns-out FILE` | none | append each `--turns` answer as JSONL |

Agents:

| Flag | Default | What it does |
|---|---|---|
| `--roster FILE` | none | JSON file naming the worker models this session may reach, listed by `/roster` |
| `--workers-autostart` | off | start the roster's spawn entries once the chat model is loaded, instead of waiting for `/worker start` |
| `--offload-ingest` | off | remember what a worker answered, as a turn of its own labeled with the worker it came from |

Memory sizing:

| Flag | Default | What it does |
|---|---|---|
| `--budget-pct P` | `0.20` | share of the remembered words the memory block may use |
| `--memory-cap N\|auto\|off` | `auto` | absolute ceiling on the block, fitted to the model's window |

Behavior already on by default:

| Flag | Default | What it does |
|---|---|---|
| `--short-turns off\|keep\|fuse` | `keep` | keep short user messages in memory, `off` restores dropping them |
| `--no-tail-exclude` | off | let the memory block repeat sentences still shown in the recent messages (`--tail-exclude` is the default and accepted as a no-op) |
| `--no-turn-labels` | off | drop the turn and speaker labels from memory excerpts |
| `--sync-ingest` | off | index messages inline instead of on the background worker |

Backends and hardware:

| Flag | Default | What it does |
|---|---|---|
| `--backend hf\|vllm\|vllm-serve` | `hf` | how the chat model runs |
| `--server-url URL` | localhost 8000 | the `vllm-serve` backend's server |
| `--device D` | auto | device for the chat model |
| `--gpu LIST` | none | card index or comma list, several cards shard the model |
| `--bge-device D` | same as model | device for the encoder |
| `--gpu-mem-util F` | per backend | fraction of each card the engine may claim |
| `--max-model-len N` | model's own | cap the context window |

Tuning values for switches above:

| Flag | Default | What it does |
|---|---|---|
| `--shift-margin COS` | `0.12` | cosine drop that counts as a topic pivot |
| `--shift-query-boost X` | `1.5` | query boost on a pivot turn while damping is on |

## saltServe

Launches a persistent `vllm serve` process for a registered model, so
the server and its cache outlive individual chats. Everything after
`--` passes to `vllm serve` unchanged.

| Flag | Default | What it does |
|---|---|---|
| `MODEL` | the single registered model | registered alias or HF id to serve |
| `--host ADDR` | `127.0.0.1` | bind address |
| `--port N` | `8000` | listen port |
| `--gpu LIST` | none | card index or comma list, several cards tensor parallel the model |
| `--gpu-mem-util F` | `0.80` multi, `0.90` single | fraction of each card the server claims |
| `--max-model-len N` | model's own | cap the context window |
| `--vllm-bin PATH` | alongside saltServe | which vllm executable to run |

## salt

One shot compression of a dataset or a single document.

| Flag | Default | What it does |
|---|---|---|
| `--data FILE` | none | LongBench style JSONL to compress |
| `--doc PATH` | none | single document instead of `--data` |
| `--query Q` | empty | question to bias a `--doc` selection |
| `--output FILE` | required | where the compressed records go |
| `--selector coverage\|legacy` | `coverage` | CELF coverage selection or the legacy heuristic |
| `--token-budget-pct P` | `0.20` | share of the document kept |
| `--model NAME` | BGE small | encoder |
| `--device D` | `cuda` | encoder device |
| `--theme-percentile P` | mode default | frequency cutoff for theme keywords |
| `--lam L` | `0.5` | coverage discount between ranking and set cover |
| `--query-mass R` | `1.0` | query weight as a ratio of document mass |
| `--synthetic` | off | paragraph unit adapter for enumerated tasks |
| `--code` | off | line record adapter for code datasets |
| `--max-samples N` | all | cap the number of records |

The legacy selector's pass knobs and the mode adapters' sub flags are
intentionally not documented here. They exist for experiments, and
their meaning lives with their code in `salt/compress.py` and
`salt/engine/dataset_modes.py`.
