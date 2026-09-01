# 🎛 Options

Every flag of `saltChat`, `saltServe` and `salt-mcp`, one line each,
plus the `salt` flags a normal compression needs. The concepts live on
the [Chatbot mode](chatbot.md), [Serving](serving.md),
[Agents](agents.md), [MCP](mcp.md) and [Architecture](architecture.md)
pages, and the full detail lives in the code. Defaults shown are what a
plain run uses.

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
| `--query-identifiers` | lets the question's dates, versions and numbers match memory directly | questions that hinge on an identifier keep missing the turn that named it |
| `--episode-gap 6` | groups memory into time episodes, splitting where exchanges sit more than this many hours apart, and gives each its own branch | a session spanning days blurs its epochs together and answers from the wrong one |
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
| `--roster FILE` | none | JSON file naming the worker models this session may reach, listed by `/roster`. `auto` fits one from the registry and the memory free right now, and writes it to the session folder |
| `--workers-autostart` | off | start the roster's spawn entries once the chat model is loaded, instead of waiting for `/worker start` |
| `--offload-timeout SECONDS` | the standard call timeout | how long to wait on a quiet worker during a delegation, for workers whose roster entry names no timeout of its own |
| `--offload-budget-pct` | the session budget | memory budget for a delegation's context, as a fraction like `--budget-pct` |
| `--offload-context-cap N` | off | cap the memory handed to a worker at N words, on top of the memory budget that already sizes it |
| `--offload-ingest-cap CHARS` | 2000 | how much of a worker's answer is remembered, cut at a sentence boundary (0 keeps all of it) |
| `--offload-ingest` | off | remember what a worker answered, as a turn of its own labeled with the worker it came from |
| `--agent-keep-think` | off | keep a worker's reasoning when its answer is remembered, instead of cutting it |
| `--agent` | off | plan every turn out instead of answering it directly, the way `/agent` does one turn |
| `--agent-quiet` | off | leave out the one-line notice an agent-routed reply carries under `--agent` |
| `--agent-rounds N` | 1 | how many rounds of delegating one turn may take, at most 2: the second lets the orchestrator ask for one more thing |
| `--agent-think MODE` | template | which parts of a round reason out loud on the models that offer the choice: `template`, `plan`, `on` or `off` |
| `--agent-max-delegations N` | 4 | how many pieces one `/agent` turn may hand out before the rest are reported as not attempted |
| `--agent-max-wall SECONDS` | 600 | how long one `/agent` turn may spend handing pieces out before it answers with what it has |
| `--log-signals` | off | write one line per turn to `signals.jsonl` in the session folder, holding the numbers this session reports about itself |
| `--route-agent` | off | let a policy decide which turns are planned out over the helpers instead of planning every one of them |
| `--route-rules FILE` | none | the rules `--route-agent` decides by, as sentences about the session, the ask and the last round |
| `--route-rules-allow-examples` | off | load the route rules a file marks as examples too, none of which has been measured |
| `--route-policy rule\|model` | rule | what decides under `--route-agent`: the rules file, or the model itself (experimental) |
| `--switch-agent` | off | let a policy decide the memory switches per turn instead of leaving them where the flags set them |
| `--switch-rules FILE` | none | the rules `--switch-agent` decides by, as sentences about the session and the switch each one changes |
| `--switch-rules-allow-examples` | off | load the rules a file marks as examples too, which are written down to be read rather than run |
| `--switch-policy rule\|model` | rule | what decides under `--switch-agent`: the rules file, or the chat model itself (experimental) |

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
| `--verbose` | off | also print the top theme keywords for each record |

The tuning knobs are intentionally not documented here: the shared trie
thresholds, the legacy selector's pass knobs, and the mode adapters' sub
flags. They exist for experiments. `salt --help` lists every one of
them, and their meaning lives with their code in `salt/compress.py` and
`salt/engine/dataset_modes.py`.

## salt-mcp

Serves the same conversation memory to an MCP client, over the tools on
the [MCP](mcp.md) page.

| Flag | Default | What it does |
|---|---|---|
| `--sessions-dir DIR` | the saltChat folder | where conversations live |
| `--max-open-sessions N` | `8` | how many conversations stay open at once |
| `--max-ingest-chars N` | `400000` | longest text one call may carry |
| `--doc-root DIR` | any file the server can read | read documents by path only from under this folder |
| `--read-only` | off | answer reads and refuse every write, leaving every conversation exactly as it was found |
| `--roster FILE` | none | roster of helper models this server may delegate to |
| `--device D` | `cuda` when there is one | encoder device |
| `--bge-device D` | same as `--device` | encoder device, winning over `--device` |
| `--gpu LIST` | none | card index or comma list, with the encoder on the last card |
| `--version` | off | print the salt version this server carries |
