# Package map

Where each stage of SALT lives, from the split of a document into
sentences to the commands that run it. The ideas behind the stages are
on the [Architecture](https://oteomamo.github.io/SALT/latest/architecture/)
page.

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
| Scoped search (branch scores, the rule, `/scope`) | `salt/chat/scope.py`, `salt/engine/session_trie.py` |
| Time windows (the parser, `/when`) | `salt/chat/when.py` |
| Summary turns (the lexicon, `/summary`) | `salt/chat/summary.py`, `salt/engine/session_trie.py` |
| Chat REPL + model registry | `salt/chat/`, `salt/models/` |
| Persistent serving (`saltServe`, serve client) | `salt/chat/serve.py`, `salt/chat/runner_serve.py` |
| MCP server (`salt-mcp`) | `salt/mcp/server.py`, `salt/mcp/pool.py`, `salt/mcp/agents.py` |
| Agents (roster, personas, delegation, orchestrator, switch policy) | `salt/agents/` |
| Multi-GPU placement (`--gpu` list) | `salt/chat/runner.py`, `salt/chat/serve.py` |
| CLI entry points | `salt` (`salt/compress.py`), `eval.py`, `saltChat`, `saltServe`, `salt-mcp` |

Each module opens with a docstring saying what it is for. The engine
directory holds the compression path the evaluation runs on, which
changes only through additive, off-by-default seams (see
[CONTRIBUTING](../CONTRIBUTING.md)), and the chat, agents and mcp
directories hold everything built on top of it.
