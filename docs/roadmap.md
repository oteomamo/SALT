# 🔭 Roadmap

In progress:

- **Bounded long sessions** - mask-based (never-delete) eviction, so
  long-running sessions stay fast and exact as conversations and
  attachments accumulate.
- **MCP server** - a `salt-mcp` entry point exposing compression and session
  memory as tools, so AI clients (Claude Code, Claude Desktop, Cursor) can use
  SALT as their conversation memory without the REPL.
- **Tail-aware memory selection** - skip sentences the model is already
  reading verbatim in the recent messages, so the memory budget buys new
  context instead of repeating what is on screen.
- **Incremental compression** - carry the previous turn's selection work
  forward on an append-only conversation, instead of redoing all of it every
  turn.
- **Graduating the memory switches** - several memory behaviors ship off
  by default (see [Options](options.md)) while `/stats` numbers from real
  sessions decide which of them become defaults.
- **Scripted conversation runs** - richer tooling around `--turns`, so
  canned conversations can drive long sessions and be scored afterward.

Next:

- **Summarization coverage** - extend the theme-coverage objective to better
  serve summarization, where recall across many minor themes matters most.
