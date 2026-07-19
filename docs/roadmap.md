# 🔭 Roadmap

In progress:

- **Bounded long sessions** - mask-based (never-delete) eviction and
  growth-stable theme bookkeeping, so long-running sessions stay fast and
  exact as conversations and attachments accumulate.
- **Graduating the memory switches** - several memory behaviors ship off
  by default (see [Options](options.md)) while `/stats` numbers from real
  sessions decide which of them become defaults.
- **Scripted conversation runs** - richer tooling around `--turns`, so
  canned conversations can drive long sessions and be scored afterward.
- **MCP server** - a `salt-mcp` entry point exposing compression and session
  memory as tools, so AI clients (Claude Code, Claude Desktop, Cursor) can use
  SALT as their conversation memory without the REPL.

Next:

- **Summarization coverage** - extend the theme-coverage objective to better
  serve summarization, where recall across many minor themes matters most.
