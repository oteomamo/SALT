# 🔭 Roadmap

In progress:

- **MCP server** - a `salt-mcp` entry point exposing compression and session
  memory as tools, so AI clients (Claude Code, Claude Desktop, Cursor) can use
  SALT as their conversation memory without the REPL.
- **Tail-aware memory selection** - skip sentences the model is already
  reading verbatim in the recent messages, so the memory budget buys new
  context instead of repeating what is on screen.
- **Graduating the memory switches** - several memory behaviors ship off
  by default (see [Options](options.md)) while `/stats` numbers from real
  sessions decide which of them become defaults.
- **Scripted conversation runs** - richer tooling around `--turns`, so
  canned conversations can drive long sessions and be scored afterward.

Next:

- **Summarization coverage** - extend the theme-coverage objective to better
  serve summarization, where recall across many minor themes matters most.
