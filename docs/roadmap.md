# 🔭 Roadmap

In progress:

- **MCP server** - a `salt-mcp` entry point exposing compression and session
  memory as tools, so AI clients (Claude Code, Claude Desktop, Cursor) can use
  SALT as their conversation memory without the REPL.
- **Scripted conversation runs** - richer tooling around `--turns`, so
  canned conversations can drive long sessions and be scored afterward.

Next:

- **Summarization coverage** - extend the theme-coverage objective to better
  serve summarization, where recall across many minor themes matters most.
