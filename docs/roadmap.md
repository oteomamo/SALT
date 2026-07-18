# 🔭 Roadmap

Active goals and next steps:

- **Summarization coverage** - extend the theme-coverage objective to better
  serve summarization, where recall across many minor themes matters most.
- **Bounded long sessions** - mask-based (never-delete) eviction and
  growth-stable theme bookkeeping, so long-running sessions stay fast and
  exact as conversations and attachments accumulate.
- **MCP server** - a `salt-mcp` entry point exposing compression and session
  memory as tools, so AI clients (Claude Code, Claude Desktop, Cursor) can use
  SALT as their conversation memory without the REPL.
