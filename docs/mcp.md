# 🔌 MCP server

`salt-mcp` puts SALT behind the Model Context Protocol, so an editor or
an agent runtime can compress text through the same engine `salt` and
`saltChat` use. It speaks JSON-RPC over stdio and is started by the
client, not by you.

## Install

The server ships with the optional extra:

```bash
pip install "salt[mcp]"
```

Check it:

```bash
salt-mcp --version
```

## Point a client at it

Most clients read a JSON file naming the servers they may start. The
entry for SALT is the command and its flags:

```json
{
  "mcpServers": {
    "salt": {
      "command": "salt-mcp",
      "args": ["--gpu", "0"]
    }
  }
}
```

The server loads the encoder on the first call that needs it, so a
client that connects and asks for nothing pays nothing.

## Server flags

| Flag | Effect |
|---|---|
| `--version` | print the SALT version this server carries and exit |
| `--device` | device for the encoder (default: `cuda` when there is one, else `cpu`) |
| `--gpu` | CUDA index or comma list, with the encoder on the last card |
| `--bge-device` | device for the encoder, winning over `--device` |
| `--sessions-dir` | where conversations live (default: the folder saltChat uses) |
| `--max-open-sessions` | how many conversations stay open at once (default: 8) |

There is no GPU requirement. On a machine without one the encoder runs
on the CPU, which is slower per call but needs nothing installed beyond
the package.

## Tools

| Tool | What it does |
|---|---|
| `salt_compress` | compress one text to a fraction of its words |
| `session_create` | start a conversation whose memory this server keeps |
| `session_resume` | open one that already exists, with the memory it had |
| `session_list` | every conversation on disk, most recently written first |
| `session_stats` | what one conversation holds |

### salt_compress

| Argument | Meaning |
|---|---|
| `text` | the text to compress (required) |
| `budget_pct` | the share of the original words to keep, `0.2` by default |
| `query` | what the compression should favor, optional |

It returns the compressed text and the numbers behind it: how many
words went in and came back, how many sentences were found and kept,
the word budget, the token count of the result, and the share of the
text's themes the result still covers.

Without a query the result keeps what covers the text as a whole. With
one, the selection is biased toward the sentences that answer it, which
is the same thing a saltChat turn does with its memory.

Compression is prose-shaped here: sentences are cleaned, split and
filtered, then selected under the budget. That is the same path the
`salt` command runs, with the same defaults, so a tool call and a
command line give the same answer for the same text.

## Conversations

`salt_compress` is a single shot: text in, shorter text out, nothing
remembered. A session is the other way of working. `session_create`
starts one, `session_resume` opens one again later, and what the
conversation has accumulated is memory the server keeps between calls.

`session_list` shows what is on disk, newest first, with the turn and
sentence counts each one recorded. `session_stats` opens one and reports
what it holds now: turns, sentences, how many are still live after any
cap, attached files and the budget it compresses under.

These are the same conversations `saltChat` keeps, in the same folder
and under the same naming rule, so a session started here can be
resumed at the prompt and the other way round.

Open sessions are held warm, up to `--max-open-sessions` of them. Past
that the one used longest ago is closed, and closing writes it first,
so a conversation is never dropped with unsaved turns in it.

One server per set of conversations. Nothing locks a session folder, so
two servers on the same folder would each save over the other. A server
that notices another one holding a session says so in its reply rather
than refusing, since a leftover marker from a crashed server must not
be what stops the next one from working.
