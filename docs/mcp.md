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
| `--read-only` | answer reads and refuse every write |

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
| `session_add_turn` | remember a message, or a whole exchange at once |
| `session_memory` | what the conversation remembers about a question |
| `salt_ingest_document` | read a document into a conversation's memory |

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

### Remembering and reading

`session_add_turn` puts something said into the conversation's memory.
It takes one `text` with a `role` of `user` or `assistant`, or an
`exchange` list of both sides in one call, which is the usual shape and
saves a round trip. The work happens on the session's own worker, in
the order it arrived. Pass `sync` to wait for it, which is worth doing
when the next thing you do is read.

`session_memory` is the read. Give it a query and it returns the
labeled memory block a chat turn would be given: excerpts grouped by
where they came from, each conversation excerpt marked with the turn
and the speaker. `budget_pct` sets how much of the conversation the
block may spend, defaulting to the session's own budget.

A read is a turn. The selection is committed the way a chat turn
commits it, so what the conversation has already surfaced shapes what
the next read surfaces. Anything submitted but not yet encoded is
finished first, so a turn added a moment earlier is never missed.

Open sessions are held warm, up to `--max-open-sessions` of them. Past
that the one used longest ago is closed, and closing writes it first,
so a conversation is never dropped with unsaved turns in it.

One server per set of conversations. Nothing locks a session folder, so
two servers on the same folder would each save over the other. A server
that notices another one holding a session says so in its reply rather
than refusing, since a leftover marker from a crashed server must not
be what stops the next one from working.

## Documents in a conversation

`salt_ingest_document` puts a file into a session's memory. Give it a
`path` and it reads the file itself, PDF or plain text, or give it
`text` you already have. Either way it is filed under a source name and
becomes its own branch of the session, so a long document and the
conversation around it never crowd each other out of the memory block.

`source_name` is what excerpts are labeled with. Only the last part of
whatever is passed is kept, so a name can never point somewhere on
disk. Ingesting twice under one name merges into that same branch, the
way attaching a file twice does at the prompt.

## A server that only reads

`--read-only` starts a server that answers questions about
conversations and changes none of them. Reads all work:
`salt_compress`, `session_list`, `session_resume`, `session_stats` and
`session_memory`. Writes refuse: `session_create`, `session_add_turn`
and `salt_ingest_document` come back as an error beginning
`read-only server:`, naming the tool and saying why, so a client can
tell a server that will not from a call that was wrong.

The tool list is the same either way. A read-only server offers all
eight and refuses three of them when called, rather than hiding them,
so what a client discovers does not depend on how the server was
started.

A memory read normally counts as a turn: the selection is committed, so
what has already surfaced shapes what surfaces next. Read-only drops
that commit, and says so in the reply with `committed: false`. Nothing
in the session folder is written at all, not even the marker a normal
server leaves to notice a second one.

This is what to point a shared or automated client at when it should
be able to look at conversations without being able to change them.

## What is not here

Some things are deliberately left out of this first version. A
conversation's turns are not written to the kv ledger the REPL keeps,
so an MCP client's traffic does not appear in a session's turn records.
There is no full-context attachment either: a document joins the memory
and is compressed with everything else, the way `salt@` works at the
prompt, rather than riding whole in every prompt like `attach@`.
