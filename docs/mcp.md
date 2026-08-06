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

Where that file lives depends on the client.

**Claude Code** takes it from the command line:

```bash
claude mcp add salt -- salt-mcp --gpu 0
```

Add `--scope project` to write it into a `.mcp.json` beside the code,
which is what you want when the conversations belong to the project
rather than to you.

**Claude Desktop** reads `claude_desktop_config.json`. Open its
settings, edit the config, and put the block above in it. The app
starts the server when it starts and stops it when it quits.

**Cursor** reads `~/.cursor/mcp.json` for every project, or
`.cursor/mcp.json` inside one project. Same block either way.

Anything else that speaks MCP works the same way: it needs a command to
run, and `salt-mcp` is on the path once the extra is installed. Use the
full path to the `salt-mcp` in your environment if the client does not
inherit your shell.

## How the server runs

One process, started by the client and living as long as it does. It
holds the encoder in memory once loaded, and loads it on the first call
that needs it, so a client that connects and asks for nothing pays
nothing.

Conversations are opened on demand and kept warm, because a client that
adds a turn is usually about to read memory. Past
`--max-open-sessions` the one used longest ago is closed, and closing
means finishing what it was still encoding, writing it if it changed,
and then letting go.

One client per server, and one server per folder of conversations.
Nothing locks a session folder, so two servers over the same folder
would each save over the other. A server that sees signs of another one
holding a conversation says so in the reply rather than refusing.

No GPU is required. On a machine without one the encoder runs on the
CPU, which is slower per call and needs nothing installed beyond the
package. `--gpu`, `--device` and `--bge-device` put it on a card when
there is one.

## Server flags

| Flag | Effect |
|---|---|
| `--version` | print the SALT version this server carries and exit |
| `--device` | device for the encoder (default: `cuda` when there is one, else `cpu`) |
| `--gpu` | CUDA index or comma list, with the encoder on the last card |
| `--bge-device` | device for the encoder, winning over `--device` |
| `--sessions-dir` | where conversations live (default: the folder saltChat uses) |
| `--max-open-sessions` | how many conversations stay open at once (default: 8) |
| `--max-ingest-chars` | longest text one call may carry (default: 400000) |
| `--roster` | roster of helper models this server may delegate to |
| `--read-only` | answer reads and refuse every write |

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
| `salt_contract` | which version of this tool contract the server speaks |
| `roster_list` | the helper models this server can reach |
| `salt_switches` | the memory switches and what each one is set to |
| `salt_delegate` | hand one task to a helper model |

`salt_contract` answers the question a client asks first: which surface
am I talking to. It returns the contract number, the SALT version, and
every tool this server offers, in order.

The surface grows one way only. Tool names are forever, a renamed tool
being a break every client feels silently, and schemas grow additively:
a new argument is optional with a default, a new response field is
added beside the others, and an existing field is never repurposed. The
contract number moves only if that promise is ever broken, so a client
that reads a `1` knows every tool it learned still means what it meant.
The version in the handshake is the SALT version, so a client log
records which build it spoke to.

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
resumed at the prompt and the other way round. What the prompt does
with them is on the [Chatbot mode](chatbot.md) page.

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

Anything a client should know about a conversation that is not a
refusal comes back in a `warnings` list on the reply: another server
looking like it holds this one, or a repair the open had to make. A
conversation whose files disagree after a crash is rolled back to its
last complete state as it opens, the dropped text is kept in
`load_repairs.jsonl` beside it, and the reply says what was dropped.

However the server ends, it ends the same way. A client hanging up, a
Ctrl-C or a kill all drain every open conversation and write the ones
that changed before the process stops.

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

## Reading a conversation as numbers

`session_stats` carries a `snapshot` block: a flat set of signals
describing the conversation as it stands. How many sentences it holds
and how many are still selectable, how many turns, how many words are
live, how many files are attached and how much of the memory they
account for, how old the conversation is, and what its last read
measured, including how far the question drifted from the recent
conversation and how much of the coverage table no longer matches
anything live.

Every value is a number, a boolean or nothing at all, and nothing at
all means this conversation cannot say rather than zero. A conversation
held here has no chat model and no verbatim tail, so the signals about
those come back empty.

`salt_switches` is the other half. It lists the memory switches, what
this server has each one set to, what it ships as, and which number in
the snapshot or in a compression's own statistics reports whether it
did anything. The switches are read-only over MCP: a client can see how
memory is being selected and measure the result, and the decision to
change it stays where the session is run.

Together they are a loop an agent can close on its own: read the
numbers, see which switch addresses them, and measure whether it
helped.

## Helper models

Start the server with `--roster FILE` and it can hand work to the
smaller models that file names. The roster is the same one `saltChat`
takes, described on the [agents](agents.md) page, and the models in it
are servers of their own that are already running.

`roster_list` returns what is in the roster: each model's name, role,
alias, whether it is attached or spawned, where it lives and how many
calls it has taken. Pass `probe` and each endpoint is contacted for the
model it is actually serving, which is the only way to find out that a
declared helper is not there.

`salt_delegate` hands one task over and waits for the whole answer.

| Argument | Meaning |
|---|---|
| `task` | what the helper is asked to do (required) |
| `conversation_id` | whose memory to send with it, optional |
| `target` | which helper, needed only when the roster names several |
| `context_query` | what to select the memory for, when the task itself is a poor search line |
| `budget_pct` | how much of the conversation the context may spend |
| `ingest` | keep the answer as a turn of the conversation |

With a `conversation_id` the task travels with that conversation's
memory, selected for the task the way a chat turn selects it. Selecting
it changes nothing: the conversation is the same after a delegation as
before it, which is what makes it safe to ask several helpers the same
question. Without one it is a task on its own, and the helper gets no
context at all.

The answer comes back with what it cost and how much context it went
out with. Every delegation under a conversation is also filed in
`delegations.jsonl` beside that conversation, so what was asked, of
whom, and how it ended is a record rather than something only the
client saw.

`ingest` is the exception to changing nothing. With it, an answer is
remembered as a turn of its own, headed with the helper it came from
rather than as something the conversation said. It is off by default:
a helper's prose in memory is a decision, not a side effect.

## When a call is refused

Every refusal opens with a fixed phrase saying what kind it is, so a
client can branch on the kind and a person can still read the sentence.

| Kind | The reason begins | When |
|---|---|---|
| invalid argument | `invalid argument:` | an argument is missing, empty or out of range |
| invalid session id | `invalid session id:` | the id is not one a conversation may be called |
| not found | `no such conversation:` | the conversation, or a file, is not there |
| too large | `too large:` | a text is past `--max-ingest-chars` |
| read only | `read-only server:` | the call would write and the server may not |
| no roster | `no roster:` | a helper was asked for and none is loaded |
| worker failed | `worker failed:` | the delegation could not be sent at all |
| failed | `the call failed:` | anything else, with the fault named |

A helper that answers badly is not a refusal. A delegation that timed
out or reached a model that is not there comes back as a normal result
with its status saying so, because the call itself was fine.

Nothing else reaches the client. An unexpected fault is reported as the
last kind with its type named, never as a traceback.

## A server that only reads

`--read-only` starts a server that answers questions about
conversations and changes none of them. Reads all work:
`salt_compress`, `session_list`, `session_resume`, `session_stats`,
`session_memory`, `roster_list` and `salt_delegate` itself. Writes
refuse: `session_create`, `session_add_turn`, `salt_ingest_document`
and a delegation asking for `ingest` come back as a refusal whose
reason begins `read-only server:`, naming the tool and saying why, so a
client can tell a server that will not from a call that was wrong.

The tool list is the same either way. A read-only server offers every
tool and refuses the writing ones when called, rather than hiding them,
so what a client discovers does not depend on how the server was
started. A delegation still runs, since asking a helper a question
reads the conversation without moving it, but nothing about it is
written down.

A memory read normally counts as a turn: the selection is committed, so
what has already surfaced shapes what surfaces next. Read-only drops
that commit, and says so in the reply with `committed: false`. Nothing
in the session folder is written at all, not even the marker a normal
server leaves to notice a second one.

This is what to point a shared or automated client at when it should
be able to look at conversations without being able to change them.

A useful pair: run your own work through `saltChat` or a normal server,
and give anything automated a read-only one over the same folder.

```json
{
  "mcpServers": {
    "salt-readonly": {
      "command": "salt-mcp",
      "args": ["--read-only", "--sessions-dir", "/path/to/conversations"]
    }
  }
}
```

It can list the conversations, read what any of them remembers about a
question, report their numbers and ask a helper model something, and it
cannot add a turn, ingest a document or leave a mark on the folder.

## What is not here

Some things are deliberately left out of this first version. A
conversation's turns are not written to the kv ledger the REPL keeps,
so an MCP client's traffic does not appear in a session's turn records.
There is no full-context attachment either: a document joins the memory
and is compressed with everything else, the way `salt@` works at the
prompt, rather than riding whole in every prompt like `attach@`.
