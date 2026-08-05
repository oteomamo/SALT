# 🤝 Agents

A chat can name other models beside the one it talks to. A roster file
lists them, `saltChat --roster` loads it, and the session keeps a handle
on each one. Nothing is contacted until you ask, and a session that
never asks behaves exactly as it did before. This page covers the
roster and the workers it starts. What a session does with a worker
once it is up is being built over the 2.10 releases.

## The roster file

```json
{
  "version": "salt-roster/1",
  "models": [
    {
      "name": "qwen05",
      "alias": "qwen05",
      "role": "worker",
      "server_url": "http://127.0.0.1:8081",
      "max_tokens": 512,
      "notes": "start the server first: saltServe qwen05 --port 8081"
    }
  ]
}
```

`name` is what you type in the REPL, `alias` is the registered model it
resolves to, and `role` is `worker` or `orchestrator`. A roster may name
one orchestrator at most. Loading validates every entry up front, so a
bad port, an unknown alias or a model whose weights are missing stops
the launch instead of failing later. This file ships as
`salt/agents/roster_sample.json`, ready to copy, with one entry of each
mode in it.

## Attach or spawn

Each entry either attaches to a server that is already running or
describes one to start. An attach entry carries `server_url` and points
at a [saltServe](serving.md) you started yourself, which is the simplest
way to begin. A spawn entry carries `spawn` instead, with the port, card
and window the session should launch it on. Exactly one of the two is
required, so an entry is never ambiguous about who owns the process.

```json
{
  "name": "helper",
  "alias": "qwen05",
  "role": "worker",
  "spawn": {
    "port": "auto",
    "gpu": "1",
    "gpu_mem_util": 0.3,
    "max_model_len": 8192,
    "ready_timeout": 180
  },
  "timeout_s": 300
}
```

`port` may be `auto`, which picks a free one. The rest is optional and
falls through to whatever `saltServe` would choose on its own.

## One server, one model

A vLLM server holds a single model, so reaching a second model means a
second server. That is the whole reason a roster exists: rather than
unloading the chat model to borrow another one, the session talks to a
model that is already loaded somewhere else, and its own model and warm
cache are never disturbed.

## Starting and stopping

```
you> /worker start helper
  helper: starting qwen05 at http://127.0.0.1:41225, up to 180s ....
  helper: ready, serving qwen05, window 8192 tokens
```

`/worker start <name>` launches the server for a spawn entry and waits
for it to answer. A cold start loads weights and builds a graph, so
minutes is normal and the dots mark the wait. `/worker start --all` does
the same for every spawn entry in the file, and `--workers-autostart`
runs that once the chat model is loaded. Attach entries are skipped
everywhere, because this session did not start those servers and so
never starts or stops them.

`/worker stop <name>` sends SIGTERM, waits, and only then SIGKILLs, so
the engine gets its chance to shut down cleanly. A worker answering a
call right now refuses to stop until that call finishes. Leaving the
REPL stops every worker the session started, whether or not you stopped
them by hand.

Each spawned worker writes to `<session>/workers/<name>.log`, with a
small `<name>.json` beside it holding its pid and port. A restart
appends to the log, so the crash that explains it is still there to
read, and the record is removed on a clean stop, so a record left on
disk always means a worker still running. When a server dies during
startup the last lines of its log come with the error, which is usually
enough to see why.

## Which card a worker lands on

A spawn onto a card that already carries the chat model or another
worker is refused, unless both sides declare a `gpu_mem_util` share.
Two servers each assuming they may take most of a card is the ordinary
way to run out of memory at load, and the refusal comes before anything
is launched. When the declared shares on one card add up past 0.95 the
start still runs and prints the total as a warning. The card holding the
BGE encoder is fine to share and says so, since the encoder needs about
130 MB. An entry naming no card at all is allowed with a note, because
then the child chooses for itself.

## When a worker is given up on

A worker that goes quiet mid-reply is given up on after `timeout_s`, 300
seconds by default, and the response is severed so the worker stops
generating too. It is not held against the worker: the next call goes to
the same one. This is where a worker differs from the chat model, which
is waited on for as long as it takes, because there the session has
nothing else to fall back to.

A refused connection is retried once, and only before any text has
arrived, since asking again for a half written reply would duplicate it.
Two failed calls in a row mark the worker `DEAD` and the session stops
sending work there. `/worker probe <name>` reconnects it and clears the
count.

## Handing over a task

```
you> /offload summarize the sizing argument for the installer
  delegating to qwen05 (qwen05), 9 sentences of context [214 words] ...
The house has 9 kW of panels against a 5 kW inverter, and the worst case
is a winter evening drawing for about 4 hours ...
  [qwen05] ok, 486 in, 92 out, 2.4s
```

`/offload <task>` sends one task to a worker together with this
conversation's memory, selected for that task the same way a chat turn
selects it. The worker answers from what it was given and its reply is
printed as it came back. With one worker in the roster it needs no
naming, and with several `/offload @NAME <task>` picks the one.

The session's own memory does not change. Delegating selects context but
commits nothing, so the coverage state, the verbatim tail and the trie
are the same after a delegation as before it, and the reply is not
remembered as part of the conversation.

Start saltChat with `--offload-ingest` to remember the answers instead. A
worker's reply then enters this session's memory as a turn of its own,
kept apart from what you and the chat model said, so a later question can
select it the way it selects anything else. It never joins the recent
messages shown word for word, and a delegation that failed leaves nothing
behind. This is off by default because a worker's prose competes with
your own conversation for the same memory budget, which is worth turning
on deliberately rather than by accident.

Ctrl-C during a delegation cuts the connection, which aborts the request
on the worker rather than leaving it generating into nothing, and returns
you to the prompt with the worker ready for the next task. Whatever the
worker had said by then is still shown.

A delegation ends in one of five ways, and the status line says which:
`ok` when the worker answered, `timeout` when it went quiet mid-reply and
is still usable, `dead` when it has stopped answering at all, `aborted`
when you interrupted it, and `error` for everything else, including a
server that refused the request. Only `ok` leaves anything behind. A
delegation that ended any other way never touches the conversation's
memory, even with `--offload-ingest` on, because there is no answer to
remember.

Every delegation is written to `delegations.jsonl` in the session folder,
one line each: which worker it went to, the task, how much context it was
given, how it ended and what it cost. The answer itself is not kept there,
because it was printed and the conversation did not remember it.
Numbering carries on from the file, so a delegation in a resumed session
never reuses an earlier number. A line left half written by a crash is
reported and skipped the next time the session opens, and the rest of the
history still loads.

How much a worker is handed is bounded twice. The memory budget sizes
the selection the way it sizes a chat turn's, and `--offload-context-cap
N` puts a word ceiling under that, which is how you keep a small
worker's prompt short without shrinking what the chat model gets. If the
result still will not fit the worker's window, the front of the context
is dropped until it does and a note says how much was kept. The task
itself is never trimmed, because a worker that lost its task would
answer the wrong question with confidence.

Two more knobs bound the same exchange from the other side.
`--offload-budget-pct` sets the memory budget a delegation selects
under, so a worker can be handed more or less of the conversation than a
chat turn gets, and `--offload-timeout` says how long to wait on a
worker that goes quiet. A roster entry that names a timeout of its own
keeps it, because how long a model takes is a fact about that model
rather than about the session asking.

## Delegating from a script

A scripted run can delegate as well as talk. An item in a `--turns` file
shaped like this goes to a worker instead of the chat model, in the
middle of the same conversation:

```json
[
  "What did we decide about the inverter?",
  {"offload": {"task": "Size the battery bank from that decision",
               "target": "qwen05", "ingest": true}}
]
```

The task is handed over exactly as `/offload` hands it over, so the
memory it is given, the ledger line it leaves and the labels on a
remembered answer are all the same. `target` picks the worker and can be
left out when the roster has only one, and `ingest` decides whether the
answer is remembered, defaulting to whatever the session was launched
with. See [scripted turns](chatbot.md#scripted-turns) for the file
format and what `--turns-out` writes for a delegated row.

A short mixed conversation ships as `salt/agents/demo_turns.json`, three
questions to the chat model and two tasks to the worker the sample
roster names. With a worker up, it runs end to end:

```
saltChat --roster salt/agents/roster_sample.json \
         --turns salt/agents/demo_turns.json --conversation-id demo
```

`/stats` adds a delegation line once a session has handed something over:
how many went out in all, and per worker the calls, how many came back,
the tokens each way and the mean time one took. The totals are read back
from the ledger when the session opens, so a resumed conversation carries
on counting instead of starting again at zero.

## Coming back to a session

Reopening a session picks its delegation history up where it stopped.
The numbering carries on, the `/stats` totals continue, and a remembered
answer is still labeled with the worker that gave it.

Workers are not restarted. A worker is a server holding a GPU, so
bringing one back is a decision to make out loud with `/worker start`,
or at launch with `--workers-autostart`. What reopening does tell you is
which servers from the earlier run are still up, and which of them left
a record behind after the process was gone. A record nobody can honour
is archived rather than believed, because trusting it would point the
session at a port nothing is serving. The worker's log is kept either
way, so the reason it went is still on disk.

## Seeing what is there

```
you> /roster probe
  NAME    ROLE    ALIAS   MODE    ENDPOINT               STATE
  qwen05  worker  qwen05  attach  http://127.0.0.1:8081  PROBED
      serving qwen05, window 32768 tokens
```

`/roster` lists what the file declared and `/roster probe` asks every
endpoint what it is actually serving, so a typo or a server holding the
wrong model shows up as `DEAD` with the reason. `/worker` reports the
live side of the same models, including how many calls each has taken
and how slow it was, and `/worker probe <name>` reconnects one of them.

The commands sit alongside the rest on the
[Chatbot mode](chatbot.md) page, and every flag is on the
[Options](options.md) page.
