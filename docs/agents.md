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
`salt/agents/roster_sample.json`, ready to copy.

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
    "gpu": 1,
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
