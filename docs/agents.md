# 🤝 Agents

A chat can name other models beside the one it talks to. A roster file
lists them, `saltChat --roster` loads it, and the session keeps a handle
on each one. Nothing is contacted until you ask, and a session that
never asks behaves exactly as it did before. This page covers the
roster itself. What a session does with a worker is being built over
the 2.10 releases.

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

## One server, one model

A vLLM server holds a single model, so reaching a second model means a
second server. That is the whole reason a roster exists: rather than
unloading the chat model to borrow another one, the session talks to a
model that is already loaded somewhere else, and its own model and warm
cache are never disturbed. Two servers on one card need an explicit
memory split, so give each worker its own card where you can.

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

Every flag is on the [Options](options.md) page.
