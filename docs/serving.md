# 🔌 Serving

The in-process backends load the chat model inside the `saltChat`
process, so the model and its cache vanish on exit. `saltServe` runs
the model as its own long-lived server instead: the model stays loaded,
the prefix cache stays warm, and chats connect, disconnect and resume
freely.

## Start a server

```bash
saltServe qwen05 --gpu 1
```

The command resolves the same model registry `saltChat` uses, prints
the exact `vllm serve` invocation it runs, and serves an OpenAI
compatible API on `http://127.0.0.1:8000`. Ctrl-C stops it. One server
serves one model. Anything after `--` passes to `vllm serve` unchanged,
and `--vllm-bin` can point at another environment's vllm, so the server
runs whichever vLLM release fits the hardware while the SALT install
stays put.

## Connect a chat

```bash
saltChat --model qwen05 --backend vllm-serve
```

The prompt is rendered and tokenized in `saltChat` itself, so the text
the server caches is exactly the text the kv ledger counts. Exit, come
back later, resume the conversation by its id: the model is still
loaded and the stable prompt head is still cached, so the first turn
only prefills what changed. `/stats` shows the measured cache reuse
every turn. `--server-url` reaches a server on another port or machine.

## Several cards

```bash
saltServe llama-3.1-8b-instruct --gpu 0,1
```

A model too big for one card tensor parallels across the list, each
card capped at `0.80` of its memory by default so the machine keeps
headroom. Give the connecting `saltChat` the same list: it pins the
same card order and puts its encoder on the last card, inside that
headroom. Cards older than Ampere are served in `float16`, which they
require.

Every flag for both sides is on the [Options](options.md) page.
