# Model registry

Chat models registered for `saltChat` live here, one directory per model:

```text
salt/models/<alias>/
  config.json   loading + generation settings
  weights       symlink to the snapshot in your HuggingFace cache
  tokenizer     only when needed, see below
```

Weights are never copied: registering downloads through
`huggingface_hub.snapshot_download` (reusing `~/.cache/huggingface/hub`) and
symlinks the snapshot, so a model already on disk registers instantly and
deleting an entry here never touches the cache. The directory scan is the
registry - there is no index file.

Register a model:

```bash
saltChat --add meta-llama/Llama-3.1-8B-Instruct
saltChat --add Qwen/Qwen2.5-0.5B-Instruct --alias qwen05
saltChat --list
```

Before the download, `--add` fetches the model's own `config.json` and
prints a warning when this environment cannot load that model, then
registers it anyway.

When the installed transformers builds a model's tokenizer differently
from the model's own `tokenizer.json`, saltChat writes a `tokenizer`
folder into the entry that hands vLLM that file instead. Removing the
entry removes the folder too.

`config.json` schema:

```json
{
  "alias": "llama-3.1-8b-instruct",
  "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
  "dtype": "bfloat16",
  "attn_implementation": "sdpa",
  "gen": {"max_new_tokens": 512, "temperature": 0.7, "do_sample": true, "top_p": 0.9},
  "registered_at": "2026-07-03T12:00:00"
}
```

The input length is not configurable here: `saltChat` reads the model's own
context window from its config and truncates only past that ceiling (minus
reply headroom).

Everything in this directory except this README is gitignored.
