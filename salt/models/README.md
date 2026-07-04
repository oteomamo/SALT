# Model registry

Chat models registered for `saltChat` live here, one directory per model:

```text
salt/models/<alias>/
  config.json   loading + generation settings
  weights       symlink to the snapshot in your HuggingFace cache
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

`config.json` schema:

```json
{
  "alias": "llama-3.1-8b-instruct",
  "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
  "dtype": "bfloat16",
  "max_input_len": 8192,
  "attn_implementation": "sdpa",
  "gen": {"max_new_tokens": 512, "temperature": 0.7, "do_sample": true, "top_p": 0.9},
  "registered_at": "2026-07-03T12:00:00"
}
```

Everything in this directory except this README is gitignored.
