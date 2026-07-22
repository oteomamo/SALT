# 🧰 scripts

Development helper scripts for working on SALT. None of these are part of
the installed `saltChat` runtime, which lives entirely under `salt/`.
Nothing here is imported or run when you use SALT. They exist for
contributors verifying a change and for reproducing the evaluation.

## 🧪 Regression tests

Fast, self-contained checks. Run the one that matches the area you touched
before opening a pull request (CONTRIBUTING.md maps each area to its
script). Each skips cleanly when an optional dependency is missing, so a
plain install stays green.

- `chat_ingest_regression.py` covers per-turn ingestion, the session
  trie, and the chat CLI.
- `chat_theme_regression.py` covers the theme-coverage selection engine.
- `chat_dedup_regression.py` covers the near-duplicate memory gate
  (`--dedup-cos`).
- `chat_pdf_regression.py` covers PDF and text ingestion.
- `chat_textclean_regression.py` covers chat-side text handling (what
  conversation memory stores verbatim).
- `chat_keystab_regression.py` covers cross-turn coverage-key stability.
- `chat_evict_regression.py` covers the bounded-session mask
  (`--max-sentences`).
- `chat_incremental_regression.py` covers the per-turn work the session
  trie carries forward instead of redoing.
- `chat_vllm_regression.py` covers the in-process `--backend vllm`.
- `chat_serve_regression.py` covers persistent serving (`saltServe` and
  `--backend vllm-serve`), including its multi-GPU command construction.

## 📦 Utilities

- `verify.sh` runs the regression suites for one area in one command
  (`bash scripts/verify.sh chat`, or `all`). Start here.
- `setup_env.sh` creates the `salt` conda environment and installs the
  dependencies.
- `run_datasets.sh` compresses the LongBench datasets and can then
  evaluate the compressed outputs.
- `longbench_categories.py` folds a LongBench result file into the
  standard per-category summary row.
