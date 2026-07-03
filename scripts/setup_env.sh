#!/usr/bin/env bash
# Minimal environment setup for SALT.
#
# Creates (or reuses) two conda environments — both Python 3.10, since SALT uses
# PEP 604 `X | None` unions:
#
#   $SALT_ENV  (default: salt)   compress.py + eval.py --backend hf
#   $VLLM_ENV  (default: vllm)   eval.py --backend vllm
#
# Existing envs are reused; deps already satisfied are left untouched (plain
# `pip install`, no -U), so this is safe to re-run.
#
# Usage:
#   bash scripts/setup_env.sh                   # both envs
#   SALT_ENV=salt-core bash scripts/setup_env.sh
#   SKIP_VLLM=1 bash scripts/setup_env.sh       # core + HF only, no vLLM
#
# The eval model (meta-llama/Llama-3.1-8B-Instruct) is gated: run
#   hf auth login            (or export HF_TOKEN=...)
# before evaluating. `pip install torch` pulls the default CUDA wheel; for a
# specific CUDA build, install torch yourself first, then re-run this script.
set -euo pipefail

SALT_ENV="${SALT_ENV:-salt}"
VLLM_ENV="${VLLM_ENV:-vllm}"
PYVER="${PYVER:-3.10}"
SKIP_VLLM="${SKIP_VLLM:-0}"

source "$(conda info --base)/etc/profile.d/conda.sh"

# Shared by both envs: compressor runtime + eval metrics + dataset tool.
#   torch/transformers/numpy  -> BGE encoder, sentence pipeline
#   tiktoken                  -> token-accurate budget logging (optional but used)
#   huggingface_hub + click   -> model + LongBench dataset download, `hf` CLI
#   rouge/fuzzywuzzy/python-Levenshtein/jieba -> eval.py metrics (metrics.py)
CORE_PKGS=(torch transformers numpy tiktoken huggingface_hub click
           rouge fuzzywuzzy python-Levenshtein jieba)

ensure_env () {   # $1 = env name
  if conda env list | awk '{print $1}' | grep -qx "$1"; then
    echo ">> env '$1' exists — ensuring deps"
  else
    echo ">> creating env '$1' (python $PYVER)"
    conda create -y -n "$1" "python=$PYVER"
  fi
}

echo "=== [1/2] SALT core + HF backend -> env '$SALT_ENV' ==="
ensure_env "$SALT_ENV"
conda run -n "$SALT_ENV" python -m pip install "${CORE_PKGS[@]}"

if [ "$SKIP_VLLM" != "1" ]; then
  echo "=== [2/2] vLLM eval backend -> env '$VLLM_ENV' ==="
  ensure_env "$VLLM_ENV"
  # vllm ships its own torch build; install it first so it wins the torch pin.
  conda run -n "$VLLM_ENV" python -m pip install vllm
  conda run -n "$VLLM_ENV" python -m pip install \
    transformers numpy huggingface_hub rouge fuzzywuzzy python-Levenshtein jieba
else
  echo "=== [2/2] vLLM env skipped (SKIP_VLLM=1) ==="
fi

echo
echo "Done."
echo "  compress:  CUDA_VISIBLE_DEVICES=1 conda run -n $SALT_ENV python compress.py --data DATA.jsonl --output OUT.jsonl --device cuda --token-budget-pct 0.20"
echo "  eval hf:   conda run -n $SALT_ENV python eval.py --backend hf   --data-dir OUT_DIR --model meta-llama/Llama-3.1-8B-Instruct --gpu 1"
echo "  eval vllm: conda run -n $VLLM_ENV python eval.py --backend vllm --data-dir OUT_DIR --model meta-llama/Llama-3.1-8B-Instruct --gpu 1 --max-input-len 14000"
echo "  datasets:  conda run -n $SALT_ENV python salt/datasets/download_datasets.py --list"
