#!/usr/bin/env bash
# Environment setup for SALT — a single conda env named `salt` (Python 3.10,
# required for PEP 604 `X | None` unions) that runs everything: compression,
# dataset download, and both eval backends (HF, and optionally vLLM).
#
# Dependencies come from requirements.txt plus the editable install, so this
# script stays in sync with the package and is safe to re-run (an existing env
# is reused, nothing is force-upgraded).
#
# Usage:
#   bash scripts/setup_env.sh                # salt env: deps + editable install
#   WITH_VLLM=1 bash scripts/setup_env.sh    # also install the vLLM eval backend
#   SALT_ENV=my-salt bash scripts/setup_env.sh
#
# The eval model (meta-llama/Llama-3.1-8B-Instruct) is gated: run
#   hf auth login            (or export HF_TOKEN=...)
# before evaluating. requirements.txt pins a CUDA torch wheel; for a different
# CUDA build install torch yourself first, then re-run this script.
set -euo pipefail

SALT_ENV="${SALT_ENV:-salt}"
PYVER="${PYVER:-3.10}"
WITH_VLLM="${WITH_VLLM:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$SALT_ENV"; then
  echo ">> env '$SALT_ENV' exists — ensuring deps"
else
  echo ">> creating env '$SALT_ENV' (python $PYVER)"
  conda create -y -n "$SALT_ENV" "python=$PYVER"
fi

echo ">> installing SALT (requirements.txt + editable package)"
conda run -n "$SALT_ENV" python -m pip install -r "$REPO_ROOT/requirements.txt"
conda run -n "$SALT_ENV" python -m pip install -e "$REPO_ROOT"

if [ "$WITH_VLLM" = "1" ]; then
  echo ">> installing vLLM eval backend into '$SALT_ENV'"
  conda run -n "$SALT_ENV" python -m pip install "vllm==0.11.0"
fi

echo
echo "Done. Next:"
echo "  conda activate $SALT_ENV"
echo "  hf auth login                              # gated eval model"
echo "  python salt/datasets/download_longbench.py # fetch LongBench"
echo "  bash scripts/run_datasets.sh               # compress + evaluate"
