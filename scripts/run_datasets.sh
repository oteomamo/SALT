#!/usr/bin/env bash
# Compress every LongBench dataset that exists at the data path, then optionally
# evaluate the compressed outputs.
#
# Per-dataset routing:
#   passage_count, passage_retrieval_en  -> compress.py --synthetic  (paragraph-unit adapter)
#   trec, triviaqa, samsum               -> few-shot bypass           (automatic, no flag)
#   everything else                      -> compress.py               (standard prose;
#                                           code datasets run through the prose path too)
#
# Usage:
#   bash scripts/run_datasets.sh                             # full run, vLLM eval
#   MAX_SAMPLES=5 RUN_EVAL=0 bash scripts/run_datasets.sh    # quick smoke test, compress only
#   EVAL_BACKEND=hf bash scripts/run_datasets.sh             # score with the HF backend
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="${DATA_DIR:-$REPO_ROOT/salt/datasets/longbench/data}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/runs/run_$(date +%Y%m%d_%H%M%S)}"
BUDGET="${BUDGET:-0.20}"
GPU="${GPU:-}"                           
MAX_SAMPLES="${MAX_SAMPLES:-}"            # empty = full dataset
SALT_PY="${SALT_PY:-$(conda info --base)/envs/salt/bin/python}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
RUN_EVAL="${RUN_EVAL:-1}"
EVAL_BACKEND="${EVAL_BACKEND:-vllm}"      # vllm | hf
MAX_INPUT_LEN="${MAX_INPUT_LEN:-14000}"   # cap so the 24GB A5000 does not OOM on the 128k window

# The 16 English LongBench tasks (matches salt/datasets/download_datasets.py).
LONGBENCH_EN=(narrativeqa qasper multifieldqa_en hotpotqa 2wikimqa musique
              gov_report qmsum multi_news trec triviaqa samsum
              passage_count passage_retrieval_en lcc repobench-p)
SYNTH_SET=" passage_count passage_retrieval_en "

[ -d "$DATA_DIR" ] || { echo "ERROR: data dir not found: $DATA_DIR" >&2
  echo "Build it with: $SALT_PY salt/datasets/download_datasets.py" >&2; exit 1; }
[ -x "$SALT_PY" ] || { echo "ERROR: salt python not found: $SALT_PY (run scripts/setup_env.sh)" >&2; exit 1; }

mkdir -p "$OUT_DIR"
echo "data:   $DATA_DIR"
echo "output: $OUT_DIR"
echo "budget: $BUDGET   gpu: $GPU   max_samples: ${MAX_SAMPLES:-full}"
echo

samples_flag=()
[ -n "$MAX_SAMPLES" ] && samples_flag=(--max-samples "$MAX_SAMPLES")

n_run=0; n_skip=0
for ds in "${LONGBENCH_EN[@]}"; do
  data="$DATA_DIR/$ds.jsonl"
  if [ ! -f "$data" ]; then
    echo "-- skip $ds (not at path)"; n_skip=$((n_skip + 1)); continue
  fi
  if [[ "$SYNTH_SET" == *" $ds "* ]]; then flags=(--synthetic); else flags=(); fi

  echo "== $ds  (${flags[*]:-prose}) =="
  CUDA_VISIBLE_DEVICES="$GPU" "$SALT_PY" compress.py \
    --data "$data" --output "$OUT_DIR/$ds.jsonl" \
    --device cuda --token-budget-pct "$BUDGET" "${flags[@]}" "${samples_flag[@]}"
  n_run=$((n_run + 1))
done

echo
echo "compressed $n_run dataset(s), skipped $n_skip  ->  $OUT_DIR"

if [ "$RUN_EVAL" = "1" ] && [ "$n_run" -gt 0 ]; then
  if [ "$EVAL_BACKEND" = vllm ] && ! "$SALT_PY" -c "import vllm" 2>/dev/null; then
    echo "ERROR: --backend vllm needs vLLM in the salt env." >&2
    echo "  install it:  WITH_VLLM=1 bash scripts/setup_env.sh   (or pip install vllm==0.11.0)" >&2
    echo "  or score with the HF backend:  EVAL_BACKEND=hf bash scripts/run_datasets.sh" >&2
    exit 1
  fi
  echo
  echo "== eval ($EVAL_BACKEND, $MODEL) =="
  "$SALT_PY" eval.py --backend "$EVAL_BACKEND" \
    --data-dir "$OUT_DIR" --model "$MODEL" --gpu "$GPU" --max-input-len "$MAX_INPUT_LEN"
  echo "scores -> $OUT_DIR/eval_all.json"
else
  echo "eval skipped (RUN_EVAL=$RUN_EVAL). To score later:"
  echo "  $SALT_PY eval.py --backend $EVAL_BACKEND --data-dir $OUT_DIR --model $MODEL --gpu $GPU --max-input-len $MAX_INPUT_LEN"
fi
