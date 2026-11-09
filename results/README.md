# SALT LongBench results

## 🧭 Versions

SALT now runs on vLLM 0.19.1. It moved up from vLLM 0.11.0 because the
newer models SALT uses as agents need it. The results in the EMNLP 2026
paper were produced with vLLM 0.11.0. The table below was run with vLLM
0.19.1.

## 📊 Scores

Each input is compressed to at most 20% of its tokens, and the model
answers from the compressed text. The legacy selector does the compressing,
except on the two code tasks noted below. One full run covers each of the 16
English LongBench tasks. Inputs are capped at 14000 tokens and decoding is
greedy.

| Dataset | Llama-3.1-8B-Instruct | Ministral-8B-Instruct-2410 |
|---|---|---|
| narrativeqa | 23.00 | 23.65 |
| qasper | 39.38 | 39.39 |
| multifieldqa_en | 51.99 | 49.99 |
| hotpotqa | 54.68 | 57.70 |
| 2wikimqa | 37.78 | 43.63 |
| musique | 24.90 | 31.95 |
| gov_report | 32.40 | 30.13 |
| qmsum | 23.86 | 22.89 |
| multi_news | 23.70 | 23.01 |
| trec | 74.00 | 76.50 |
| triviaqa | 82.44 | 90.38 |
| samsum | 43.46 | 44.09 |
| passage_count | 10.50 | 12.50 |
| passage_retrieval_en | 99.50 | 97.00 |
| lcc | 48.50 | 46.91 |
| repobench-p | 41.85 | 50.67 |
| **Average** | **44.50** | **46.27** |

| Category | Llama-3.1-8B-Instruct | Ministral-8B-Instruct-2410 |
|---|---|---|
| Single-Doc QA | 38.12 | 37.68 |
| Multi-Doc QA | 39.12 | 44.43 |
| Summarization | 26.65 | 25.34 |
| Few-Shot | 66.63 | 70.32 |
| Synthetic | 55.00 | 54.75 |
| Code | 45.17 | 48.79 |

Some tasks get options made for them:

- `passage_count` and `passage_retrieval_en` use `--synthetic`, which
  treats each numbered paragraph as one unit.
- `trec`, `triviaqa` and `samsum` use `--fewshot`, which keeps whole
  examples instead of pieces of them.
- `lcc` and `repobench-p` use `--code`, which reads the input as code.
  The code reader works only with the coverage selector, so these two tasks
  run with it.

General notes:

- Under vLLM 0.19, Ministral is scored with its Hugging Face tokenizer
  (`--tokenizer-mode hf`). The default there loads Mistral's own
  tokenizer, which leaves many answers empty.
- The legacy selector breaks ties in the order of Python's string hashing,
  so the script fixes `PYTHONHASHSEED=0` and every run keeps the same
  sentences.

## 🚀 Reproduce

The script runs one model at a time on one GPU, with every option above
already set. It expects SALT and vLLM to be installed. The LongBench data
comes from `python -m salt.datasets.download_longbench`, which writes it
to `salt/datasets/longbench/data` where the script reads it. The script
also runs that download on its own for any task it does not find there.

```bash
python results/run_longbench.py --model llama --gpu 0
python results/run_longbench.py --model ministral --gpu 1
```

Results land in `runs/longbench_<model>/`, with one score file per task in
`eval_parts/` and all of them together in `eval_all.json`. Other options:
`--out-dir` picks the output folder, `--datasets` runs a subset of tasks,
and `--max-samples N` takes the first N samples of each task for a quick
check.
