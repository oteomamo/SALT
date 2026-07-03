# Datasets

Each script fetches one dataset and writes it under `salt/datasets/`. All are
idempotent: existing files are skipped (`--force` rebuilds).

| script | dataset | consumed by | output dir |
|---|---|---|---|
| `download_longbench.py` | [LongBench](https://huggingface.co/datasets/THUDM/LongBench) (16 EN tasks) | `compress.py` / `eval.py` | `longbench/data/` |
| `download_quality.py` | [QuALITY](https://github.com/nyu-mll/quality) (MCQ) / [LooGLE](https://huggingface.co/datasets/bigai-nlco/LooGLE) (free-form) | `results/quality_multiturn.py` | `quality/`, `loogle/` |
| `download_niah.py` | Needle-in-a-haystack from [PG-19](https://huggingface.co/datasets/emozilla/pg19) | `results/niah_ttft.py` | `niah/` |

## Usage

```bash
python download_longbench.py                       # all 16 English tasks LongBench
python download_quality.py                         # QuALITY subset (--dataset loogle for LooGLE)
python download_niah.py --tokenizer meta-llama/Llama-3.1-8B-Instruct
```

The LongBench download needs `huggingface_hub` (`pip install huggingface_hub`); `--from-dir` skips it.
`download_niah.py` needs `datasets` + `transformers` and packs prompts to exact token lengths for the
given `--tokenizer` — match it to your eval LLM.

## Canonical schema

One JSON object per line — every SALT tool reads exactly these fields:

| field | type | notes |
|---|---|---|
| `_id` | str | stable sample id |
| `input` | str | query/instruction (`""` for context-only summarization) |
| `context` | str | the long document to compress |
| `answers` | list[str] | gold answers |
| `length` | int | reported token length |
| `dataset` | str | task name, e.g. `gov_report` |
| `language` | str | `en` |
| `all_classes` | list \| null | label set for classification (`trec`), else null |

