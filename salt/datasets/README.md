# Datasets

Each script fetches one dataset and writes it under `salt/datasets/`. All are
idempotent: existing files are skipped (`--force` rebuilds).

| script | dataset | consumed by | output dir |
|---|---|---|---|
| `download_longbench.py` | [LongBench](https://huggingface.co/datasets/THUDM/LongBench) (16 EN tasks) | `salt` / `eval.py` | `longbench/data/` |
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

## Canonical schema (LongBench)

One JSON object per line — this is the shape `salt` and `eval.py` consume:

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

## QuALITY / LooGLE record schema

`download_quality.py` writes one pretty-printed JSON file per subset (a list of
per-document records, not JSONL) consumed by `results/quality_multiturn.py`:

| field | type | notes |
|---|---|---|
| `article_id` | str | stable document id (QuALITY article id / LooGLE title) |
| `title` | str | document title |
| `source_dataset` | str | `quality` or `loogle_longdep_qa` |
| `format` | str | `mcq` (QuALITY) or `freeform` (LooGLE) |
| `article` | str | the full document text |
| `word_count` | int | whitespace word count of `article` |
| `questions` | list | per-question records, below |

Each question record: `question_unique_id`, `question`, `options` (4 strings,
MCQ only), `gold_label` (1-indexed into `options`, MCQ only), `answer`
(free-form gold, LooGLE only), `difficult` (0/1), `writer_id`, `writer_label`,
`validation`.

## NIAH record schema

`download_niah.py` writes one JSONL file per target length, consumed by
`results/niah_ttft.py`:

| field | type | notes |
|---|---|---|
| `target_length` | int | exact packed token length |
| `sample_id` | int | index within the length file |
| `prompt` | str | filler text with needle inserted and question appended |
| `token_count_built` | int | tokens at build time (== `target_length`) |
| `token_count_retokenized` | int | tokens after a decode/re-encode round trip |
| `tokenizer` | str | HF tokenizer the lengths were packed for |
| `needle_city` / `needle_number` / `needle_depth` | str / str / float | absent with `--no-needle` |

