# Datasets

`download_datasets.py` fetches [LongBench](https://huggingface.co/datasets/THUDM/LongBench)
and normalizes it to SALT's canonical JSONL — one `<task>.jsonl` per dataset in
`longbench/data/`. Idempotent: existing files are skipped (`--force` rebuilds).

## Usage

```bash
python download_datasets.py                        # all 16 English tasks LongBench
```

The download path needs `huggingface_hub` (`pip install huggingface_hub`); `--from-dir` skips it.

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

