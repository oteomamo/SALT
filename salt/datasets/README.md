# Datasets

`download_longbench.py` fetches LongBench and writes it under `salt/datasets/`.
It is idempotent: existing files are skipped (`--force` rebuilds).

| script | dataset | consumed by | output dir |
|---|---|---|---|
| `download_longbench.py` | [LongBench](https://huggingface.co/datasets/THUDM/LongBench) (16 EN tasks) | `salt` / `eval.py` | `longbench/data/` |

## Usage

```bash
python download_longbench.py                       # all 16 English tasks LongBench
```

The LongBench download needs `huggingface_hub` (`pip install huggingface_hub`); `--from-dir` skips it.

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
