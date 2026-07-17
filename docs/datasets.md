# 📚 Datasets

SALT evaluates on the 16 English tasks of
[LongBench](https://huggingface.co/datasets/THUDM/LongBench). If the data is not
already present, fetch and normalize it with:

```bash
python salt/datasets/download_longbench.py
```

Existing files are skipped (`--force` rebuilds, `--list` shows status). The
canonical JSONL schema and options are documented in
[`salt/datasets/README.md`](https://github.com/oteomamo/SALT/blob/main/salt/datasets/README.md).
