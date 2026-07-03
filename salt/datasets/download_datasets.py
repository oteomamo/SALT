"""Download + normalize LongBench into the SALT canonical JSONL schema. See README.md."""

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

# The 16 English LongBench tasks used throughout SALT (paper + eval), in the
# canonical display order shared with eval.py / the compressors.
LONGBENCH_EN = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa",
    "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
    "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p",
]

HF_REPO = "THUDM/LongBench"
HF_FILE = "data.zip"                      # the published bundle; members are data/<task>.jsonl
DEFAULT_OUT = Path(__file__).resolve().parent / "longbench" / "data"

# Canonical field order written to disk (purely cosmetic; loaders are key-based).
CANONICAL_FIELDS = ["_id", "input", "context", "answers", "length",
                    "dataset", "language", "all_classes"]


def normalize_record(rec, dataset):
    """Coerce a raw LongBench (or user) record into SALT's canonical schema.

    Reference implementation of the on-disk format: missing fields are filled
    with safe defaults so the result is always usable by the compressors /
    eval.py regardless of the source's quirks.
    """
    context = rec.get("context", "")
    answers = rec.get("answers", [])
    if isinstance(answers, str):
        answers = [answers]
    out = {
        "_id": rec.get("_id") or hashlib.sha1(context.encode("utf-8")).hexdigest(),
        "input": rec.get("input", "") or "",
        "context": context,
        "answers": list(answers),
        "length": int(rec.get("length") or len(context.split())),
        "dataset": rec.get("dataset") or dataset,
        "language": rec.get("language") or "en",
        "all_classes": rec.get("all_classes", None),
    }
    return {k: out[k] for k in CANONICAL_FIELDS}


def _iter_raw_lines(dataset, *, from_dir, zip_obj):
    """Yield raw JSON strings for one dataset, from a local dir or the HF zip."""
    if from_dir is not None:
        path = Path(from_dir) / f"{dataset}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found in --from-dir")
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield line
    else:
        member = f"data/{dataset}.jsonl"
        if member not in zip_obj.namelist():
            raise KeyError(f"{member} not in {HF_FILE} (unknown LongBench task {dataset!r})")
        for line in zip_obj.read(member).decode("utf-8").splitlines():
            if line.strip():
                yield line


def write_dataset(dataset, out_dir, *, from_dir=None, zip_obj=None):
    """Normalize one dataset and write <out_dir>/<dataset>.jsonl. Returns count."""
    out_path = Path(out_dir) / f"{dataset}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".jsonl.tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as w:
        for raw in _iter_raw_lines(dataset, from_dir=from_dir, zip_obj=zip_obj):
            rec = normalize_record(json.loads(raw), dataset)
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(out_path)               # atomic: never leave a half-written file
    return n


def _open_zip():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is required to download LongBench: "
                 "pip install huggingface_hub  (or pass --from-dir to skip the download)")
    print(f"Fetching {HF_REPO}/{HF_FILE} from the Hugging Face Hub (cached after first run)...")
    zip_path = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset")
    return zipfile.ZipFile(zip_path, "r")


def prepare(datasets, out_dir, *, from_dir=None, force=False):
    out_dir = Path(out_dir)
    todo = []
    skipped = []
    for ds in datasets:
        if not force and (out_dir / f"{ds}.jsonl").exists():
            skipped.append(ds)
        else:
            todo.append(ds)

    if skipped:
        print(f"Already present (skipped): {', '.join(skipped)}")
    if not todo:
        print(f"\nNothing to do — all {len(datasets)} requested datasets are in {out_dir}/")
        return

    zip_obj = None if from_dir is not None else _open_zip()
    try:
        print(f"\nWriting {len(todo)} dataset(s) -> {out_dir}/")
        total = 0
        for ds in todo:
            n = write_dataset(ds, out_dir, from_dir=from_dir, zip_obj=zip_obj)
            total += n
            print(f"  {ds:<22} {n:>5} records")
        print(f"\nDone: {total} records across {len(todo)} dataset(s) in {out_dir}/")
    finally:
        if zip_obj is not None:
            zip_obj.close()


def main():
    ap = argparse.ArgumentParser(
        description="Download + prepare LongBench in SALT's canonical JSONL format.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", type=str, default=None,
                    help="Comma-separated subset (default: all 16 English tasks).")
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT),
                    help=f"Output dir (default: {DEFAULT_OUT}).")
    ap.add_argument("--from-dir", type=str, default=None,
                    help="Normalize an already-extracted LongBench data/ dir "
                         "instead of downloading.")
    ap.add_argument("--force", action="store_true",
                    help="Re-create datasets even if already present.")
    ap.add_argument("--list", action="store_true",
                    help="List the tasks and what is already present, then exit.")
    args = ap.parse_args()

    requested = ([d.strip() for d in args.datasets.split(",") if d.strip()]
                 if args.datasets else list(LONGBENCH_EN))
    unknown = [d for d in requested if d not in LONGBENCH_EN]
    if unknown:
        ap.error(f"unknown LongBench task(s): {unknown}. Known: {LONGBENCH_EN}")

    out_dir = Path(args.out_dir)
    if args.list:
        print(f"Canonical out-dir: {out_dir}/\n{'task':<24} present")
        for ds in LONGBENCH_EN:
            mark = "yes" if (out_dir / f"{ds}.jsonl").exists() else "no"
            print(f"  {ds:<22} {mark}")
        return

    prepare(requested, out_dir, from_dir=args.from_dir, force=args.force)


if __name__ == "__main__":
    main()
