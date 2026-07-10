"""Download + build the QuALITY / LooGLE reading-comprehension subsets used by
the multi-turn SALT experiment (`salt/results/quality_multiturn.py`). See README.md.

Two source datasets share one builder; select with `--dataset`:

  quality (default)
    - Source: github.com/nyu-mll/quality (v1.0.1 htmlstripped dev)
    - Format: MCQ, 4 options, integer gold_label in {1,2,3,4}.
    - The two writer-rows per article are merged into one document.

  loogle
    - Source: huggingface.co/datasets/bigai-nlco/LooGLE (data/longdep_qa.jsonl)
    - Format: free-form, single string answer, grouped by `title`.

Both are written as a single JSON file: a list of per-document records with the
schema documented in README.md ("QuALITY / LooGLE record schema"). Idempotent:
an existing subset is skipped (`--force` rebuilds). The raw source download is
cached next to it.
"""

import argparse
import json
import os
import urllib.request
from collections import defaultdict
from pathlib import Path

# ---- QuALITY ----
QUALITY_URL = ("https://raw.githubusercontent.com/nyu-mll/quality/main/"
               "data/v1.0.1/QuALITY.v1.0.1.htmlstripped.dev")

# ---- LooGLE longdep_qa ----
# Direct file in the HF repo; the resolve URL bypasses the datasets lib for a
# plain JSONL download.
LOOGLE_URL = ("https://huggingface.co/datasets/bigai-nlco/LooGLE/"
              "resolve/main/data/longdep_qa.jsonl")

# One output dir per dataset, alongside longbench/ under salt/datasets/.
DATASETS = {
    "quality": {
        "url": QUALITY_URL,
        "raw": "quality_raw_dev.jsonl",
        "dir": Path(__file__).resolve().parent / "quality",
    },
    "loogle": {
        "url": LOOGLE_URL,
        "raw": "loogle_longdep_qa.jsonl",
        "dir": Path(__file__).resolve().parent / "loogle",
    },
}


def download(url, path):
    """Fetch `url` to `path` (cached: skipped if already present)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"[skip] raw source already present: {path}")
        return
    print(f"[download] {url}")
    try:
        import requests
        # Try the system CA bundle first (picks up corporate / HPC root certs),
        # then fall back to certifi's default.
        verify = (os.environ.get("REQUESTS_CA_BUNDLE")
                  or os.environ.get("SSL_CERT_FILE")
                  or "/etc/ssl/certs/ca-certificates.crt")
        if not os.path.exists(verify):
            verify = True
        with requests.get(url, stream=True, verify=verify, timeout=120) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
    except ImportError:
        urllib.request.urlretrieve(url, path)
    print(f"[ok] saved -> {path} ({path.stat().st_size / 1e6:.1f} MB)")


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ============================== QuALITY =====================================
def quality_group(lines):
    """Two lines per article (one per writer). Merge them into one document."""
    by_article = defaultdict(lambda: {"questions": [], "meta": None})
    for entry in lines:
        aid = entry["article_id"]
        if by_article[aid]["meta"] is None:
            by_article[aid]["meta"] = {
                "article_id": str(aid),
                "title": entry.get("title", ""),
                "source_dataset": "quality",
                "format": "mcq",
                "article": entry["article"],
                "word_count": len(entry["article"].split()),
            }
        writer_id = entry.get("set_unique_id", "")
        for q in entry["questions"]:
            by_article[aid]["questions"].append({
                "question_unique_id": q.get("question_unique_id", ""),
                "question": q["question"],
                "options": q["options"],
                "gold_label": q.get("gold_label"),
                "answer": None,
                "difficult": q.get("difficult", 0),
                "writer_id": writer_id,
                "writer_label": q.get("writer_label"),
                "validation": q.get("validation", []),
            })
    return by_article


def build_quality(raw_path, n_articles):
    lines = load_lines(raw_path)
    print(f"[info] loaded {len(lines)} writer-rows")
    by_article = quality_group(lines)
    print(f"[info] {len(by_article)} unique articles")
    ranked = sorted(
        by_article.values(),
        key=lambda a: (-len(a["questions"]), -a["meta"]["word_count"]),
    )
    out = []
    for a in ranked[:n_articles]:
        rec = dict(a["meta"])
        rec["questions"] = a["questions"]
        out.append(rec)
    return out


# ============================== LooGLE ======================================
def loogle_group(lines):
    """Flat schema: one row per (document, question). Group by `title`."""
    by_doc = defaultdict(lambda: {"questions": [], "meta": None})
    for i, entry in enumerate(lines):
        title = entry.get("title", "") or f"unknown_{i}"
        context = entry["context"]
        if by_doc[title]["meta"] is None:
            by_doc[title]["meta"] = {
                "article_id": title,           # title used as id
                "title": title,
                "source_dataset": "loogle_longdep_qa",
                "format": "freeform",
                "article": context,
                "word_count": len(context.split()),
            }
        # qa_pairs is sometimes a stringified python list, sometimes a real list.
        qa_field = entry.get("qa_pairs", None)
        if qa_field:
            if isinstance(qa_field, str):
                try:
                    qa_list = json.loads(qa_field.replace("'", '"'))
                except Exception:
                    qa_list = []
            else:
                qa_list = qa_field
            for j, qa in enumerate(qa_list):
                by_doc[title]["questions"].append({
                    "question_unique_id": f"{title}__{j}",
                    "question": qa.get("Q") or qa.get("question") or "",
                    "options": [],
                    "gold_label": None,
                    "answer": qa.get("A") or qa.get("answer") or "",
                    "difficult": 0,
                    "writer_id": "",
                    "writer_label": None,
                    "validation": [],
                })
        else:
            by_doc[title]["questions"].append({
                "question_unique_id": f"{title}__{len(by_doc[title]['questions'])}",
                "question": entry.get("question", ""),
                "options": [],
                "gold_label": None,
                "answer": entry.get("answer", ""),
                "difficult": 0,
                "writer_id": "",
                "writer_label": None,
                "validation": [],
            })
    return by_doc


def build_loogle(raw_path, n_articles):
    lines = load_lines(raw_path)
    print(f"[info] loaded {len(lines)} rows")
    by_doc = loogle_group(lines)
    print(f"[info] {len(by_doc)} unique documents")
    ranked = sorted(
        by_doc.values(),
        key=lambda a: (-len(a["questions"]), -a["meta"]["word_count"]),
    )
    out = []
    for a in ranked[:n_articles]:
        rec = dict(a["meta"])
        rec["questions"] = a["questions"]
        out.append(rec)
    return out


BUILDERS = {"quality": build_quality, "loogle": build_loogle}


def print_stats(subset, dataset, out_path):
    n_q = sum(len(a["questions"]) for a in subset)
    avg_q = n_q / max(len(subset), 1)
    avg_w = sum(a["word_count"] for a in subset) / max(len(subset), 1)
    min_w = min((a["word_count"] for a in subset), default=0)
    max_w = max((a["word_count"] for a in subset), default=0)
    print(f"[ok] wrote {out_path}")
    print(f"      dataset: {dataset}")
    print(f"      articles: {len(subset)}")
    print(f"      total questions: {n_q}")
    print(f"      avg questions/article: {avg_q:.1f}")
    print(f"      article words: avg={avg_w:.0f} min={min_w} max={max_w}")
    print(f"      file size: {out_path.stat().st_size / 1e6:.2f} MB")


def prepare(dataset, n_articles, out_dir, *, force=False):
    spec = DATASETS[dataset]
    out_dir = Path(out_dir)
    subset_path = out_dir / f"{dataset}_subset_{n_articles}.json"
    if subset_path.exists() and not force:
        print(f"Already present (skipped): {subset_path}\n"
              f"Pass --force to rebuild.")
        return subset_path

    raw_path = out_dir / spec["raw"]
    download(spec["url"], raw_path)

    subset = BUILDERS[dataset](raw_path, n_articles)

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = subset_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)
    tmp.replace(subset_path)               # atomic: never leave a half-written file

    print_stats(subset, dataset, subset_path)
    return subset_path


def main():
    ap = argparse.ArgumentParser(
        description="Download + build the QuALITY / LooGLE subsets for the "
                    "multi-turn SALT experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="quality",
                    help="Which source to build (default: quality).")
    ap.add_argument("--n-articles", type=int, default=50,
                    help="Number of top-ranked documents to keep (default: 50).")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Output dir (default: salt/datasets/<dataset>/).")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild the subset even if already present.")
    ap.add_argument("--list", action="store_true",
                    help="List the datasets and what is already present, then exit.")
    args = ap.parse_args()

    if args.list:
        print(f"{'dataset':<10} {'default out-dir':<44} present")
        for name, spec in DATASETS.items():
            d = spec["dir"]
            present = "yes" if d.exists() and any(d.glob(f"{name}_subset_*.json")) else "no"
            print(f"  {name:<8} {str(d):<44} {present}")
        return

    out_dir = args.out_dir or DATASETS[args.dataset]["dir"]
    prepare(args.dataset, args.n_articles, out_dir, force=args.force)


if __name__ == "__main__":
    main()
