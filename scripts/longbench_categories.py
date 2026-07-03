#!/usr/bin/env python3
"""Fold a LongBench `eval_all.json` into the standard category summary row:

    Single-Doc QA | Multi-Doc QA | Summarization | Few-Shot | Synthetic | Code | Avg.

Each category cell is the mean of its tasks' scores; `Avg.` is the mean over all
evaluated tasks (equal task weighting, the LongBench convention). Missing tasks
are simply skipped, so partial runs still summarize (coverage is printed below).

Usage:
    python scripts/longbench_categories.py runs/run_XXXX/eval_all.json
    python scripts/longbench_categories.py runs/run_XXXX          # dir with eval_all.json
"""
import argparse
import json
from pathlib import Path

# Canonical LongBench (English) task -> category grouping.
CATEGORIES = [
    ("Single-Doc QA", ["narrativeqa", "qasper", "multifieldqa_en"]),
    ("Multi-Doc QA",  ["hotpotqa", "2wikimqa", "musique"]),
    ("Summarization", ["gov_report", "qmsum", "multi_news"]),
    ("Few-Shot",      ["trec", "triviaqa", "samsum"]),
    ("Synthetic",     ["passage_count", "passage_retrieval_en"]),
    ("Code",          ["lcc", "repobench-p"]),
]


def load_scores(path):
    """Return (scores_by_dataset, model, resolved_path) from an eval_all.json."""
    path = Path(path)
    if path.is_dir():
        path = path / "eval_all.json"
    if not path.exists():
        raise SystemExit(f"not found: {path}")
    data = json.loads(path.read_text())
    results = data.get("results", data)   # tolerate a bare results dict
    scores = {ds: v["score"] for ds, v in results.items()
              if isinstance(v, dict) and "score" in v}
    return scores, data.get("model", "?"), path


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def summarize(scores):
    """Returns (rows, avg, n_tasks) where rows = [(name, mean, n_present, n_total)]."""
    rows, all_task_scores = [], []
    for name, tasks in CATEGORIES:
        present = [scores[t] for t in tasks if t in scores]
        all_task_scores.extend(present)
        rows.append((name, _mean(present), len(present), len(tasks)))
    return rows, _mean(all_task_scores), len(all_task_scores)


def _fmt(v):
    return f"{v:.2f}" if v is not None else "-"


def render(scores, model, src):
    rows, avg, n = summarize(scores)
    cols = [name for name, _ in CATEGORIES] + ["Avg."]
    vals = [_fmt(m) for _, m, _, _ in rows] + [_fmt(avg)]
    widths = [max(len(c), len(v), 6) for c, v in zip(cols, vals)]

    print(f"LongBench category scores  (model: {model})")
    print(f"source: {src}")
    print()
    print("  ".join(c.rjust(w) for c, w in zip(cols, widths)))
    print("  ".join(v.rjust(w) for v, w in zip(vals, widths)))
    print()
    cov = ", ".join(f"{name} {p}/{t}" for name, _, p, t in rows)
    print(f"coverage: {cov}")
    print(f"Avg. over {n} evaluated task(s)"
          + ("" if n == 16 else "  (partial run: fewer than 16 tasks)"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="eval_all.json file, or a run dir containing it.")
    args = ap.parse_args()
    scores, model, src = load_scores(args.path)
    render(scores, model, src)


if __name__ == "__main__":
    main()
