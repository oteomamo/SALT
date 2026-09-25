"""Compare an eval smoke run with a reference run.

Within one environment every output file must match byte for byte.
With --across (the dependencies themselves moved) the compressed
records must match field for field, except the greedy mode label of a
selection whose other fields, objective and chosen sentences included,
are identical. Metadata files must match byte for byte in both modes.

    python scripts/smoke_compare.py BASE_DIR NEW_DIR [--across]
"""

import argparse
import filecmp
import json
import sys
from pathlib import Path


def _without_tie_label(record):
    stats = (record.get("compression_stats") or {}).get("selection_stats")
    if isinstance(stats, dict):
        stats.pop("greedy_mode", None)
    return record


def _records_match(base, new):
    a = base.read_text().splitlines()
    b = new.read_text().splitlines()
    if len(a) != len(b):
        return f"{len(a)} records vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y and (_without_tie_label(json.loads(x))
                       != _without_tie_label(json.loads(y))):
            return f"record {i} differs"
    return None


def compare(base_dir, new_dir, across=False):
    base_dir, new_dir = Path(base_dir), Path(new_dir)
    problems = []
    names = sorted(p.name for p in base_dir.glob("*.jsonl"))
    if not names:
        problems.append(f"no .jsonl files in {base_dir}")
    for name in names:
        base, new = base_dir / name, new_dir / name
        if not new.is_file():
            problems.append(f"{name}: missing")
        elif filecmp.cmp(base, new, shallow=False):
            continue
        elif across and not name.endswith("_metadata.jsonl"):
            why = _records_match(base, new)
            if why:
                problems.append(f"{name}: {why}")
        else:
            problems.append(f"{name}: bytes differ")
    return names, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base")
    ap.add_argument("new")
    ap.add_argument("--across", action="store_true",
                    help="the runs come from different environments")
    args = ap.parse_args(argv)
    names, problems = compare(args.base, args.new, args.across)
    for line in problems:
        print(f"smoke compare: {line}")
    mode = "across environments" if args.across else "byte for byte"
    if problems:
        print(f"smoke compare: FAILED ({mode}, {len(problems)} of "
              f"{len(names)} files)")
        return 1
    print(f"smoke compare: {len(names)} files match ({mode})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
