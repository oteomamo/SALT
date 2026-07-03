"""Build the needle-in-a-haystack long-context set used by the SALT TTFT /
memory benchmark (`salt/results/niah_ttft.py`). See README.md.

Streams PG-19 (`emozilla/pg19`) and packs prompts to EXACT token lengths for a
given tokenizer. Each sample gets a unique, non-overlapping slice of the corpus
(no repeated text). A "needle" — a magic number tied to a city — is inserted at
a random depth, followed by the question that asks for it, so the same file
doubles as a retrieval-accuracy probe. Decoded text (not token ids) is saved,
one JSONL file per length.

With `--no-needle` the prompts are plain long-context filler (pure TTFT / memory
scaling, no retrieval probe).

The needle / question templates live here as module constants so the benchmark
consumer imports them instead of duplicating the strings.
"""

import argparse
import json
import random
from pathlib import Path

NEEDLE_TPL = "One of the special magic numbers for {city} is: {number}."
QUESTION_TPL = ("\n\nWhat is the special magic number for {city} "
                "mentioned in the provided text?")
CITIES = ["San Francisco", "Tokyo", "Paris", "Berlin", "Sydney",
          "Cairo", "Mumbai", "Toronto", "Lima", "Oslo"]

DEFAULT_OUT = Path(__file__).resolve().parent / "niah"
DEFAULT_LENGTHS = [32000, 64000, 128000, 256000]


def collect_buffer(tok, total_needed):
    """Stream PG-19 until at least `total_needed` tokens are buffered."""
    from datasets import load_dataset
    print(f"Streaming PG-19 for {total_needed:,} unique tokens...")
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    buf = []
    for ex in ds:
        buf.extend(tok.encode(ex["text"], add_special_tokens=False))
        if len(buf) >= total_needed:
            break
    print(f"Collected {len(buf):,} tokens")
    return buf


def fmt_k(n):
    """32000 -> '32k', 128000 -> '128k', 1500 -> '1500' (fallback)."""
    return f"{n // 1000}k" if n % 1000 == 0 else str(n)


def out_path_for(out_dir, prefix, length, samples_per_length):
    return Path(out_dir) / f"{prefix}_{fmt_k(length)}_{samples_per_length}.jsonl"


def build_length(buf, cursor, L, samples_per_length, rng, tok, tokenizer_name,
                 no_needle):
    """Materialize `samples_per_length` records of exactly `L` tokens.
    Returns (records, new_cursor)."""
    records = []
    for i in range(samples_per_length):
        if no_needle:
            ids = buf[cursor:cursor + L]
            cursor += L
            meta = {}
        else:
            city = rng.choice(CITIES)
            number = rng.randint(1_000_000, 9_999_999)
            depth = rng.random()
            n_ids = tok.encode(NEEDLE_TPL.format(city=city, number=number),
                               add_special_tokens=False)
            q_ids = tok.encode(QUESTION_TPL.format(city=city),
                               add_special_tokens=False)
            hay_len = L - len(n_ids) - len(q_ids)
            hay_ids = buf[cursor:cursor + hay_len]
            cursor += hay_len
            insert = int(hay_len * depth)
            ids = hay_ids[:insert] + n_ids + hay_ids[insert:] + q_ids
            meta = {"needle_city": city, "needle_number": number,
                    "needle_depth": round(depth, 4)}

        assert len(ids) == L, f"got {len(ids)}, expected {L}"
        text = tok.decode(ids)
        records.append({
            "target_length": L,
            "sample_id": i,
            "prompt": text,
            "token_count_built": L,
            "token_count_retokenized": len(tok.encode(text, add_special_tokens=False)),
            "tokenizer": tokenizer_name,
            **meta,
        })
    return records, cursor


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)                       # atomic: never leave a half-written file


def prepare(tokenizer_name, lengths, samples_per_length, out_dir, prefix, seed,
            no_needle, *, force=False):
    from transformers import AutoTokenizer

    out_dir = Path(out_dir)
    todo = [L for L in lengths
            if force or not out_path_for(out_dir, prefix, L, samples_per_length).exists()]
    skipped = [L for L in lengths if L not in todo]
    if skipped:
        print("Already present (skipped): "
              + ", ".join(fmt_k(L) for L in skipped) + "  (--force rebuilds)")
    if not todo:
        print(f"\nNothing to do — all {len(lengths)} lengths present in {out_dir}/")
        return

    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    tok.model_max_length = int(1e9)

    # One shared buffer covers every length so slices never overlap.
    total_needed = sum(L * samples_per_length for L in todo) + 10_000
    buf = collect_buffer(tok, total_needed)

    cursor = 0
    for L in todo:
        records, cursor = build_length(
            buf, cursor, L, samples_per_length, rng, tok, tokenizer_name, no_needle)
        path = out_path_for(out_dir, prefix, L, samples_per_length)
        write_jsonl(path, records)
        print(f"  {L:>7} tokens: {samples_per_length} samples -> {path}")
    print(f"\nDone: {len(todo)} length(s) in {out_dir}/")


def main():
    ap = argparse.ArgumentParser(
        description="Build needle-in-a-haystack long-context prompts from PG-19.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True,
                    help="Tokenizer to pack exact token lengths against "
                         "(match your eval LLM, e.g. meta-llama/Llama-3.1-8B).")
    ap.add_argument("--lengths", nargs="+", type=int, default=DEFAULT_LENGTHS,
                    help=f"Target token lengths (default: {DEFAULT_LENGTHS}).")
    ap.add_argument("--samples-per-length", type=int, default=10)
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT),
                    help=f"Output dir (default: {DEFAULT_OUT}).")
    ap.add_argument("--prefix", default="longctx",
                    help="Output filename prefix (default: longctx).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-needle", action="store_true",
                    help="Plain long-context filler (no needle / no probe).")
    ap.add_argument("--force", action="store_true",
                    help="Rebuild lengths even if already present.")
    ap.add_argument("--list", action="store_true",
                    help="List target lengths and what is already present, then exit.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.list:
        print(f"Out-dir: {out_dir}/\n{'length':<10} present")
        for L in args.lengths:
            mark = "yes" if out_path_for(
                out_dir, args.prefix, L, args.samples_per_length).exists() else "no"
            print(f"  {fmt_k(L):<8} {mark}")
        return

    prepare(args.tokenizer, args.lengths, args.samples_per_length, out_dir,
            args.prefix, args.seed, args.no_needle, force=args.force)


if __name__ == "__main__":
    main()
