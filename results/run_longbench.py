#!/usr/bin/env python3
"""Reproduce the SALT LongBench table for one model on one GPU.

Compresses the 16 English LongBench tasks with the legacy selector at a
20% budget, scores them with eval.py's vLLM backend, and prints the
per-dataset and per-category scores. Assumes SALT and vLLM are installed,
and downloads any LongBench task missing from --data-dir.

Usage:
    python results/run_longbench.py --model llama --gpu 0
    python results/run_longbench.py --model ministral --gpu 1
"""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATASETS = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa",
            "2wikimqa", "musique", "gov_report", "qmsum", "multi_news",
            "trec", "triviaqa", "samsum", "passage_count",
            "passage_retrieval_en", "lcc", "repobench-p"]

DATASET_FLAGS = {
    "passage_count": ["--synthetic"],
    "passage_retrieval_en": ["--synthetic"],
    "trec": ["--fewshot"],
    "triviaqa": ["--fewshot"],
    "samsum": ["--fewshot"],
    "lcc": ["--selector", "coverage", "--code"],
    "repobench-p": ["--selector", "coverage", "--code"],
}

MODELS = {
    "llama": {"id": "meta-llama/Llama-3.1-8B-Instruct",
              "tokenizer_mode": "auto"},
    "ministral": {"id": "mistralai/Ministral-8B-Instruct-2410",
                  "tokenizer_mode": "hf"},
}


def load_eval_module():
    spec = importlib.util.spec_from_file_location("salt_eval",
                                                  REPO_ROOT / "eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def supported_tokenizer_mode(mode):
    if mode == "auto":
        return mode
    try:
        import typing
        from vllm.config.model import TokenizerMode
        if mode in typing.get_args(TokenizerMode):
            return mode
    except Exception:
        pass
    return "auto"


def ensure_data(args):
    missing = [ds for ds in args.datasets
               if not (Path(args.data_dir) / f"{ds}.jsonl").exists()]
    if not missing:
        return
    print(f"downloading LongBench data for: {' '.join(missing)}", flush=True)
    rc = subprocess.run([sys.executable, "-m", "salt.datasets.download_longbench",
                         "--datasets", ",".join(missing),
                         "--out-dir", str(args.data_dir)],
                        cwd=REPO_ROOT).returncode
    still = [ds for ds in missing
             if not (Path(args.data_dir) / f"{ds}.jsonl").exists()]
    if rc != 0 or still:
        sys.exit(f"LongBench download failed for: {' '.join(still or missing)}")


def check_settings(args, model, out_dir):
    settings = {"model": model["id"], "budget": args.budget,
                "max_samples": args.max_samples,
                "max_input_len": args.max_input_len}
    stamp = out_dir / "run_settings.json"
    if stamp.exists():
        old = json.loads(stamp.read_text())
        if old != settings:
            sys.exit(f"{out_dir} holds a run with other settings {old}. "
                     f"Pick another --out-dir or remove it.")
    else:
        stamp.write_text(json.dumps(settings, indent=2))


def compress(args, out_dir, log_dir):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               PYTHONHASHSEED="0")
    for ds in args.datasets:
        src = Path(args.data_dir) / f"{ds}.jsonl"
        dst = out_dir / f"{ds}.jsonl"
        if dst.exists():
            print(f"compress {ds}: already done", flush=True)
            continue
        if not src.exists():
            sys.exit(f"missing LongBench data: {src}")
        partial = out_dir / "partial"
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir()
        cmd = [sys.executable, "-m", "salt.compress",
               "--data", str(src), "--output", str(partial / dst.name),
               "--selector", "legacy", "--device", "cuda",
               "--token-budget-pct", str(args.budget)] + DATASET_FLAGS.get(ds, [])
        if args.max_samples:
            cmd += ["--max-samples", str(args.max_samples)]
        t0 = time.time()
        with open(log_dir / f"compress_{ds}.log", "w") as log:
            rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        if rc != 0 or not (partial / dst.name).exists():
            sys.exit(f"compress {ds} failed (rc={rc}), see "
                     f"{log_dir / f'compress_{ds}.log'}")
        for f in partial.iterdir():
            f.replace(out_dir / f.name)
        partial.rmdir()
        print(f"compress {ds}: {time.time() - t0:.0f}s", flush=True)


def evaluate(args, model, out_dir):
    parts = out_dir / "eval_parts"
    parts.mkdir(exist_ok=True)
    todo = [ds for ds in args.datasets
            if not (parts / f"{ds}_eval.json").exists()]
    if todo:
        ev = load_eval_module()
        mode = supported_tokenizer_mode(model["tokenizer_mode"])
        backend = ev.VLLMBackend(argparse.Namespace(
            backend="vllm", model=model["id"], gpu=args.gpu,
            max_input_len=args.max_input_len, gpu_mem_util=args.gpu_mem_util,
            tensor_parallel_size=1, tokenizer_mode=mode))
        for ds in todo:
            results = ev.eval_jsonl(str(out_dir / f"{ds}.jsonl"), ds, backend)
            tmp = parts / f"{ds}_eval.json.tmp"
            tmp.write_text(json.dumps(
                {"model": model["id"], "backend": "vllm",
                 "tokenizer_mode": mode, "results": results},
                indent=2, ensure_ascii=False))
            tmp.replace(parts / f"{ds}_eval.json")

    results = {}
    for ds in args.datasets:
        results.update(json.loads(
            (parts / f"{ds}_eval.json").read_text())["results"])
    eval_all = out_dir / "eval_all.json"
    eval_all.write_text(json.dumps(
        {"model": model["id"], "backend": "vllm", "target": str(out_dir),
         "kind": "folder", "results": results}, indent=2, ensure_ascii=False))
    return results, eval_all


def main():
    ap = argparse.ArgumentParser(
        description="SALT LongBench reproduction, one model per run.")
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out-dir", default=None,
                    help="default: runs/longbench_<model>")
    ap.add_argument("--data-dir",
                    default=str(REPO_ROOT / "salt/datasets/longbench/data"))
    ap.add_argument("--datasets", nargs="+", default=DATASETS,
                    choices=DATASETS)
    ap.add_argument("--budget", type=float, default=0.20)
    ap.add_argument("--max-input-len", type=int, default=14000)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="first N samples per dataset, for a quick check")
    args = ap.parse_args()

    model = MODELS[args.model]
    args.data_dir = str(Path(args.data_dir).resolve())
    out_dir = Path(args.out_dir or REPO_ROOT / "runs" /
                   f"longbench_{args.model}").resolve()
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    check_settings(args, model, out_dir)
    ensure_data(args)
    compress(args, out_dir, log_dir)
    results, eval_all = evaluate(args, model, out_dir)

    print(f"\n{model['id']}  (legacy selector, {args.budget:.0%} budget)")
    for ds in args.datasets:
        print(f"  {ds:<22} {results[ds]['score']:6.2f}")
    print(f"  {'average':<22} "
          f"{sum(results[d]['score'] for d in args.datasets) / len(args.datasets):6.2f}\n")
    subprocess.run([sys.executable,
                    str(REPO_ROOT / "scripts/longbench_categories.py"),
                    str(eval_all)], cwd=REPO_ROOT)


if __name__ == "__main__":
    main()
