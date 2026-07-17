# -*- coding: utf-8 -*-
"""saltServe: launch a persistent vllm serve process for saltChat.

Resolves a registered model and execs ``vllm serve`` so the server owns the
model and its prefix cache outlives every saltChat run. The server is a
foreground process in its own terminal - saltChat never manages its
lifecycle, it only connects. ``--vllm-bin`` points at another environment's
vllm binary, so the server can run whichever vLLM release fits the GPU
while the SALT environment stays pinned.
"""

import argparse
import math
import os
import shlex
import shutil
import subprocess
import sys

from salt.chat.registry import RegistryError, list_models, resolve_model


def compute_capability(gpu_index):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap",
             "--format=csv,noheader", "-i", str(gpu_index)],
            capture_output=True, text=True, timeout=10)
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def resolve_dtype(dtype, cap):
    # bfloat16 is hard-unsupported below compute capability 8.0 on every
    # vLLM release; the server would refuse to start
    if dtype == "bfloat16" and cap is not None and cap < 8.0:
        return "float16"
    return dtype


def target_gpu(gpu_arg):
    if gpu_arg is not None:
        return gpu_arg
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = vis.split(",")[0].strip()
    if first:
        # nvidia-smi -i accepts GPU-<uuid> and MIG ids directly
        return int(first) if first.isdigit() else first
    return 0 if not vis else None


def build_parser():
    ap = argparse.ArgumentParser(
        prog="saltServe",
        description="Launch a persistent vllm serve process for a "
                    "registered saltChat model. Everything after -- is "
                    "passed to vllm serve unchanged.")
    ap.add_argument("model", nargs="?", default=None,
                    help="registered alias or HF id (default: the single "
                         "registered model)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default 127.0.0.1, this machine only)")
    ap.add_argument("--port", type=int, default=8000,
                    help="listen port (default 8000)")
    ap.add_argument("--gpu", type=int, default=None,
                    help="GPU index the server runs on (PCI bus order, the "
                         "same numbering nvidia-smi shows)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.90,
                    help="fraction of GPU memory the server claims "
                         "(default 0.90)")
    ap.add_argument("--max-model-len", type=int, default=0,
                    help="cap the context window (0 = the model's own)")
    ap.add_argument("--vllm-bin", default=None,
                    help="vllm executable to run (default: the vllm next "
                         "to saltServe, then PATH); point it at another "
                         "environment's vllm to serve with a different "
                         "vLLM release")
    return ap


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    extra = []
    if "--" in argv:
        cut = argv.index("--")
        argv, extra = argv[:cut], argv[cut + 1:]
    args = build_parser().parse_args(argv)

    if not 1 <= args.port <= 65535:
        sys.exit("--port must be in 1..65535")
    if not (math.isfinite(args.gpu_mem_util) and 0 < args.gpu_mem_util <= 1):
        sys.exit("--gpu-mem-util must be in (0, 1]")
    if args.max_model_len < 0:
        sys.exit("--max-model-len must be >= 0 (0 = the model's own window)")
    if args.gpu is not None and args.gpu < 0:
        sys.exit("--gpu must be >= 0")

    try:
        if args.model:
            cfg = resolve_model(args.model)
        else:
            models = list_models()
            if len(models) != 1:
                known = (", ".join(m["alias"] for m in models)
                         or "none registered")
                sys.exit(f"Pick a model to serve (registered: {known}), "
                         f"e.g. saltServe <alias>")
            cfg = models[0]
    except RegistryError as exc:
        sys.exit(str(exc))
    if not cfg.get("downloaded"):
        sys.exit(f"{cfg['alias']}'s weights are not downloaded - register "
                 f"again with: saltChat --add {cfg['hf_id']} --force")

    if args.vllm_bin:
        vllm_bin = os.path.expanduser(args.vllm_bin)
        if not (os.path.isfile(vllm_bin) and os.access(vllm_bin, os.X_OK)):
            sys.exit(f"{vllm_bin} is not an executable vllm binary")
    else:
        # prefer the env saltServe itself runs from, so an unactivated env
        # still finds its own vllm; PATH is the fallback
        sibling = os.path.join(os.path.dirname(sys.executable), "vllm")
        if os.path.isfile(sibling) and os.access(sibling, os.X_OK):
            vllm_bin = sibling
        else:
            vllm_bin = shutil.which("vllm")
        if vllm_bin is None:
            sys.exit("no vllm found next to saltServe or on PATH - install "
                     "the optional vLLM backend (Installation step 5), or "
                     "point --vllm-bin at another environment's vllm")

    dtype = cfg.get("dtype", "bfloat16")
    cap = compute_capability(target_gpu(args.gpu))
    served_dtype = resolve_dtype(dtype, cap)
    if served_dtype != dtype:
        print(f"note: this GPU predates bfloat16 support - serving "
              f"{cfg['alias']} in float16")
    elif dtype == "bfloat16" and cap is None:
        print("note: could not detect the GPU's compute capability - if "
              "the server rejects bfloat16, add: -- --dtype float16")

    cmd = [vllm_bin, "serve", cfg["path"],
           "--served-model-name", cfg["alias"],
           "--enable-prompt-tokens-details",
           "--host", args.host,
           "--port", str(args.port),
           "--dtype", served_dtype,
           "--gpu-memory-utilization", str(args.gpu_mem_util)]
    if args.max_model_len:
        cmd += ["--max-model-len", str(args.max_model_len)]
    cmd += extra

    env = os.environ.copy()
    if args.gpu is not None:
        # PCI order makes --gpu N mean the same card nvidia-smi (and the
        # capability probe above) call N
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"Serving {cfg['alias']} ({cfg['hf_id']}) at "
          f"http://{args.host}:{args.port} - Ctrl-C stops the server")
    print(" ".join(shlex.quote(c) for c in cmd))
    sys.stdout.flush()
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
