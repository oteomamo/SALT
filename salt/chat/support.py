# -*- coding: utf-8 -*-
"""Why a chat model did not load, read from its config.json and the
installed versions, never from the model's own code."""

import json
import re
from importlib import metadata, util
from pathlib import Path

from packaging.version import InvalidVersion, Version

QUANT_PACKAGES = {"awq": ("awq", "gptqmodel"),
                  "gptq": ("gptqmodel", "auto_gptq")}
VLLM_FOR_TF5 = Version("0.17")
VALIDATION_HEAD = re.compile(r"\d+ validation errors? for ")


def read_config(folder):
    try:
        config = json.loads((Path(folder) / "config.json").read_text())
    except (OSError, TypeError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def installed(name):
    try:
        return Version(metadata.version(name))
    except (metadata.PackageNotFoundError, InvalidVersion):
        return None


def importable(name):
    return util.find_spec(name) is not None


def vllm_knows(model_type):
    try:
        from vllm.transformers_utils.config import _CONFIG_REGISTRY
    except Exception:
        return False
    return model_type in _CONFIG_REGISTRY


def load_hint(config, backend="hf"):
    try:
        return _hint(config, backend)
    except Exception:
        return None


def _hint(config, backend):
    if not isinstance(config, dict) or backend not in ("hf", "vllm"):
        return None
    tf = installed("transformers")
    if backend == "vllm":
        vllm = installed("vllm")
        if vllm and tf and vllm < VLLM_FOR_TF5 and tf.major >= 5:
            return (f"vLLM {vllm} cannot run next to transformers {tf}. "
                    f"Install the pair the Installation page names.")
    from transformers.models.auto.configuration_auto import \
        CONFIG_MAPPING_NAMES
    top = config.get("model_type")
    text = config.get("text_config")
    nested = text.get("model_type") if isinstance(text, dict) else None
    unknown = [t for t in (top, nested)
               if isinstance(t, str) and t not in CONFIG_MAPPING_NAMES]
    if unknown:
        if "auto_map" in config or (backend == "vllm"
                                    and vllm_knows(unknown[0])):
            return None
        return (f"transformers {tf} does not know the model type "
                f"{unknown[0]!r}. Newer model families need a newer SALT "
                f"environment, see the Installation page.")
    if backend != "hf":
        return None
    from transformers.models.auto.modeling_auto import \
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
    if isinstance(top, str) and top not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
        return (f"transformers {tf} has no text-generation class for "
                f"{top!r}. Try --backend vllm.")
    quant = config.get("quantization_config")
    method = quant.get("quant_method") if isinstance(quant, dict) else None
    if method in QUANT_PACKAGES and not any(
            importable(p) for p in QUANT_PACKAGES[method]):
        return (f"This {method.upper()} checkpoint needs a package the hf "
                f"backend lacks here. Use --backend vllm.")
    return None


def failure_detail(exc):
    try:
        first = exc.errors()[0]
        loc = first.get("loc") or ()
        text = f"{loc[0]}: {first['msg']}" if loc else str(first["msg"])
    except Exception:
        text = str(exc)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1 and VALIDATION_HEAD.match(lines[0]):
        return lines[1].split(" [type=")[0]
    return lines[0] if lines else ""


def failure_line(alias, exc):
    msg = failure_detail(exc)
    head = f"{alias} did not load: {type(exc).__name__}"
    return f"{head}: {msg}" if msg else head


def report_failure(cfg, exc, backend, file=None):
    print(failure_line(cfg["alias"], exc), file=file)
    hint = load_hint(read_config(cfg.get("path")), backend)
    if hint:
        print(hint, file=file)
