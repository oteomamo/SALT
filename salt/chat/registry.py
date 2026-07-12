# -*- coding: utf-8 -*-
"""Model registry for saltChat.

Each registered chat model lives under ``salt/models/<alias>/``:

    config.json   loading + generation settings for the model
    weights       symlink to the resolved snapshot in the user's HF cache

Weights are never copied: ``register_model`` downloads through
``huggingface_hub.snapshot_download`` (which reuses the normal HF cache, so a
model already on disk registers instantly) and symlinks the snapshot
directory. Removing a registry entry never touches the cache. There is no
index file - the directory scan IS the registry, so it cannot fall out of
sync.
"""

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

VALID_DTYPES = ("bfloat16", "float16", "float32")
_ALIAS_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class RegistryError(Exception):
    """User-facing registry failure (bad name, collision, missing entry)."""


def default_alias(hf_id):
    return hf_id.split("/")[-1].lower()


def _entry_dir(alias):
    return MODELS_DIR / alias


def _load_entry(alias):
    cfg_path = _entry_dir(alias) / "config.json"
    try:
        cfg = json.loads(cfg_path.read_text())
    except (OSError, ValueError):
        # missing or corrupt config.json; register_model reports the
        # collision as "an unreadable entry"
        return None
    link = _entry_dir(alias) / "weights"
    cfg["path"] = str(link)
    cfg["downloaded"] = link.is_symlink() and link.resolve().exists()
    return cfg


def register_model(hf_id, alias=None, *, dtype="bfloat16",
                   attn_implementation="sdpa", max_new_tokens=512,
                   temperature=0.7, force=False):
    """Download ``hf_id`` (into the normal HF cache) and register it.

    Creates ``salt/models/<alias>/`` with a ``weights`` symlink to the cached
    snapshot plus a ``config.json``. Returns the entry dict (with ``path``).
    """
    hf_id = (hf_id or "").strip()
    if "/" not in hf_id:
        raise RegistryError(
            f"{hf_id!r} is not a full HuggingFace id (expected 'org/name', "
            f"e.g. meta-llama/Llama-3.1-8B-Instruct).")
    if dtype not in VALID_DTYPES:
        raise RegistryError(f"dtype must be one of {VALID_DTYPES}, got {dtype!r}")
    alias = alias or default_alias(hf_id)
    if not _ALIAS_RE.fullmatch(alias):
        raise RegistryError(
            f"alias {alias!r} may only contain letters, digits, '.', '_', '-'.")
    entry = _entry_dir(alias)
    if entry.exists() and not entry.is_dir():
        raise RegistryError(
            f"Alias {alias!r} collides with the file {entry}; pick another alias.")
    if entry.exists() and not force:
        existing = _load_entry(alias)
        held = existing["hf_id"] if existing else "an unreadable entry"
        raise RegistryError(
            f"Alias {alias!r} already registered (holds {held}). "
            f"Pick another with --alias, or pass --force to overwrite.")

    from huggingface_hub import snapshot_download
    try:
        snapshot_path = snapshot_download(repo_id=hf_id)
    except Exception as exc:
        msg = f"Download failed for {hf_id!r} ({type(exc).__name__}: {exc})"
        low = str(exc).lower()
        if "gated" in low or "401" in low or "403" in low or "restricted" in low:
            msg += ("\nThis repo is gated: request access on huggingface.co, "
                    "then run `hf auth login` or `export HF_TOKEN=hf_...`.")
        raise RegistryError(msg) from exc

    entry.mkdir(parents=True, exist_ok=True)
    link = entry / "weights"
    if link.is_symlink() or link.is_file():
        link.unlink()
    elif link.is_dir():
        shutil.rmtree(link)
    os.symlink(snapshot_path, link)

    cfg = {
        "alias": alias,
        "hf_id": hf_id,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "gen": {"max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "do_sample": temperature > 0,
                "top_p": 0.9},
        "registered_at": datetime.now().isoformat(timespec="seconds"),
    }
    (entry / "config.json").write_text(json.dumps(cfg, indent=2))
    return {**cfg, "path": str(link), "downloaded": True}


def list_models():
    """All registered models, sorted by alias."""
    if not MODELS_DIR.exists():
        return []
    entries = []
    for d in sorted(MODELS_DIR.iterdir()):
        if not d.is_dir():
            continue
        cfg = _load_entry(d.name)
        if cfg is not None:
            entries.append(cfg)
    return entries


def resolve_model(name):
    """Resolve an alias or a full HF id to a registry entry."""
    cfg = _load_entry(name) if _ALIAS_RE.fullmatch(name or "") else None
    if cfg is not None:
        return cfg
    for entry in list_models():
        if entry["hf_id"].lower() == (name or "").lower():
            return entry
    known = ", ".join(m["alias"] for m in list_models()) or "none registered"
    raise RegistryError(
        f"No registered model matches {name!r} (known: {known}). "
        f"Register it with: saltChat --add <hf_id>")


def remove_model(alias):
    """Delete the registry entry only; the HF cache is untouched."""
    entry = _entry_dir(alias)
    if not entry.exists():
        raise RegistryError(f"No registry entry named {alias!r}.")
    shutil.rmtree(entry)
