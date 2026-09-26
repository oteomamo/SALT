# -*- coding: utf-8 -*-
"""The one tokenizer loader every chat runner uses."""

import json
import os

PROBE = [
    {"role": "system",
     "content": "You are a helpful assistant. Answer briefly."},
    {"role": "user",
     "content": "It is 2026. Set the margin to 10pt and greet: नमस्ते दुनिया, "
                "สวัสดีชาวโลก, مَرْحَبًا بِالْعَالَمِ."},
    {"role": "assistant",
     "content": "Done.\n\n10pt margins,  two  spaces\tand a tab. naïve café 🙂"},
    {"role": "user", "content": "And 1234567 items at 12:30pm?"},
]

_warned = set()


def _render(tok):
    for messages in (PROBE, PROBE[1:]):
        try:
            text = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
        except Exception:
            continue
        if isinstance(text, str):
            return text
    return "\n\n".join(m["content"] for m in PROBE)


def faithful(tok, spec):
    try:
        text = _render(tok)
        return (tok(text, add_special_tokens=False).input_ids
                == spec.encode(text, add_special_tokens=False).ids)
    except Exception:
        return False


def _config(path):
    try:
        with open(os.path.join(path, "tokenizer_config.json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _kwargs(config):
    names = config.get("model_input_names") or ["input_ids", "attention_mask"]
    return {"extra_special_tokens": {}, "model_input_names": names}


def _verbatim(path):
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(
        path, **_kwargs(_config(path)))


def _rescue(path):
    from transformers import AutoTokenizer
    config = _config(path)
    if isinstance(config.get("extra_special_tokens"), list):
        try:
            return AutoTokenizer.from_pretrained(path, **_kwargs(config))
        except Exception:
            pass
    return _verbatim(path)


def resolve_tokenizer(path):
    """(tokenizer, reason), reason None when AutoTokenizer's own is kept."""
    from transformers import AutoTokenizer
    spec_path = os.path.join(path, "tokenizer.json")
    reason = None
    try:
        tok = AutoTokenizer.from_pretrained(path)
    except Exception as exc:
        if not os.path.isfile(spec_path):
            raise
        try:
            tok = _rescue(path)
        except Exception:
            raise exc from None
        reason = f"AutoTokenizer failed ({type(exc).__name__}: {exc})"[:240]
    try:
        from tokenizers import Tokenizer
        spec = Tokenizer.from_file(spec_path)
    except Exception:
        return tok, reason
    if faithful(tok, spec):
        return tok, reason
    try:
        alt = _verbatim(path)
    except Exception:
        alt = None
    if alt is not None and faithful(alt, spec):
        return alt, f"{type(tok).__name__} ids differ from tokenizer.json"
    return tok, "unfaithful"


def load_tokenizer(path):
    tok, reason = resolve_tokenizer(path)
    if reason == "unfaithful" and path not in _warned:
        _warned.add(path)
        print(f"note: the tokenizer at {path} does not reproduce its "
              f"tokenizer.json, so prompts may not reach the model exactly "
              f"as written")
    return tok
