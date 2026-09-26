# -*- coding: utf-8 -*-
"""Regression harness for chat model loading.

CPU only, no model weights. Asserts:

  1. The tokenizer loader keeps what AutoTokenizer gives when it encodes
     a chat-rendered probe exactly as tokenizer.json does, and otherwise
     loads tokenizer.json itself, on tokenizers built here: a plain config,
     extra special tokens written as a list, the TokenizersBackend class,
     and a Llama label over a byte-level tokenizer. A tokenizer it
     substitutes hands the model input ids and an attention mask only.
  2. Every registered model's tokenizer stays faithful to its
     tokenizer.json, with any fallback named (SKIP when none is registered).

Usage:
    python scripts/chat_models_regression.py
"""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if not __debug__:
    sys.exit("this harness is assert-based - run it without python -O")

import transformers
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        trainers)
from transformers import AutoTokenizer

from salt.chat import tokload
from salt.chat.registry import list_models
from salt.chat.tokload import (PROBE, faithful, load_tokenizer,
                               resolve_tokenizer)

transformers.logging.set_verbosity_error()

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n"
          "{{ m['content'] }}<|im_end|>\n{% endfor %}"
          "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
BASE = {"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>",
        "bos_token": None, "unk_token": None, "model_max_length": 4096}
FIXTURES = {
    "plain": {"tokenizer_class": "PreTrainedTokenizerFast",
              "chat_template": CHATML},
    "listed": {"tokenizer_class": "PreTrainedTokenizerFast",
               "chat_template": CHATML,
               "extra_special_tokens": ["<|im_start|>", "<|im_end|>"]},
    "backend": {"tokenizer_class": "TokenizersBackend"},
    "llama": {"tokenizer_class": "LlamaTokenizerFast",
              "chat_template": CHATML, "bos_token": "<|endoftext|>",
              "add_bos_token": True},
}
WARNING = "does not reproduce its tokenizer.json"


def byte_level_bpe():
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=600, special_tokens=SPECIALS, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    corpus = [m["content"] for m in PROBE] * 20 + [
        "the quick brown fox jumps over the lazy dog",
        "system user assistant"]
    tok.train_from_iterator(corpus, trainer)
    return tok


def build(root, name, spec):
    path = root / name
    path.mkdir()
    spec.save(str(path / "tokenizer.json"))
    cfg = {**BASE, **FIXTURES[name]}
    (path / "tokenizer_config.json").write_text(json.dumps(cfg))
    if "chat_template" not in cfg:
        (path / "chat_template.jinja").write_text(CHATML)
    return str(path)


def plain_outcome(path, spec):
    try:
        tok = AutoTokenizer.from_pretrained(path)
    except Exception as exc:
        return None, f"raised {type(exc).__name__}"
    return tok, "faithful" if faithful(tok, spec) else "unfaithful"


def check_fixture(path, name, spec):
    plain, how = plain_outcome(path, spec)
    tok, reason = resolve_tokenizer(path)
    assert faithful(tok, spec), f"{name}: loader result is unfaithful"
    assert tok.eos_token == "<|im_end|>", (name, tok.eos_token)
    if reason:
        assert sorted(tok("hello there")) == ["attention_mask", "input_ids"], (
            name, sorted(tok("hello there")))
    text = tok.apply_chat_template(PROBE, tokenize=False,
                                   add_generation_prompt=True)
    assert text.endswith("<|im_start|>assistant\n"), (name, text[-40:])
    if plain is None:
        assert reason and reason.startswith("AutoTokenizer failed"), (
            name, reason)
    elif how == "unfaithful":
        want = f"{type(plain).__name__} ids differ from tokenizer.json"
        assert reason == want, (name, reason)
    else:
        assert reason is None and type(tok) is type(plain), (
            name, reason, type(tok), type(plain))
        assert (tok(text, add_special_tokens=False).input_ids
                == plain(text, add_special_tokens=False).input_ids)
    print(f"  {name}: AutoTokenizer {how}, loader gives "
          f"{type(tok).__name__}" + (f" ({reason})" if reason else ""))


def check_warns_once(path):
    kept = tokload.faithful
    tokload.faithful = lambda tok, spec: False
    try:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            tok, reason = resolve_tokenizer(path)
            load_tokenizer(path)
            load_tokenizer(path)
    finally:
        tokload.faithful = kept
    assert reason == "unfaithful", reason
    assert out.getvalue().count(WARNING) == 1, out.getvalue()


def check_fixtures():
    spec = byte_level_bpe()
    with tempfile.TemporaryDirectory() as tmp:
        for name in FIXTURES:
            check_fixture(build(Path(tmp), name, spec), name, spec)
        check_warns_once(str(Path(tmp) / "plain"))
    print(f"1. fixtures: {len(FIXTURES)} tokenizers faithful under "
          f"transformers {transformers.__version__}, an unfaithful one "
          f"warns once")


def check_registered():
    entries = [m for m in list_models() if m.get("downloaded")]
    if not entries:
        print("2. registered models: SKIP (none registered)")
        return
    bad = []
    for m in entries:
        path = Path(m["path"])
        try:
            tok, reason = resolve_tokenizer(str(path))
        except Exception as exc:
            raise AssertionError(f"{m['alias']}: the tokenizer does not "
                                 f"load: {type(exc).__name__}: {exc}")
        if not (path / "tokenizer.json").is_file():
            print(f"  {m['alias']}: {type(tok).__name__}, no tokenizer.json")
            continue
        ok = faithful(tok, Tokenizer.from_file(str(path / "tokenizer.json")))
        if not ok:
            bad.append(m["alias"])
        print(f"  {m['alias']}: {type(tok).__name__}"
              + (f", fallback: {reason}" if reason else "")
              + ("" if ok else ", UNFAITHFUL"))
    assert not bad, f"tokenizers unfaithful to tokenizer.json: {bad}"
    print(f"2. registered models: {len(entries)} tokenizers faithful")


def main():
    check_fixtures()
    check_registered()
    print("PASS")


if __name__ == "__main__":
    main()
