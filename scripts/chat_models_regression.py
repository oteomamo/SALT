# -*- coding: utf-8 -*-
"""Regression harness for chat model loading.

CPU only, no model weights. Asserts:

  1. The tokenizer loader keeps what AutoTokenizer gives when it encodes
     a chat-rendered probe exactly as tokenizer.json does, and otherwise
     loads tokenizer.json itself, on tokenizers built here: a plain config,
     extra special tokens written as a list, the TokenizersBackend class,
     and a Llama label over a byte-level tokenizer. A tokenizer it
     substitutes hands the model input ids and an attention mask only.
  2. The vLLM engine's tokenizer, on the same fixtures: none for a
     tokenizer the loader keeps or cannot make faithful, or one that
     ships its own code, otherwise a directory outside the snapshot whose
     config names PreTrainedTokenizerFast and which encodes and decodes
     as tokenizer.json does. It lives in the registry entry for a
     registered model, is kept while current and rebuilt when stale.
  3. Every registered model's tokenizer stays faithful to its
     tokenizer.json, with any fallback named, and the vLLM engine gets a
     corrected tokenizer exactly when the loader fell back (SKIP when none
     is registered).
  4. The HF runner also stops at the tokenizer's end token: generate gets
     the union of the model's stop ids and the tokenizer's only when the
     tokenizer's id is missing, and nothing extra when they agree.
  5. The HF runner names the dtype the way the installed transformers
     reads it: torch_dtype before 4.56, dtype from 4.56 on.

Usage:
    python scripts/chat_models_regression.py
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if not __debug__:
    sys.exit("this harness is assert-based - run it without python -O")

import torch
import transformers
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        trainers)
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from salt.chat import runner as runner_mod
from salt.chat import registry, tokload
from salt.chat.registry import list_models
from salt.chat.runner import ChatRunner, dtype_keyword, eos_union
from salt.chat.tokload import (PROBE, engine_tokenizer, faithful,
                               load_tokenizer, resolve_tokenizer)

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
AUTO_MAP = {"AutoTokenizer": [None, "tokenization_custom.CustomTokenizerFast"]}
CUSTOM = ("from transformers import PreTrainedTokenizerFast\n\n\n"
          "class CustomTokenizerFast(PreTrainedTokenizerFast):\n    pass\n")


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


def snapshot(path):
    return {n: Path(path, n).read_bytes() for n in sorted(os.listdir(path))}


def check_engine_dir(out, spec):
    config = json.loads(Path(out, "tokenizer_config.json").read_text())
    assert config["tokenizer_class"] == "PreTrainedTokenizerFast", config
    assert not isinstance(config.get("extra_special_tokens"), list), config
    tok = AutoTokenizer.from_pretrained(out)
    assert faithful(tok, spec), f"{out}: the engine tokenizer is unfaithful"
    ids = spec.encode(PROBE[2]["content"], add_special_tokens=False).ids
    assert (tok.decode(ids, clean_up_tokenization_spaces=False)
            == spec.decode(ids)), tok.decode(ids)


def own_code(root, source):
    for name, where in (("code", "tokenizer_config.json"),
                        ("code-config", "config.json")):
        path = root / name
        shutil.copytree(source, path)
        (path / "tokenization_custom.py").write_text(CUSTOM)
        config = {"auto_map": AUTO_MAP}
        if where == "tokenizer_config.json":
            config = {**json.loads((path / where).read_text()), **config,
                      "tokenizer_class": "CustomTokenizerFast"}
        (path / where).write_text(json.dumps(config))
        yield str(path)


def check_engine(root, spec):
    kept = (tokload.CACHE_DIR, registry.MODELS_DIR, tokload.faithful,
            tokload.resolve_tokenizer)
    tokload.CACHE_DIR = str(root / "cache")
    registry.MODELS_DIR = root / "models"
    needed = []
    try:
        for name in FIXTURES:
            path = str(root / name)
            before = snapshot(path)
            out = engine_tokenizer(path)
            assert snapshot(path) == before, f"{name}: the snapshot changed"
            if resolve_tokenizer(path)[1] is None:
                assert out is None, (name, out)
                continue
            needed.append(name)
            assert out.startswith(tokload.CACHE_DIR + os.sep), out
            check_engine_dir(out, spec)
            inode = os.stat(out).st_ino
            assert engine_tokenizer(path) == out
            assert os.stat(out).st_ino == inode, (
                f"{name}: rebuilt while current")
            Path(out, "tokenizer_config.json").write_text("{}")
            assert engine_tokenizer(path) == out
            check_engine_dir(out, spec)
        assert needed, "no fixture needed an engine tokenizer"
        source = root / needed[0]
        before = snapshot(source)
        entry = root / "models" / "zz"
        entry.mkdir(parents=True)
        os.symlink(source, entry / "weights")
        out = engine_tokenizer(str(entry / "weights"))
        assert out == str(entry / "tokenizer"), out
        check_engine_dir(out, spec)
        assert snapshot(source) == before
        assert engine_tokenizer(str(root / "missing")) is None
        calls = []
        tokload.resolve_tokenizer = lambda path: (
            calls.append(path) or (None, "AutoTokenizer failed (stub)"))
        cached = sorted(os.listdir(tokload.CACHE_DIR))
        for path in own_code(root, source):
            assert engine_tokenizer(path) is None, path
        assert not calls and sorted(os.listdir(tokload.CACHE_DIR)) == cached
        assert engine_tokenizer(str(source)) and calls == [str(source)]
        tokload.resolve_tokenizer = kept[3]
        tokload.faithful = lambda tok, spec: False
        assert engine_tokenizer(str(source)) is None
    finally:
        (tokload.CACHE_DIR, registry.MODELS_DIR, tokload.faithful,
         tokload.resolve_tokenizer) = kept
    return needed


def check_fixtures():
    spec = byte_level_bpe()
    with tempfile.TemporaryDirectory() as tmp:
        for name in FIXTURES:
            check_fixture(build(Path(tmp), name, spec), name, spec)
        check_warns_once(str(Path(tmp) / "plain"))
        needed = check_engine(Path(tmp), spec)
    print(f"1. fixtures: {len(FIXTURES)} tokenizers faithful under "
          f"transformers {transformers.__version__}, an unfaithful one "
          f"warns once")
    print(f"2. engine tokenizer: none for a kept or unfaithful tokenizer or "
          f"one that ships its own code, a corrected directory for "
          f"{', '.join(needed)}, in the registry entry when registered, the "
          f"snapshot untouched, rebuilt when stale")


def check_registered():
    entries = [m for m in list_models() if m.get("downloaded")]
    if not entries:
        print("3. registered models: SKIP (none registered)")
        return
    bad, engines = [], 0
    for m in entries:
        path = Path(m["path"])
        try:
            tok, reason = resolve_tokenizer(str(path))
        except Exception as exc:
            raise AssertionError(f"{m['alias']}: the tokenizer does not "
                                 f"load: {type(exc).__name__}: {exc}")
        engine = engine_tokenizer(str(path))
        assert (engine is None) == (reason in (None, "unfaithful")), (
            m["alias"], reason, engine)
        if not (path / "tokenizer.json").is_file():
            print(f"  {m['alias']}: {type(tok).__name__}, no tokenizer.json")
            continue
        spec = Tokenizer.from_file(str(path / "tokenizer.json"))
        ok = faithful(tok, spec)
        if not ok:
            bad.append(m["alias"])
        if engine:
            check_engine_dir(engine, spec)
            engines += 1
        print(f"  {m['alias']}: {type(tok).__name__}"
              + (f", fallback: {reason}" if reason else "")
              + ("" if ok else ", UNFAITHFUL")
              + (f", vLLM gets {engine}" if engine else ""))
    assert not bad, f"tokenizers unfaithful to tokenizer.json: {bad}"
    print(f"3. registered models: {len(entries)} tokenizers faithful, "
          f"{engines} handed to vLLM corrected")


class StubModel:
    def __init__(self, eos):
        self.generation_config = SimpleNamespace(eos_token_id=eos)
        self.seen = None

    def generate(self, **kwargs):
        self.seen = kwargs
        kwargs["streamer"].end()


def generate_kwargs(tokenizer, eos):
    runner = ChatRunner.__new__(ChatRunner)
    runner.cfg = {"alias": "stub", "gen": {"max_new_tokens": 4,
                                           "temperature": 0}}
    runner.tokenizer = tokenizer
    runner.model = StubModel(eos)
    runner.input_device = "cpu"
    runner.max_input_len = None
    list(runner.stream_chat(PROBE[:2]))
    return runner.model.seen


def check_eos():
    assert eos_union(5, 7) == [5, 7]
    assert eos_union([5, 6], 7) == [5, 6, 7]
    assert eos_union((5,), 7) == [5, 7]
    assert eos_union(None, 7) == [7]
    for same in ((5, 5), ([5, 6], 6), ([6, 5], 6), (5, None),
                 (None, None), ([5, 6], None)):
        assert eos_union(*same) is None, same

    tok = PreTrainedTokenizerFast(tokenizer_object=byte_level_bpe(),
                                  eos_token="<|im_end|>",
                                  pad_token="<|endoftext|>")
    tok.chat_template = CHATML
    end = tok.eos_token_id
    other = tok.convert_tokens_to_ids("<|endoftext|>")
    assert generate_kwargs(tok, other).get("eos_token_id") == [other, end]
    assert generate_kwargs(tok, None).get("eos_token_id") == [end]
    for own in ([end, other], end, [other, end]):
        assert "eos_token_id" not in generate_kwargs(tok, own), own
    print("4. stop ids: the tokenizer's end token joins the model's own "
          "only when missing, and a model that has it passes nothing new")


class StubLoader:
    seen = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.seen = kwargs
        return SimpleNamespace(eval=lambda: None, config=None,
                               get_input_embeddings=None, device="cpu")


def check_dtype():
    for version, want in (("4.55.2", "torch_dtype"), ("4.55.0", "torch_dtype"),
                          ("3.9.0", "torch_dtype"), ("4.56.0", "dtype"),
                          ("4.56.0.dev0", "dtype"), ("4.57.6", "dtype"),
                          ("5.0.0rc1", "dtype"), ("5.5.3", "dtype"),
                          ("5.17.0", "dtype")):
        assert dtype_keyword(version) == want, (version, want)

    kept = runner_mod.AutoModelForCausalLM, runner_mod.load_tokenizer
    runner_mod.AutoModelForCausalLM = StubLoader
    runner_mod.load_tokenizer = lambda path: SimpleNamespace(
        pad_token="<pad>", eos_token="<eos>", model_max_length=None)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ChatRunner({"alias": "stub", "hf_id": "stub/stub",
                        "path": "/nonexistent", "dtype": "float16"},
                       device="cpu")
    finally:
        runner_mod.AutoModelForCausalLM, runner_mod.load_tokenizer = kept
    want = dtype_keyword(transformers.__version__)
    other = {"dtype": "torch_dtype", "torch_dtype": "dtype"}[want]
    assert StubLoader.seen.get(want) is torch.float16, StubLoader.seen
    assert other not in StubLoader.seen, StubLoader.seen
    print(f"5. dtype: torch_dtype below transformers 4.56, dtype from it, "
          f"and this {transformers.__version__} load passes {want}")


def main():
    check_fixtures()
    check_registered()
    check_eos()
    check_dtype()
    print("PASS")


if __name__ == "__main__":
    main()
