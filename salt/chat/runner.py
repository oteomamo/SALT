# -*- coding: utf-8 -*-
"""ChatRunner: the chat LLM pinned on GPU for a whole saltChat session.

One HF-transformers model + tokenizer, loaded once from a registry entry and
kept resident. ``stream_chat`` yields decoded text pieces as they are
generated (``TextIteratorStreamer`` fed by a background generate thread, the
same pattern as C2C's demo). ``unload`` frees the GPU so a ``/model`` switch
never holds two models at once. Sibling runners slot in behind the same
surface: load / stream_chat / unload plus the ``max_input_len``,
``input_budget()``, ``last_prompt_tokens``, ``tokenizer``, ``alias``, and
``cfg`` members the REPL reads. ``make_runner`` maps a backend name to a
runner class; ``render_prompt`` / ``input_budget_for`` are shared so every
backend renders identical prompts and honors the same reply headroom.
"""

import gc
import threading

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList,
                          TextIteratorStreamer)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}


def _model_input_limit(config, tokenizer):
    """The model's own context window, read from its config; the tokenizer
    limit is the fallback (it over-claims on some models, so config wins).
    Returns None when neither source knows one (no truncation then). This is
    deliberately NOT a per-machine knob: the only input ceiling saltChat
    honors is the published model length."""
    candidates = [getattr(config, attr, None)
                  for attr in ("max_position_embeddings", "n_positions",
                               "seq_length", "max_seq_len",
                               "max_sequence_length")]
    candidates.append(getattr(tokenizer, "model_max_length", None))
    for v in candidates:
        # accept float-written ints from hand-edited configs; reject bool
        # (an int in Python) and HF's huge "unset" sentinel
        if (isinstance(v, (int, float)) and not isinstance(v, bool)
                and 0 < v < 10 ** 9):
            return int(v)
    return None


# A generation setting that reaches the chat template rather than the
# sampler or the request body. Named here so a caller can set it the way
# it sets anything else, and popped by every backend before the settings
# it does not belong in: a sampler handed a dict it cannot use is an
# error, and a request body carrying it would be dead weight, since
# every backend renders its prompt locally
TEMPLATE_KEY = "chat_template_kwargs"


def render_prompt(tokenizer, messages, template_kwargs=None):
    """Chat-template the messages; plain-concat fallback for models without
    a template (mirrors eval.apply_chat). Returns (text, used_chat_template).

    ``template_kwargs`` are handed to the template itself, for the few
    settings a model exposes there rather than through sampling. Empty is
    the same bytes as passing nothing, which is what keeps a prompt this
    session has always sent unchanged and its prefix cache warm. A model
    with no template has no template settings either, so the fallback
    ignores them rather than pretending to honour them.

    A setting the template refuses costs the setting and not the
    template: it is asked again with nothing, and only a model with no
    usable template at all reaches the plain-concat fallback. Losing the
    template over an optional setting would rewrite the whole prompt,
    which is a far larger change than the one that was asked for.
    """
    if template_kwargs:
        try:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **template_kwargs), True
        except Exception:
            pass
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True), True
    except Exception:
        text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return text + "\n\nassistant:", False


def input_budget_for(max_input_len, gen_cfg, max_new_tokens=None):
    """Effective prompt ceiling: the context window minus reply headroom.
    Headroom is capped at half the window so a broken gen config can never
    un-cap the prompt. None when the window is unknown."""
    if not max_input_len:
        return None
    if max_new_tokens is None:
        max_new_tokens = int((gen_cfg or {}).get("max_new_tokens", 512))
    return max_input_len - min(max_new_tokens, max_input_len // 2)


def sampling_for(gen_cfg):
    """Temperature and whether to sample, from a merged gen config.

    Temperature 0 means greedy decoding whatever the config's do_sample
    says. A registry entry that samples by default would otherwise turn
    a caller's 0 into a sampler asked to draw at zero temperature,
    which the generate call refuses - and 0 is exactly what a planning
    call asks for."""
    temperature = float(gen_cfg.get("temperature", 0.7))
    do_sample = bool(gen_cfg.get("do_sample", temperature > 0))
    if temperature <= 0:
        do_sample = False
    return temperature, do_sample


def _cuda_total_memory(idx):
    return torch.cuda.get_device_properties(idx).total_memory


def hf_placement(gpus, device, fraction, total_memory=_cuda_total_memory):
    """Choose how the HF chat model loads. One card (or none) keeps the
    plain device string, unchanged. Several cards return device_map
    'balanced' plus a max_memory cap per card - a fraction of each card's
    total - so accelerate shards the weights across exactly the listed
    cards and leaves room for activations and the BGE encoder on the last
    one. total_memory(idx) yields a card's total bytes. Returns
    (device_map, max_memory)."""
    if not gpus or len(gpus) <= 1:
        return device, None
    max_memory = {int(g): int(total_memory(int(g)) * fraction) for g in gpus}
    return "balanced", max_memory


def make_runner(cfg, device="cuda", backend="hf", **backend_opts):
    if backend == "hf":
        return ChatRunner(cfg, device=device, **backend_opts)
    if backend == "vllm":
        from salt.chat.runner_vllm import VLLMChatRunner  # lazy: optional dep
        return VLLMChatRunner(cfg, device=device, **backend_opts)
    if backend == "vllm-serve":
        from salt.chat.runner_serve import VLLMServeChatRunner
        return VLLMServeChatRunner(cfg, device=device, **backend_opts)
    raise ValueError(f"Unknown chat backend {backend!r} "
                     "(available: hf, vllm, vllm-serve)")


class _StopFlag(StoppingCriteria):
    """Lets the consumer abort the background generate thread (Ctrl-C)."""

    def __init__(self):
        self.stop = False

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop


class ChatRunner:
    kind = "hf"

    def __init__(self, cfg, device="cuda", gpus=None,
                 gpu_memory_utilization=0.80):
        self.cfg = cfg
        self.device = device
        self.alias = cfg["alias"]
        dtype = DTYPES.get(cfg.get("dtype"), torch.bfloat16)
        device_map, max_memory = hf_placement(gpus, device,
                                              gpu_memory_utilization)
        where = (f"sharded across GPUs {','.join(gpus)}"
                 if max_memory is not None else f"on {device}")
        print(f"Loading chat model {cfg['hf_id']} {where} "
              f"[{cfg.get('dtype', 'bfloat16')}, "
              f"attn={cfg.get('attn_implementation', 'sdpa')}]")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=dtype, device_map=device_map,
            max_memory=max_memory,
            attn_implementation=cfg.get("attn_implementation", "sdpa"))
        self.model.eval()
        # generation inputs start on the first shard (where the embeddings
        # live); with a device_map the model spans several cards, so
        # model.device is not a single reliable target
        try:
            self.input_device = \
                self.model.get_input_embeddings().weight.device
        except Exception:
            self.input_device = getattr(self.model, "device", None) or device
        self.max_input_len = _model_input_limit(self.model.config,
                                                self.tokenizer)
        if self.max_input_len:
            print(f"Context window: {self.max_input_len} tokens")
        if cfg.get("max_input_len"):
            print("note: this entry's max_input_len is no longer used - "
                  "the model's own context window is the only ceiling")

    def input_budget(self, max_new_tokens=None):
        """Effective prompt ceiling; see input_budget_for."""
        return input_budget_for(self.max_input_len, self.cfg.get("gen"),
                                max_new_tokens)

    def stream_chat(self, messages, **overrides):
        """Yield decoded text pieces for a chat turn as they are generated."""
        gen_cfg = dict(self.cfg.get("gen") or {})
        gen_cfg.update(overrides)
        template_kwargs = gen_cfg.pop(TEMPLATE_KEY, None)
        temperature, do_sample = sampling_for(gen_cfg)
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 512))

        prompt, used_chat = render_prompt(self.tokenizer, messages,
                                          template_kwargs)
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                add_special_tokens=not used_chat)
        # the only input ceiling is the model's own context window, minus
        # headroom for the reply so generation never runs past the window
        max_input = self.input_budget(max_new_tokens) or 0
        n_prompt = int(inputs["input_ids"].shape[-1])
        if max_input > 0 and n_prompt > max_input:
            # keep the tail: the generation prompt and recent turns must
            # survive; smarter budget-aware trimming is a later seam
            inputs = {k: v[:, -max_input:] for k, v in inputs.items()}
            print(f"\nnote: prompt ({n_prompt} tokens) exceeds the context "
                  f"window - truncated to the last {max_input} tokens, "
                  f"earliest content (system prompt first) dropped")
        # observed prompt size after truncation, for the KV-trace ledger
        self.last_prompt_tokens = int(inputs["input_ids"].shape[-1])

        gen_kwargs = dict(max_new_tokens=max_new_tokens,
                          do_sample=do_sample,
                          pad_token_id=self.tokenizer.pad_token_id)
        if do_sample:
            gen_kwargs["temperature"] = temperature
            if "top_p" in gen_cfg:
                gen_kwargs["top_p"] = float(gen_cfg["top_p"])

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)
        stop_flag = _StopFlag()
        gen_kwargs.update(
            {k: v.to(self.input_device) for k, v in inputs.items()},
            streamer=streamer,
            stopping_criteria=StoppingCriteriaList([stop_flag]))
        self._gen_error = None
        thread = threading.Thread(
            target=self._generate, args=(gen_kwargs, streamer), daemon=True)
        thread.start()
        try:
            yield from streamer
        finally:
            stop_flag.stop = True
            thread.join()
        if self._gen_error is not None:
            raise self._gen_error

    def _generate(self, gen_kwargs, streamer):
        try:
            with torch.no_grad():
                self.model.generate(**gen_kwargs)
        except Exception as exc:
            self._gen_error = exc
            streamer.end()

    def unload(self):
        """Free the GPU before loading a different model."""
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
