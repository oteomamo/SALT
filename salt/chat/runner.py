# -*- coding: utf-8 -*-
"""ChatRunner: the chat LLM pinned on GPU for a whole saltChat session.

One HF-transformers model + tokenizer, loaded once from a registry entry and
kept resident. ``stream_chat`` yields decoded text pieces as they are
generated (``TextIteratorStreamer`` fed by a background generate thread, the
same pattern as C2C's demo). ``unload`` frees the GPU so a ``/model`` switch
never holds two models at once. A vLLM runner can slot in later as a sibling
class behind the same load / stream_chat / unload surface.
"""

import gc
import threading

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList,
                          TextIteratorStreamer)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}


class _StopFlag(StoppingCriteria):
    """Lets the consumer abort the background generate thread (Ctrl-C)."""

    def __init__(self):
        self.stop = False

    def __call__(self, input_ids, scores, **kwargs):
        return self.stop


class ChatRunner:
    kind = "hf"

    def __init__(self, cfg, device="cuda"):
        self.cfg = cfg
        self.device = device
        self.alias = cfg["alias"]
        dtype = DTYPES.get(cfg.get("dtype"), torch.bfloat16)
        print(f"Loading chat model {cfg['hf_id']} on {device} "
              f"[{cfg.get('dtype', 'bfloat16')}, "
              f"attn={cfg.get('attn_implementation', 'sdpa')}]")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=dtype, device_map=device,
            attn_implementation=cfg.get("attn_implementation", "sdpa"))
        self.model.eval()

    def _to_prompt(self, messages):
        """Chat-template the messages; plain-concat fallback for models
        without a template (mirrors eval.apply_chat)."""
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True), True
        except Exception:
            text = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)
            return text + "\n\nassistant:", False

    def stream_chat(self, messages, **overrides):
        """Yield decoded text pieces for a chat turn as they are generated."""
        prompt, used_chat = self._to_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt",
                                add_special_tokens=not used_chat)
        max_input = int(self.cfg.get("max_input_len") or 0)
        if max_input > 0 and inputs["input_ids"].shape[-1] > max_input:
            # keep the tail: the generation prompt and recent turns must
            # survive; smarter budget-aware trimming is a later seam
            inputs = {k: v[:, -max_input:] for k, v in inputs.items()}
        # observed prompt size after truncation, for the KV-trace ledger
        self.last_prompt_tokens = int(inputs["input_ids"].shape[-1])

        gen_cfg = dict(self.cfg.get("gen") or {})
        gen_cfg.update(overrides)
        temperature = float(gen_cfg.get("temperature", 0.7))
        do_sample = bool(gen_cfg.get("do_sample", temperature > 0))
        gen_kwargs = dict(max_new_tokens=int(gen_cfg.get("max_new_tokens", 512)),
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
            {k: v.to(self.model.device) for k, v in inputs.items()},
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
