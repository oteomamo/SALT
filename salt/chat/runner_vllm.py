# -*- coding: utf-8 -*-
"""VLLMChatRunner: vLLM sibling of ChatRunner behind the same seam.

In-process V1 AsyncLLM with automatic prefix caching: the stable prompt head
(system message + verbatim tail) is reused from the GPU KV cache across
turns, so the per-turn prefill is only the SALT selection and the newest
exchange. Blocking generate for now; the streaming bridge lands next.
"""

import asyncio
import gc
import os
import threading

import torch
from transformers import AutoConfig, AutoTokenizer

from salt.chat.runner import (_model_input_limit, input_budget_for,
                              render_prompt)


class VLLMChatRunner:
    kind = "vllm"

    def __init__(self, cfg, device="cuda", gpu_memory_utilization=0.85,
                 max_model_len=0):
        try:
            from vllm import AsyncEngineArgs
            from vllm.v1.engine.async_llm import AsyncLLM
        except ImportError as exc:
            raise RuntimeError(
                "the vllm backend needs the optional vLLM install: "
                "pip install vllm==0.11.0 (README step 5)") from exc
        self.cfg = cfg
        self.device = device
        self.alias = cfg["alias"]
        if ":" in device:
            # parent CUDA is already live; this only steers the engine's
            # child process
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        print(f"Loading chat model {cfg['hf_id']} on {device} "
              f"[vLLM, {cfg.get('dtype', 'bfloat16')}, prefix caching on]")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever,
                                             daemon=True)
        self._loop_thread.start()
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=cfg["path"], tokenizer=cfg["path"],
            dtype=cfg.get("dtype", "bfloat16"),
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len or None,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            enforce_eager=bool(cfg.get("enforce_eager", False)),
            trust_remote_code=True))
        self.max_input_len = self._resolved_window()
        if self.max_input_len:
            print(f"Context window: {self.max_input_len} tokens")
        self._request_no = 0

    def _resolved_window(self):
        try:
            return int(self.engine.model_config.max_model_len)
        except Exception:
            try:
                config = AutoConfig.from_pretrained(self.cfg["path"])
            except Exception:
                config = None
            return _model_input_limit(config, self.tokenizer)

    def input_budget(self, max_new_tokens=None):
        return input_budget_for(self.max_input_len, self.cfg.get("gen"),
                                max_new_tokens)

    def stream_chat(self, messages, **overrides):
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        gen_cfg = dict(self.cfg.get("gen") or {})
        gen_cfg.update(overrides)
        temperature = float(gen_cfg.get("temperature", 0.7))
        do_sample = bool(gen_cfg.get("do_sample", temperature > 0))
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 512))

        prompt, used_chat = render_prompt(self.tokenizer, messages)
        ids = self.tokenizer(prompt, add_special_tokens=not used_chat).input_ids
        max_input = self.input_budget(max_new_tokens) or 0
        n_prompt = len(ids)
        if max_input > 0 and n_prompt > max_input:
            ids = ids[-max_input:]
            print(f"\nnote: prompt ({n_prompt} tokens) exceeds the context "
                  f"window - truncated to the last {max_input} tokens, "
                  f"earliest content (system prompt first) dropped")
        self.last_prompt_tokens = len(ids)

        if do_sample:
            params = SamplingParams(temperature=temperature,
                                    top_p=float(gen_cfg.get("top_p", 1.0)),
                                    max_tokens=max_new_tokens)
        else:
            params = SamplingParams(temperature=0.0,
                                    max_tokens=max_new_tokens)

        self._request_no += 1
        request_id = f"{self.alias}-{self._request_no}"

        async def _collect():
            final = None
            async for out in self.engine.generate(
                    TokensPrompt(prompt_token_ids=ids), params, request_id):
                final = out
            return final

        fut = asyncio.run_coroutine_threadsafe(_collect(), self._loop)
        try:
            final = fut.result()
        finally:
            if not fut.done():
                fut.cancel()
                asyncio.run_coroutine_threadsafe(
                    self.engine.abort(request_id), self._loop)
        if final is not None and final.outputs:
            yield final.outputs[0].text

    def unload(self):
        if self.engine is not None:
            try:
                self.engine.shutdown()
            except Exception:
                pass
        self.engine = None
        self.tokenizer = None
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
