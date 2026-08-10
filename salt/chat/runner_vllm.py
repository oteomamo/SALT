# -*- coding: utf-8 -*-
"""VLLMChatRunner: vLLM sibling of ChatRunner behind the same seam.

In-process V1 AsyncLLM with automatic prefix caching: the stable prompt head
(system message + verbatim tail) is reused from the GPU KV cache across
turns, so the per-turn prefill is only the SALT selection and the newest
exchange. A background thread owns the asyncio loop; text deltas cross to
the sync REPL through a queue, and dropping the generator aborts the request
(the Ctrl-C contract ChatRunner implements with _StopFlag).
"""

import asyncio
import gc
import os
import queue
import threading

import torch
from transformers import AutoConfig, AutoTokenizer

from salt.chat.runner import (_model_input_limit, input_budget_for,
                              render_prompt, TEMPLATE_KEY)


class VLLMChatRunner:
    kind = "vllm"

    def __init__(self, cfg, device="cuda", gpu_memory_utilization=0.85,
                 max_model_len=0, gpus=None):
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
        # several cards tensor-parallel the weights; one (or none) leaves the
        # engine's default of 1
        tp = len(gpus) if gpus and len(gpus) > 1 else 1
        note = f", tensor parallel x{tp}" if tp > 1 else ""
        print(f"Loading chat model {cfg['hf_id']} on {device} "
              f"[vLLM, {cfg.get('dtype', 'bfloat16')}, prefix caching "
              f"on{note}]")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever,
                                             daemon=True)
        self._loop_thread.start()
        # CUDA_VISIBLE_DEVICES steers the engine's worker processes (parent
        # CUDA is already live) and is restored right after: leaving it
        # mutated would hide GPUs from every later subprocess. A --gpu list
        # also pins PCI order so an index means the card nvidia-smi calls N,
        # the same numbering saltServe uses
        prev = {k: os.environ.get(k)
                for k in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")}
        if gpus:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        elif ":" in device:
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        mutated = bool(gpus) or ":" in device
        try:
            self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
                model=cfg["path"], tokenizer=cfg["path"],
                dtype=cfg.get("dtype", "bfloat16"),
                tensor_parallel_size=tp,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len or None,
                enable_prefix_caching=True,
                enable_chunked_prefill=True,
                enforce_eager=bool(cfg.get("enforce_eager", False)),
                trust_remote_code=True))
        except BaseException:
            self._loop.call_soon_threadsafe(self._loop.stop)
            raise
        finally:
            if mutated:
                for k, v in prev.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
        self.max_input_len = self._resolved_window()
        if self.max_input_len:
            print(f"Context window: {self.max_input_len} tokens")
        self._request_no = 0
        self.last_engine_stats = None

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
        template_kwargs = gen_cfg.pop(TEMPLATE_KEY, None)
        temperature = float(gen_cfg.get("temperature", 0.7))
        do_sample = bool(gen_cfg.get("do_sample", temperature > 0))
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 512))

        prompt, used_chat = render_prompt(self.tokenizer, messages,
                                          template_kwargs)
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
        q = queue.Queue()
        done = object()
        self.last_engine_stats = None
        last = {}

        async def _pump():
            try:
                async for out in self.engine.generate(
                        TokensPrompt(prompt_token_ids=ids), params,
                        request_id):
                    last["out"] = out
                    q.put(out.outputs[0].text if out.outputs else "")
            except Exception as exc:
                q.put(exc)
            finally:
                q.put(done)

        fut = asyncio.run_coroutine_threadsafe(_pump(), self._loop)
        sent = 0
        try:
            while True:
                item = q.get()
                if item is done:
                    break
                if isinstance(item, Exception):
                    raise item
                if len(item) > sent:
                    piece, sent = item[sent:], len(item)
                    yield piece
        finally:
            if not fut.done():
                # cancelling the consuming task aborts the request in V1
                fut.cancel()
            out = last.get("out")
            if out is not None:
                self.last_engine_stats = {
                    "engine_backend": "vllm",
                    "apc_cached_tokens": getattr(out, "num_cached_tokens",
                                                 None),
                    "apc_prompt_tokens": (len(out.prompt_token_ids)
                                          if out.prompt_token_ids
                                          else self.last_prompt_tokens),
                }

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
