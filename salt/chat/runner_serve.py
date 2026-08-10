# -*- coding: utf-8 -*-
"""VLLMServeChatRunner: HTTP client sibling of ChatRunner behind the seam.

Talks to a long-lived ``vllm serve`` process instead of loading a model, so
the server's prefix cache outlives the REPL: exit saltChat, resume the
session later, and the stable prompt head is still warm on the GPU. Prompts
are rendered and tokenized locally with the shared helpers and submitted as
token ids, byte-identical to the in-process backends. ``unload`` only
closes the connection - the server keeping its model is the point.
"""

import json

import requests
from transformers import AutoConfig, AutoTokenizer

from salt.chat.runner import (_model_input_limit, input_budget_for,
                              render_prompt, TEMPLATE_KEY)


# what a caller may put on the request body instead of into sampling:
# vLLM's structured-output demands. Named here rather than guessed at,
# so a stray generation setting can never become a body key
BODY_EXTRAS = ("guided_json", "guided_regex", "guided_choice",
               "guided_grammar", "response_format")


class VLLMServeChatRunner:
    kind = "vllm-serve"

    def __init__(self, cfg, device="cuda", server_url="http://127.0.0.1:8000",
                 read_timeout=None):
        self.cfg = cfg
        self.device = device
        self.alias = cfg["alias"]
        self.server_url = server_url.rstrip("/")
        # None = wait as long as the server takes, which is what the chat
        # model's own client has always done: a slow reply is still a reply
        # and there is nobody else to fall back to. A caller that has
        # somewhere else to go (an agent worker) passes a number.
        self.read_timeout = read_timeout
        self.http = requests.Session()
        try:
            resp = self.http.get(f"{self.server_url}/v1/models", timeout=5)
            resp.raise_for_status()
            cards = resp.json().get("data", [])
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(
                f"no vLLM server answering at {self.server_url} ({exc}); "
                f"start one with: saltServe {self.alias}") from exc
        accepted = {cfg.get(k) for k in ("alias", "hf_id", "path")}
        card = next((c for c in cards if c.get("id") in accepted), None)
        if card is None:
            served = ", ".join(str(c.get("id")) for c in cards) or "nothing"
            raise RuntimeError(
                f"the server at {self.server_url} serves {served}, not "
                f"{self.alias!r}; start one for it with: "
                f"saltServe {self.alias}")
        self.served_model = card["id"]
        print(f"Connected to vLLM server at {self.server_url} "
              f"[{self.served_model}, prefix cache lives with the server]")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["path"])
        self.max_input_len = self._resolved_window(card)
        if self.max_input_len:
            print(f"Context window: {self.max_input_len} tokens")
        self.last_prompt_tokens = None
        self.last_engine_stats = None

    def _resolved_window(self, card):
        v = card.get("max_model_len")
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
        try:
            config = AutoConfig.from_pretrained(self.cfg["path"])
        except Exception:
            config = None
        return _model_input_limit(config, self.tokenizer)

    def input_budget(self, max_new_tokens=None):
        return input_budget_for(self.max_input_len, self.cfg.get("gen"),
                                max_new_tokens)

    def stream_chat(self, messages, **overrides):
        gen_cfg = dict(self.cfg.get("gen") or {})
        gen_cfg.update(overrides)
        template_kwargs = gen_cfg.pop(TEMPLATE_KEY, None)
        temperature = float(gen_cfg.get("temperature", 0.7))
        do_sample = bool(gen_cfg.get("do_sample", temperature > 0))
        max_new_tokens = int(gen_cfg.get("max_new_tokens", 512))

        prompt, used_chat = render_prompt(self.tokenizer, messages,
                                          template_kwargs)
        ids = self.tokenizer(prompt,
                             add_special_tokens=not used_chat).input_ids
        max_input = self.input_budget(max_new_tokens) or 0
        n_prompt = len(ids)
        if max_input > 0 and n_prompt > max_input:
            ids = ids[-max_input:]
            print(f"\nnote: prompt ({n_prompt} tokens) exceeds the context "
                  f"window - truncated to the last {max_input} tokens, "
                  f"earliest content (system prompt first) dropped")
        self.last_prompt_tokens = len(ids)
        if self.max_input_len:
            # the server rejects prompt+reply past the window outright;
            # shrink the reply room like the in-process backends do
            max_new_tokens = min(max_new_tokens,
                                 self.max_input_len - len(ids))

        payload = {"model": self.served_model, "prompt": ids,
                   "max_tokens": max_new_tokens, "stream": True,
                   "stream_options": {"include_usage": True}}
        # a caller that wants the server to hold this reply to a shape
        # says so with one of these, and they ride the body rather than
        # the sampling parameters. Absent unless somebody sets one, so a
        # turn that asks for nothing sends exactly what it always sent
        for key in BODY_EXTRAS:
            if gen_cfg.get(key) is not None:
                payload[key] = gen_cfg[key]
        if do_sample:
            payload["temperature"] = temperature
            payload["top_p"] = float(gen_cfg.get("top_p", 1.0))
        else:
            payload["temperature"] = 0.0

        self.last_engine_stats = None
        usage = {}
        resp = self.http.post(f"{self.server_url}/v1/completions",
                              json=payload, stream=True,
                              timeout=(5, self.read_timeout))
        try:
            if resp.status_code != 200:
                detail = resp.text[:300]
                resp.close()
                raise RuntimeError("vLLM server rejected the request "
                                   f"(HTTP {resp.status_code}): {detail}")
            # bytes on purpose: only \n and \r\n end SSE lines, while str
            # splitting would also break on U+2028-class codepoints inside
            # the streamed text and shear the JSON frame
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace")
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                error = chunk.get("error")
                if error:
                    msg = (error.get("message") if isinstance(error, dict)
                           else None) or str(error)
                    raise RuntimeError(
                        f"vLLM server error mid-reply: {msg[:300]}")
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    piece = choice.get("text") or ""
                    if piece:
                        yield piece
        finally:
            # closing the response mid-stream disconnects the client, which
            # aborts the request server-side (the Ctrl-C contract)
            resp.close()
            stats = {"engine_backend": "vllm-serve",
                     "apc_cached_tokens": None,
                     "apc_prompt_tokens": self.last_prompt_tokens}
            if usage:
                stats["apc_prompt_tokens"] = usage.get(
                    "prompt_tokens", self.last_prompt_tokens)
                details = usage.get("prompt_tokens_details") or {}
                stats["apc_cached_tokens"] = details.get("cached_tokens")
            self.last_engine_stats = stats

    def unload(self):
        """The server keeps the model and its cache; only the client lets go."""
        self.http.close()
        self.tokenizer = None
