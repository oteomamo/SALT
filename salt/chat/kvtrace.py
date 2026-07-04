# -*- coding: utf-8 -*-
"""Per-conversation KV-cache read/write ledger for saltChat.

Records, per chat turn, which SALT-selected sentences were carried over from
the previous turn's context (KV READ - their cache would be reusable) versus
freshly selected (KV WRITE - new prefill), plus the generated reply (OUTPUT).
The per-turn totals use Langfuse's native usage-details key semantics
(`input` = write, `input_cached_tokens` = read, `output`, `total`;
every `input_*` key is an additive component of input), so a downstream
emitter can map one event line onto one `generation-create` observation.

Layout, alongside the session's state.pkl / embeddings.npy / attachments/:

    <session>/kvtrace/
      manifest.json    format version + column/enum/semantics documentation
      events.jsonl     append-only, one JSON record per turn
      tokens.npy       int32 matrix (n_tokens, 3): [turn, kind, sent_idx]
                       kind: 0=read, 1=write, 2=output; sent_idx=-1 for output

The event-stream + token-count shape follows the convergent industry design
(TensorRT-LLM KV events / vLLM BlockStored / Mooncake trace files); the
matrix is rewritten atomically each turn (same pattern as SessionTrie's
embeddings.npy) and events are line-appended, so a crash loses at most the
in-flight turn.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
KIND_READ, KIND_WRITE, KIND_OUTPUT = 0, 1, 2


class KVTrace:
    """Append-only per-turn KV read/write ledger under one session dir."""

    def __init__(self, session_dir, conversation_id=""):
        self.dir = Path(session_dir) / "kvtrace"
        self.conversation_id = conversation_id
        self.turn = 0
        self.prev_selected = set()
        self.totals = {"input": 0, "input_cached_tokens": 0, "output": 0}
        self.tokens = np.zeros((0, 3), dtype=np.int32)
        self.last_event = None
        self._load()

    # ── persistence ───────────────────────────────────────────────────────
    def _p(self, name):
        return self.dir / name

    def _load(self):
        ep = self._p("events.jsonl")
        if ep.exists():
            for line in ep.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue  # torn tail line from a crash - skip
                self.last_event = rec
                self.turn = rec.get("turn", self.turn) + 1
                self.prev_selected = set(rec.get("selected_sent_idx", []))
                for k in self.totals:
                    self.totals[k] += int(rec.get("usage", {}).get(k, 0))
        tp = self._p("tokens.npy")
        if tp.exists():
            try:
                self.tokens = np.load(tp)
            except Exception:
                self.tokens = np.zeros((0, 3), dtype=np.int32)
        # Reconcile the matrix with the last event's recorded span: a crash
        # between the matrix rewrite and the event append must not leave
        # orphan rows that shift or overlap future spans; external damage
        # (missing/short matrix) is padded with -1 sentinel rows instead.
        expected = (int(self.last_event["token_rows"][1])
                    if self.last_event else 0)
        if self.tokens.shape[0] > expected:
            self.tokens = self.tokens[:expected]
        elif self.tokens.shape[0] < expected:
            pad = np.full((expected - self.tokens.shape[0], 3), -1,
                          dtype=np.int32)
            self.tokens = (np.vstack([self.tokens, pad])
                           if self.tokens.size else pad)

    def _write_manifest(self):
        mp = self._p("manifest.json")
        if mp.exists():
            try:
                json.loads(mp.read_text())
                return
            except ValueError:
                pass  # torn first write - repair by rewriting atomically
        manifest = {
            "format_version": FORMAT_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "conversation_id": self.conversation_id,
            "events_file": "events.jsonl",
            "tokens_file": "tokens.npy",
            "tokens_columns": ["turn", "kind", "sent_idx"],
            "kind_enum": {"read": KIND_READ, "write": KIND_WRITE,
                          "output": KIND_OUTPUT},
            "lost_row_sentinel": "[-1,-1,-1] rows pad regions lost to "
                                 "external file damage",
            "token_counting": "active model tokenizer, add_special_tokens=False",
            "usage_semantics": {
                "input": "KV WRITE: newly selected (freshly prefilled) sentences",
                "input_cached_tokens": "KV READ: sentences re-selected from "
                                       "the previous turn's context",
                "output": "generated reply tokens",
                "total": "input + input_cached_tokens + output",
            },
        }
        tmp = self._p("manifest.json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2))
        os.replace(tmp, mp)

    def _save_tokens(self):
        tmp = self._p("tokens.npy.tmp")
        with open(tmp, "wb") as fh:
            np.save(fh, self.tokens)
        os.replace(tmp, self._p("tokens.npy"))

    # ── recording ─────────────────────────────────────────────────────────
    def record_turn(self, *, tokenizer, trie, selected_idx, reply_text,
                    model_id, ts_start, ts_end, prompt_tokens=None,
                    extra=None):
        """Append one turn's ledger entry; returns the event record."""
        self.dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest()

        def n_tok(text):
            try:
                return len(tokenizer(text, add_special_tokens=False).input_ids)
            except Exception:
                return max(1, len(text.split()))

        curr = set(selected_idx)
        read_idx = sorted(curr & self.prev_selected)
        write_idx = sorted(curr - self.prev_selected)

        rows, read_tok, write_tok = [], 0, 0
        for i in sorted(curr):  # document order, mirroring the prompt
            kind = KIND_READ if i in self.prev_selected else KIND_WRITE
            n = n_tok(trie.texts[i])
            if kind == KIND_READ:
                read_tok += n
            else:
                write_tok += n
            rows.extend((self.turn, kind, i) for _ in range(n))
        out_tok = n_tok(reply_text) if reply_text else 0
        rows.extend((self.turn, KIND_OUTPUT, -1) for _ in range(out_tok))

        usage = {"input": write_tok, "input_cached_tokens": read_tok,
                 "output": out_tok,
                 "total": write_tok + read_tok + out_tok}
        row_start = int(self.tokens.shape[0])
        event = {
            "v": FORMAT_VERSION,
            "turn": self.turn,
            "conversation_id": self.conversation_id,
            "model": model_id,
            "ts_start": ts_start,
            "ts_end": ts_end,
            "usage": usage,
            "words": {
                "read": int(sum(trie.n_words[i] for i in read_idx)),
                "write": int(sum(trie.n_words[i] for i in write_idx)),
            },
            "selected_sent_idx": sorted(curr),
            "read_sent_idx": read_idx,
            "write_sent_idx": write_idx,
            "prompt_tokens": prompt_tokens,
            "token_rows": [row_start, row_start + len(rows)],
        }
        if extra:
            event.update(extra)
        try:
            if rows:
                self.tokens = np.vstack(
                    [self.tokens, np.array(rows, dtype=np.int32)])
            self._save_tokens()
            with open(self._p("events.jsonl"), "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
        except Exception:
            # nothing references these rows - roll the matrix back so the
            # turn id is not silently reused over orphan rows (disk rollback
            # is best-effort; load-time reconciliation is the safety net)
            self.tokens = self.tokens[:row_start]
            try:
                self._save_tokens()
            except Exception:
                pass
            raise

        self.prev_selected = curr
        self.turn += 1
        for k in self.totals:
            self.totals[k] += usage[k]
        self.last_event = event
        return event
