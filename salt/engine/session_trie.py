# -*- coding: utf-8 -*-
"""
SessionTrie: a persistent, continuously-growing keyword-trie cache for
multi-turn chat, built on SALT's default `coverage` selector.

The one-shot connector (`compress.py`) rebuilds the trie from scratch for every
request and throws all state away. A chat tool instead wants ONE trie per
conversation that is (1) reused every turn, (2) grown every turn as new content
arrives (user messages, attached docs, assistant replies), and (3) stored in a
reusable on-disk cache. SessionTrie provides exactly that, reusing the shared
primitives in `salt.engine.trie_core` so selection behavior is identical to the
batch compressor.

Design:
  * `add_turn(text, role)` runs the two expensive model passes (dense-attention
    keywords + BGE [CLS] embeddings) ONLY on the new sentences, then appends them
    to an append-only corpus. Old sentences are never re-embedded.
  * `compress(query, budget_pct)` rebuilds the cheap trie over ALL sentences,
    SEEDS the prior-turn per-node coverage so already-surfaced material is
    discounted, runs `coverage_select`, and persists the merged coverage. Each
    turn therefore favors still-uncovered content (cross-turn submodular memory).
  * Cross-turn coverage is keyed by the canonical frozenset of each trie node's
    root->node keyword path, so it survives the trie being rebuilt (and grown)
    every turn — see `coverage_select(seed_coverage=...)` in salt.engine.celf.

Persistence mirrors `GroupStore`: a per-conversation `cache_dir/<id>/` holding
`embeddings.npy` + `state.pkl` + `config.json`, written atomically.

Usage:
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    mdl = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5",
                                    output_attentions=True).eval()
    st = SessionTrie("conv1", cache_dir="/tmp/salt_sessions")
    st.add_turn(long_document, role="doc", tokenizer=tok, model=mdl)
    out = st.compress("What were the main findings?", 0.2, tokenizer=tok, model=mdl)
    print(out["context"])
"""

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from salt.engine.embedder import split_sentences as embed_split_sentences
from salt.engine.sentence_filter import filter_texts, clean_text_for_embedding
from salt.engine.trie_core import (
    run_dense_attention, get_bge_sentence_embeddings, embed_query,
    profile_themes, is_content_word,
    extract_query_keywords, extract_proper_nouns_in_query,
)
from salt.engine.celf import coverage_select

VALID_ROLES = ("user", "assistant", "doc")
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class SessionTrie:
    """Persistent, growing per-conversation trie for multi-turn compression."""

    def __init__(self, conversation_id, cache_dir=None,
                 model_name=DEFAULT_MODEL, theme_percentile=0.9,
                 lam=0.5, query_mass_ratio=1.0, max_keywords_ratio=0.4,
                 budget_pct_default=0.20, max_sentences=None):
        base = Path(cache_dir) if cache_dir else (Path(__file__).parent / "session_cache")
        self.conversation_id = conversation_id
        self.cache_dir = base / conversation_id
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # --- append-only corpus (one entry per kept sentence) ---
        self.texts = []                 # list[str]
        self.roles = []                 # list[str] in VALID_ROLES
        self.turns = []                 # list[int] turn index the sentence entered on
        self.n_words = []               # list[int]
        self.keyword_weights = []       # list[dict[str, float]]  (cached attention keywords)
        self.embeddings = None          # np.ndarray (n, dim) float32  (cached BGE [CLS])

        # --- cross-turn state ---
        self.coverage = {}              # dict[frozenset[str], float]  accumulated per-node coverage
        self._seen_hashes = set()       # cross-turn sentence dedupe

        # --- counters / config ---
        self.dim = None
        self._next_sentence_index = 0
        self._next_turn_index = 0
        self.config = {
            "conversation_id": conversation_id,
            "model_name": model_name,
            "dim": None,
            "theme_percentile": theme_percentile,
            "lam": lam,
            "query_mass_ratio": query_mass_ratio,
            "max_keywords_ratio": max_keywords_ratio,
            "budget_pct_default": budget_pct_default,
            "max_sentences": max_sentences,
        }

        self._loaded = self.load()

    # ── properties ────────────────────────────────────────────────────────
    @property
    def is_loaded(self):
        return self._loaded

    @property
    def n_sentences(self):
        return len(self.texts)

    @property
    def n_turns(self):
        return self._next_turn_index

    # ── ingest ────────────────────────────────────────────────────────────
    @staticmethod
    def _norm_hash(text):
        return hashlib.sha1(" ".join(text.lower().split()).encode("utf-8")).hexdigest()

    def add_turn(self, text, role="user", *, tokenizer, model, device="cpu"):
        """Split/filter/encode NEW text and append it to the growing corpus.

        Runs the dense-attention keyword pass and the BGE [CLS] embedding pass on
        the new sentences only (old sentences are never re-encoded). `role` is
        stored for provenance; it does not affect weighting (user/assistant/doc
        text is ingested identically).
        """
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")
        text = (text or "").strip()
        if not text:
            return {"added": 0, "filtered": 0, "n_total": self.n_sentences,
                    "turn": self._next_turn_index}

        cleaned = clean_text_for_embedding(text)
        raw = embed_split_sentences(cleaned, max_tokens=400, tokenizer=tokenizer)
        all_texts = [s for s, _, _ in raw]
        sentences, *_ = filter_texts(all_texts, aggressive=True,
                                     remove_urls=True, deduplicate=True)
        # cross-turn dedupe: drop sentences already present in the corpus
        fresh = []
        for s in sentences:
            h = self._norm_hash(s)
            if h not in self._seen_hashes:
                self._seen_hashes.add(h)
                fresh.append(s)
        n_filtered = len(all_texts) - len(fresh)
        if not fresh:
            turn = self._next_turn_index
            self._next_turn_index += 1
            return {"added": 0, "filtered": n_filtered, "n_total": self.n_sentences,
                    "turn": turn}

        # --- the two expensive model passes, on NEW sentences only ---
        attn = run_dense_attention(
            fresh, tokenizer, model,
            max_keywords_ratio=self.config["max_keywords_ratio"], overlap_sents=2)
        bge = np.asarray(
            get_bge_sentence_embeddings(fresh, tokenizer, model, device),
            dtype=np.float32)

        if self.dim is None:
            self.dim = int(bge.shape[1])
            self.config["dim"] = self.dim
        elif bge.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dim {bge.shape[1]} != session dim {self.dim}. "
                f"Clear the cache dir if changing the embedding model.")

        turn = self._next_turn_index
        for i, sent in enumerate(fresh):
            ar = attn[i]
            kw = {}
            for j, w in enumerate(ar["word_labels"]):
                if w in ar["important_words"] and is_content_word(w):
                    if j < len(ar["word_attns"]):
                        kw[w] = max(kw.get(w, 0.0), float(ar["word_attns"][j]))
            self.texts.append(sent)
            self.roles.append(role)
            self.turns.append(turn)
            self.n_words.append(len(sent.split()))
            self.keyword_weights.append(kw)
            self._next_sentence_index += 1

        self.embeddings = (bge if self.embeddings is None
                           else np.vstack([self.embeddings, bge]))
        self._next_turn_index += 1
        self._evict_if_needed()
        self.save()
        return {"added": len(fresh), "filtered": n_filtered,
                "n_total": self.n_sentences, "turn": turn}

    def _evict_if_needed(self):
        """v1 keeps everything; `max_sentences` is a hook for a future bounded
        policy (e.g. keep last K turns + all `doc` sentences). Stale coverage
        keys are harmless, so eviction would only prune the corpus columns."""
        cap = self.config.get("max_sentences")
        if cap is None or self.n_sentences <= cap:
            return
        # TODO: bounded eviction (drop oldest non-doc sentences) once needed.

    # ── compression ───────────────────────────────────────────────────────
    def _sent_data(self):
        return [{"sent_idx": i, "text": self.texts[i], "n_words": self.n_words[i],
                 "keyword_weights": self.keyword_weights[i],
                 "embedding_l2": self.embeddings[i]}
                for i in range(self.n_sentences)]

    def compress(self, query="", budget_pct=None, *, tokenizer, model,
                 device="cpu", delimiter=" "):
        """Compress the accumulated corpus for `query`, reusing the persisted
        trie + cross-turn coverage.

        Returns {context, stats, selected_sent_idx, n_total_sentences, n_turns}.
        When `query` is empty, selection is document-coverage only (no query
        nodes). Coverage from this turn is merged into the persisted state.
        """
        budget_pct = self.config["budget_pct_default"] if budget_pct is None else budget_pct
        if self.n_sentences == 0:
            return {"context": "", "stats": {}, "selected_sent_idx": [],
                    "n_total_sentences": 0, "n_turns": self.n_turns}

        sent_data = self._sent_data()
        kw_df, theme_keywords = profile_themes(
            sent_data, theme_percentile=self.config["theme_percentile"])
        orig_words = sum(self.n_words)
        word_budget = int(orig_words * budget_pct)

        query = (query or "").strip()
        q_kws = q_pns = q_emb = None
        if query:
            q_kws = extract_query_keywords(query)
            q_pns = extract_proper_nouns_in_query(query)
            q_emb = embed_query(query, tokenizer, model, device)

        selected, stats, cov = coverage_select(
            sent_data, dict(kw_df), theme_keywords, word_budget,
            query_keywords=q_kws, query_embedding=q_emb, query_proper_nouns=q_pns,
            lam=self.config["lam"], query_mass_ratio=self.config["query_mass_ratio"],
            seed_coverage=self.coverage, return_coverage=True)

        self.coverage = cov["coverage"]
        self.save()

        sel_idx = [sr.sent_idx for sr in selected]
        context = delimiter.join(sr.text for sr in selected)
        return {"context": context, "stats": stats, "selected_sent_idx": sel_idx,
                "n_total_sentences": self.n_sentences, "n_turns": self.n_turns}

    # ── persistence ───────────────────────────────────────────────────────
    def _p(self, name):
        return self.cache_dir / name

    def _atomic_write(self, path, write_fn):
        tmp = path.with_suffix(path.suffix + ".tmp")
        write_fn(tmp)
        os.replace(tmp, path)

    def save(self):
        self.config["dim"] = self.dim
        self.config["n_sentences"] = self.n_sentences
        self.config["n_turns"] = self.n_turns
        if self.embeddings is not None:
            # np.save appends ".npy" to a path arg; write via a handle so the
            # atomic ".tmp" name is preserved.
            def _save_npy(p):
                with open(p, "wb") as fh:
                    np.save(fh, self.embeddings)
            self._atomic_write(self._p("embeddings.npy"), _save_npy)
        state = {
            "texts": self.texts, "roles": self.roles, "turns": self.turns,
            "n_words": self.n_words, "keyword_weights": self.keyword_weights,
            "coverage": self.coverage, "seen_hashes": self._seen_hashes,
            "next_sentence_index": self._next_sentence_index,
            "next_turn_index": self._next_turn_index,
        }
        self._atomic_write(self._p("state.pkl"),
                           lambda p: p.write_bytes(pickle.dumps(state)))
        self._atomic_write(self._p("config.json"),
                           lambda p: p.write_text(json.dumps(self.config, indent=2)))

    def load(self):
        sp = self._p("state.pkl")
        if not sp.exists():
            return False
        cfg = json.loads(self._p("config.json").read_text())
        # consistency guard (mirrors GroupStore's dim/model check)
        if cfg.get("model_name") != self.config["model_name"]:
            raise ValueError(
                f"Cache model {cfg.get('model_name')!r} != requested "
                f"{self.config['model_name']!r}. Clear {self.cache_dir} to switch models.")
        state = pickle.loads(sp.read_bytes())
        self.texts = state["texts"]; self.roles = state["roles"]
        self.turns = state["turns"]; self.n_words = state["n_words"]
        self.keyword_weights = state["keyword_weights"]
        self.coverage = state["coverage"]; self._seen_hashes = state["seen_hashes"]
        self._next_sentence_index = state["next_sentence_index"]
        self._next_turn_index = state["next_turn_index"]
        self.dim = cfg.get("dim")
        # preserve persisted hyperparams but keep this process's overrides where set
        for k in ("theme_percentile", "lam", "query_mass_ratio",
                  "max_keywords_ratio", "budget_pct_default", "max_sentences"):
            if k in cfg:
                self.config[k] = cfg[k]
        self.config["dim"] = self.dim
        ep = self._p("embeddings.npy")
        self.embeddings = np.load(ep) if ep.exists() else None
        return True
