# -*- coding: utf-8 -*-
"""
SessionTrie: a persistent, continuously-growing keyword-trie cache for
multi-turn chat, built on SALT's default `coverage` selector.

The one-shot connector (`salt/compress.py`) rebuilds the trie from scratch for every
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
import time
from pathlib import Path

import numpy as np

from salt.engine.chat_text import (clean_chat_text, is_protected_chat_unit,
                                   resolve_chat_urls)
from salt.engine.embedder import split_sentences as embed_split_sentences
from salt.engine.sentence_filter import filter_texts
from salt.engine.trie_core import (
    run_dense_attention, get_bge_sentence_embeddings, embed_query,
    profile_themes, is_content_word,
    extract_query_keywords, extract_proper_nouns_in_query,
)
from salt.engine.celf import coverage_select

VALID_ROLES = ("user", "assistant", "doc")
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# Synthetic root keyword for per-file trie branches. Sentences ingested with a
# `source` get this token injected at compress time with a document frequency
# above every real keyword, so path SF-ordering places it first and the whole
# file hangs off the shared root as ONE branch (its own sub-trie). CELF's
# branch discounting then spreads budget across files + conversation themes
# instead of letting one large attachment flood selection. The section sign
# cannot appear in tokenized text keywords, so collisions are impossible.
FILE_TOKEN_PREFIX = "§file:"

# Coverage-decay floor: entries whose decayed count falls below this are
# dropped from the persisted dict entirely. Content is never deleted — only
# a node's residual suppression — and the floor is what keeps the coverage
# dict bounded over long sessions instead of growing forever.
COVERAGE_DECAY_FLOOR = 0.05

# Topic-shift drift detection. Each query's BGE cosine against the mean of
# the last DRIFT_WINDOW conversation-sentence embeddings is compared to the
# session's own EMA baseline — BGE cosines have a high, corpus-shaped floor,
# so an absolute threshold would be fragile across models and domains.
# Detection runs whenever the turn has a query and the conversation holds at
# least DRIFT_MIN_SENTENCES sentences (below that the baseline is noise);
# it only ACTS (seed damping + query boost) behind `shift_damping`.
DRIFT_WINDOW = 12
DRIFT_MIN_SENTENCES = 6
DRIFT_EMA_ALPHA = 0.3

# On a shift turn only STALE suppression is damped: coverage keys that were
# incremented within the last SHIFT_FRESH_WINDOW compress calls keep their
# full counts. Scaling the whole seed uniformly would hand the freed
# budget straight back to the topic being pivoted AWAY from — its
# suppression is the largest and freshest, so lifting it too cancels the
# very discount that was keeping it out. Freshness, not source, is the
# discriminator: a stale attachment branch is damped like a stale
# conversation theme, so a pivot back to a document also resurfaces it.
#
# Known granularity limits of the freshness proxy:
# staleness is per trie NODE, not per topic — a long single-topic phase
# leaves its own early, already-covered nodes stale, so the pivot-away turn
# also briefly relaxes them; and the damped turn's own selections re-stamp
# the returned topic fresh, so the trie-side relief lasts one turn per
# return (a sustained return rides the query channels and the tail until
# its keys go stale again). With coverage decay (A) also active, decayed
# conversation keys shrink toward the floor while doc branches stay exempt,
# so most remaining stale MASS sits in attachments — damping then mostly
# re-opens documents on pivot turns.
SHIFT_FRESH_WINDOW = 3


def file_token(source):
    return FILE_TOKEN_PREFIX + source


def _chunk_by_tokens(sent, tokenizer, max_tokens=400, _depth=0):
    """Word-boundary bisection for over-long pre-split sentences (BGE
    truncates past ~512 tokens, so an over-budget unit would lose its tail).
    Bisects recursively because token density is uneven — a proportional
    word split cannot guarantee the cap when some words tokenize heavily."""
    try:
        n = len(tokenizer.encode(sent, add_special_tokens=False))
    except Exception:
        n = len(sent) // 4
    words = sent.split()
    if n <= max_tokens or len(words) < 2 or _depth >= 8:
        return [sent]
    mid = len(words) // 2
    return (_chunk_by_tokens(" ".join(words[:mid]), tokenizer, max_tokens, _depth + 1)
            + _chunk_by_tokens(" ".join(words[mid:]), tokenizer, max_tokens, _depth + 1))


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
        self.sources = []               # list[str|None]: None = conversation,
                                        #   else attachment id -> per-file branch
        # wall-clock ingest time, one stamp per add_turn call shared by that
        # message's sentences: this is when the text was FILED, not when it
        # was authored, and a message is filed once
        self.timestamps = []            # list[float|None] (None = pre-2.9.20)
        self.n_words = []               # list[int]
        self.keyword_weights = []       # list[dict[str, float]]  (cached attention keywords)
        self.embeddings = None          # np.ndarray (n, dim) float32  (cached BGE [CLS])

        # --- cross-turn state ---
        self.coverage = {}              # dict[frozenset[str], float]  accumulated per-node coverage
        self._seen_hashes = set()       # cross-turn sentence dedupe
        self.drift_ema = None           # EMA of query-vs-recent-conversation cosine
        self.coverage_turn = {}         # dict[key -> compress call of last increment]
        self._n_compress = 0            # completed compress() calls (freshness clock)
        self.n_near_dups = 0            # sentences suppressed by the near-dup gate
        self.dirty = False              # unsaved add_turn(save=False) mutations
        self.load_repair = None         # last load's reconcile record (None = clean)

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

    @property
    def attached_sources(self):
        return sorted({s for s in self.sources if s})

    # ── ingest ────────────────────────────────────────────────────────────
    @staticmethod
    def _norm_hash(text):
        return hashlib.sha1(" ".join(text.lower().split()).encode("utf-8")).hexdigest()

    def add_turn(self, text, role="user", *, tokenizer, model, device="cpu",
                 source=None, sentences=None, keep=None, dedup_cos=None,
                 save=True):
        """Split/filter/encode NEW text and append it to the growing corpus.

        Runs the dense-attention keyword pass and the BGE [CLS] embedding pass on
        the new sentences only (old sentences are never re-encoded). `role` is
        stored for provenance; it does not affect weighting (user/assistant/doc
        text is ingested identically). `source` (e.g. an attached file's name)
        groups the sentences into their own trie branch at compress time —
        see FILE_TOKEN_PREFIX; None means the main conversation trie.

        `sentences`, when given, is an already-segmented unit list (e.g. from
        `salt.chat.pdfio.split_document_sentences`) used INSTEAD of the
        built-in clean+split pass; the junk filter, cross-turn dedupe and the
        per-unit token cap still apply. `keep` (a text predicate, e.g.
        `pdfio.is_protected_unit`) exempts structural units — headings,
        grouped tables, equations — from the junk filter. Raw conversation
        ingest with no predicate defaults to `is_protected_chat_unit`, so
        table rows, pipelines, rescued link sentences and code-shaped lines
        survive the junk filter's length gates.

        Chat text is stored verbatim apart from whitespace normalization
        and the <url> substitution (url-dominated lines drop): code,
        tables, pipes and markup survive as typed, and the URL decision
        runs before any junk gate. The eval compressor path
        (salt/engine/compressor.py) keeps its own cleaner and filter
        defaults and is unaffected.

        `dedup_cos` (opt-in, None = off) is a near-duplicate gate for
        conversation ingest: a new user/assistant sentence whose BGE cosine
        against an earlier conversation sentence of the SAME role reaches
        the threshold is not appended. Restatements and re-asked questions
        otherwise inflate keyword document frequency — the statistic that
        mints themes and orders the trie — so boilerplate phrasing can
        become a theme branch and steal selection budget. Same-role-only
        means an assistant restatement can never suppress the user's own
        words; doc-role and attachment (`source` set) ingest is exempt.
        A suppressed sentence's hash is withdrawn from the cross-turn
        dedupe — the text lives in the corpus nowhere, so an exact re-send
        is judged afresh (re-gated and logged while the gate is on,
        ingestible again once it is off or the threshold raised). The turn
        still advances; each suppression is appended to `near_dups.jsonl`
        in the session dir (with its cosine, so the threshold can be tuned
        against real data) and counted in the persisted `n_near_dups`.

        `save=False` skips the end-of-call `save()` and marks the trie
        `dirty` instead. The caller owns persisting the batch: `dirty`
        clears on the next `save()`.
        """
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")
        text = (text or "").strip()
        if sentences is None and not text:
            return {"added": 0, "filtered": 0, "near_dups": 0,
                    "n_total": self.n_sentences,
                    "turn": self._next_turn_index}

        if sentences is not None:
            all_texts = []
            for s in sentences:
                s = " ".join((s or "").split())
                if s:
                    all_texts.extend(_chunk_by_tokens(s, tokenizer))
        else:
            cleaned = clean_chat_text(text)
            raw = embed_split_sentences(cleaned, max_tokens=400, tokenizer=tokenizer)
            all_texts = resolve_chat_urls([s for s, _, _ in raw])
        if keep is None and sentences is None:
            keep = is_protected_chat_unit
        sentences, *_ = filter_texts(all_texts, aggressive=True,
                                     remove_urls=True, deduplicate=True,
                                     strip_urls=True, lenient=True, keep=keep)
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
            return {"added": 0, "filtered": n_filtered, "near_dups": 0,
                    "n_total": self.n_sentences, "turn": turn}

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

        # Near-duplicate gate — deliberately AFTER the model passes (the
        # kept sentences' attention contexts must not change with the gate
        # on) and BEFORE the appends. Conversation roles only: documents
        # are legitimately repetitive, and gating them would silently
        # thin the very content attachments exist to preserve.
        n_near_dups = 0
        if (dedup_cos is not None and source is None
                and role in ("user", "assistant")):
            keep_rows, records = self._near_dup_gate(fresh, bge, role,
                                                     dedup_cos)
            n_near_dups = len(records)
            if n_near_dups:
                self.n_near_dups += n_near_dups
                self._log_near_dups(records)
                # A suppressed sentence lives in the corpus NOWHERE, so its
                # hash must be withdrawn from the cross-turn dedupe: an
                # exact re-send is then judged afresh — re-gated (and
                # logged, and counted) while the gate is on, ingestible
                # again once it is off or the threshold raised. Leaving the
                # hash would drop re-sends silently forever, even across
                # launches with the gate disabled. The hash was added THIS
                # call (a pre-existing hash never reaches the gate), so the
                # withdrawal cannot expose an older corpus sentence.
                for r in records:
                    self._seen_hashes.discard(self._norm_hash(r["text"]))
                fresh = [fresh[i] for i in keep_rows]
                attn = [attn[i] for i in keep_rows]
                bge = bge[keep_rows]
        if not fresh:
            # all near-duplicates: the turn still advances, counter changed
            turn = self._next_turn_index
            self._next_turn_index += 1
            if save:
                self.save()
            else:
                self.dirty = True
            return {"added": 0, "filtered": n_filtered,
                    "near_dups": n_near_dups,
                    "n_total": self.n_sentences, "turn": turn}

        turn = self._next_turn_index
        filed_at = time.time()
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
            self.sources.append(source)
            self.timestamps.append(filed_at)
            self.n_words.append(len(sent.split()))
            self.keyword_weights.append(kw)
            self._next_sentence_index += 1

        self.embeddings = (bge if self.embeddings is None
                           else np.vstack([self.embeddings, bge]))
        self._next_turn_index += 1
        self._evict_if_needed()
        if save:
            self.save()
        else:
            self.dirty = True
        return {"added": len(fresh), "filtered": n_filtered,
                "near_dups": n_near_dups,
                "n_total": self.n_sentences, "turn": turn}

    def _near_dup_gate(self, fresh, bge, role, threshold):
        """Rows of `fresh` to keep, plus one log record per suppressed
        sentence. A sentence is suppressed when its max BGE cosine against
        prior conversation sentences of the SAME role — or an earlier kept
        sentence of this same batch — reaches `threshold`. Keep-first inside
        the batch mirrors the cross-turn direction: the first phrasing wins,
        the restatement is the duplicate. Embeddings are L2-normalized, so
        cosine is a plain dot product."""
        prior_idx = [j for j in range(self.n_sentences)
                     if self.sources[j] is None and self.roles[j] == role]
        prior = self.embeddings[prior_idx] if prior_idx else None
        keep_rows, records = [], []
        for i, v in enumerate(bge):
            best, matched = -1.0, None
            if prior is not None:
                sims = prior @ v
                j = int(np.argmax(sims))
                best, matched = float(sims[j]), self.texts[prior_idx[j]]
            for r in keep_rows:
                c = float(np.dot(bge[r], v))
                if c > best:
                    best, matched = c, fresh[r]
            if matched is not None and best >= threshold:
                records.append({"turn": self._next_turn_index, "role": role,
                                "cos": round(best, 4), "text": fresh[i],
                                "matched": matched})
            else:
                keep_rows.append(i)
        return keep_rows, records

    def _log_near_dups(self, records):
        # Append-only diagnostic, one JSON line per suppressed sentence,
        # so the threshold can be tuned against real suppressions before
        # it is trusted. Best-effort: a logging failure must never take
        # down ingest.
        try:
            with open(self._p("near_dups.jsonl"), "a", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        except OSError:
            pass

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
                 device="cpu", delimiter=" ",
                 coverage_half_life=None, coverage_decay_docs=False,
                 shift_damping=None, shift_margin=0.12,
                 shift_query_boost=1.5):
        """Compress the accumulated corpus for `query`, reusing the persisted
        trie + cross-turn coverage.

        Returns {context, stats, selected_sent_idx, n_total_sentences, n_turns}.
        When `query` is empty, selection is document-coverage only (no query
        nodes). Coverage from this turn is merged into the persisted state.

        `coverage_half_life` (in compress calls, i.e. chat turns) enables
        Ebbinghaus-style forgetting of the persisted coverage before it seeds
        selection: a theme's accumulated suppression halves every
        `coverage_half_life` turns it goes unselected, so topics the
        conversation left behind resurface gradually instead of staying
        discounted forever. Entries decayed below COVERAGE_DECAY_FLOOR are
        dropped, which also bounds the dict. Attachment branches (§file:)
        are exempt unless `coverage_decay_docs` is set — decaying them makes
        selection oscillate back to a document's head themes instead of
        progressively covering the file. Default None keeps the original
        accumulate-forever behavior exactly.

        Topic-shift damping (`shift_damping`): drift detection always runs
        on query turns (surfaced in stats as drift_cos/drift_ema/topic_shift
        so the margin can be tuned before trusting it); when a shift is
        detected AND `shift_damping` is set, the STALE part of the coverage
        seed handed to CELF is scaled by that factor FOR THIS TURN ONLY —
        keys incremented within the last SHIFT_FRESH_WINDOW compress calls
        keep their counts, so the topic being pivoted away from stays
        suppressed while a long-quiet topic's discount is lifted — and the
        query-mass ratio is multiplied by `shift_query_boost`. The persisted
        coverage is then reconstructed increments-only: the scaled seed
        never reaches disk. Relief is per shift turn: the damped turn's own
        selections re-stamp the returned topic fresh, so its second turn is
        carried by the query channels and the verbatim tail, not by a
        lifted discount. Default None keeps selection and persisted
        coverage bit-identical to the flag-off behavior.
        """
        budget_pct = self.config["budget_pct_default"] if budget_pct is None else budget_pct
        if self.n_sentences == 0:
            return {"context": "", "stats": {}, "selected_sent_idx": [],
                    "n_total_sentences": 0, "n_turns": self.n_turns}

        # Decay BEFORE seeding; compress() runs once per chat turn, so one
        # multiplicative step per call is per-turn decay (no last-touched
        # bookkeeping). The decayed dict stays LOCAL until the commit after
        # selection: the REPL survives a mid-compress Ctrl-C/error and would
        # otherwise persist an extra decay step for a turn that never
        # selected anything.
        seed = self.coverage
        if coverage_half_life and coverage_half_life > 0 and seed:
            factor = 0.5 ** (1.0 / coverage_half_life)
            kept = {}
            for k, v in seed.items():
                if not coverage_decay_docs and any(
                        t.startswith(FILE_TOKEN_PREFIX) for t in k):
                    kept[k] = v                   # doc branches exempt
                    continue
                v *= factor
                if v >= COVERAGE_DECAY_FLOOR:
                    kept[k] = v
            seed = kept

        sent_data = self._sent_data()
        kw_df, theme_keywords = profile_themes(
            sent_data, theme_percentile=self.config["theme_percentile"])

        # Per-file branches: inject each attachment's synthetic root keyword
        # AFTER theme profiling (so the percentile threshold sees only real
        # keywords) with df just above the real maximum — high enough that
        # SF-ordering roots every one of the file's paths at its token, low
        # enough that real node weights (df/max_df) barely shift.
        attached = sorted({s for s in self.sources if s})
        if attached:
            kw_df = dict(kw_df)
            top_df = max(kw_df.values(), default=1) + 1
            theme_keywords = set(theme_keywords)
            for s in attached:
                kw_df[file_token(s)] = top_df
                theme_keywords.add(file_token(s))
            for i, sd in enumerate(sent_data):
                if self.sources[i]:
                    kw = dict(sd["keyword_weights"])
                    kw[file_token(self.sources[i])] = max(kw.values(), default=1.0)
                    sd["keyword_weights"] = kw

        orig_words = sum(self.n_words)
        word_budget = int(orig_words * budget_pct)

        query = (query or "").strip()
        q_kws = q_pns = q_emb = None
        if query:
            q_kws = extract_query_keywords(query)
            q_pns = extract_proper_nouns_in_query(query)
            q_emb = embed_query(query, tokenizer, model, device)

        # Topic-shift detection: query cosine vs the mean of recent
        # conversation embeddings (attachments excluded — a stable document
        # must not anchor the "what we were just talking about" baseline),
        # compared against the session's own EMA. Like the decay above, the
        # EMA update is computed here but committed only after selection, so
        # an interrupted turn perturbs neither coverage nor the baseline.
        drift_cos, drift_baseline, shifted = None, self.drift_ema, False
        if q_emb is not None:
            conv_idx = [i for i in range(self.n_sentences)
                        if self.sources[i] is None]
            if len(conv_idx) >= DRIFT_MIN_SENTENCES:
                recent = self.embeddings[conv_idx[-DRIFT_WINDOW:]]
                drift_cos = float(np.dot(recent.mean(axis=0), q_emb))
                if drift_baseline is not None:
                    shifted = drift_cos < drift_baseline - shift_margin

        # On a shift turn (and only behind the opt-in), hand CELF a seed
        # with STALE suppression scaled down and boost the query channels,
        # for this turn only: a pivot back to an old topic must not find
        # precisely that topic the most-discounted content in the trie.
        # Fresh keys (incremented within SHIFT_FRESH_WINDOW compress calls)
        # keep their counts — the topic being pivoted away from must stay
        # suppressed, or it wins the freed budget right back.
        seed_passed = seed
        query_mass_ratio = self.config["query_mass_ratio"]
        damped = bool(shifted and shift_damping)
        n_damped_keys = 0
        if damped:
            fresh_after = self._n_compress - SHIFT_FRESH_WINDOW
            seed_passed = {}
            for k, v in seed.items():
                # A missing stamp reads STALE: only pre-feature resumes lack
                # stamps (every key this code persists was stamped when it
                # was incremented), and coverage of unknown age must damp.
                # A numeric default would classify those keys fresh exactly
                # while fresh_after < 0 — the first shifts after a resume.
                stamp = self.coverage_turn.get(k)
                if stamp is not None and stamp >= fresh_after:
                    seed_passed[k] = v
                else:
                    seed_passed[k] = v * shift_damping
                    n_damped_keys += 1
            query_mass_ratio *= shift_query_boost

        selected, stats, cov = coverage_select(
            sent_data, dict(kw_df), theme_keywords, word_budget,
            query_keywords=q_kws, query_embedding=q_emb, query_proper_nouns=q_pns,
            lam=self.config["lam"], query_mass_ratio=query_mass_ratio,
            seed_coverage=seed_passed, return_coverage=True)

        if damped:
            # Increments-only merge — the one real trap of seed scaling:
            # CELF's returned coverage is seed-AS-PASSED + this turn's
            # increments, so persisting it wholesale here would persist the
            # scaled counts too — a permanent forgetting event on every
            # shift turn. Rebuild from the unscaled base instead; untouched
            # keys subtract to exactly 0.0 (the dict is copied, never
            # re-derived), so the guard only filters float noise.
            merged = dict(seed)
            for k, v in cov["coverage"].items():
                d = v - seed_passed.get(k, 0.0)
                if d > 1e-12:
                    merged[k] = merged.get(k, 0.0) + d
            self.coverage = merged
        else:
            self.coverage = cov["coverage"]
        # Freshness clock for stale-only damping: stamp every key CELF
        # actually incremented this call (returned vs seed-as-passed is the
        # true in/out diff on both paths), then drop stamps for keys the
        # decay floor garbage-collected so the dict stays bounded with the
        # coverage itself.
        for k, v in cov["coverage"].items():
            if v > seed_passed.get(k, 0.0) + 1e-12:
                self.coverage_turn[k] = self._n_compress
        if len(self.coverage_turn) > len(self.coverage):
            self.coverage_turn = {k: t for k, t in self.coverage_turn.items()
                                  if k in self.coverage}
        self._n_compress += 1
        if drift_cos is not None:
            self.drift_ema = (drift_cos if drift_baseline is None
                              else DRIFT_EMA_ALPHA * drift_cos
                              + (1.0 - DRIFT_EMA_ALPHA) * drift_baseline)
        self.save()

        stats["coverage_keys"] = len(self.coverage)
        stats["coverage_half_life"] = coverage_half_life
        stats["drift_cos"] = drift_cos
        stats["drift_ema"] = drift_baseline
        stats["topic_shift"] = shifted
        stats["shift_damped"] = damped
        stats["shift_damped_keys"] = n_damped_keys

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
        # embeddings-first on purpose: a crash in the gap leaves orphan
        # matrix rows load() can drop; state-first would leave real
        # sentences with no vectors, which nothing could repair.
        if self.embeddings is not None:
            # np.save appends ".npy" to a path arg; write via a handle so the
            # atomic ".tmp" name is preserved.
            def _save_npy(p):
                with open(p, "wb") as fh:
                    np.save(fh, self.embeddings)
            self._atomic_write(self._p("embeddings.npy"), _save_npy)
        state = {
            "texts": self.texts, "roles": self.roles, "turns": self.turns,
            "sources": self.sources, "timestamps": self.timestamps,
            "n_words": self.n_words, "keyword_weights": self.keyword_weights,
            "coverage": self.coverage, "seen_hashes": self._seen_hashes,
            "drift_ema": self.drift_ema,
            "coverage_turn": self.coverage_turn,
            "n_compress": self._n_compress,
            "n_near_dups": self.n_near_dups,
            "next_sentence_index": self._next_sentence_index,
            "next_turn_index": self._next_turn_index,
        }
        self._atomic_write(self._p("state.pkl"),
                           lambda p: p.write_bytes(pickle.dumps(state)))
        self._atomic_write(self._p("config.json"),
                           lambda p: p.write_text(json.dumps(self.config, indent=2)))
        self.dirty = False

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
        # sessions saved before per-file branches existed have no sources,
        # and those saved before ingest time was recorded have no timestamps
        self.sources = state.get("sources", [None] * len(self.texts))
        self.timestamps = state.get("timestamps", [None] * len(self.texts))
        self.keyword_weights = state["keyword_weights"]
        self.coverage = state["coverage"]; self._seen_hashes = state["seen_hashes"]
        # sessions saved before topic-shift detection have no drift baseline
        # and no freshness clock (every key then counts as stale, which is
        # the right reading for coverage of unknown age)
        self.drift_ema = state.get("drift_ema")
        self.coverage_turn = state.get("coverage_turn", {})
        self._n_compress = state.get("n_compress", 0)
        # sessions saved before the near-dup gate have no counter
        self.n_near_dups = state.get("n_near_dups", 0)
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
        self.load_repair = self._reconcile_after_load()
        return True

    def _reconcile_after_load(self):
        """Reconcile embeddings.npy against state.pkl (the same safety net
        KVTrace runs for tokens.npy vs events.jsonl). The save set is
        per-file atomic with no cross-file barrier, so a crash between the
        two writes leaves the matrix and the corpus lists disagreeing on
        length; unrepaired, every later sentence scores against another
        sentence's vector for the rest of the session's life. state.pkl is
        the authority (config.json is written last, so its count is stale
        inside its own crash window). Healthy sessions return None with
        nothing mutated. Turn ids never rewind: a repair may leave a gap,
        same as the all-near-dups path."""
        n_rows = 0 if self.embeddings is None else int(self.embeddings.shape[0])
        n_txt = len(self.texts)
        if n_rows == n_txt:
            return None
        n = min(n_rows, n_txt)
        if n_rows > n:
            self.embeddings = self.embeddings[:n]
        if n_txt > n:
            dropped = list(self.texts[n:])
            self.texts = self.texts[:n]
            self.roles = self.roles[:n]
            self.turns = self.turns[:n]
            self.sources = self.sources[:n]
            self.timestamps = self.timestamps[:n]
            self.n_words = self.n_words[:n]
            self.keyword_weights = self.keyword_weights[:n]
            for t in dropped:
                self._seen_hashes.discard(self._norm_hash(t))
        self._next_sentence_index = n
        self.dirty = True
        return {"kept": n, "orphan_rows": max(0, n_rows - n),
                "dropped_sentences": max(0, n_txt - n)}
