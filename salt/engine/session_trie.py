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
    Per-turn work that only the newest sentences can change — each sentence's
    expanded lexical token set, and the keyword document frequencies the theme
    profile reads — is derived at ingest and carried forward; the trie itself
    is still rebuilt every turn.
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
    profile_themes, is_content_word, clean_text_words, expand_with_stems,
    extract_query_keywords, extract_proper_nouns_in_query,
    build_trie_paths,
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
# a node's residual suppression. The floor bounds the dict ONLY under its
# conditions: --coverage-half-life must be on (off by default), attachment
# branches are exempt unless --coverage-decay-docs, and keys orphaned by
# trie churn never decay-match anything — those need --coverage-gc, and an
# unconditional bound needs --coverage-max-keys.
COVERAGE_DECAY_FLOOR = 0.05

# Per-source theme profiling (opt-in): each source's df values are rescaled
# onto this common scale before the max-merge, so a 400-sentence attachment
# and a 12-sentence conversation meet as equals. Buckets smaller than
# MIN_SOURCE_SENTENCES fold into the conversation bucket — a 2-sentence
# attachment has every keyword at df 1 and would mint them all as themes.
DF_SCALE = 1000
MIN_SOURCE_SENTENCES = 3

# Orphan-GC grace (compress calls): an orphaned coverage key is inert
# while orphaned, so the only cost of dropping one is that a later
# reordering could resurrect the same prefix and find its suppression
# gone. The grace window keeps recently-touched keys around long enough
# for that to be rare.
COVERAGE_GC_GRACE = 8

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
        # A bounded session masks rows dead rather than deleting them. Row
        # indices are permanent — kvtrace references sent_idx for the whole
        # life of the session and coverage keys are df-rank-sensitive, so a
        # removal would renumber both — and a masked row keeps its text,
        # its vector and its place.
        self.alive = []                 # list[bool] (False = masked out)
        self.embeddings = None          # np.ndarray (n, dim) float32  (cached BGE [CLS])

        # --- derived caches: reconstructible, never persisted, never
        # authoritative. See _lex_tokens for why a miss is only ever slow.
        self._lex = {}                  # text -> expanded lexical token set
        self._kw_df = None              # dict[str, int] over LIVING rows
        self._kw_df_rows = None         # (n_rows, n_alive) the counts describe

        # --- cross-turn state ---
        self.coverage = {}              # dict[frozenset[str], float]  accumulated per-node coverage
        self._seen_hashes = set()       # cross-turn sentence dedupe
        self.drift_ema = None           # EMA of query-vs-recent-conversation cosine
        self.coverage_turn = {}         # dict[key -> compress call of last increment]
        self.kw_order = []              # frozen keyword order (stable_keys, append-only)
        self.theme_admitted = set()     # sticky theme membership (stable_keys)
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
    def n_alive(self):
        return sum(1 for a in self.alive if a)

    @property
    def n_masked(self):
        return len(self.alive) - self.n_alive

    @property
    def live_words(self):
        """The selection budget's base. One definition on purpose: the CLI
        sizes its prompt warning from the same number compress() divides,
        and a second copy of the filter is how the two drift apart."""
        return sum(w for w, a in zip(self.n_words, self.alive) if a)

    @property
    def n_turns(self):
        return self._next_turn_index

    @property
    def attached_sources(self):
        return sorted({s for s in self.sources if s})

    # ── derived caches ────────────────────────────────────────────────────
    def _lex_tokens(self, text):
        """One sentence's expanded lexical tokens, derived once and reused.

        `coverage_select` scores the query's lexical channel against this set
        for every living sentence on every turn, and re-derives it from the
        text each time. The answer cannot change:
        `expand_with_stems(clean_text_words(...))` is pure and a stored
        sentence is immutable, so the whole per-turn pass is recomputation.

        Ingest fills the cache while the two model passes are already
        running, which is why the cost disappears rather than moves. Text
        that arrives another way — a session resumed from disk, a corpus
        assembled field by field — fills on its first miss with the value
        the selector would have built itself. The cache is therefore never
        authoritative and a miss costs time, not accuracy. It is not
        persisted: refilling is cheaper than a state format change."""
        toks = self._lex.get(text)
        if toks is None:
            toks = expand_with_stems(clean_text_words(text))
            self._lex[text] = toks
        return toks

    def _live_kw_df(self):
        """Keyword document frequency over the living corpus, carried
        forward instead of recounted.

        `profile_themes` derives df[kw] = how many living rows carry kw by
        walking every row's keyword dict on every turn, but the living set
        moves by exactly the rows one turn appends or masks. This count is
        maintained at those two sites and REBUILT whenever the rows it
        describes stop matching the corpus, so a resumed session, a
        crash-repaired one, or a corpus assembled field by field falls back
        to a full count instead of trusting stale numbers. Each maintenance
        site updates the row pair last, so an interrupt between a row and
        its count leaves a mismatch that rebuilds rather than a skew that
        survives."""
        rows = (len(self.texts), self.n_alive)
        if self._kw_df is None or self._kw_df_rows != rows:
            df = {}
            for kw, alive in zip(self.keyword_weights, self.alive):
                if alive:
                    for k in kw:
                        df[k] = df.get(k, 0) + 1
            self._kw_df, self._kw_df_rows = df, rows
        return self._kw_df

    def _kw_df_add(self, kw):
        if self._kw_df is None:
            return
        for k in kw:
            self._kw_df[k] = self._kw_df.get(k, 0) + 1
        self._kw_df_rows = (self._kw_df_rows[0] + 1, self._kw_df_rows[1] + 1)

    def _kw_df_drop(self, kw):
        # Delete at zero: profile_themes never mints a key for a row that
        # is not there, and its theme threshold is an index into the count
        # of keys, so a zero left behind would move the cutoff.
        if self._kw_df is None:
            return
        for k in kw:
            n = self._kw_df.get(k, 0) - 1
            if n > 0:
                self._kw_df[k] = n
            else:
                self._kw_df.pop(k, None)
        self._kw_df_rows = (self._kw_df_rows[0], self._kw_df_rows[1] - 1)

    def _themes_from_df(self, df):
        """(kw_df, theme_keywords) from a maintained count, mirroring
        `profile_themes` line for line: the same percentile index into the
        sorted df values, the same `>=` membership, the same answer for a
        corpus that mints no keywords at all (a living corpus can be all
        table rows, whose words are no content words). The dict handed back
        is a copy on purpose — compress() injects the §file: tokens into
        what it gets — so the maintained count is never the thing a caller
        mutates."""
        if not df:
            return {}, set()
        values = sorted(df.values())
        idx = int(len(values) * self.config["theme_percentile"])
        threshold = values[min(idx, len(values) - 1)]
        return dict(df), {k for k, v in df.items() if v >= threshold}

    # ── ingest ────────────────────────────────────────────────────────────
    @staticmethod
    def _norm_hash(text):
        return hashlib.sha1(" ".join(text.lower().split()).encode("utf-8")).hexdigest()

    def add_turn(self, text, role="user", *, tokenizer, model, device="cpu",
                 source=None, sentences=None, keep=None, dedup_cos=None,
                 max_sentences=None, save=True):
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

        `max_sentences` (opt-in, None = off) caps the living conversation
        corpus: once this ingest pushes it past the cap, the oldest living
        conversation rows are masked out of memory until it meets the cap
        again. Attachments are never masked. Nothing is deleted, so the
        text, the vector and above all the row index of a masked sentence
        stay valid forever — see `_evict_if_needed`, which also explains
        why a bounded session wants one of the coverage bounds on too.
        The returned `masked` count says how many rows this call retired.
        A masked sentence's verbatim-dedupe hash is withdrawn with it, so
        re-sending it word for word stores it again rather than reading as
        a duplicate of a row no longer in memory.

        `save=False` skips the end-of-call `save()` and marks the trie
        `dirty` instead. The caller owns persisting the batch: `dirty`
        clears on the next `save()`.
        """
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")
        text = (text or "").strip()
        if sentences is None and not text:
            return {"added": 0, "filtered": 0, "near_dups": 0, "masked": 0,
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
                    "masked": 0, "n_total": self.n_sentences, "turn": turn}

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
                    "near_dups": n_near_dups, "masked": 0,
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
            self._lex_tokens(sent)      # while the model passes already cost
            self.roles.append(role)
            self.turns.append(turn)
            self.sources.append(source)
            self.timestamps.append(filed_at)
            self.n_words.append(len(sent.split()))
            self.keyword_weights.append(kw)
            self.alive.append(True)
            self._kw_df_add(kw)
            self._next_sentence_index += 1

        self.embeddings = (bge if self.embeddings is None
                           else np.vstack([self.embeddings, bge]))
        self._next_turn_index += 1
        n_masked = self._evict_if_needed(max_sentences)
        if save:
            self.save()
        else:
            self.dirty = True
        return {"added": len(fresh), "filtered": n_filtered,
                "near_dups": n_near_dups, "masked": n_masked,
                "n_total": self.n_sentences, "turn": turn}

    def _near_dup_gate(self, fresh, bge, role, threshold):
        """Rows of `fresh` to keep, plus one log record per suppressed
        sentence. A sentence is suppressed when its max BGE cosine against
        prior conversation sentences of the SAME role — or an earlier kept
        sentence of this same batch — reaches `threshold`. Keep-first inside
        the batch mirrors the cross-turn direction: the first phrasing wins,
        the restatement is the duplicate. Embeddings are L2-normalized, so
        cosine is a plain dot product."""
        # living rows only: a masked sentence is out of memory, so a later
        # restatement of it is the only copy left and must be ingestible
        prior_idx = [j for j in range(self.n_sentences)
                     if self.alive[j] and self.sources[j] is None
                     and self.roles[j] == role]
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

    def _evict_if_needed(self, cap=None):
        """Mask the oldest living CONVERSATION rows once they outnumber
        `cap`, so a long session stops growing without bound. Nothing is
        deleted — a masked row keeps its text, its vector and its index
        (see `alive`) — and attachment rows are never masked and never
        counted: a document is a bounded cost the user chose, while the
        conversation is the part that grows forever. Returns how many rows
        this call retired. `cap` of None falls back to the persisted
        `max_sentences`; anything <= 0 is off, matching `coverage_max_keys`.
        Masking also withdraws the row's verbatim-dedupe hash, so
        re-sending a masked sentence word for word stores it again instead
        of dropping it against a row no longer in memory.

        What this does NOT bound is the coverage dict. Retiring a theme's
        last living rows leaves its keys in the persisted dict, inert but
        copied into every seed and every save. A conversation key decays
        away once `coverage_half_life` is on, since nothing re-increments
        it; a `§file:` key is exempt unless `coverage_decay_docs`. With
        decay off neither is collected, so a bounded session wants
        `coverage_gc`, `coverage_max_keys` or the `stable_keys` reconcile
        on as well."""
        cap = self.config.get("max_sentences") if cap is None else cap
        if cap is None or int(cap) <= 0:
            return 0
        live_conv = [i for i in range(self.n_sentences)
                     if self.alive[i] and self.sources[i] is None]
        n_evict = len(live_conv) - int(cap)
        if n_evict <= 0:
            return 0
        for i in live_conv[:n_evict]:
            self.alive[i] = False
            self._kw_df_drop(self.keyword_weights[i])
            # Derived state follows the row out of memory, or the cap
            # would bound what selection reads while the tokens of every
            # sentence ever said stayed resident. Two rows can hold the
            # same text (a withdrawn near-dup hash, a repaired load), and
            # dropping a living row's entry with them costs that row one
            # re-derivation, never a wrong answer.
            self._lex.pop(self.texts[i], None)
            # The verbatim-dedupe hash goes with it: a masked sentence
            # lives in memory nowhere, so an exact re-send must be judged
            # afresh, not dropped against a dead row (the near-dup path
            # withdraws for the same reason). Masking is oldest-first and
            # two living rows never share a hash, so no living row's copy
            # is exposed by the discard.
            self._seen_hashes.discard(self._norm_hash(self.texts[i]))
        return n_evict

    # ── compression ───────────────────────────────────────────────────────
    def _sent_data(self):
        """The living corpus, as CELF's record dicts. `sent_idx` stays the
        row's own index, so a masked row leaves a GAP rather than shifting
        the rows after it — kvtrace and the persisted coverage keys both
        read those indices. CELF orders records positionally and touches
        sent_idx only for the final document-order sort, which monotonic
        gaps leave correct."""
        return [{"sent_idx": i, "text": self.texts[i], "n_words": self.n_words[i],
                 "keyword_weights": self.keyword_weights[i],
                 "embedding_l2": self.embeddings[i]}
                for i in range(self.n_sentences) if self.alive[i]]

    def _profile(self, sent_data, per_source=False):
        """One (kw_df, theme_keywords) pair for CELF. Global mode is the
        frozen path: a single pooled profile_themes call. Per-source mode
        profiles each attachment and the conversation separately with the
        SAME untouched profile_themes, rescales every bucket's df onto
        DF_SCALE, and max-merges — so a large attachment can no longer set
        the percentile cutoff that evicts the conversation's own themes.
        Deliberate compromise: kw_rank downstream still orders paths from
        the one merged dict, so a keyword shared across sources takes its
        max-merged df in both."""
        if not per_source:
            self._profile_diag = {"sources": 1, "keywords_conv": None}
            # The maintained count describes the LIVING corpus and nothing
            # else, so a caller handing over some other list of records
            # gets the full recount it asked for.
            if len(sent_data) == self.n_alive:
                return self._themes_from_df(self._live_kw_df())
            return profile_themes(
                sent_data, theme_percentile=self.config["theme_percentile"])
        buckets = {}
        for sd in sent_data:
            buckets.setdefault(self.sources[sd["sent_idx"]], []).append(sd)
        conv = buckets.pop(None, [])
        for src in [s for s, b in buckets.items()
                    if len(b) < MIN_SOURCE_SENTENCES]:
            conv.extend(buckets.pop(src))
        grouped = ([conv] if conv else []) + list(buckets.values())
        merged_df, merged_themes = {}, set()
        keywords_conv = 0
        for bucket in grouped:
            df, themes = profile_themes(
                bucket, theme_percentile=self.config["theme_percentile"])
            top = max(df.values(), default=1)
            for k, v in df.items():
                s = max(1, int(round(DF_SCALE * v / top)))
                if s > merged_df.get(k, 0):
                    merged_df[k] = s
            merged_themes |= themes
            if bucket is conv:
                keywords_conv = len(themes)
        self._profile_diag = {"sources": len(grouped),
                              "keywords_conv": keywords_conv}
        return merged_df, merged_themes

    def _plan_commit(self, cov, seed, seed_passed, damped, drift_cos,
                     drift_baseline, stable_keys, coverage_gc,
                     coverage_max_keys, node_universe=None):
        """Compute the turn's coverage/EMA commit without touching any
        session state. Returns (new_coverage, new_coverage_turn,
        new_drift_ema, diag); _apply_commit() makes it real.
        `node_universe` is the key universe the bookkeeping classifies
        live-vs-orphan against; None reads this turn's trie keys."""
        if damped:
            # Increments-only merge — the one real trap of seed scaling:
            # CELF's returned coverage is seed-AS-PASSED + this turn's
            # increments, so persisting it wholesale here would persist the
            # scaled counts too — a permanent forgetting event on every
            # shift turn. Rebuild from the unscaled base instead; untouched
            # keys subtract to exactly 0.0 (the dict is copied, never
            # re-derived), so the guard only filters float noise.
            new_coverage = dict(seed)
            for k, v in cov["coverage"].items():
                d = v - seed_passed.get(k, 0.0)
                if d > 1e-12:
                    new_coverage[k] = new_coverage.get(k, 0.0) + d
        else:
            # same object, not a copy: the non-damped path has always
            # adopted CELF's returned dict wholesale
            new_coverage = cov["coverage"]
        new_coverage_turn = dict(self.coverage_turn)
        # Reconcile (stable_keys only): with the order frozen and
        # membership sticky over an append-only corpus, a persisted key
        # absent from this turn's node set can only be pre-flag history
        # or eviction residue - no node exists for it to discount this
        # turn or any later one, so carrying it forever is pure bloat.
        universe_now = (node_universe if node_universe is not None
                        else cov.get("node_keys") or set())
        n_orphans_dropped = 0
        if stable_keys:
            before_n = len(new_coverage)
            new_coverage = {k: v for k, v in new_coverage.items()
                            if k in universe_now}
            n_orphans_dropped = before_n - len(new_coverage)
            if n_orphans_dropped:
                new_coverage_turn = {k: t for k, t
                                     in new_coverage_turn.items()
                                     if k in new_coverage}
        # Persisted-dict liveness measurement (free: the live set is this
        # turn's node keys, already built by coverage_select). Orphans are
        # keys no current trie node can match - inert for selection but
        # copied forward and saved every turn.
        persisted_orphans = [k for k in new_coverage
                             if k not in universe_now]
        n_orphan_doc = sum(1 for k in persisted_orphans
                           if any(t.startswith(FILE_TOKEN_PREFIX)
                                  for t in k))
        persisted_orphan_mass = sum(new_coverage[k]
                                    for k in persisted_orphans)
        # Freshness clock for stale-only damping: stamp every key CELF
        # actually incremented this call (returned vs seed-as-passed is the
        # true in/out diff on both paths), then drop stamps for keys the
        # decay floor garbage-collected so the dict stays bounded with the
        # coverage itself. Stamps carry the PRE-increment compress counter.
        for k, v in cov["coverage"].items():
            if v > seed_passed.get(k, 0.0) + 1e-12:
                new_coverage_turn[k] = self._n_compress
        if len(new_coverage_turn) > len(new_coverage):
            new_coverage_turn = {k: t for k, t in new_coverage_turn.items()
                                 if k in new_coverage}
        # Orphan GC (opt-in): a key absent from the live node set cannot
        # discount anything this turn either way; the grace window only
        # hedges a later reordering resurrecting the same prefix. Doc keys
        # get no immortality here - the decay exemption protects
        # progressive coverage of a LIVE document branch, never orphans.
        n_gc_dropped = 0
        if coverage_gc:
            cutoff = self._n_compress - COVERAGE_GC_GRACE
            drop = [k for k in new_coverage
                    if k not in universe_now
                    and (new_coverage_turn.get(k) is None
                         or new_coverage_turn[k] < cutoff)]
            for k in drop:
                del new_coverage[k]
                new_coverage_turn.pop(k, None)
            n_gc_dropped = len(drop)
        # Hard cap (opt-in): the only unconditional bound with decay off.
        # Orphans go first regardless of grace, then live keys stalest
        # and weakest first.
        n_cap_dropped = 0
        if coverage_max_keys is not None and int(coverage_max_keys) > 0:
            cap = int(coverage_max_keys)
            if len(new_coverage) > cap:
                victims = sorted(
                    new_coverage,
                    key=lambda k: (k in universe_now,
                                   new_coverage_turn.get(k, -1),
                                   new_coverage[k]))
                for k in victims[:len(new_coverage) - cap]:
                    del new_coverage[k]
                    new_coverage_turn.pop(k, None)
                    n_cap_dropped += 1
        new_drift_ema = None
        if drift_cos is not None:
            new_drift_ema = (drift_cos if drift_baseline is None
                             else DRIFT_EMA_ALPHA * drift_cos
                             + (1.0 - DRIFT_EMA_ALPHA) * drift_baseline)
        diag = {"orphans_dropped": n_orphans_dropped,
                "persisted_orphans": len(persisted_orphans),
                "orphan_doc_keys": n_orphan_doc,
                "persisted_orphan_mass": persisted_orphan_mass,
                "gc_dropped": n_gc_dropped,
                "cap_dropped": n_cap_dropped}
        return new_coverage, new_coverage_turn, new_drift_ema, diag

    def _apply_commit(self, new_coverage, new_coverage_turn, new_drift_ema,
                      drift_cos, new_theme_admitted=None, new_kw_order=None,
                      save=True):
        """Make a planned commit real: coverage, freshness stamps, the
        compress counter, the drift EMA, and under stable_keys the frozen
        keyword order and sticky theme set, then persist (save=False marks
        the state dirty for the caller's own save path)."""
        self.coverage = new_coverage
        self.coverage_turn = new_coverage_turn
        self._n_compress += 1
        if drift_cos is not None:
            self.drift_ema = new_drift_ema
        # stable_keys only: both are computed during selection to build
        # kw_rank, but applied HERE so a discarded defer_commit turn cannot
        # widen the append-only order or the sticky set (both persisted).
        if new_theme_admitted is not None:
            self.theme_admitted = new_theme_admitted
        if new_kw_order is not None:
            self.kw_order = new_kw_order
        if save:
            self.save()
        else:
            self.dirty = True

    def compress(self, query="", budget_pct=None, *, tokenizer, model,
                 device="cpu", delimiter=" ",
                 coverage_half_life=None, coverage_decay_docs=False,
                 shift_damping=None, shift_margin=0.12,
                 shift_query_boost=1.5, per_source_themes=False,
                 max_words=None, stable_keys=False, coverage_gc=False,
                 coverage_max_keys=None, defer_commit=False,
                 exclude_sent_idx=None):
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
        dropped, which bounds the decaying part of the dict (exempt
        attachment branches and orphaned keys are outside that bound —
        see `coverage_gc` and `coverage_max_keys`). Attachment branches (§file:)
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

        `max_words` (opt-in): absolute ceiling on the selection word
        budget. The budget is normally a fraction of the ever-growing
        corpus, so the memory block grows without bound; a positive
        `max_words` clamps it (word_budget = min(fraction, max_words)) so
        the block stops growing once the corpus outruns the cap. Stats
        report word_budget and word_budget_capped. Default None reproduces
        today's selection exactly.

        `stable_keys` (opt-in): freezes the session's keyword order so
        cross-turn coverage keys survive trie rebuilds. New theme keywords
        append to the tail of a persisted global order and paths are built
        with that order, so existing path prefixes (the coverage keys)
        never re-key when document frequencies move. Default False keeps
        the per-turn df ordering exactly.

        `defer_commit` (opt-in): when true, nothing of this turn is
        committed inside the call — coverage, the freshness stamps, the
        drift EMA, the compress counter and, under stable_keys, the frozen
        keyword order and sticky theme set stay untouched — and the
        returned dict carries a one-shot `commit` callable that applies
        and persists the turn (`commit(save=False)` only marks the state
        dirty, for callers that own the save). Never calling it discards
        the turn's accumulation, as if the compress never ran; a second
        call is a no-op. The empty-session early exit returns
        `commit: None`. Stats always describe the planned commit.
        Default False commits inside the call exactly as before.

        `exclude_sent_idx` (opt-in): row indices removed from selection
        CANDIDACY for this call, meant for sentences the model is already
        reading verbatim in the chat tail. Exclusion narrows what can be
        picked and nothing else: the theme profile, the trie ordering and
        every piece of coverage bookkeeping (the stable-keys reconcile,
        orphan GC, the hard cap and the orphan diagnostics) still see the
        full living corpus, so an excluded row's accumulated discount is
        carried, never collected, and its themes cannot be stamped by a
        turn that could not select them. An exclusion that would empty
        the candidate set is ignored outright — the caller asked to avoid
        duplication, not to blank memory (and an empty selection under
        `stable_keys` would reconcile the whole coverage dict away).
        Stats report `excluded_sent`, the candidacy rows actually
        removed. Default None reproduces today's selection exactly.
        """
        budget_pct = self.config["budget_pct_default"] if budget_pct is None else budget_pct
        if self.n_alive == 0:
            out = {"context": "", "stats": {}, "selected_sent_idx": [],
                   "n_total_sentences": self.n_sentences,
                   "n_turns": self.n_turns}
            if defer_commit:
                out["commit"] = None
            return out

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
        kw_df, theme_keywords = self._profile(sent_data, per_source_themes)

        # Sticky theme membership (stable_keys only): a keyword that
        # already earned a place in the tree keeps it while its remembered
        # counts are alive, so branches cannot vanish from under their
        # discounts when the percentile cutoff moves. Bounded by coverage
        # mass: once the decay floor GCs a keyword's last key, it stops
        # being sticky. Runs BEFORE the file-token injection, which
        # re-copies theme_keywords.
        n_sticky = 0
        new_theme_admitted = None
        if stable_keys:
            alive = set().union(*self.coverage) if self.coverage else set()
            sticky = {k for k in self.theme_admitted
                      if k in alive and k not in theme_keywords}
            n_sticky = len(sticky)
            theme_keywords = set(theme_keywords) | sticky
            new_theme_admitted = set(theme_keywords)   # applied at commit

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
            for sd in sent_data:
                src = self.sources[sd["sent_idx"]]
                if src:
                    kw = dict(sd["keyword_weights"])
                    kw[file_token(src)] = max(kw.values(), default=1.0)
                    sd["keyword_weights"] = kw

        kw_rank = None
        new_kw_order = None
        if stable_keys:
            fresh_kws = sorted(set(theme_keywords) - set(self.kw_order),
                               key=lambda k: (-kw_df.get(k, 0), k))
            new_kw_order = self.kw_order + fresh_kws   # applied at commit
            kw_rank = {kw: r for r, kw in enumerate(new_kw_order)}

        # Tail exclusion: drop the caller's rows from CANDIDACY only. The
        # profile above and the bookkeeping below keep seeing the full
        # living corpus, so the trie shape and the coverage keys stay
        # exactly what the flag-off turn would build — an excluded row
        # simply cannot be selected while the model already reads it.
        n_excluded = 0
        excluded_node_keys = None
        if exclude_sent_idx:
            # normalize first: the two scans below would consume a one-shot
            # iterable, and set membership keeps them O(1) either way
            exclude_sent_idx = set(exclude_sent_idx)
            kept = [sd for sd in sent_data
                    if sd["sent_idx"] not in exclude_sent_idx]
            if kept and len(kept) < len(sent_data):
                excluded = [sd for sd in sent_data
                            if sd["sent_idx"] in exclude_sent_idx]
                n_excluded = len(excluded)
                sent_data = kept
                # The excluded rows' own node keys, unioned into the
                # commit universe below: a coverage key whose only
                # carriers ride in the tail is temporarily unselectable,
                # not orphaned, and must survive the stable-keys
                # reconcile, the GC and the cap's victim ordering. Path
                # prefixes are per-record under the shared keyword
                # ordering, so this union equals the full corpus's keys.
                epaths, _, _, enode_kw = build_trie_paths(
                    [set(sd["keyword_weights"]) & set(theme_keywords)
                     for sd in excluded],
                    kw_df, theme_keywords, kw_rank=kw_rank)
                excluded_node_keys = set()
                for pid in epaths:
                    acc = []
                    for v in pid:
                        acc.append(enode_kw[v])
                        excluded_node_keys.add(frozenset(acc))

        orig_words = self.live_words
        word_budget = int(orig_words * budget_pct)
        word_budget_capped = False
        if max_words is not None and int(max_words) > 0:
            word_budget_capped = word_budget > int(max_words)
            word_budget = min(word_budget, int(max_words))

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
                        if self.alive[i] and self.sources[i] is None]
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
            token_fn=self._lex_tokens,
            seed_coverage=seed_passed, return_coverage=True,
            kw_rank=kw_rank)

        universe = cov.get("node_keys") or set()
        if excluded_node_keys:
            universe = universe | excluded_node_keys
        (new_coverage, new_coverage_turn, new_drift_ema,
         commit_diag) = self._plan_commit(
             cov, seed, seed_passed, damped, drift_cos, drift_baseline,
             stable_keys, coverage_gc, coverage_max_keys,
             node_universe=universe)
        commit_cb = None
        if defer_commit:
            committed = []

            def commit_cb(save=True):
                # one-shot: a second call must not re-tick the counter
                if committed:
                    return
                committed.append(True)
                self._apply_commit(new_coverage, new_coverage_turn,
                                   new_drift_ema, drift_cos,
                                   new_theme_admitted, new_kw_order,
                                   save=save)
        else:
            self._apply_commit(new_coverage, new_coverage_turn,
                               new_drift_ema, drift_cos,
                               new_theme_admitted, new_kw_order)

        stats["coverage_keys"] = len(new_coverage)
        stats["coverage_half_life"] = coverage_half_life
        stats["drift_cos"] = drift_cos
        stats["drift_ema"] = drift_baseline
        stats["topic_shift"] = shifted
        stats["shift_damped"] = damped
        stats["shift_damped_keys"] = n_damped_keys
        stats["theme_scope"] = "source" if per_source_themes else "global"
        stats["theme_sources"] = self._profile_diag["sources"]
        stats["theme_keywords_conv"] = self._profile_diag["keywords_conv"]
        stats["theme_keywords_sticky"] = n_sticky
        stats["word_budget"] = word_budget
        stats["word_budget_capped"] = word_budget_capped
        stats["excluded_sent"] = n_excluded
        seed_matched = sum(1 for k in seed_passed if k in universe)
        stats["coverage_seed_keys"] = len(seed_passed)
        stats["coverage_seed_matched"] = seed_matched
        stats["coverage_orphan_keys"] = len(seed_passed) - seed_matched
        stats["coverage_orphan_mass"] = round(
            sum(v for k, v in seed_passed.items() if k not in universe), 4)
        stats["coverage_orphans_dropped"] = commit_diag["orphans_dropped"]
        stats["coverage_persisted_live"] = (len(new_coverage)
                                            - commit_diag["persisted_orphans"])
        stats["coverage_persisted_orphans"] = commit_diag["persisted_orphans"]
        stats["coverage_orphan_doc_keys"] = commit_diag["orphan_doc_keys"]
        stats["coverage_persisted_orphan_mass"] = round(
            commit_diag["persisted_orphan_mass"], 4)
        stats["coverage_gc_dropped"] = commit_diag["gc_dropped"]
        stats["coverage_capped_dropped"] = commit_diag["cap_dropped"]

        sel_idx = [sr.sent_idx for sr in selected]
        context = delimiter.join(sr.text for sr in selected)
        out = {"context": context, "stats": stats, "selected_sent_idx": sel_idx,
               "n_total_sentences": self.n_sentences, "n_turns": self.n_turns}
        if defer_commit:
            out["commit"] = commit_cb
        return out

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
        self.config["n_alive"] = self.n_alive
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
            "alive": self.alive,
            "coverage": self.coverage, "seen_hashes": self._seen_hashes,
            "drift_ema": self.drift_ema,
            "coverage_turn": self.coverage_turn,
            "n_compress": self._n_compress,
            "n_near_dups": self.n_near_dups,
            "kw_order": self.kw_order,
            "theme_admitted": self.theme_admitted,
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
        # sessions saved before bounded eviction carry no mask: all alive
        self.alive = state.get("alive", [True] * len(self.texts))
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
        # sessions saved before stable coverage keys have no frozen order
        self.kw_order = state.get("kw_order", [])
        self.theme_admitted = state.get("theme_admitted", set())
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
        # A whole new corpus arrived: drop what the old one derived rather
        # than derive into it, and do it AFTER the repair above, which can
        # shorten the rows. A stale token entry would be unreachable
        # rather than wrong, but it would also never be freed.
        self._lex = {}
        self._kw_df = self._kw_df_rows = None
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
        dropped = []
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
            self.alive = self.alive[:n]
            for t in dropped:
                self._seen_hashes.discard(self._norm_hash(t))
        self._next_sentence_index = n
        self.dirty = True
        repair = {"kept": n, "orphan_rows": max(0, n_rows - n),
                  "dropped_sentences": max(0, n_txt - n)}
        self._log_load_repair(dict(repair, dropped_texts=dropped)
                              if dropped else repair)
        return repair

    def _log_load_repair(self, record):
        # Append-only diagnostic, one JSON line per repair, carrying the
        # dropped sentence texts so external damage never destroys
        # transcript text with no trace. Best-effort: a logging failure
        # must never block opening the session.
        try:
            with open(self._p("load_repairs.jsonl"), "a",
                      encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass
