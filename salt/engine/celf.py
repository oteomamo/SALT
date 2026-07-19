# -*- coding: utf-8 -*-
"""
SALT coverage selector: the CELF (lazy-greedy) submodular sentence selector.

`coverage_select` maximizes a probabilistic trie-coverage objective under a word
budget, replacing the phase/boost heuristics of the legacy `trie_select`
(`salt.engine.retrieval`) with a single submodular objective. It is the default
selector of the `salt/compress.py` connector (`--selector coverage`) and the selector
used by the persistent multi-turn store (`salt.engine.session_trie.SessionTrie`).

The shared primitives it builds on (trie construction, theme profiling, query
parsing) live in `salt.engine.trie_core`.
"""
import numpy as np

from salt.engine.trie_core import (
    SentenceRecord, clean_text_words, expand_with_stems, soft_stem,
    build_trie_paths,
)


def coverage_select(sent_data, kw_df, theme_keywords, word_budget,
                    query_keywords=None, query_embedding=None,
                    query_proper_nouns=None,
                    lam=0.5, query_mass_ratio=1.0, token_fn=None,
                    seed_coverage=None, return_coverage=False,
                    kw_rank=None):
    """Budget-constrained sentence selection by maximizing the probabilistic
    trie-coverage objective with CELF lazy greedy. Replaces the phase/boost
    heuristics of trie_select with a single submodular objective.

    Multi-turn extensions (both default off -> one-shot behavior unchanged):
      seed_coverage: {frozenset(path keywords): count} of prior-turn per-node
        coverage. Seeds the CELF coverage counters so already-surfaced material
        is discounted this turn (cross-turn submodular memory).
      return_coverage: when True, also return {"coverage": merged_counts,
        "node_keys": the set of node keys in this turn's trie} where the
        merge is prior seed + this turn's selected sentences."""
    import heapq

    if not (0.0 < lam < 1.0):
        raise ValueError(f"lam must be in (0, 1), got {lam}: lam>=1 voids "
                         "monotone submodularity (gains vanish or flip sign)")

    records = []
    for sd in sent_data:
        kw_set = set(sd["keyword_weights"].keys())
        sr = SentenceRecord(sent_idx=sd["sent_idx"], text=sd["text"],
                            n_words=sd["n_words"],
                            keyword_weights=sd["keyword_weights"],
                            theme_keywords=kw_set & set(theme_keywords),
                            embedding_l2=sd.get("embedding_l2"))
        records.append(sr)
    n = len(records)
    if n == 0:
        empty = {"n_selected": 0, "words_used": 0, "n_branches": 0,
                 "pass1_selected": 0, "pass2_selected": 0,
                 "pass3_query_anchored": 0, "avg_words_per_sent": 0,
                 "theme_keywords_total": 0, "theme_keywords_covered": 0,
                 "theme_coverage_pct": 0.0, "n_precision_skips": 0,
                 "n_redundancy_skips": 0, "trie": {},
                 "selected_details": []}
        if return_coverage:
            return [], empty, {"coverage": dict(seed_coverage or {}),
                               "node_keys": set()}
        return [], empty
    costs = np.array([max(sr.n_words, 1) for sr in records], dtype=np.float64)

    # --- Document term: trie node paths, weighted by normalized SF. ---
    paths, node_w, n_branches, node_kw = build_trie_paths(
        [sr.theme_keywords for sr in records], kw_df, theme_keywords,
        kw_rank=kw_rank)
    doc_mass = float(node_w.sum())

    # Canonical, rebuild-stable key per node id: the frozenset of keywords on the
    # root->node path. A trie node IS uniquely its ancestor-context + keyword set
    # (multi-anchor semantics), so the frozenset is order-insensitive and survives
    # df-rank reordering as the corpus grows -- the key that lets cross-turn
    # coverage (seed_coverage) map onto a freshly rebuilt, larger trie.
    node_key = [None] * len(node_w)
    for pid in paths:
        acc = []
        for v in pid:
            acc.append(node_kw[v])
            if node_key[v] is None:
                node_key[v] = frozenset(acc)

    # --- Query terms: surprisal-weighted lexical nodes + semantic node. ---
    term_w = np.array([], dtype=np.float64)
    term_hits = [[] for _ in range(n)]
    rel = np.zeros(n)
    emb_mass = 0.0
    q_terms = set()
    if query_keywords:
        q_terms |= {w for w in query_keywords if len(w) >= 2}
    if query_proper_nouns:
        q_terms |= {p.lower() for p in query_proper_nouns}
    has_emb = query_embedding is not None
    if q_terms or has_emb:
        base_mass = doc_mass if doc_mass > 0.0 else 1.0
        query_mass = query_mass_ratio * base_mass
        lex_mass = 0.0
        if q_terms:
            sent_words_exp = [token_fn(sr.text) if token_fn
                              else expand_with_stems(clean_text_words(sr.text))
                              for sr in records]
            term_list, term_df, hits_per_term = [], [], []
            for w in sorted(q_terms):
                ws = soft_stem(w)
                hits = [i for i in range(n)
                        if w in sent_words_exp[i] or ws in sent_words_exp[i]]
                if hits:
                    term_list.append(w)
                    term_df.append(len(hits))
                    hits_per_term.append(hits)
            if term_list:
                surp = np.log((n + 1.0) / (np.array(term_df, dtype=np.float64) + 1.0))
                surp = np.maximum(surp, 1e-6)
                # Gate the ABSOLUTE lexical mass by realized surprisal against
                # the all-terms-rare scale, so a lone generic/spurious match
                # cannot absorb the whole lexical channel; unspent mass flows
                # to the semantic node when an embedding exists.
                full_scale = max(len(q_terms) * np.log((n + 1.0) / 2.0), 1e-9)
                match_frac = min(1.0, float(surp.sum()) / full_scale)
                lex_mass = query_mass * (0.5 if has_emb else 1.0) * match_frac
                term_w = lex_mass * surp / surp.sum()
                for t, hits in enumerate(hits_per_term):
                    for i in hits:
                        term_hits[i].append(t)
        if has_emb:
            emb_mass = query_mass - lex_mass
            cos_raw = np.zeros(n)
            for i, sr in enumerate(records):
                if sr.embedding_l2 is not None:
                    cos_raw[i] = max(0.0, float(np.dot(query_embedding,
                                                       np.array(sr.embedding_l2))))
            # Encoder cosines have a high anisotropic floor (~0.5 even for
            # unrelated text), so raw accumulation would saturate the semantic
            # node with background similarity after a handful of picks. Measure
            # relevance ABOVE the document median and rescale to [0, 1].
            floor = float(np.median(cos_raw))
            rel = np.maximum(0.0, cos_raw - floor)
            rmax = float(rel.max())
            if rmax > 1e-9:
                rel = rel / rmax
            # Scale-free saturation: count semantic coverage in budget
            # fractions, not picks, or lam^sigma dies after ~10 selections and
            # leaves the rest of a 1000-sentence budget query-blind. sigma
            # reaches ~1 when a budget's worth of average-relevance text is
            # selected; increments stay modular, so submodularity and the
            # greedy guarantee are unchanged.
            total_cost = float(costs.sum())
            rel_norm = float((rel * costs).sum()) * (
                min(1.0, word_budget / max(total_cost, 1.0)))
            if rel_norm > 1e-9:
                rel = rel * costs / rel_norm

    # --- CELF lazy greedy on F; run cost-benefit and unit-cost variants. ---
    def run_greedy(ratio_mode):
        n_node = np.zeros(len(node_w))
        if seed_coverage:
            # Seed prior-turn coverage: nodes already surfaced have higher n_node,
            # so lam**n_node shrinks their marginal gain -> this turn favors
            # still-uncovered material. m_term / sigma are query-specific (the
            # query changes each turn) and are NOT seeded.
            for v in range(1, len(node_w)):
                c = seed_coverage.get(node_key[v])
                if c:
                    n_node[v] = c
        m_term = np.zeros(len(term_w)) if len(term_w) else None
        sigma = 0.0

        def gain(i):
            g = 0.0
            for v in paths[i]:
                g += node_w[v] * (lam ** n_node[v])
            for t in term_hits[i]:
                g += term_w[t] * (lam ** m_term[t])
            g *= (1.0 - lam)
            if emb_mass > 0.0 and rel[i] > 0.0:
                g += emb_mass * (lam ** sigma) * (1.0 - lam ** rel[i])
            return g

        heap = []
        for i in range(n):
            g = gain(i)
            if g > 0.0:
                pri = g / costs[i] if ratio_mode else g
                heap.append((-pri, i, g, 0))
        heapq.heapify(heap)
        selected, used, fval, tick = [], 0.0, 0.0, 0
        while heap:
            neg_pri, i, g, t = heapq.heappop(heap)
            if t < tick:
                g = gain(i)
                if g <= 0.0:
                    continue
                pri = g / costs[i] if ratio_mode else g
                heapq.heappush(heap, (-pri, i, g, tick))
                continue
            if used + costs[i] > word_budget:
                continue  # drop: budget only shrinks
            selected.append(i)
            used += costs[i]
            fval += g
            tick += 1
            for v in paths[i]:
                n_node[v] += 1.0
            for tt in term_hits[i]:
                m_term[tt] += 1.0
            sigma += rel[i]
        return selected, used, fval

    sel_r, used_r, f_r = run_greedy(True)
    sel_u, used_u, f_u = run_greedy(False)
    if f_u > f_r:
        sel_idx, used_words, f_best, mode = sel_u, used_u, f_u, "unit"
    else:
        sel_idx, used_words, f_best, mode = sel_r, used_r, f_r, "ratio"

    # Zero-gain budget fill: remaining sentences in document order, first-fit.
    chosen = set(sel_idx)
    for i in range(n):
        if i in chosen:
            continue
        if used_words + costs[i] <= word_budget:
            chosen.add(i)
            used_words += costs[i]
    selected = sorted((records[i] for i in chosen), key=lambda sr: sr.sent_idx)
    n_fill = len(chosen) - len(sel_idx)
    used_words = int(used_words)

    # Accumulated per-node coverage (prior seed + every node on this turn's kept
    # sentences, incl. the zero-gain fill -- those are shown to the model too).
    new_cov = None
    node_keys = None
    if return_coverage:
        new_cov = dict(seed_coverage) if seed_coverage else {}
        for i in chosen:
            for v in paths[i]:
                k = node_key[v]
                new_cov[k] = new_cov.get(k, 0.0) + 1.0
        node_keys = {k for k in node_key[1:] if k is not None}

    all_theme_kws, covered_kws = set(), set()
    for sd in sent_data:
        all_theme_kws |= (set(sd["keyword_weights"].keys()) & set(theme_keywords))
    for sr in selected:
        covered_kws |= sr.theme_keywords

    stats = {
        "n_selected": len(selected), "pass1_selected": len(sel_idx),
        "pass2_selected": n_fill, "pass3_query_anchored": 0,
        "n_branches": n_branches, "words_used": used_words,
        "avg_words_per_sent": round(used_words / max(len(selected), 1), 1),
        "theme_keywords_total": len(all_theme_kws),
        "theme_keywords_covered": len(covered_kws),
        "theme_coverage_pct": round(len(covered_kws) / max(len(all_theme_kws), 1), 4),
        "n_precision_skips": 0, "n_redundancy_skips": 0,
        "trie": {"n_branches": n_branches, "n_nodes": len(node_w) - 1},
        "selector": "coverage", "lam": lam,
        "query_mass_ratio": query_mass_ratio if (q_terms or has_emb) else 0.0,
        "objective": round(f_best, 4), "doc_mass": round(doc_mass, 4),
        "greedy_mode": mode,
        "n_query_terms_active": int(len(term_w)),
        "qtype": None,
        "n_query_proper_nouns": len(query_proper_nouns or set()),
        "selected_details": [{"sent_idx": sr.sent_idx, "n_words": sr.n_words,
                              "n_theme_kw": len(sr.theme_keywords),
                              "theme_precision": round(sr.theme_precision, 3)}
                             for sr in selected],
    }
    if return_coverage:
        return selected, stats, {"coverage": new_cov, "node_keys": node_keys}
    return selected, stats
