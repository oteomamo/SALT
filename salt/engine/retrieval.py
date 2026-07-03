# -*- coding: utf-8 -*-
"""
SALT selector: keyword-trie guided-traversal selection.

`trie_select` is a multi-phase heuristic selector (query anchoring + neighbor
expansion, df-weighted branch budgeting, thematic fill). The shared trie / 
attention / theme / query primitives
it builds on live in `salt.engine.trie_core`.
"""
import re
import numpy as np

from salt.engine.trie_core import (
    SentenceRecord, clean_text_words, expand_with_stems,
)


# =============================================================================
# Question-type detection boosts
# =============================================================================
_NAME_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")
_DIGIT_RE = re.compile(r"\b\d+\b")
_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December|Monday|Tuesday|Wednesday|Thursday|Friday|"
    r"Saturday|Sunday|morning|afternoon|evening|night|yesterday|tomorrow|"
    r"today|year|years|month|months|week|weeks|day|days|hour|hours|"
    r"minute|minutes|century|centuries|decade|decades|ago|later|earlier)\b",
    re.IGNORECASE)
_CAUSAL_RE = re.compile(
    r"\b(?:because|since|due\s+to|therefore|thus|hence|reason|reasons|"
    r"caused?|causing|result(?:ed|ing|s)?|so\s+that|in\s+order\s+to|"
    r"led\s+to|leading\s+to|owing\s+to)\b", re.IGNORECASE)
_LOC_PREP_RE = re.compile(
    r"\b(?:in|at|near|inside|outside|by|toward|towards|upon|behind|"
    r"beside|beneath|above|below|across|along)\s+(?:the\s+)?[A-Z][a-z]{2,}\b")
_NUM_WORD_RE = re.compile(
    r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion|dozen|score)\b", re.IGNORECASE)
_DIE_RE = re.compile(
    r"\b(?:shot|killed|stabbed|hanged?|hung|fell|jumped|drowned|poisoned|"
    r"murdered|died|death|sword|knife|gun|pistol|throat|blood)\b",
    re.IGNORECASE)


def question_type_boost(text, qtype):
    """Per-question-type multiplier on a sentence based on text-level patterns
    that correlate with the gold-answer surface form for that question class."""
    if not qtype: return 1.0
    if qtype == "who" and _NAME_RE.search(text): return 1.4
    if qtype == "when":
        if _DIGIT_RE.search(text) or _DATE_RE.search(text): return 1.45
    if qtype == "where":
        if _LOC_PREP_RE.search(text): return 1.35
        if _NAME_RE.search(text): return 1.15
    if qtype == "how_many":
        if _DIGIT_RE.search(text): return 1.6
        if _NUM_WORD_RE.search(text): return 1.3
    if qtype == "why" and _CAUSAL_RE.search(text): return 1.4
    if qtype == "how_die" and _DIE_RE.search(text): return 1.5
    return 1.0


def _length_reward(n_words):
    if n_words < 15: return 0.30
    elif n_words < 20: return 0.65
    elif n_words < 28: return 0.85
    elif n_words <= 45: return 1.00
    return 0.95 / (1.0 + 0.01 * (n_words - 45))


# =============================================================================
# Selection via trie traversal (Tier-1 enhanced scoring)
# =============================================================================
def trie_select(sent_data, kw_df, theme_keywords, word_budget,
                query_keywords=None, query_embedding=None,
                query_proper_nouns=None, qtype=None,
                min_keywords=2, pass1_budget_pct=0.65,
                theme_prec_min=0.0, redundancy_thresh=1.0, intro_pct=0.05,
                neighbor_window=1, query_budget_uncapped=False,
                query_budget_pct=0.60, branch_floor=0):
    effective_themes = set(theme_keywords)
    if query_keywords:
        effective_themes |= query_keywords
        _max_df = max(kw_df.values()) if kw_df else 1
        for qk in query_keywords:
            kw_df[qk] = _max_df   # Fix 1: unconditional max-df promotion
    max_df = max(kw_df.values()) if kw_df else 1
    kw_df_norm = {kw: df / max_df for kw, df in kw_df.items()}

    records = []
    for sd in sent_data:
        kw_set = set(sd["keyword_weights"].keys())
        theme_kws = kw_set & effective_themes
        sr = SentenceRecord(sent_idx=sd["sent_idx"], text=sd["text"],
                            n_words=sd["n_words"],
                            keyword_weights=sd["keyword_weights"],
                            theme_keywords=theme_kws,
                            embedding_l2=sd.get("embedding_l2"))
        records.append(sr)
    n_total_sents = len(records)
    query_kw_set = query_keywords if query_keywords else set()
    pn_lower = {p.lower() for p in (query_proper_nouns or set())}

    # Precompute per-sentence text-token set, PN flag, qtype multiplier.
    sent_text_words = {}
    sent_text_words_exp = {}
    sent_pn_match = {}
    sent_qtype_mult = {}
    for sr in records:
        tw = clean_text_words(sr.text)
        sent_text_words[id(sr)] = tw
        sent_text_words_exp[id(sr)] = expand_with_stems(tw)
        sent_pn_match[id(sr)] = bool(tw & pn_lower) if pn_lower else False
        sent_qtype_mult[id(sr)] = question_type_boost(sr.text, qtype) if qtype else 1.0

    # Stem-expanded query keywords (for soft lexical matching).
    query_kw_exp = expand_with_stems(query_kw_set)

    def score_sentence(sr):
        df_mass = 0.0
        for kw in sr.theme_keywords:
            w = kw_df_norm.get(kw, 0.0)
            if kw in query_kw_set: w *= 3.0
            df_mass += w
        lr = _length_reward(sr.n_words)
        s = df_mass * lr
        # Proper-noun protection: query named entity present in sentence text.
        if sent_pn_match.get(id(sr)): s *= 1.3
        # Question-type pattern boost.
        s *= sent_qtype_mult.get(id(sr), 1.0)
        # Intro position prior.
        if n_total_sents > 0 and sr.sent_idx < n_total_sents * intro_pct:
            s *= 1.3
        return s

    selected = []; selected_set = set(); used_words = 0; p3_count = 0
    has_query = bool(query_kw_set)

    if has_query:
        query_phase_budget = int(word_budget * query_budget_pct)
        idx_to_record = {sr.sent_idx: sr for sr in records}
        query_candidates = []
        for sr in records:
            if id(sr) in selected_set: continue
            tw_exp = sent_text_words_exp[id(sr)]
            overlap = tw_exp & query_kw_exp     # soft (stem) lexical overlap
            kw_score = len(overlap) / max(len(query_kw_exp), 1)
            cos_score = 0.0
            if query_embedding is not None and sr.embedding_l2 is not None:
                emb = np.array(sr.embedding_l2)
                cos_score = max(0.0, float(np.dot(query_embedding, emb)))
            # Proper-noun bonus in query phase (small, additive).
            pn_count = len(sent_text_words[id(sr)] & pn_lower) if pn_lower else 0
            pn_score = (pn_count / max(len(pn_lower), 1)) if pn_lower else 0.0
            combined = kw_score + cos_score + 0.6 * pn_score
            if combined > 0.05:
                query_candidates.append((combined, score_sentence(sr), sr, len(overlap)))
        query_candidates.sort(key=lambda x: (-x[0], -x[1]))
        keyword_anchored_indices = set()
        for combined, _, sr, n_overlap in query_candidates:
            if used_words + sr.n_words > query_phase_budget: continue
            selected.append(sr); selected_set.add(id(sr))
            used_words += sr.n_words; p3_count += 1
            if n_overlap > 0: keyword_anchored_indices.add(sr.sent_idx)
        cosine_anchored_indices = set()
        for combined, _, sr, _ in query_candidates[:10]:
            if id(sr) in selected_set: cosine_anchored_indices.add(sr.sent_idx)
        all_anchored = keyword_anchored_indices | cosine_anchored_indices
        for idx in sorted(all_anchored):
            for neighbor_idx in range(idx - neighbor_window, idx + neighbor_window + 1):
                if neighbor_idx == idx or neighbor_idx not in idx_to_record: continue
                nbr = idx_to_record[neighbor_idx]
                if id(nbr) not in selected_set and used_words + nbr.n_words <= query_phase_budget:
                    selected.append(nbr); selected_set.add(id(nbr))
                    used_words += nbr.n_words; p3_count += 1

    branches = {}
    for sr in records:
        if sr.theme_keywords and len(sr.keyword_weights) >= min_keywords:
            primary = max(sr.theme_keywords, key=lambda k: kw_df.get(k, 0))
            branches.setdefault(primary, []).append(sr)
    for bkw in branches:
        branches[bkw].sort(key=lambda sr: -score_sentence(sr))
    branch_mass = {bkw: sum(score_sentence(sr) for sr in srs[:10])
                   for bkw, srs in branches.items()}
    total_mass = sum(branch_mass.values()) or 1.0
    remaining_budget = word_budget - used_words
    phase1_budget = int(remaining_budget * (0.50 if has_query else pass1_budget_pct))
    branch_budgets = {bkw: max(int(phase1_budget * mass / total_mass), branch_floor)
                      for bkw, mass in branch_mass.items()}
    sorted_branches = sorted(branch_budgets.keys(), key=lambda k: -branch_mass[k])
    p1_start = len(selected); phase1_cap = used_words + phase1_budget
    branch_selected = {}   # bkw -> (selected_sent_idxs, words_used)
    for bkw in sorted_branches:
        bb = branch_budgets[bkw]; branch_used = 0
        sel_idxs = []
        for sr in branches[bkw]:
            if id(sr) in selected_set: continue
            if branch_used + sr.n_words > bb: continue
            if used_words + sr.n_words > phase1_cap: break
            selected.append(sr); selected_set.add(id(sr))
            branch_used += sr.n_words; used_words += sr.n_words
            sel_idxs.append(int(sr.sent_idx))
        branch_selected[bkw] = (sel_idxs, branch_used)
    p1_count = len(selected) - p1_start

    remaining = [(score_sentence(sr), sr) for sr in records
                 if id(sr) not in selected_set and sr.theme_keywords
                 and len(sr.keyword_weights) >= min_keywords]
    remaining.sort(key=lambda x: -x[0])
    for score, sr in remaining:
        if used_words >= word_budget: break
        if used_words + sr.n_words > word_budget: continue
        selected.append(sr); selected_set.add(id(sr))
        used_words += sr.n_words
    p2_count = len(selected) - p1_count - p3_count
    selected.sort(key=lambda sr: sr.sent_idx)

    all_theme_kws = set(); covered_kws = set()
    for sr in selected: covered_kws |= sr.theme_keywords
    for sd in sent_data:
        all_theme_kws |= (set(sd["keyword_weights"].keys()) & effective_themes)

    stats = {
        "n_selected": len(selected), "pass1_selected": p1_count,
        "pass2_selected": p2_count, "pass3_query_anchored": p3_count,
        "n_branches": len(branches), "words_used": used_words,
        "avg_words_per_sent": round(used_words / max(len(selected), 1), 1),
        "theme_keywords_total": len(all_theme_kws),
        "theme_keywords_covered": len(covered_kws),
        "theme_coverage_pct": round(len(covered_kws) / max(len(all_theme_kws), 1), 4),
        "n_precision_skips": 0, "n_redundancy_skips": 0,
        "trie": {"n_branches": len(branches)},
        "query_budget_uncapped": query_budget_uncapped,
        "neighbor_window": neighbor_window,
        "qtype": qtype,
        "n_query_proper_nouns": len(pn_lower),
        "selected_details": [{"sent_idx": sr.sent_idx, "n_words": sr.n_words,
                              "n_theme_kw": len(sr.theme_keywords),
                              "theme_precision": round(sr.theme_precision, 3)}
                             for sr in selected],
    }
    return selected, stats
