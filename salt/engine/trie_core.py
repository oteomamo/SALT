# -*- coding: utf-8 -*-
"""
SALT core primitives: dense-attention keyword extraction, BGE embedding, theme
profiling, and query parsing. These are the building blocks the `trie_select`
selector (`salt.engine.retrieval`) stands on.
"""
import re
import string
import numpy as np
import torch
from collections import Counter, defaultdict



# =============================================================================
# PHASE 1 HELPERS
# =============================================================================

STOPWORDS = frozenset({"the"})


def l2_norm(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, 1e-12)


def is_content_word(w):
    return len(w) > 2 and w.isalpha() and w not in STOPWORDS


def find_kneedle(steps, max_keywords_ratio=0.3):
    n = len(steps)
    if n < 3:
        return max(0, n - 1)
    x = np.array([s["step"] for s in steps], dtype=float)
    y = np.array([s["cos"] for s in steps], dtype=float)
    x_norm = (x - x[0]) / max(x[-1] - x[0], 1e-12)
    y_norm = (y - y[0]) / max(y[-1] - y[0], 1e-12)
    distances = np.abs(y_norm - x_norm) / np.sqrt(2)
    max_knee = max(1, int(n * max_keywords_ratio))
    return int(np.argmax(distances[:max_knee + 1]))


def merge_subwords(content_tokens, cls_attn_content):
    if len(content_tokens) == 0:
        return [], np.array([]), []
    word_indices = []
    word_labels = []
    current_idxs = [0]
    current_label = content_tokens[0].replace("\u2581", "").replace("##", "")
    for i in range(1, len(content_tokens)):
        tok = content_tokens[i]
        if tok.startswith("\u2581") or (not tok.startswith("##")):
            word_indices.append(current_idxs)
            word_labels.append(current_label.lower())
            current_idxs = [i]
            current_label = tok.replace("\u2581", "").replace("##", "")
        else:
            current_idxs.append(i)
            current_label += tok.replace("\u2581", "").replace("##", "")
    word_indices.append(current_idxs)
    word_labels.append(current_label.lower())
    word_attns = np.array([cls_attn_content[idxs].sum() for idxs in word_indices])
    return word_labels, word_attns, word_indices


def pack_sentences_dense(sentences, tokenizer, max_length=512, overlap_sents=2):
    sent_token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in sentences]
    max_content = max_length - 2
    chunks = []
    start_idx = 0
    while start_idx < len(sentences):
        chunk_ids, chunk_spans, chunk_sent_indices = [], [], []
        cursor, i = 0, start_idx
        while i < len(sentences):
            ids = sent_token_ids[i]
            if cursor + len(ids) > max_content and cursor > 0:
                break
            trimmed = ids[:max_content - cursor]
            if not trimmed:
                i += 1
                continue
            chunk_ids.extend(trimmed)
            chunk_spans.append((cursor, cursor + len(trimmed)))
            chunk_sent_indices.append(i)
            cursor += len(trimmed)
            i += 1
        if chunk_ids:
            chunks.append({"sentence_indices": chunk_sent_indices,
                           "token_ids": chunk_ids, "spans": chunk_spans})
        if not chunk_sent_indices:
            start_idx += 1
        else:
            last = chunk_sent_indices[-1]
            start_idx = max(last - overlap_sents + 1, last)
            if start_idx <= chunk_sent_indices[0]:
                start_idx = last + 1
    return chunks


def build_window_inputs(sentences, tokenizer, model_device, max_length=512):
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    sent_token_ids = [tokenizer.encode(s, add_special_tokens=False) for s in sentences]
    max_content_len = max_length - 2
    kept_ids, spans, included = [], [], []
    cursor = 0
    for ids in sent_token_ids:
        if cursor >= max_content_len:
            included.append(False); spans.append((cursor, cursor)); continue
        room = max_content_len - cursor
        ids_cut = ids[:room]
        if len(ids_cut) == 0:
            included.append(False); spans.append((cursor, cursor)); continue
        start = cursor; end = cursor + len(ids_cut)
        kept_ids.extend(ids_cut); spans.append((start, end))
        included.append(True); cursor = end
    input_ids = [cls_id] + kept_ids + [sep_id]
    inputs = {
        "input_ids": torch.tensor([input_ids], device=model_device),
        "attention_mask": torch.tensor([[1] * len(input_ids)], device=model_device),
        "token_type_ids": torch.tensor([[0] * len(input_ids)], device=model_device),
    }
    content_tokens = tokenizer.convert_ids_to_tokens(kept_ids)
    return inputs, content_tokens, spans, included


def run_dense_attention(sentences, tokenizer, model, max_keywords_ratio=0.4,
                        overlap_sents=2):
    n_sents = len(sentences)
    if n_sents == 0: return []
    chunks = pack_sentences_dense(sentences, tokenizer, max_length=512,
                                  overlap_sents=overlap_sents)
    cls_id = tokenizer.cls_token_id; sep_id = tokenizer.sep_token_id
    sent_attn_max = [None] * n_sents
    sent_attn_renorm = [None] * n_sents
    sent_word_labels = [None] * n_sents
    sent_word_indices = [None] * n_sents
    sent_content_embs = [None] * n_sents
    sent_best_mass = [0.0] * n_sents
    for chunk in chunks:
        token_ids = chunk["token_ids"]; spans = chunk["spans"]
        sent_indices = chunk["sentence_indices"]
        input_ids = [cls_id] + token_ids + [sep_id]
        model_inputs = {
            "input_ids": torch.tensor([input_ids], device=model.device),
            "attention_mask": torch.tensor([[1] * len(input_ids)], device=model.device),
            "token_type_ids": torch.tensor([[0] * len(input_ids)], device=model.device),
        }
        with torch.no_grad():
            outputs = model(**model_inputs, output_attentions=True)
        hidden = outputs.last_hidden_state[0].cpu().numpy()
        content_embs_all = hidden[1:1 + len(token_ids)]
        last_layer_attn = outputs.attentions[-1][0].cpu()
        cls_attn_raw = last_layer_attn[:, 0, :].mean(dim=0).numpy()
        cls_attn_content = cls_attn_raw[1:1 + len(token_ids)]
        content_tokens_all = tokenizer.convert_ids_to_tokens(token_ids)
        for local_idx, global_idx in enumerate(sent_indices):
            start, end = spans[local_idx]
            if end <= start: continue
            s_tokens = content_tokens_all[start:end]
            s_attn_raw = cls_attn_content[start:end]
            s_embs = content_embs_all[start:end]
            if len(s_tokens) == 0: continue
            attn_sum = s_attn_raw.sum()
            s_renorm = s_attn_raw / attn_sum if attn_sum > 1e-12 else s_attn_raw.copy()
            wlabels, wattns_renorm, windices = merge_subwords(s_tokens, s_renorm)
            _, wattns_raw, _ = merge_subwords(s_tokens, s_attn_raw)
            if sent_word_labels[global_idx] is None:
                sent_word_labels[global_idx] = wlabels
                sent_word_indices[global_idx] = windices
            ref_n = len(sent_word_labels[global_idx])
            aligned_raw = np.zeros(ref_n); aligned_renorm = np.zeros(ref_n)
            for j in range(min(len(wattns_raw), ref_n)):
                aligned_raw[j] = wattns_raw[j]
                aligned_renorm[j] = wattns_renorm[j]
            if sent_attn_max[global_idx] is None:
                sent_attn_max[global_idx] = aligned_raw.copy()
            else:
                sent_attn_max[global_idx] = np.maximum(
                    sent_attn_max[global_idx], aligned_raw)
            window_mass = float(aligned_raw.sum())
            if window_mass > sent_best_mass[global_idx]:
                sent_best_mass[global_idx] = window_mass
                sent_attn_renorm[global_idx] = aligned_renorm.copy()
                sent_content_embs[global_idx] = s_embs
    for i in range(n_sents):
        if sent_word_labels[i] is not None: continue
        inputs, content_tokens, spans, included = build_window_inputs(
            [sentences[i]], tokenizer, model.device, max_length=512)
        if len(content_tokens) == 0:
            sent_word_labels[i] = []; sent_word_indices[i] = []
            sent_attn_max[i] = np.array([]); sent_attn_renorm[i] = np.array([])
            sent_content_embs[i] = np.array([]).reshape(0, 1); continue
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        hidden = outputs.last_hidden_state[0].cpu().numpy()
        cembs = hidden[1:1 + len(content_tokens)]
        cattn = outputs.attentions[-1][0].cpu()[:, 0, :].mean(dim=0).numpy()
        cattn_content = cattn[1:1 + len(content_tokens)]
        attn_sum = cattn_content.sum()
        cattn_renorm = cattn_content / attn_sum if attn_sum > 1e-12 else cattn_content.copy()
        wl, wa_raw, wi = merge_subwords(content_tokens, cattn_content)
        _, wa_renorm, _ = merge_subwords(content_tokens, cattn_renorm)
        sent_word_labels[i] = wl; sent_word_indices[i] = wi
        sent_attn_max[i] = wa_raw; sent_attn_renorm[i] = wa_renorm
        sent_content_embs[i] = cembs
    results = []
    for i in range(n_sents):
        wlabels = sent_word_labels[i]
        wattns = sent_attn_renorm[i] if sent_attn_renorm[i] is not None else sent_attn_max[i]
        windices = sent_word_indices[i]; cembs = sent_content_embs[i]
        if not wlabels or wattns is None or len(wattns) == 0 \
                or cembs is None or len(cembs) == 0:
            results.append({"sent_idx": i, "word_labels": [], "word_attns": np.array([]),
                            "word_indices": [], "content_embs": np.array([]).reshape(0, 1),
                            "important_words": set(), "knee": 0, "kneedle_steps": [],
                            "avg_cosine_at_knee": 0.0, "total_attn_mass": 0.0})
            continue
        final_mean = l2_norm(cembs.mean(axis=0, keepdims=True))[0]
        ranked = np.argsort(wattns)[::-1]
        accumulated_idxs = []; steps = []
        for step, word_idx in enumerate(ranked):
            if word_idx < len(windices): accumulated_idxs.extend(windices[word_idx])
            valid_idxs = [idx for idx in accumulated_idxs if idx < len(cembs)]
            if not valid_idxs: continue
            subset_emb = cembs[valid_idxs].mean(axis=0)
            subset_normed = subset_emb / max(np.linalg.norm(subset_emb), 1e-12)
            cos = float(np.dot(subset_normed, final_mean))
            steps.append({"step": step + 1, "cos": cos})
        knee = find_kneedle(steps, max_keywords_ratio=max_keywords_ratio) if steps else 0
        avg_cos_at_knee = steps[knee]["cos"] if steps and knee < len(steps) else 0.0
        content_ranked = [idx for idx in ranked
                          if idx < len(wlabels) and is_content_word(wlabels[idx])]
        n_keep = min(len(content_ranked), knee + 1)
        important_words = {wlabels[idx] for idx in content_ranked[:n_keep]}
        raw_attns = sent_attn_max[i] if sent_attn_max[i] is not None else wattns
        total_mass = float(raw_attns.sum())
        results.append({"sent_idx": i, "word_labels": wlabels, "word_attns": wattns,
                        "word_indices": windices, "content_embs": cembs,
                        "important_words": important_words, "knee": knee,
                        "kneedle_steps": steps, "avg_cosine_at_knee": avg_cos_at_knee,
                        "total_attn_mass": total_mass})
    return results


# =============================================================================
# Query keyword extraction (extended stopwords + proper nouns)
# =============================================================================
QUERY_STOPWORDS = frozenset({
    "the","a","an","is","are","was","were","be","been","being",
    "do","does","did","have","has","had","having",
    "will","would","shall","should","may","might","can","could",
    "that","this","these","those","it","its",
    "and","or","but","not","no","nor",
    "in","on","at","to","for","of","by","with","from",
    "as","if","so","than","too","very",
    "what","which","who","whom","whose","where","when","how","why"})

# Extra QA-boilerplate stopwords: verbs/nouns of asking that aren't content kws.
QUERY_STOPWORDS_EXTRA = frozenset({
    "describe","describes","described","describing",
    "tell","tells","told","telling",
    "say","says","said","saying",
    "happen","happens","happened","happening",
    "call","calls","called","calling",
    "name","names","named","naming",
    "passage","text","document","article","story","context",
    "mention","mentions","mentioned","according","based",
    "follow","follows","followed","following",
    "main","thing","things","way","ways","like","given",
    "show","shows","shown","showing",
})

QUESTION_WORDS = frozenset({"who","what","where","when","why","how",
                            "which","whose","whom"})

_POSSESSIVE_RE = re.compile(r"[\u2019\u2018']s$|[\u2019\u2018']$")

def clean_query_word(w):
    w = w.lower().strip(".,;:!?()[]\"'\u201c\u201d\u2018\u2019")
    return _POSSESSIVE_RE.sub("", w)

def extract_query_keywords(query_text, use_extended=True):
    sset = (QUERY_STOPWORDS | QUERY_STOPWORDS_EXTRA) if use_extended else QUERY_STOPWORDS
    keywords = set()
    for w in query_text.split():
        cleaned = clean_query_word(w)
        if cleaned and len(cleaned) >= 2 and cleaned.isalpha() and cleaned not in sset:
            keywords.add(cleaned)
    return keywords

def extract_proper_nouns_in_query(qt):
    """
    Capitalized non-stopword tokens OR hyphenated lowercase compounds in the query.

    """
    nouns = set()
    for tok in qt.split():
        raw = tok.strip(".,;:!?()[]\"'\u201c\u201d\u2018\u2019")
        if not raw or len(raw) < 2:
            continue

        # Hyphenated compound (e.g., "jittery-hospital"): treat as named entity.
        if "-" in raw and len(raw) > 5:
            parts = raw.split("-")
            if len(parts) >= 2 and all(p.isalpha() and len(p) >= 2 for p in parts):
                nouns.add(raw.lower())
                continue

        # Original path: capitalized token = proper noun.
        if raw.isupper() and len(raw) > 3:
            continue  # likely acronym
        if not raw[0].isupper():
            continue
        c = clean_query_word(raw)
        if not c or c in QUESTION_WORDS or c in QUERY_STOPWORDS:
            continue
        nouns.add(c)
    return nouns

def clean_text_words(text):
    words = set()
    for w in text.lower().split():
        w = w.strip(".,;:!?()[]\"'\u201c\u201d\u2018\u2019")
        w = _POSSESSIVE_RE.sub("", w)
        if w: words.add(w)
    return words


# =============================================================================
# Soft stemming for lexical overlap
# =============================================================================
_STEM_SUFFIXES = ("ingly", "edly", "ings", "ies", "ied", "ing", "ed", "es", "ly")

def soft_stem(w):
    w = w.lower()
    if len(w) < 4: return w
    for suf in _STEM_SUFFIXES:
        if w.endswith(suf) and len(w) > len(suf) + 2:
            return w[:-len(suf)]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    if w.endswith("e") and len(w) > 3:
        return w[:-1]
    return w

def expand_with_stems(words):
    out = set(words)
    for w in words:
        s = soft_stem(w)
        if s and s != w: out.add(s)
    return out

def detect_question_type(qt):
    """Returns one of: who, what, where, when, why, how, how_many, how_die, None.
    Looks at first 4 query tokens for a wh-word, then refines with the next token."""
    if not qt: return None
    toks = qt.lower().strip().split()
    if not toks: return None
    fq, fi = None, -1
    for i, t in enumerate(toks[:4]):
        tc = t.strip(".,?!:;'\"")
        if tc in QUESTION_WORDS:
            fq, fi = tc, i; break
    if fq is None: return None
    if fi + 1 < len(toks):
        second = toks[fi+1].strip(".,?!:;'\"")
        if fq == "how":
            if second in ("many","much"): return "how_many"
            if second in ("long","old"): return "when"
            for d in ("die","died","dies","kill","killed","kills","murder","murdered"):
                if d in toks[fi:fi+6]:
                    return "how_die"
        elif fq == "what":
            if second in ("year","time","date","age","century","decade","month","day"):
                return "when"
    return fq


# =============================================================================
# Query embedding (BGE retrieval prefix on by default)
# =============================================================================
BGE_RETRIEVAL_PREFIX = "Represent this sentence for searching relevant passages: "

@torch.no_grad()
def embed_query(query_text, tokenizer, model, device="cpu", retrieval=True):
    if not query_text or not query_text.strip(): return None
    q = query_text.strip()
    if retrieval and not q.startswith(BGE_RETRIEVAL_PREFIX):
        q = BGE_RETRIEVAL_PREFIX + q
    encoded = tokenizer([q], padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
    outputs = model(**{k: v for k, v in encoded.items()
                       if k in ("input_ids", "attention_mask", "token_type_ids")})
    emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 1e-12 else emb

@torch.no_grad()
def get_bge_sentence_embeddings(sentences, tokenizer, model, device="cpu", batch_size=32):
    """
    Batched [CLS] passage embeddings, L2-normalized. Each sentence is encoded
    as an independent sequence to align with the BGE query space (unlike the
    dense-attention pass, which packs sentences together).
    """
    all_embs = []
    for i in range(0, len(sentences), batch_size):
        batch_texts = sentences[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(device)
        outputs = model(**{k: v for k, v in encoded.items()
                           if k in ("input_ids", "attention_mask", "token_type_ids")})
        cls_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        norms = np.linalg.norm(cls_embs, axis=1, keepdims=True)
        all_embs.extend((cls_embs / np.maximum(norms, 1e-12)).tolist())
    return all_embs


# =============================================================================
# Sentence record
# =============================================================================
class SentenceRecord:
    __slots__ = ["sent_idx", "text", "n_words", "keyword_weights",
                 "theme_keywords", "embedding_l2", "theme_precision",
                 "n_total_keywords"]
    def __init__(self, sent_idx, text, n_words, keyword_weights,
                 theme_keywords, embedding_l2):
        self.sent_idx = sent_idx; self.text = text; self.n_words = n_words
        self.keyword_weights = keyword_weights; self.theme_keywords = theme_keywords
        self.embedding_l2 = embedding_l2
        self.n_total_keywords = len(keyword_weights)
        self.theme_precision = len(theme_keywords) / max(len(keyword_weights), 1)


# =============================================================================
# Document theme profiling
# =============================================================================
def profile_themes(sent_data, theme_percentile=0.75):
    kw_df = Counter()
    for sd in sent_data:
        seen = set()
        for kw in sd["keyword_weights"]:
            if kw not in seen: kw_df[kw] += 1; seen.add(kw)
    if not kw_df: return {}, set()
    df_values = sorted(kw_df.values())
    threshold_idx = int(len(df_values) * theme_percentile)
    df_threshold = df_values[min(threshold_idx, len(df_values) - 1)]
    theme_keywords = {kw for kw, df in kw_df.items() if df >= df_threshold}
    return dict(kw_df), theme_keywords
