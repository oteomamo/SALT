"""
Sentence filter: drop junk sentences before embedding — fragments, wiki markup,
citations, URL-heavy lines, and near-duplicates.
"""

import re


# Patterns that indicate junk sentences
JUNK_PATTERNS = [
    r'^\s*\.\s*$',                          # lone periods
    r'^\s*\d+[\.\)]\s*$',                   # lone numbers "1." or "2)"
    r'^\([^)]*\)\s*$',                       # lone parenthetical "(2003)"
    r'^[A-Z][a-z]+\s*\(\d{4}',              # citation starts "Adams (2003"
    r'^\d+\.\d+\s+for\s+',                  # fragments like "17.9 for females)"
    r'^(ibid|op\.?\s*cit|et\.?\s*al)',       # Latin citation abbreviations
    r'^\s*[\[\(]?\d+[\]\)]?\s*$',           # bare reference numbers "[3]"
    r'^\[\[File:',                            # Wikipedia file links
    r'^\[\[Image:',                           # Wikipedia image links
    r'^\[\[Category:',                        # Wikipedia category tags
    r'^Category:',                            # Category tags without brackets
    r'^\{\{',                                 # Wiki template markup
    r'^#REDIRECT',                            # Wiki redirects
    r'^\s*<ref',                              # HTML reference tags
    r'^\s*</ref',                             # HTML ref closing tags
    r'^(See also|References|Bibliography)\s*:?\s*$',  # section headers
    r'^(Sources|External links|Further reading)\s*:?\s*$',
    r'^thumb\b',                              # Wiki image captions after pipe cleanup
    r'^\d+x\d+px\b',                         # Pixel dimensions "311x311px"
]

# Patterns that can appear anywhere in the text (not just start)
JUNK_CONTAINS = [
    r'\|(?:thumb|right|left|center|upright|frameless|frame|border|baseline)',  # wiki image params
    r'\|upright\s*=',                         # |upright=1.35|
    r'\|\d+x\d+px',                           # |300x200px
    r'\|\d+px',                                # |250px
    r'\bthumb\b.*\bupright\b',                # "thumb upright Augustus..." after cleanup
    r'\bright\b\s+thumb\b',                   # "right thumb upright=..."
    r'\bupright\s*=\s*[\d.]+',                # "upright=1.35" anywhere
    r'\b\d+x\d+px\b',                         # "311x311px" anywhere
]

MIN_CHAR_LENGTH = 30          # sentences shorter than this are likely junk
MIN_WORD_COUNT = 5            # fewer words than this is likely a fragment


# ── URL detection ──────────────────────────────────────────────────────────
_URL_PATTERN = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)


def contains_url(text: str) -> bool:
    return bool(_URL_PATTERN.search(text))


def _url_placeholder(m):
    # the URL pattern swallows trailing sentence punctuation; keep it
    trail = re.search(r"[).,;:\]]+$", m.group(0))
    return "<url>" + (trail.group(0) if trail else "")


def is_url_dominated(text: str, threshold: float = 0.40) -> bool:
    text_stripped = text.strip()
    if not text_stripped:
        return False
    urls = _URL_PATTERN.findall(text_stripped)
    if not urls:
        return False
    url_chars = sum(len(u) for u in urls)
    return url_chars / len(text_stripped) > threshold


# ── Near-duplicate normalization ───────────────────────────────────────────
_NORMALIZE_PUNCT_RE = re.compile(r'[^a-z0-9\s]')
_NORMALIZE_WS_RE = re.compile(r'\s+')


def _normalize_for_dedup(text: str) -> str:
    t = text.lower()
    t = _NORMALIZE_PUNCT_RE.sub(' ', t)
    t = _NORMALIZE_WS_RE.sub(' ', t).strip()
    return t


# ── Aggressive domain-aware filters (opt-in only) ──────────────────────────
_URL_RE = re.compile(r'https?://\S+')

_CONGRESS_META_RE = re.compile(
    r'\d{1,3}\s*(?:th|st|nd|rd)\s+Cong\.\s*,?\s*\d+\s*(?:st|nd|rd|th)\s+sess\.',
    re.IGNORECASE,
)

_GPO_RE = re.compile(r'\(Washington,?\s*D\.?C\.?:\s*GPO', re.IGNORECASE)

_BIBLIO_RE = re.compile(
    r'^[A-Z][a-z]+(?:[-\'][A-Z]?[a-z]+)?,\s+'
    r'(?:[A-Z][a-z]*\.?\s*){1,3}'
    r'(?:and\s+[A-Z][a-z]+(?:[-\'][A-Z]?[a-z]+)?,?\s+(?:[A-Z][a-z]*\.?\s*){1,3})?'
    r'"',
)

_NEWS_OUTLETS = re.compile(
    r'(?:Associated Press|Reuters|Washington Post|New York Times|'
    r'Army Times|Military\.com|Military Times|DOD News|Air Force News|'
    r'Defense Media Activity|Stars and Stripes|The Hill|Politico)',
    re.IGNORECASE,
)

# ── V3: Additional aggressive patterns ──────────────────────────────────────

_CRS_REPORT_RE = re.compile(
    r'(?:^|\b)(?:See|For)\s+(?:also\s+|current\s+information[^,]*,\s*)?'
    r'(?:see\s+)?CRS\s+Report\s+[A-Z]?\d+',
    re.IGNORECASE,
)

_FORWARD_REF_RE = re.compile(
    r'^(?:For\s+(?:current|legislative|additional|further|more)\s+'
    r'(?:information|initiatives|details|analysis|discussion|data))',
    re.IGNORECASE,
)

_SUPERSEDES_RE = re.compile(
    r'(?:supersedes|replaces|updates|see\s+also)\s+CRS\s+Report',
    re.IGNORECASE,
)

_CITATION_CHAIN_RE = re.compile(
    r'(?:Pub\.?\s*L\.?\s*(?:No\.?)?\s*\d+[-–]\d+|'
    r'\d+\s+U\.?S\.?C\.?\s*§?\s*\d+|'
    r'\d+\s+Stat\.?\s*\d+)',
)

_LEG_REF_RE = re.compile(
    r'^(?:H\.?\s*(?:Rept|Res|Con)\.?\s*\d+|'
    r'S\.?\s*(?:Rept|Res|Con)\.?\s*\d+|'
    r'P\.?L\.?\s*\d+[-–]\d+)',
    re.IGNORECASE,
)

_DATE_ONLY_RE = re.compile(
    r'^(?:January|February|March|April|May|June|July|August|'
    r'September|October|November|December)\s+\d{1,2},?\s+\d{4}\s*[-–—]\s*$',
    re.IGNORECASE,
)

_GPO_BOILERPLATE_RE = re.compile(
    r'^\(?\s*Washington,?\s*D\.?C\.?:\s*(?:GPO|Government Publishing|'
    r'Government Printing|U\.?S\.?\s*Government)',
    re.IGNORECASE,
)


def is_aggressive_junk(text: str) -> bool:
    """Extended junk detection for legislative/government report text.

    Only called when --pre-filter-aggressive is enabled. Does NOT modify
    the base is_junk() behavior.
    """
    text_stripped = text.strip()
    n_words = len(text_stripped.split())

    # URL-dominated lines (>40% URL characters)
    if is_url_dominated(text_stripped, threshold=0.40):
        return True

    # Congressional hearing metadata with GPO citation
    if _CONGRESS_META_RE.search(text_stripped) and _GPO_RE.search(text_stripped):
        return True

    # Bibliographic entry with news outlet or URL
    if _BIBLIO_RE.match(text_stripped):
        if _NEWS_OUTLETS.search(text_stripped) or _URL_RE.search(text_stripped):
            return True

    # Short congressional metadata lines
    if _CONGRESS_META_RE.search(text_stripped) and n_words < 30:
        return True

    # Short GPO reference lines
    if _GPO_RE.search(text_stripped) and n_words < 25:
        return True

    # ── V3 additions ────────────────────────────────────────────────────

    # CRS Report cross-references
    if _CRS_REPORT_RE.search(text_stripped):
        if n_words < 50:
            return True

    # Forward references / supersedes lines
    if _FORWARD_REF_RE.match(text_stripped) and n_words < 40:
        return True
    if _SUPERSEDES_RE.search(text_stripped) and n_words < 40:
        return True

    # Pure legislative reference lines
    if _LEG_REF_RE.match(text_stripped) and n_words < 20:
        return True

    # Date-only header fragments
    if _DATE_ONLY_RE.match(text_stripped):
        return True

    # GPO boilerplate lines
    if _GPO_BOILERPLATE_RE.match(text_stripped) and n_words < 25:
        return True

    # Heavy citation chains
    citation_matches = _CITATION_CHAIN_RE.findall(text_stripped)
    if citation_matches:
        citation_chars = sum(len(m) for m in citation_matches)
        if citation_chars / max(len(text_stripped), 1) > 0.50 and n_words < 30:
            return True

    return False


def clean_wiki_markup(text: str) -> str:
    """Strip common wiki markup from text, keeping the actual content."""
    text = re.sub(r'^(?:\s*(?:thumb|right|left|center|upright(?:\s*=[^|]*)?|'
                  r'frameless|frame|border|baseline|\d+(?:x\d+)?px)\s*\|)+',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\[(?:File|Image):[^\]]*\|', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.strip().strip('|').strip()
    return text


# Fragment-shaped junk patterns that also match full prose sentences
# ("Wang (2024) proposed ...", "2.61 for SALT versus ..."); under
# lenient=True they only fire for genuinely short units.
_LENIENT_GATED = {r'^[A-Z][a-z]+\s*\(\d{4}', r'^\d+\.\d+\s+for\s+'}
_LENIENT_MIN_WORDS = 8


def is_junk(text: str, lenient: bool = False) -> bool:
    """Check if a sentence is likely junk/noise."""
    text = text.strip()

    if len(text) < MIN_CHAR_LENGTH:
        return True

    if len(text.split()) < MIN_WORD_COUNT:
        return True

    for pattern in JUNK_PATTERNS:
        if re.match(pattern, text, re.IGNORECASE):
            if (lenient and pattern in _LENIENT_GATED
                    and len(text.split()) >= _LENIENT_MIN_WORDS):
                continue
            return True

    for pattern in JUNK_CONTAINS:
        if re.search(pattern, text, re.IGNORECASE):
            cleaned = clean_wiki_markup(text)
            if len(cleaned) < MIN_CHAR_LENGTH or len(cleaned.split()) < MIN_WORD_COUNT:
                return True
            return False

    return False


def clean_text_for_embedding(text: str) -> str:
    text = re.sub(r'\[\[(?:File|Image):[^\]]*\]\]', '', text)
    text = re.sub(r'\[\[Category:[^\]]*\]\]', '', text)
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^/]*/>', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\b(?:thumb|frameless|frame)\b\s*'
                  r'(?:(?:left|right|center|upright(?:\s*=\s*[\d.]+)?|border|baseline|\d+(?:x\d+)?px)\s*)*',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'(?:^|\s)(?:right|left|center)\s+(?:thumb|upright)', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\bupright\s*=\s*[\d.]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d+x\d+px\b', '', text)
    text = re.sub(r'\b\d+px\b', '', text)
    text = re.sub(r'\|+', ' ', text)
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def filter_texts(
    texts: list[str],
    aggressive: bool = False,
    remove_urls: bool = True,
    deduplicate: bool = True,
    strip_urls: bool = False,
    lenient: bool = False,
    keep=None,
) -> tuple[list[str], int, int, int, int]:
    """Drop junk texts. All new behavior is opt-in and off by default:
    `strip_urls` replaces a URL with <url> and keeps the sentence (only
    URL-dominated lines still drop wholesale); `lenient` length-gates the
    fragment-shaped junk patterns; `keep` is a predicate exempting a unit
    from every junk test (it is still deduplicated)."""
    kept = []
    n_junk = 0
    n_url = 0
    n_aggressive = 0
    n_dedup = 0
    seen_normalized = set()

    for text in texts:
        protected = keep is not None and keep(text)

        if not protected:
            if is_junk(text, lenient=lenient):
                n_junk += 1
                continue

            if remove_urls and contains_url(text):
                if not strip_urls or is_url_dominated(text):
                    n_url += 1
                    continue
                text = _URL_PATTERN.sub(_url_placeholder, text)

            if aggressive and is_aggressive_junk(text):
                n_aggressive += 1
                continue

        if deduplicate:
            norm = _normalize_for_dedup(text)
            if norm in seen_normalized:
                n_dedup += 1
                continue
            seen_normalized.add(norm)

        kept.append(text)

    return kept, n_junk, n_url, n_aggressive, n_dedup