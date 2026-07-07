"""
Sentence splitter for SALT.

split_sentences() breaks document text into sentences with character offsets,
protecting known abbreviations ("U.S.", "Gen.", "H.R.") from false splits and
keeping over-long sentences within a token budget.

Despite the module name, nothing here embeds — the BGE encoder lives in
trie_core.py; this module only splits.
"""

import re


# ── Abbreviation protection ─────────────────────────────────────────────────
# Sorted longest-first so "H.R." is replaced before "H." would be.

ABBREVIATIONS = sorted([
    # Titles / ranks
    "Dr.", "Mr.", "Mrs.", "Ms.", "Jr.", "Sr.", "Prof.", "Rev.",
    "Gen.", "Lt.", "Col.", "Sgt.", "Cpl.", "Pvt.", "Maj.", "Capt.", "Cmdr.",
    "Adm.", "Brig.",
    # Political / government
    "Rep.", "Sen.", "Gov.", "Pres.", "Sec.", "Dept.", "Div.", "Assn.",
    "Cong.", "Sess.", "Rept.", "Res.", "Stat.", "Pub.",
    "H.R.", "S.J.", "H.J.",
    # U.S. and geographic
    "U.S.", "D.C.", "U.K.", "E.U.",
    # Corporate
    "Corp.", "Inc.", "Ltd.", "Co.", "Bros.",
    # Months
    "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.",
    "Sep.", "Sept.", "Oct.", "Nov.", "Dec.",
    # Volume / number / page
    "Vol.", "No.", "Fig.", "Ch.", "Ed.", "Eq.",
    "p.", "pp.", "vs.", "etc.", "approx.",
    # Military / DOD specific
    "P.L.", "Exec.", "Amdt.",
], key=len, reverse=True)

# Compile a single regex that matches any known abbreviation.
# We escape each abbreviation for regex safety and word-boundary anchor.
_ABBREV_RE = re.compile(
    r'(?:' + '|'.join(re.escape(a) for a in ABBREVIATIONS) + r')',
)

_PLACEHOLDER = '\x00'  # never appears in normal text


def _protect_abbreviations(text: str) -> str:
    """
    Replace periods in known abbreviations with a placeholder
    so the sentence splitter doesn't break on them.
    """
    return _ABBREV_RE.sub(lambda m: m.group(0).replace('.', _PLACEHOLDER), text)


def _restore_abbreviations(text: str) -> str:
    """Restore placeholders back to periods."""
    return text.replace(_PLACEHOLDER, '.')


# ── Sentence splitting ──────────────────────────────────────────────────────

def split_sentences(text: str, max_tokens: int = 400, tokenizer=None) -> list[tuple[str, int, int]]:
    """
    Split text into sentences with character offsets.
    """
    # Protect abbreviations before splitting
    protected = _protect_abbreviations(text.strip())

    pattern = r'(?<=[.!?;])\s+|\n+'
    raw_parts = re.split(pattern, protected)

    chunks = []
    offset = 0
    for part in raw_parts:
        part = part.strip()
        if not part:
            idx = protected.find('\n', offset)
            if idx != -1:
                offset = idx + 1
            continue
        start = protected.find(part, offset)
        if start == -1:
            start = offset
        end = start + len(part)
        # Restore abbreviations in the chunk text and find in ORIGINAL text
        restored = _restore_abbreviations(part)
        orig_start = text.find(restored, max(0, start - 5))
        if orig_start == -1:
            orig_start = start
        orig_end = orig_start + len(restored)
        chunks.append((restored, orig_start, orig_end))
        offset = end

    if tokenizer is None:
        def token_len(t): return len(t) // 4
    else:
        def token_len(t): return len(tokenizer.encode(t, add_special_tokens=False))

    max_chars = max_tokens * 5
    final = []
    for (chunk_text, char_start, char_end) in chunks:
        if len(chunk_text) <= max_chars and token_len(chunk_text) <= max_tokens:
            final.append((chunk_text, char_start, char_end))
            continue

        # Try splitting at commas first
        sub_parts = re.split(r',\s*', chunk_text)
        if len(sub_parts) > 1:
            sub_offset = char_start
            for sp in sub_parts:
                sp = sp.strip()
                if not sp:
                    continue
                s = text.find(sp, sub_offset)
                if s == -1:
                    s = sub_offset
                e = s + len(sp)
                final.append((sp, s, e))
                sub_offset = e
        else:
            # Hard split by character budget
            chunk_size = max_tokens * 4
            pos = char_start
            while pos < char_end:
                end_pos = min(pos + chunk_size, char_end)
                if end_pos < char_end:
                    space_idx = text.rfind(' ', pos, end_pos)
                    if space_idx > pos:
                        end_pos = space_idx
                segment = text[pos:end_pos].strip()
                if segment:
                    final.append((segment, pos, end_pos))
                pos = end_pos

    # Deduplicate: keep first occurrence of each sentence text
    seen_texts = set()
    deduped = []
    for (chunk_text, char_start, char_end) in final:
        normalized = chunk_text.strip().lower()
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        deduped.append((chunk_text, char_start, char_end))
    return deduped
