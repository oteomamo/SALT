# -*- coding: utf-8 -*-
"""Document-to-text extraction and sentence segmentation for saltChat
attachments.

PDFs are read whole with pypdf (text only — embedded images are ignored),
then cleaned line-by-line before reflow:

  * extraction artifacts are normalized per line: ligatures (ﬁ → fi),
    (cid:NN) glyph failures, soft hyphens, exotic Unicode spaces, and
    spaces pypdf inserts before punctuation ("FastKV ," → "FastKV,")
  * running headers/footers: lines from the top/bottom zone of a page whose
    digit-normalized form repeats on at least half the pages are dropped
    everywhere (catches "SALT preprint", "Page 3 of 12", dates, ...)
  * standalone page-number lines are dropped
  * margin line-number rails (ACL/NeurIPS-style draft numbering, which
    pypdf glues onto line ends: "process in-002") are stripped when a
    monotonic 1,2,3,... sequence is detected — real trailing numbers
    survive because they don't continue the sequence
  * words hyphenated across line breaks are re-joined
  * wrapped lines are reflowed into paragraph blocks; blank lines,
    headings, captions, bullets and table rows keep real newlines, so the
    extracted text preserves document structure instead of one flat line

`split_document_sentences` turns that structured text into clean sentence
units for trie ingestion (the salt@ / /doc path): citation-safe boundaries
(no splits at semicolons or inside balanced parentheses, extended
abbreviation protection, initials), reference-list entries and
number-dominated table rows dropped, headings kept as their own units,
bullet markers stripped and the items sentence-split.

Plain-text files (.txt/.md and friends) skip extraction but share the
reflow + sentence stage.
"""

import re
from collections import Counter
from pathlib import Path

from salt.engine.embedder import ABBREVIATIONS as _BASE_ABBREVIATIONS

# lines this close to a page's top/bottom are header/footer candidates
_ZONE = 3
# a normalized zone line repeating on this fraction of pages is furniture
_REPEAT_FRAC = 0.5
_PAGE_NO_RE = re.compile(r"^[\s\-–—.]*\d{1,4}[\s\-–—.]*$")
_MIN_TEXT_CHARS = 200

PLAIN_SUFFIXES = {".txt", ".md", ".rst", ".text"}


class ExtractionError(Exception):
    """User-facing extraction failure (unreadable, encrypted, image-only)."""


# ── per-line artifact scrub ─────────────────────────────────────────────────

_LIG_TABLE = str.maketrans({
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl",
    "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "ft", "ﬆ": "st",
})
_CID_RE = re.compile(r"\(cid:\d+\)")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\u200b\u200c\u200d\ufeff]")
_ODD_SPACE_RE = re.compile(r"[\u00a0\u2000-\u200a\u202f\u205f\u3000]")
_PUNCT_GAP_RE = re.compile(r"\s+([,;:.!?)\]])")


def _scrub_line(line):
    line = line.translate(_LIG_TABLE)
    line = _CID_RE.sub(" ", line)
    line = line.replace("\u00ad", "")  # soft hyphen
    line = _ODD_SPACE_RE.sub(" ", line)
    line = _CTRL_RE.sub(" ", line)
    line = _PUNCT_GAP_RE.sub(r"\1", line)
    return re.sub(r"\s+", " ", line).strip()


# ── repeated header/footer furniture ────────────────────────────────────────

def _normalize_furniture(line):
    """Digit-insensitive fingerprint so 'Page 3' and 'Page 11' match."""
    return re.sub(r"\d+", "#", line.strip().lower())


def _zone_size(n_lines):
    """Header/footer zone shrinks on short pages (slide decks often extract
    3-5 lines/page) so real content keeps a protected middle."""
    return min(_ZONE, max(1, n_lines // 3))


def _drop_repeated_furniture(pages):
    counts = Counter()
    for lines in pages:
        z = _zone_size(len(lines))
        zone = lines[:z] + lines[-z:]
        for fp in {_normalize_furniture(l) for l in zone if l.strip()}:
            counts[fp] += 1
    threshold = max(3, int(len(pages) * _REPEAT_FRAC))
    furniture = {fp for fp, c in counts.items() if c >= threshold}

    cleaned = []
    for lines in pages:
        z = _zone_size(len(lines))
        kept = []
        for j, line in enumerate(lines):
            in_zone = j < z or j >= len(lines) - z
            if in_zone and (_normalize_furniture(line) in furniture
                            or _PAGE_NO_RE.match(line)):
                continue
            kept.append(line)
        cleaned.append(kept)
    return cleaned


# ── margin line-number rail ─────────────────────────────────────────────────
# ACL/NeurIPS draft templates number every typeset line in the margin; pypdf
# glues the number onto the end (sometimes the start) of each text line.
# Detection requires a long 1,2,3,... sequence, and stripping only removes a
# number that continues the sequence, so years, table values and counts in
# prose ("972 turns") are left alone.

_RAIL_MIN_RUN_PAIRS = 20
_RAIL_GAP_TOLERANCE = 3   # small gaps stripped unconditionally
_RAIL_MAX_JUMP = 300      # larger jumps need the next candidate to confirm
_RAIL_MAX_MISSES = 8      # consecutive non-continuing candidates ending a chain


def _trailing_rail(line):
    m = re.search(r"(\d{1,4})\s*$", line)
    if not m:
        return None
    before = line[:m.start(1)]
    if before[-1:].isdigit() or re.search(r"\d[.,]$", before):
        return None  # part of a larger or decimal number ("16.70")
    return int(m.group(1)), m.start(1), len(line)


def _leading_rail(line):
    m = re.match(r"(\d{1,4})\s+", line)
    if not m:
        return None
    return int(m.group(1)), 0, m.end()


def _strip_number_rail(pages):
    """Chains are anchored only at a candidate that itself starts a +1,+2 run
    (a prose number ahead of the real rail cannot steal the lock), tolerate
    small gaps, accept long forward jumps only when the next candidate
    confirms the resync, and re-anchor after a dead chain so per-page
    restarting rails and post-gap remainders are still stripped. Trailing and
    leading rails are handled independently (some layouts mix both)."""
    for extract in (_trailing_rail, _leading_rail):
        cands = []  # (page_i, line_i, value)
        for pi, page in enumerate(pages):
            for li, line in enumerate(page):
                r = extract(line)
                if r:
                    cands.append((pi, li, r[0]))
        vals = [v for _, _, v in cands]
        if sum(1 for a, b in zip(vals, vals[1:]) if b == a + 1) < _RAIL_MIN_RUN_PAIRS:
            continue

        strip, n, i = set(), len(cands), 0
        while i < n - 2:
            if not (vals[i + 1] == vals[i] + 1 and vals[i + 2] == vals[i] + 2):
                i += 1
                continue
            expected, j, misses, last_hit = vals[i], i, 0, i
            while j < n and misses <= _RAIL_MAX_MISSES:
                v = vals[j]
                confirmed_jump = (
                    expected + _RAIL_GAP_TOLERANCE < v <= expected + _RAIL_MAX_JUMP
                    and j + 1 < n and vals[j + 1] == v + 1)
                if expected <= v <= expected + _RAIL_GAP_TOLERANCE or confirmed_jump:
                    strip.add(j)
                    expected, misses, last_hit = v + 1, 0, j
                else:
                    misses += 1
                j += 1
            i = max(last_hit + 1, i + 1)

        for k in strip:
            pi, li, _ = cands[k]
            line = pages[pi][li]
            _, start, end = extract(line)
            pages[pi][li] = (line[:start].rstrip() if extract is _trailing_rail
                             else line[end:].lstrip())
    return pages


# ── reflow: physical lines -> logical blocks ────────────────────────────────

_MD_HEAD_RE = re.compile(r"^#{1,6}\s")
_NUM_HEAD_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?|[A-Z]\.?|[IVXLC]+\.?|Appendix\s+[A-Z]\d*\.?)\s+\S")
_CAPTION_RE = re.compile(r"^(?:Figure|Table|Algorithm|Listing|Chart)\s+\d+\s*[:.]")
_SUBCAP_RE = re.compile(r"^\([a-z]\)\s")
_BULLET_RE = re.compile(
    r"^(?:[-–—•*·▪◦‣]|\(?\d{1,2}[.)]|\(?[ivx]{1,4}\)|\(?[a-hj-z]\))\s+")
_KNOWN_HEADS = {
    "abstract", "references", "bibliography", "acknowledgments",
    "acknowledgements", "appendix", "introduction", "conclusion",
    "conclusions", "related work", "limitations", "ethics statement",
    "contents", "index", "glossary",
}
_MINOR_WORDS = {"a", "an", "the", "of", "in", "on", "for", "and", "or", "to",
                "with", "at", "by", "from", "via", "as", "is", "are"}
_NUMERIC_TOKEN_RE = re.compile(
    r"^[±+-]?\d+(?:[.,]\d+)*[%kKMBx×]?$|^(?:OOM|N/A|NaN|[-–—✓✗×])$")


def _title_ratio(words):
    major = [w for w in words if w.lower() not in _MINOR_WORDS]
    if not major:
        return 0.0
    return sum(1 for w in major if w[:1].isupper() or w[:1].isdigit()) / len(major)


def _is_heading(line):
    words = line.split()
    if not words or len(words) > 10:
        return False
    if _MD_HEAD_RE.match(line):
        return True
    if line.rstrip(":").lower() in _KNOWN_HEADS:
        return True
    if line[-1] in ".!?,;:":
        return False
    if len(words) <= 8 and line.isupper() and any(c.isalpha() for c in line):
        return True
    if _NUM_HEAD_RE.match(line):
        # section numbers are small; a leading year ("2022 Conference of...",
        # a wrapped citation line) must not read as a numbered heading
        first = words[0].rstrip(".")
        if first[:1].isdigit() and any(
                not p.isdigit() or int(p) >= 100 for p in first.split(".")):
            return False
        rest = words[2:] if line.startswith("Appendix") else words[1:]
        return _title_ratio(rest) >= 0.6
    return False


def _is_table_line(line):
    toks = line.split()
    if len(toks) < 6:
        return False
    numish = sum(1 for t in toks if _NUMERIC_TOKEN_RE.match(t))
    return numish / len(toks) >= 0.5


def _reflow_blocks(lines):
    """Group physical lines into logical (kind, text) blocks. Wrapped lines
    join (re-joining end-of-line hyphenation); blank lines, headings,
    captions, bullets and table rows start new blocks."""
    blocks = []
    kind, buf = None, ""

    def flush():
        nonlocal kind, buf
        if buf:
            blocks.append((kind or "body", buf))
        kind, buf = None, ""

    for raw in lines:
        line = raw.strip()
        if not line:
            flush()
            continue
        # table check first: an all-caps row like "EXIT 31.50 23.77 ..."
        # would otherwise pass the ALL-CAPS heading test
        if _is_table_line(line):
            flush()
            blocks.append(("table", line))
            continue
        if _is_heading(line):
            flush()
            blocks.append(("heading", line))
            continue
        if _CAPTION_RE.match(line) or _SUBCAP_RE.match(line) or _BULLET_RE.match(line):
            flush()
            kind = "bullet" if _BULLET_RE.match(line) else "caption"
            buf = line
            continue
        if not buf:
            buf = line
        elif re.search(r"[A-Za-z]-$", buf) and line[:1].isalpha():
            # wrapped at a hyphen: lowercase continuation is a broken word
            # (re-join), capitalized is a hyphenated compound (keep hyphen)
            buf = (buf[:-1] + line) if line[:1].islower() else (buf + line)
        else:
            buf += " " + line
    flush()
    return blocks


# ── sentence segmentation (trie-ingestion path) ─────────────────────────────

_EXTRA_ABBREVIATIONS = [
    "e.g.", "i.e.", "et al.", "cf.", "ca.", "resp.", "viz.",
    "Sec.", "Secs.", "Eq.", "Eqs.", "Fig.", "Figs.", "Tab.", "Alg.", "App.",
    "Thm.", "Prop.", "Lem.", "Def.", "Cor.", "Rem.",
    "sec.", "eq.", "fig.", "tab.", "no.", "nos.", "vol.", "pt.",
    "Ph.D.", "M.Sc.", "B.Sc.", "a.m.", "p.m.", "St.", "Mt.", "Univ.",
    "ed.", "eds.", "trans.", "n.d.", "w.r.t.", "i.i.d.", "s.t.",
    "acc.", "avg.", "std.", "max.", "min.",
]
_ALL_ABBREVIATIONS = sorted(set(_BASE_ABBREVIATIONS) | set(_EXTRA_ABBREVIATIONS),
                            key=len, reverse=True)
# unlike the engine splitter, anchor on a non-letter so "trip." is never
# mistaken for the abbreviation "p."
_ABBR_RE = re.compile(
    r"(?<![A-Za-z])(?:" + "|".join(re.escape(a) for a in _ALL_ABBREVIATIONS) + r")")
_INITIAL_RE = re.compile(r"\b([A-Z])\.(?=\s)")
_PH = "\x00"  # placeholder for protected periods; never appears in text
_BOUND_RE = re.compile(r"([.!?]+[\"'”’)\]]*)\s+(?=[\"'“‘(\[]*[A-Z0-9])")


def _split_block_sentences(text):
    """Sentence-split one reflowed block. Boundaries need terminal .!? plus
    a following capital/digit; abbreviations and single-letter initials are
    protected; when the block's brackets are balanced, boundaries inside
    parentheses/brackets (citation lists, asides) are skipped."""
    prot = _ABBR_RE.sub(lambda m: m.group(0).replace(".", _PH), text)
    prot = _INITIAL_RE.sub(lambda m: m.group(1) + _PH, prot)

    # gate only on properly nested brackets: equal counts with a stray ")"
    # before "(" would otherwise leave depth stuck > 0 for the block's tail
    gated, depth, d = True, [], 0
    for ch in prot:
        depth.append(d)
        if ch in "([":
            d += 1
        elif ch in ")]":
            d -= 1
            if d < 0:
                gated = False
                break
    if d != 0:
        gated = False

    sents, prev = [], 0
    for m in _BOUND_RE.finditer(prot):
        if gated and depth[m.start()] > 0:
            continue
        sents.append(prot[prev:m.end(1)])
        prev = m.end()
    sents.append(prot[prev:])
    return [s.replace(_PH, ".").strip() for s in sents if s.strip()]


_REFS_HEAD_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*|[IVXLC]+|[A-Z])[.)]?\s+)?"
    r"(?:references|bibliography|works cited)\s*:?\s*$", re.IGNORECASE)
_CITATION_HINT_RE = re.compile(
    r"arXiv|In Proceedings|pages? \d+[-–]\d+|doi:|vol\.\s*\d+|[A-Z][a-z]+, [A-Z]\.")


def split_document_sentences(text):
    """Structured document text -> clean sentence units for trie ingestion.

    Reference-list entries (between a References/Bibliography heading —
    numbered or not — and the next real heading) and table-like number rows
    are dropped; headings stay whole units, bullet markers are stripped and
    the items sentence-split (short units fall to the downstream filter)."""
    blocks = _reflow_blocks(_CTRL_RE.sub(" ", text).splitlines())
    out = []
    in_refs = False
    for bi, (kind, btext) in enumerate(blocks):
        if kind == "heading":
            if _REFS_HEAD_RE.match(btext):
                in_refs = True
                continue
            if in_refs:
                # a wrapped citation-title line can look like a heading; only
                # leave the zone if what follows doesn't read as a citation
                nxt = next((b for k2, b in blocks[bi + 1:bi + 3]
                            if k2 != "table"), "")
                if (_CITATION_HINT_RE.search(btext)
                        or _CITATION_HINT_RE.search(nxt[:400])):
                    continue
                in_refs = False
            out.append(btext)
            continue
        if in_refs or kind == "table":
            continue
        if kind == "bullet":
            btext = _BULLET_RE.sub("", btext, count=1)
        out.extend(_split_block_sentences(btext))
    return out


# ── PDF extraction ──────────────────────────────────────────────────────────

def clean_pdf_pages(page_texts):
    """Apply the cleanup protocol to raw per-page text; returns structured
    text with real newlines at paragraph/heading/bullet/table boundaries."""
    pages = [[_scrub_line(l) for l in t.splitlines()] for t in page_texts]
    pages = _drop_repeated_furniture(pages)
    pages = _strip_number_rail(pages)
    # no separator between pages: paragraphs (and hyphenated words) continue
    # across page breaks and are re-joined by the reflow
    lines = [l for page in pages for l in page]
    blocks = _reflow_blocks(lines)
    return "\n".join(btext for _, btext in blocks)


def extract_pdf_text(path):
    """Whole-document text of a PDF, cleaned. Raises ExtractionError."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ExtractionError(
            "PDF support needs pypdf - install it with: pip install pypdf") from exc
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            # pypdf returns PasswordType.NOT_DECRYPTED (falsy) rather than
            # raising when the empty password fails
            try:
                unlocked = reader.decrypt("")
            except Exception as exc:
                raise ExtractionError(f"{path}: PDF is encrypted ({exc})") from exc
            if not unlocked:
                raise ExtractionError(f"{path}: PDF is password-protected")
        page_texts = [(page.extract_text() or "") for page in reader.pages]
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(f"Could not read {path}: {exc}") from exc

    text = clean_pdf_pages(page_texts)
    if len(text) < _MIN_TEXT_CHARS:
        raise ExtractionError(
            f"{path}: no extractable text ({len(text)} chars) - "
            f"likely a scanned/image-only PDF, which is not supported yet.")
    return text, len(page_texts)


def read_document(path):
    """Dispatch by suffix: (text, n_pages|None). Raises ExtractionError."""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(p)
    if suffix not in PLAIN_SUFFIXES:
        raise ExtractionError(
            f"{p}: unsupported file type {suffix or '(no suffix)'!r} - attach "
            f".pdf or plain text ({', '.join(sorted(PLAIN_SUFFIXES))}).")
    try:
        return p.read_text(encoding="utf-8", errors="replace"), None
    except OSError as exc:
        raise ExtractionError(f"Could not read {p}: {exc}") from exc
