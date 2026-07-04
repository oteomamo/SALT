# -*- coding: utf-8 -*-
"""Document-to-text extraction for saltChat attachments.

PDFs are read whole with pypdf (text only — embedded images are ignored),
then cleaned before ingestion:

  * running headers/footers: lines from the top/bottom zone of a page whose
    digit-normalized form repeats on at least half the pages are dropped
    everywhere (catches "SALT preprint", "Page 3 of 12", dates, ...)
  * standalone page-number lines are dropped
  * words hyphenated across line breaks are re-joined
  * control characters and whitespace runs collapse to single spaces —
    SALT selects at sentence level, so layout carries no signal downstream

Plain-text files (.txt/.md and friends) pass through as-is.
"""

import re
from collections import Counter
from pathlib import Path

# lines this close to a page's top/bottom are header/footer candidates
_ZONE = 3
# a normalized zone line repeating on this fraction of pages is furniture
_REPEAT_FRAC = 0.5
_PAGE_NO_RE = re.compile(r"^[\s\-–—.]*\d{1,4}[\s\-–—.]*$")
_MIN_TEXT_CHARS = 200

PLAIN_SUFFIXES = {".txt", ".md", ".rst", ".text"}


class ExtractionError(Exception):
    """User-facing extraction failure (unreadable, encrypted, image-only)."""


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


def clean_pdf_pages(page_texts):
    """Apply the cleanup protocol to raw per-page text; returns one string."""
    pages = [t.splitlines() for t in page_texts]
    pages = _drop_repeated_furniture(pages)
    text = "\n".join("\n".join(lines) for lines in pages)
    # de-hyphenate words wrapped at line ends: letters only (never digit
    # ranges like "10-\n12") and never across blank lines (paragraph breaks)
    text = re.sub(r"([A-Za-z])-[ \t]*\n[ \t]*([a-z])", r"\1\2", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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
