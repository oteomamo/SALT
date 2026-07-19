"""
Chat-side text handling: conversation ingest stores what was typed.
Whitespace is normalized and URLs become <url>; markup, code, tables and
px values pass through untouched. Deliberate loss versus the eval
cleaner: pasted MediaWiki source keeps its templates and links in chat.
"""

import re

from salt.engine.sentence_filter import (_URL_PATTERN, _url_placeholder,
                                         is_url_dominated)

_MULTISPACE_RE = re.compile(r'[ \t]{2,}')
_MULTINEWLINE_RE = re.compile(r'\n{3,}')

_CODEISH_RE = re.compile(
    r'</?[A-Za-z][\w:.-]*[^>]*/?>|::|=>|->|\)\s*\{'
    r'|\b(?:def|function|class|fn|impl)\s+\w+\s*[(<]|^\s{4,}\S')


def clean_chat_text(text):
    return _MULTINEWLINE_RE.sub('\n\n', _MULTISPACE_RE.sub(' ', text)).strip()


def resolve_chat_urls(units):
    out = []
    for u in units:
        if is_url_dominated(u):
            continue
        out.append(_URL_PATTERN.sub(_url_placeholder, u))
    return out


def is_protected_chat_unit(text):
    t = text.strip()
    if not t:
        return False
    if t.count('|') >= 2:
        return True
    if '<url>' in t and len(t.split()) >= 3:
        return True
    return len(t) >= 10 and bool(_CODEISH_RE.search(t))
