# -*- coding: utf-8 -*-
"""When a turn searches: the rows of the conversation filed inside the
time the question names.

A question about "what we talked about on 8 May" is answered from the
whole history unless something holds the other days out. The rule here
reads a time off the user's line, a day, a month, a year or a span
relative to the moment of asking, and hands every conversation row
filed outside it to the compressor as out of candidacy for that turn.
Attached files are never held out: their rows are filed when they are
attached, and a date on the line says nothing about them. A window
that holds no conversation rows is not applied.
"""

import calendar
import re
import time
from datetime import datetime, timedelta

MODES = ("off", "auto")
# the third mode, reachable only by /when: a window pinned by hand
PINNED = "pinned"
# what a turn records about its window, in this order
RECORD_KEYS = ("mode", "label", "start", "end", "rows_in", "rows_out",
               "note")
# the census of a session's time windows, one counter per outcome
CENSUS_KEYS = ("asked", "windowed", "pinned", "plain", "empty")
EMPTY = "no conversation rows in that window"

_MONTHS = {n.lower(): i for i, n in enumerate(calendar.month_name) if n}
_MONTHS.update({n.lower(): i for i, n in enumerate(calendar.month_abbr) if n})
_MONTHS["sept"] = 9
_WEEKDAYS = {n.lower(): i for i, n in enumerate(calendar.day_name)}
_WEEKDAYS.update({n.lower(): i for i, n in enumerate(calendar.day_abbr)})
_NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4,
            "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
            "ten": 10, "couple": 2, "couple of": 2, "few": 3}
_M = "|".join(sorted(_MONTHS, key=len, reverse=True))
_W = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))
_N = r"\d+|" + "|".join(sorted(_NUMBERS, key=len, reverse=True))
_BEFORE = r"(?:in|on|during|of|since|from|until|till|through|early|late|mid|last|this|for|around|about)"
# "the 8 may be too many": a day before the month "may" with no year is
# read as a date only when the next word does not carry the modal on
_MODAL_NEXT = {"be", "have", "not", "also", "still", "never", "well", "need",
               "want", "get", "come", "go", "take", "make", "seem", "only",
               "just", "even", "already", "as", "or", "and"}


def _num(text):
    return int(text) if text.isdigit() else _NUMBERS[text]


def _modal(text, m):
    """Whether a day-month match reads as the modal "may" instead."""
    if _MONTHS[m.group(2)] != 5 or m.group(3):
        return False
    after = text[m.end():].split()
    return bool(after) and after[0] in _MODAL_NEXT


def _day(dt):
    return datetime(dt.year, dt.month, dt.day)


def _week(dt):
    start = _day(dt) - timedelta(days=dt.weekday())
    return start, start + timedelta(days=7)


def _month(year, month):
    start = datetime(year, month, 1)
    end = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    return start, end


def _label(start, end):
    last = end - timedelta(days=1)
    if last.date() == start.date():
        return start.strftime("%-d %B %Y")
    if (start.day, last.day) == (1, calendar.monthrange(last.year, last.month)[1]):
        if start.month == last.month and start.year == last.year:
            return start.strftime("%B %Y")
        if (start.month, start.day, last.month) == (1, 1, 12) and start.year == last.year:
            return str(start.year)
    if start.year == last.year and start.month == last.month:
        return f"{start.day} to {last.strftime('%-d %B %Y')}"
    if start.year == last.year:
        return f"{start.strftime('%-d %B')} to {last.strftime('%-d %B %Y')}"
    return f"{start.strftime('%-d %B %Y')} to {last.strftime('%-d %B %Y')}"


def parse(line, anchor=None):
    """The time a line names, as a spec, or None when it names none.
    A spec is {"kind": day|month|year|span, ...}: a day or a month
    without a year carries year None and takes one when resolved.
    Relative terms are read against `anchor` (epoch seconds, the
    moment of asking; None means now)."""
    text = " ".join((line or "").lower().replace(",", " ").split())
    if not text:
        return None
    now = datetime.fromtimestamp(time.time() if anchor is None else anchor)
    today = _day(now)

    def span(start, end):
        return {"kind": "span", "start": start.timestamp(),
                "end": end.timestamp()}

    if re.search(r"\bday before yesterday\b", text):
        return span(today - timedelta(days=2), today - timedelta(days=1))
    if re.search(r"\byesterday\b", text):
        return span(today - timedelta(days=1), today)
    if re.search(r"\btoday\b", text):
        return span(today, today + timedelta(days=1))
    m = re.search(rf"\b({_N})\s+(day|week|month)s?\s+ago\b", text)
    if m:
        n, unit = _num(m.group(1)), m.group(2)
        if unit == "day":
            d = today - timedelta(days=n)
            return span(d, d + timedelta(days=1))
        if unit == "week":
            return span(*_week(today - timedelta(days=7 * n)))
        y, mo = now.year, now.month - n
        while mo < 1:
            mo += 12
            y -= 1
        return span(*_month(y, mo))
    m = re.search(rf"\b(?:the\s+)?(?:last|past)\s+({_N})\s+(day|week|month)s?\b", text)
    if m:
        n, unit = _num(m.group(1)), m.group(2)
        end = today + timedelta(days=1)
        if unit == "day":
            return span(today - timedelta(days=n - 1), end)
        if unit == "week":
            return span(today - timedelta(days=7 * n), end)
        y, mo = now.year, now.month - n
        while mo < 1:
            mo += 12
            y -= 1
        return span(datetime(y, mo, 1), end)
    if re.search(r"\blast week\b", text):
        return span(*_week(today - timedelta(days=7)))
    if re.search(r"\bthis week\b", text):
        return span(*_week(today))
    if re.search(r"\blast month\b", text):
        y, mo = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
        return span(*_month(y, mo))
    if re.search(r"\bthis month\b", text):
        return span(*_month(now.year, now.month))
    if re.search(r"\blast year\b", text):
        return span(datetime(now.year - 1, 1, 1), datetime(now.year, 1, 1))
    if re.search(r"\bthis year\b", text):
        return span(datetime(now.year, 1, 1), datetime(now.year + 1, 1, 1))
    m = re.search(rf"\b(last|on|this)\s+({_W})\b", text)
    if m:
        wd = _WEEKDAYS[m.group(2)]
        back = (today.weekday() - wd) % 7
        if m.group(1) == "last" and back == 0:
            back = 7
        d = today - timedelta(days=back)
        return span(d, d + timedelta(days=1))
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", text)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        if 1 <= mo <= 12 and 1 <= d <= calendar.monthrange(y, mo)[1]:
            return {"kind": "day", "day": d, "month": mo, "year": y}
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_M})(?:\s+(\d{{4}}))?\b", text)
    if m and 1 <= int(m.group(1)) <= 31 and not _modal(text, m):
        return {"kind": "day", "day": int(m.group(1)),
                "month": _MONTHS[m.group(2)],
                "year": int(m.group(3)) if m.group(3) else None}
    m = re.search(rf"\b({_M})\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:\s+(\d{{4}})\b)?", text)
    if m and 1 <= int(m.group(2)) <= 31:
        return {"kind": "day", "day": int(m.group(2)),
                "month": _MONTHS[m.group(1)],
                "year": int(m.group(3)) if m.group(3) else None}
    m = (re.search(rf"\b({_M})\s+(\d{{4}})\b", text)
         or re.search(rf"\b{_BEFORE}\s+({_M})\b()", text))
    if m:
        return {"kind": "month", "month": _MONTHS[m.group(1)],
                "year": int(m.group(2)) if m.group(2) else None}
    m = re.search(r"\b(?:in|during|of|since|from|through)\s+(\d{4})\b", text)
    if m and 1900 <= int(m.group(1)) <= 2100:
        return {"kind": "year", "year": int(m.group(1))}
    return None


def window(spec, year=None):
    """(start, end, label) for a spec, epoch seconds, end exclusive.
    `year` fills a day or a month that named none. None when the day
    does not exist in that year."""
    if spec["kind"] == "span":
        start, end = spec["start"], spec["end"]
        return start, end, _label(datetime.fromtimestamp(start),
                                  datetime.fromtimestamp(end))
    y = spec.get("year") or year
    if y is None:
        return None
    if spec["kind"] == "year":
        s, e = datetime(y, 1, 1), datetime(y + 1, 1, 1)
    elif spec["kind"] == "month":
        s, e = _month(y, spec["month"])
    else:
        if spec["day"] > calendar.monthrange(y, spec["month"])[1]:
            return None
        s = datetime(y, spec["month"], spec["day"])
        e = s + timedelta(days=1)
    return s.timestamp(), e.timestamp(), _label(s, e)


def rows_outside(trie, start, end):
    """(held out, rows in): the living conversation rows filed outside
    [start, end), and how many are inside. A row without a time is
    never held out and never counted in; a file row is never touched."""
    out, n_in = set(), 0
    for i in range(len(trie.texts)):
        if not trie.alive[i] or trie.sources[i] is not None:
            continue
        ts = trie.timestamps[i]
        if ts is None:
            continue
        if start <= ts < end:
            n_in += 1
        else:
            out.add(i)
    return out, n_in


def years_of(trie):
    """The years the conversation has rows in, latest first."""
    years = set()
    for i in range(len(trie.texts)):
        if trie.alive[i] and trie.sources[i] is None and trie.timestamps[i] is not None:
            years.add(datetime.fromtimestamp(trie.timestamps[i]).year)
    return sorted(years, reverse=True)


def decide(mode, line, trie, anchor=None, pinned=None):
    """The window for one turn as (held out, record): held out is None
    when the turn is not windowed; the record says what was decided, or
    None when there was nothing to decide. A pinned spec wins over the
    mode; under `auto` the line decides; under `off` nothing else does.
    A day or a month without a year tries the conversation's own years,
    latest first, and the anchor's year last. Never raises into a turn."""
    try:
        spec = pinned if pinned is not None else (
            parse(line, anchor) if mode == "auto" else None)
        if spec is None:
            return None, None
        how = PINNED if pinned is not None else mode
        years = [spec["year"]] if spec.get("year") else years_of(trie) + [
            datetime.fromtimestamp(time.time() if anchor is None else anchor).year]
        seen, first = set(), None
        for y in years:
            if y in seen:
                continue
            seen.add(y)
            w = window(spec, y)
            if w is None:
                continue
            start, end, label = w
            out, n_in = rows_outside(trie, start, end)
            if first is None:
                first = (start, end, label)
            if n_in:
                return out, {"mode": how, "label": label, "start": start,
                             "end": end, "rows_in": n_in, "rows_out": len(out),
                             "note": None}
        start, end, label = first if first else (None, None, "")
        return None, {"mode": how, "label": label, "start": start, "end": end,
                      "rows_in": 0, "rows_out": 0, "note": EMPTY}
    except Exception as exc:
        return None, {"mode": mode, "label": "", "start": None, "end": None,
                      "rows_in": 0, "rows_out": 0,
                      "note": f"{type(exc).__name__}: {exc}"}


def census():
    """A fresh census: turns looked at, windows applied by the rule and
    by hand, turns naming no time, and times that held no rows.
    Session-lifetime, never persisted."""
    return {"asked": 0, "windowed": 0, "pinned": 0, "plain": 0, "empty": 0}


def count(census, rec):
    """One more turn looked at: `rec` is that turn's record, None when
    the line named no time."""
    census["asked"] += 1
    if rec is None:
        census["plain"] += 1
    elif rec["note"]:
        census["empty"] += 1
    else:
        census["pinned" if rec["mode"] == PINNED else "windowed"] += 1


def census_lines(census):
    """The /stats lines for a session's census; nothing when the session
    does not look."""
    if census is None:
        return []
    assert tuple(census) == CENSUS_KEYS
    return [f"time census: {census['asked']} turns looked at, "
            f"{census['windowed']} windowed by the rule, {census['pinned']} "
            f"pinned by hand, {census['plain']} naming no time, "
            f"{census['empty']} with no rows in the time named"]


def lines(rec):
    """The /stats lines for one turn's window; nothing on a turn without
    one."""
    if rec is None:
        return []
    assert tuple(rec) == RECORD_KEYS
    if rec["note"]:
        named = f" {rec['label']!r}" if rec["label"] else ""
        return [f"time window ({rec['mode']}): not applied{named}, {rec['note']}"]
    return [f"time window ({rec['mode']}): {rec['label']}, {rec['rows_in']} "
            f"conversation rows in, {rec['rows_out']} held out"]
