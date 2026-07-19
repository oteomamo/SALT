# -*- coding: utf-8 -*-
"""Regression harness for chat-side text handling (salt/engine/chat_text.py).

Pure functions only - no encoder, runs in under a second. Asserts:

  1. clean_chat_text is identity modulo whitespace on the probes the
     eval cleaner is known to mangle: generics, JSX, markdown tables,
     shell pipelines, px values, bare "frame"/"thumb", angle-bracket
     comparisons, wiki-shaped braces and brackets.
  2. clean_text_for_embedding still mangles those probes exactly as
     before, pinning the eval path against accidental drift.
  3. resolve_chat_urls substitutes <url> inside prose (keeping trailing
     punctuation) and drops only url-dominated lines.
  4. is_protected_chat_unit protects table rows, pipelines, rescued
     link sentences and code-shaped lines, and nothing else.

Usage:
    python scripts/chat_textclean_regression.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if not __debug__:
    sys.exit("this harness is assert-based - run it without python -O")

from salt.engine.chat_text import (clean_chat_text, is_protected_chat_unit,
                                   resolve_chat_urls)
from salt.engine.sentence_filter import clean_text_for_embedding


def main():
    # group 1: chat cleaning never rewrites content
    probes = [
        "std::vector<std::string> parse(const Map<Key, Value>& in)",
        "HashMap<String, Vec<u8>>",
        "| Model | Score | Latency |",
        "cat access.log | grep ERROR | sort | uniq -c",
        "<ProviderContext value={cfg}>",
        "padding to 16px",
        "The frame rate dropped",
        "the thumb drive",
        "Compare A < B and B > C",
        "{{a, b}} is a set with two members",
        "[[wiki link]] syntax survives in chat",
        "if (a && b || c) { return; }",
        "SELECT * FROM users WHERE id = 3;",
        "template <typename T> struct Node",
        "const f = (x) => x * 2",
        "impl Display for Token",
        "x -> y -> z is a chain",
        "a 300px banner next to a 16px icon",
        "run `make -j8` and then `make install`",
        "The quick brown fox jumps over the lazy dog.",
    ]
    for p in probes:
        assert clean_chat_text(p) == p, (
            f"clean_chat_text rewrote content: {p!r} -> {clean_chat_text(p)!r}")
    assert clean_chat_text("a  b\t\tc") == "a b c"
    assert clean_chat_text("l1\n\n\n\n\nl2") == "l1\n\nl2"
    assert clean_chat_text("  padded  ") == "padded"
    print("clean_chat_text: identity modulo whitespace on "
          f"{len(probes)} probes")

    # group 2: the eval cleaner still mangles them the same way
    evalpin = {
        "Compare A < B and B > C": "Compare A C",
        "std::vector<std::string> parse(const Map<Key, Value>& in)":
            "std::vector parse(const Map& in)",
        "| Model | Score | Latency |": "Model Score Latency",
        "padding to 16px": "padding to",
        "The frame rate dropped": "The rate dropped",
        "the thumb drive": "the drive",
        "{{a, b}} is a set with two members": "is a set with two members",
        "[[wiki link]] syntax survives in chat":
            "wiki link syntax survives in chat",
    }
    for src, want in evalpin.items():
        got = clean_text_for_embedding(src)
        assert got == want, (
            f"eval cleaner drifted: {src!r} -> {got!r}, pinned {want!r}")
    print(f"clean_text_for_embedding: eval mangling pinned on "
          f"{len(evalpin)} probes")

    # group 3: URL resolution before any junk gate
    out = resolve_chat_urls(
        ["Check the docs at https://x.io before the meeting tomorrow.",
         "Details for the deploy are written at https://x.io/docs.",
         "https://github.com/oteomamo/SALT/blob/main/README.md",
         "See https://github.com/oteomamo/SALT/blob/main/README.md now.",
         "No links in this sentence at all."])
    assert out == [
        "Check the docs at <url> before the meeting tomorrow.",
        "Details for the deploy are written at <url>.",
        "No links in this sentence at all."], out
    print("resolve_chat_urls: <url> substituted with punctuation kept, "
          "url-dominated lines dropped, order preserved")

    # group 4: the protection predicate stays narrow
    protected = [
        "| Model | Score | Latency |",
        "cat access.log | grep ERROR | sort | uniq -c",
        "See <url> for details.",
        "std::vector<std::string> parse(const Map<Key, Value>& in)",
        "HashMap<String, Vec<u8>>",
        "const f = (x) => x * 2",
        "fn main() { println!(); }",
        "def build(self, deep=True):",
        "<ProviderContext value={cfg}>",
    ]
    for t in protected:
        assert is_protected_chat_unit(t), f"not protected: {t!r}"
    unprotected = [
        "ok",
        "(2003)",
        "thumb 250px",
        "a | b",
        "<url>",
        "go with option B",
        "The quick brown fox jumps over the lazy dog.",
        "We agreed to ship the feature on Tuesday.",
        "Latency dropped by ten percent last week.",
        "Please summarize the meeting notes.",
        "That restaurant was far too loud.",
        "The deploy finished without errors.",
        "Rain is expected tomorrow afternoon.",
        "My favorite editor is configured already.",
        "She presented the roadmap to the team.",
        "This paragraph is ordinary prose.",
    ]
    for t in unprotected:
        assert not is_protected_chat_unit(t), f"over-protected: {t!r}"
    print(f"is_protected_chat_unit: {len(protected)} protected, "
          f"{len(unprotected)} plain units untouched")

    print("PASS")


if __name__ == "__main__":
    main()
