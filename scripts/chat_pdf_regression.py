# -*- coding: utf-8 -*-
"""Regression harness for PDF ingestion (salt/chat/pdfio.py + the filter).

Asserts, against both synthetic snippets and the staged SALT.pdf, that the
following invariants hold:

  reflow    paragraphs held across float interruptions (captions, tables,
            footnotes) and resumed below, incl. hyphen re-joins and pypdf's
            glued interior table labels; footnotes isolated as units
  classify  lone operator glyphs and pseudocode steps are not headings;
            ALL-CAPS assignments are not headings; C.1 appendix headings
            recognized; (a)/(b) panel labels keep markers; an isolated
            numeric prose line does not sever its paragraph; Algorithm
            floats group caption-first like tables
  filter    chat-mode ingest keeps headings/equations/table units (keep
            predicate), keeps URL sentences with <url>, length-gates the
            citation-start and decimal-for junk patterns; the eval-mode
            call signature (no keep/lenient/strip_urls) keeps the default
            drop behavior
  math      Σ/∪ restored from pypdf look-alikes inside equations, spaced
            decimals don't split sentences, (N) equation tags reattach,
            dangling "=" units absorb their severed right-hand side
  refs      structural/appendix headings exit the references zone even
            when followed by citation-scented text; citation-title lines
            keep it latched; references never leak

CPU-only (no models); the trie-level path is covered by
scripts/chat_theme_regression.py. SALT.pdf checks run twice: pypdf line
extraction is nondeterministic and the assertions must hold on every
variant.
"""

import sys
from pathlib import Path

from salt.chat.pdfio import (_restore_math_glyphs, is_protected_unit,
                             read_document, split_document_sentences)
from salt.engine.sentence_filter import filter_texts

PDF = Path(__file__).resolve().parents[1] / "salt" / "files" / "SALT.pdf"

n_checks = 0


def check(name, ok):
    global n_checks
    n_checks += 1
    if not ok:
        print(f"FAIL  {name}")
        sys.exit(1)
    print(f"ok    {name}")


def synthetic_checks():
    units = split_document_sentences(
        "Serving traffic is latency-bound and batch sizes stay small, while\n"
        "offline inference for scoring or distillation emphasizes high\n"
        "1Code at https://github.com/example/repo under Apache 2.0.\n"
        "throughput and low cost per token across the fleet.\n")
    check("footnote splice: host paragraph re-joined",
          any("high throughput and low cost" in u for u in units))
    check("footnote splice: footnote isolated",
          any("Code at" in u for u in units))

    units = split_document_sentences(
        "The attention map at full context materializes as a\n"
        "4096 × 4096 × 32 tensor per layer, which\n"
        "exceeds device memory on commodity accelerators today.\n")
    check("isolated numeric line stays in its paragraph",
          any("4096" in u and "exceeds device memory" in u for u in units))

    units = split_document_sentences(
        "Results are shown below.\n"
        "Table 3: Accuracy by method and budget on the dev set today.\n"
        "SALT 40.05 41.42 26.95 62.21 53.37 37.06 43.51\n"
        "EXIT 31.50 23.77 24.94 58.89 13.50 37.45 31.68\n"
        "Prose continues after the table normally.\n")
    check("adjacent numeric rows still group under their caption",
          any("Table 3" in u and "SALT 40.05" in u and "|" in u
              for u in units))

    units = split_document_sentences(
        "5 Conclusion\nWe conclude the method works well across benchmarks.\n\n"
        "References\nAshish Vaswani and others. 2017. Attention is all you\n"
        "need. In Proceedings of NeurIPS, pages 5998-6008.\n\n"
        "C Accuracy\n"
        "All models are the arXiv-released checkpoints served with vLLM and\n"
        "we set temperature to zero for the entire evaluation run today.\n")
    joined = "\n".join(units)
    check("appendix after References survives despite citation hints",
          "arXiv-released checkpoints" in joined and "C Accuracy" in units)
    check("references never leak", "In Proceedings" not in joined)

    units = split_document_sentences(
        "References\nJacob Devlin and others. 2019. BERT pre-training. In\n"
        "Proceedings of NAACL, pages 4171-4186.\n\n"
        "A Survey Of Compression Methods For Long Context\n"
        "Alec Radford and others. 2019. Language models. In Proceedings of\n"
        "ICML, pages 1-12.\n")
    check("citation-title 'A ...' heading keeps the refs zone latched",
          not any("Radford" in u for u in units))

    check("Σ restored from lone X in equations",
          _restore_math_glyphs("Uv(R) = X w∈Γ(v) cSF(w).")
          == "Uv(R) = Σ w∈Γ(v) cSF(w).")
    check("Σ restored from glued P before a bound",
          "Σe−1 k=s" in _restore_math_glyphs("a(si) j = aj Pe−1 k=s ak"))
    check("∪ restored from lone [ in equations",
          "∪ si∈D(v)" in _restore_math_glyphs("Γ(v) = [ si∈D(v) Ti"))
    check("no glyph restoration outside equations",
          _restore_math_glyphs("X marks the spot on the map.")
          == "X marks the spot on the map.")

    units = split_document_sentences(
        "SALT reduces peak memory to 16. 70 GB at 32k tokens while EXIT\n"
        "needs 23. 70 GB.\n")
    check("spaced decimal does not split the sentence", len(units) == 1)

    units = split_document_sentences(
        "The uncovered mass is Uv(R) = X w∈Γ(v) cSF(w). (3) Thus, a branch\n"
        "can be represented compactly today.\n")
    check("(N) equation tag reattaches to its equation",
          any(u.rstrip().endswith("(3)") and "cSF" in u for u in units)
          and any(u.startswith("Thus, a branch") for u in units))

    sample = [
        "Short frag.",
        "Wang (2024) proposed a novel approach to context compression for "
        "long documents.",
        "We evaluate on LongBench (https://github.com/THUDM/LongBench) "
        "using the official splits.",
        "E = mc2", "3 Method",
        "A perfectly normal prose sentence that should survive every "
        "filter today.",
    ]
    old, *_ = filter_texts(sample, aggressive=True, remove_urls=True,
                           deduplicate=True)
    check("eval-mode filter call keeps the default drop behavior",
          old == ["A perfectly normal prose sentence that should survive "
                  "every filter today."])
    kept, *_ = filter_texts(sample, aggressive=True, remove_urls=True,
                            deduplicate=True, strip_urls=True, lenient=True,
                            keep=is_protected_unit)
    check("chat-mode filter keeps citation sentence, <url> sentence, "
          "equation, heading; junk stays junk",
          "Wang (2024) proposed a novel approach to context compression "
          "for long documents." in kept
          and "We evaluate on LongBench (<url>) using the official "
              "splits." in kept
          and {"E = mc2", "3 Method"} <= set(kept)
          and "Short frag." not in kept)
    dominated, _, n_url, _, _ = filter_texts(
        ["see https://a.io/x/y/z-long-path?token=abcdef0123456789 for the "
         "raw dump"], strip_urls=True)
    check("URL-dominated line still drops wholesale",
          not dominated and n_url == 1)


def real_pdf_checks():
    if not PDF.is_file():
        print(f"skip  real-PDF checks ({PDF} not staged)")
        return
    for attempt in (1, 2):  # pypdf extraction varies run to run
        text, _ = read_document(PDF)
        sents = split_document_sentences(text)
        joined = "\n".join(sents)
        check(f"[{attempt}] unit count sane", 300 <= len(sents) <= 400)
        check(f"[{attempt}] float-severed sentence re-joined",
              "at much higher accuracy" in joined)
        check(f"[{attempt}] hyphen re-join across table splice",
              "evaluation metrics following" in joined)
        check(f"[{attempt}] caption not fused with paragraph continuation",
              not any("Overview of the SALT pipeline" in s
                      and "where ni" in s for s in sents))
        check(f"[{attempt}] LongBench table unit intact (43.51 + caption)",
              any("43.51" in s and "Table 1" in s for s in sents))
        check(f"[{attempt}] all tables grouped",
              sum(1 for s in sents
                  if s.startswith(("Table", "[table]")) and "|" in s) >= 10)
        check(f"[{attempt}] Algorithm grouped incl. steps 3, 9 and 12",
              any("Algorithm 1" in s and "ActiveBranches" in s
                  for s in sents)
              and any("GlobalFill" in s and "|" in s for s in sents)
              and any("QueryIndex" in s and "|" in s for s in sents))
        check(f"[{attempt}] title intact, not shredded by the label split",
              any("Salience-Aware Lexical Trie for Long-Context "
                  "Compression" in s for s in sents)
              and "Longsubmission" not in joined)
        check(f"[{attempt}] acronym heading survives (4.5 NIAH)",
              any("Needle-in-a-Haystack (NIAH)" in s for s in sents)
              and "Needle-in-aevaluate" not in joined)
        check(f"[{attempt}] hyphenated wraps never split as labels",
              "Fewing" not in joined)
        check(f"[{attempt}] Eq. (3) whole with restored Σ",
              any("Uv(R) = Σ" in s for s in sents))
        check(f"[{attempt}] panel labels keep their markers",
              any(s.startswith("(a) ") for s in sents))
        check(f"[{attempt}] references dropped",
              "In Proceedings" not in joined)
        kept, *_ = filter_texts(sents, aggressive=True, remove_urls=True,
                                deduplicate=True, strip_urls=True,
                                lenient=True, keep=is_protected_unit)
        heads = [u for u in kept if u in ("Abstract", "1 Introduction",
                                          "3 Method", "5 Conclusion",
                                          "Limitations")]
        check(f"[{attempt}] section headings survive the chat filter",
              len(heads) >= 4)


def main():
    synthetic_checks()
    real_pdf_checks()
    print(f"PASS ({n_checks} checks)")


if __name__ == "__main__":
    main()
