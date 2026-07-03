# -*- coding: utf-8 -*-
"""
Few-shot context detection and block-level preservation.

Benchmark datasets like TREC, SAMSum, and TriviaQA encode their context
as few-shot exemplars rather than documents to be summarized. Compressing
these with content-relevance methods (keyword trie, sentence scoring, etc.)
destroys the label structure the model needs to infer the output format.

This module detects structured few-shot patterns in raw context strings
and provides block-level selection that preserves exemplar integrity.
It is compression-method-agnostic: any compressor can import ``detect()``
and ``select_blocks()`` without circular dependencies.

Supported patterns (extensible via EXEMPLAR_PATTERNS):
  - question_type:     Question + Type pairs       (TREC)
  - dialogue_summary:  Dialogue + Summary pairs     (SAMSum)
  - passage_qa:        Passage + Question + Answer   (TriviaQA)

Architecture:
  detect(context) → FewshotResult   (pure classification, no compression)
  select_blocks(blocks, budget) → (text, n_words, n_kept, n_total)

"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# =============================================================================
# Pattern registry
# =============================================================================

@dataclass(frozen=True)
class ExemplarPattern:
    """
    Defines a few-shot exemplar structure to detect.

    Attributes:
        name:       Human-readable identifier (for logging/stats).
        delimiter:  String that starts each exemplar block.
        label:      String that marks the supervised label within a block.
        min_pairs:  Minimum delimiter+label co-occurrences to trigger.
        strategy:   'bypass' = select complete blocks within budget.
    """
    name: str
    delimiter: str
    label: str
    min_pairs: int = 5
    strategy: str = "bypass"


# Order matters: more specific patterns first to avoid false positives.
# Each pattern is tried in sequence; first match wins.
EXEMPLAR_PATTERNS: list[ExemplarPattern] = [
    ExemplarPattern(
        name="question_type",
        delimiter="Question:",
        label="Type:",
        min_pairs=5,
    ),
    ExemplarPattern(
        name="dialogue_summary",
        delimiter="Dialogue:",
        label="Summary:",
        min_pairs=3,
    ),
    ExemplarPattern(
        name="numbered_paragraphs",
        delimiter="Paragraph ",
        label="Paragraph ",
        min_pairs=10,
    ),
    ExemplarPattern(
        name="passage_qa",
        delimiter="Passage:",
        label="Answer:",
        min_pairs=3,
    ),
]


# =============================================================================
# Detection result
# =============================================================================

@dataclass
class FewshotResult:
    """
    Result of few-shot pattern detection.

    Attributes:
        detected:     True if a known few-shot pattern was found.
        pattern_name: Name of the matched pattern (or 'none').
        strategy:     How to handle: 'bypass' or 'none'.
        blocks:       List of atomic exemplar blocks (strings).
                      Empty if not detected.
    """
    detected: bool = False
    pattern_name: str = "none"
    strategy: str = "none"
    blocks: list[str] = field(default_factory=list)


# =============================================================================
# Detection
# =============================================================================

def detect(context: str) -> FewshotResult:
    """
    Classify whether a context string contains few-shot exemplars.
    """
    if not context or not context.strip():
        return FewshotResult()

    for pattern in EXEMPLAR_PATTERNS:
        result = _try_pattern(context, pattern)
        if result.detected:
            return result

    return FewshotResult()


def _try_pattern(context: str, pattern: ExemplarPattern) -> FewshotResult:
    n_delimiters = context.count(pattern.delimiter)
    n_labels = context.count(pattern.label)

    if n_delimiters < pattern.min_pairs or n_labels < pattern.min_pairs:
        return FewshotResult()

    # Require rough balance: the counts should be within a factor of 3.
    # This prevents false positives where e.g. "Question:" appears many
    # times but "Type:" appears only in unrelated contexts.
    ratio = max(n_delimiters, n_labels) / max(min(n_delimiters, n_labels), 1)
    if ratio > 3.0:
        return FewshotResult()

    blocks = _extract_blocks(context, pattern.delimiter)

    # Validate: at least min_pairs blocks should contain the label
    labeled_blocks = sum(1 for b in blocks if pattern.label in b)
    if labeled_blocks < pattern.min_pairs:
        return FewshotResult()

    return FewshotResult(
        detected=True,
        pattern_name=pattern.name,
        strategy=pattern.strategy,
        blocks=blocks,
    )


def _extract_blocks(context: str, delimiter: str) -> list[str]:
    """
    Split context into atomic blocks on delimiter boundaries.
    """
    parts = context.split(delimiter)
    blocks = []
    for part in parts[1:]:  # skip text before first delimiter
        block = (delimiter + part).strip()
        if block and block != delimiter.strip():
            blocks.append(block)
    return blocks


# =============================================================================
# Block selection
# =============================================================================

def select_blocks(
    blocks: list[str],
    word_budget: int,
    sampling: str = "uniform",
) -> tuple[str, int, int, int]:
    """
    Select complete exemplar blocks within a word budget.

    """
    if not blocks:
        return "", 0, 0, 0

    block_words = [(b, len(b.split())) for b in blocks]
    total_words = sum(bw for _, bw in block_words)
    n_blocks = len(blocks)

    # If everything fits, keep it all
    if total_words <= word_budget:
        return "\n".join(blocks), total_words, n_blocks, n_blocks

    if sampling == "uniform":
        selected, used = _uniform_select(block_words, word_budget)
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling}")

    return "\n".join(selected), used, len(selected), n_blocks


def _uniform_select(
    block_words: list[tuple[str, int]],
    word_budget: int,
) -> tuple[list[str], int]:
    """Uniform-spacing selection with greedy backfill.

    1. Estimate capacity, compute step size, pick evenly spaced blocks.
    2. Greedily fill remaining budget from unselected blocks.

    """
    n_blocks = len(block_words)
    total_words = sum(bw for _, bw in block_words)

    # Estimate how many blocks fit
    avg_bw = total_words / n_blocks if n_blocks else 1
    est_fit = max(1, int(word_budget / avg_bw))
    step = max(1.0, n_blocks / est_fit)

    # Phase 1: uniformly spaced selection
    selected = []
    selected_indices = set()
    used = 0
    idx = 0.0

    while idx < n_blocks:
        i = int(idx)
        block, bw = block_words[i]
        if used + bw <= word_budget:
            selected.append(block)
            selected_indices.add(i)
            used += bw
        idx += step

    # Phase 2: greedy backfill from unselected
    for i, (block, bw) in enumerate(block_words):
        if i not in selected_indices and used + bw <= word_budget:
            selected.append(block)
            selected_indices.add(i)
            used += bw

    return selected, used