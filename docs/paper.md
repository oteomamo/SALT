# 📝 Paper

!!! success "Accepted at EMNLP 2026"

    **SALT: Salience-Aware Lexical Trie for Long-Context Compression**
    is a **Main Conference** paper at the 2026 Conference on Empirical
    Methods in Natural Language Processing, held in Budapest.

<div class="salt-buttons" markdown>
[Read it on OpenReview](https://openreview.net/forum?id=UTJqOhkSqi){ .md-button .md-button--primary }
[arXiv preprint](https://arxiv.org/abs/2607.17486){ .md-button }
</div>

Oteo Mamo, Hyunjin Yi, Joydhriti Choudhury, Shangqian Gao, Weikuan Yu.

## 📄 Abstract

> As large language models (LLMs) process increasingly longer prompts,
> computation and KV-cache memory costs have emerged as major
> bottlenecks in inference systems. Existing input-level prompt
> compression methods address this, but rank each sentence by a scalar
> relevance score, treating the document as an unstructured pool of
> words and sentences. Under tight budgets, this causes theme collapse,
> where the dominant theme(s) of a document consumes the budget,
> discarding less-frequent yet task-relevant themes. Preserving thematic
> coverage instead requires allocating the budget across recurring
> themes rather than scoring sentences in isolation. To this end, we
> propose SALT, a model-agnostic extractive framework that organizes
> per-sentence keywords into a trie ordered by sentence frequency (SF),
> a lightweight, reusable proxy for document thematic structure. This
> trie-based organization smooths memory allocation and prevents
> dominant themes from monopolizing the budget. Multi-anchor retrieval
> activates trie nodes labeled by query keywords at any depth, and the
> trie persists across dialogue turns, supporting multi-turn use without
> re-encoding the document. By preserving document themes, SALT reduces
> the prefill computation and memory cost of long-context prompts while
> remaining composable with KV-cache methods that target decoding-time
> latency and memory.

## 🧠 The idea in short

A prompt that does not fit has to lose sentences. The usual way to
choose is to give every sentence one relevance score and keep the top of
that list until the budget runs out. Under a tight budget this hands the
budget to whatever the document is mostly about, so the smaller themes
go first, even when one of them holds the sentence the question needs.

SALT organizes each sentence's keywords into a trie ordered by sentence
frequency, which is a cheap and reusable map of a document's recurring
themes, and then spreads the budget across the branches of that map
instead of down a ranked list. Two things follow from building the map
once.

- **A query does not rebuild it.** Multi-anchor retrieval activates the
  trie nodes labeled by the query's keywords at any depth, so one index
  serves any question asked of the document.
- **A conversation does not rebuild it either.** The trie persists
  across turns, which is what lets [chatbot mode](chatbot.md) keep a
  long session and its attached files recallable without re-encoding
  them every turn.

Because SALT acts on the input, before the model reads anything, it
stays model-agnostic and composes with the KV-cache methods that work at
decoding time rather than competing with them.

## 🔀 The paper and this repository

The selector the paper describes is tagged
[v1.0.0](https://github.com/oteomamo/SALT/releases/tag/v1.0.0), which is
where a run that follows the paper starts. It is also still reachable on
`main` as `--selector legacy`.

Since 2.0.0 the default on `main` is the coverage selector. It keeps the
same trie and the same reason for it, and chooses sentences with a CELF
lazy-greedy pass in which a sentence is worth less once its theme
branches have already been served. The [Architecture](architecture.md)
page describes what ships today and the [Results](results.md) page
carries the current LongBench table.

Everything the repository grew after the paper, which is conversation
memory, persistent serving, the agent layer and the MCP server, is built
on that same index.

## 📌 How to cite

```bibtex
@inproceedings{mamo2026salt,
      title={{SALT}: Salience-Aware Lexical Trie for Long-Context Compression},
      author={Oteo Mamo and Hyunjin Yi and Joydhriti Choudhury and Shangqian Gao and Weikuan Yu},
      booktitle={The 2026 Conference on Empirical Methods in Natural Language Processing},
      year={2026},
      url={https://openreview.net/forum?id=UTJqOhkSqi}
}
```
