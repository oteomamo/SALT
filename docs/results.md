# 🔬 Results

SALT (coverage/CELF selector) on LongBench with Llama-3.1-8B-Instruct at a 20%
token budget. These numbers were produced with torch 2.8.0, transformers
4.55.2 and vLLM 0.11.0, the stack SALT pinned up to 3.0.26. The current
stack (torch 2.10, transformers 5.5.3, vLLM 0.19.1) compresses to the same
text, and rerunning five of the datasets on it moved their average by about
a tenth of a point. The trie these numbers rest on is the subject of the
[EMNLP 2026 paper](paper.md), which reports the earlier legacy
selector. The measurements behind saltChat's memory features, on
conversation benchmarks rather than documents, are described on the
[Architecture](architecture.md) page beside the feature each one
decided.

| Category | Dataset | Metric | SALT |
|---|---|---|---:|
| Single-Doc QA | `narrativeqa` | qa_f1 | 25.89 |
| | `qasper` | qa_f1 | 42.61 |
| | `multifieldqa_en` | qa_f1 | 51.03 |
| | **average** | | **39.84** |
| Multi-Doc QA | `hotpotqa` | qa_f1 | 56.09 |
| | `2wikimqa` | qa_f1 | 44.26 |
| | `musique` | qa_f1 | 31.76 |
| | **average** | | **44.04** |
| Summarization | `gov_report` | rouge | 31.59 |
| | `qmsum` | rouge | 23.89 |
| | `multi_news` | rouge | 23.78 |
| | **average** | | **26.42** |
| Few-Shot | `trec` | classification | 61.00 |
| | `triviaqa` | qa_f1 | 81.83 |
| | `samsum` | rouge | 42.94 |
| | **average** | | **61.92** |
| Synthetic | `passage_count` | count | 10.00 |
| | `passage_retrieval_en` | retrieval | 97.00 |
| | **average** | | **53.50** |
| Code | `lcc` | code_sim | 48.50 |
| | `repobench-p` | code_sim | 41.38 |
| | **average** | | **44.94** |
| **Overall** | | | **44.60** |
