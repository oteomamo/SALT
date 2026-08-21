# 🔬 Results

SALT (coverage/CELF selector) on LongBench with Llama-3.1-8B-Instruct at a 20%
token budget. More datasets coming soon. The trie these numbers rest
on is the subject of the [EMNLP 2026 paper](paper.md), which reports
the earlier legacy selector.

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
