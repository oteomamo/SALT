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

## 🧮 Coverage selector

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

## 📜 Legacy selector

The legacy selector described in the [EMNLP 2026 paper](paper.md), run on
the current stack with vLLM 0.19.1 at the same 20% budget, for
Llama-3.1-8B-Instruct and Ministral-8B-Instruct-2410. The paper's own
numbers were produced with vLLM 0.11.0. The two code tasks run the
coverage selector, because the code reader works only with it. The
[results folder](https://github.com/oteomamo/SALT/tree/main/results) holds
the script that reproduces this table, one model per run.

| Category | Dataset | Metric | Llama | Ministral |
|---|---|---|---:|---:|
| Single-Doc QA | `narrativeqa` | qa_f1 | 23.00 | 23.65 |
| | `qasper` | qa_f1 | 39.38 | 39.39 |
| | `multifieldqa_en` | qa_f1 | 51.99 | 49.99 |
| | **average** | | **38.12** | **37.68** |
| Multi-Doc QA | `hotpotqa` | qa_f1 | 54.68 | 57.70 |
| | `2wikimqa` | qa_f1 | 37.78 | 43.63 |
| | `musique` | qa_f1 | 24.90 | 31.95 |
| | **average** | | **39.12** | **44.43** |
| Summarization | `gov_report` | rouge | 32.40 | 30.13 |
| | `qmsum` | rouge | 23.86 | 22.89 |
| | `multi_news` | rouge | 23.70 | 23.01 |
| | **average** | | **26.65** | **25.34** |
| Few-Shot | `trec` | classification | 74.00 | 76.50 |
| | `triviaqa` | qa_f1 | 82.44 | 90.38 |
| | `samsum` | rouge | 43.46 | 44.09 |
| | **average** | | **66.63** | **70.32** |
| Synthetic | `passage_count` | count | 10.50 | 12.50 |
| | `passage_retrieval_en` | retrieval | 99.50 | 97.00 |
| | **average** | | **55.00** | **54.75** |
| Code | `lcc` | code_sim | 48.50 | 46.91 |
| | `repobench-p` | code_sim | 41.85 | 50.67 |
| | **average** | | **45.17** | **48.79** |
| **Overall** | | | **44.50** | **46.27** |
