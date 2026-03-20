# Experiment Results

This document summarizes the main benchmark results used in the current repository narrative.

## Benchmark Setup

- Domain: Chinese legal and policy documents
- Evaluation style: benchmark-mode retrieval and generation evaluation
- Retrieval focus: Recall@k and MRR
- Generation focus: answer correctness, citation precision/recall, abstain accuracy

## Retrieval Summary

Best-performing retrieval pipeline:

- BM25
- DeepSeek Multi-Query
- BGE cross-encoder reranker

| Variant | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.2457 | 0.4162 | 0.4933 | 0.3396 |
| BM25 + MQ3 | 0.2457 | 0.4162 | 0.4952 | 0.3410 |
| BM25 + MQ3 + HyDE | 0.2457 | 0.3943 | 0.4848 | 0.3320 |
| BM25 + MQ3 + BGE Reranker | 0.3705 | 0.5257 | 0.5781 | 0.4511 |

Observations:

- Multi-Query gave a mild positive gain.
- HyDE hurt retrieval quality in this setting and was removed from the final stack.
- BGE reranking produced the largest retrieval improvement.

## Generation Summary

Best-performing generation pipeline:

- Processed contexts
- LLM grounded generation
- Wider prompt context window

| Variant | Answer Correctness | Citation Precision | Citation Recall | Abstain Accuracy |
|---|---:|---:|---:|---:|
| Extractive Raw | 0.2995 | 0.1781 | 0.5495 | 0.8114 |
| Extractive Processed | 0.3182 | 0.1935 | 0.5000 | 0.8114 |
| Extractive Processed Loose | 0.3049 | 0.1934 | 0.5000 | 0.8114 |
| LLM Processed | 0.3647 | 0.3286 | 0.3629 | 0.7790 |
| LLM Processed Wide | 0.3679 | 0.3354 | 0.3686 | 0.7886 |

Observations:

- Context processing improved the extractive generator.
- LLM grounded generation improved answer correctness and citation precision over extractive generation.
- Expanding the usable context window gave a small additional gain.

## Final Stack

### Retrieval

- BM25
- DeepSeek Multi-Query
- BGE reranker

### Generation

- Processed contexts
- LLM grounded generation
- Wide prompt context setting

## Practical Notes

- The current dense retriever is a research-style in-memory baseline rather than a production ANN index, so the most stable large-scale experiments were run on the BM25-centered pipeline.
- Faithfulness in the current generation report is a lightweight proxy and should not be overstated as a full semantic groundedness metric.
- The strongest claims in this repository should be made around retrieval gains, citation precision gains, and end-to-end grounded generation quality trends.
