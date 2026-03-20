# Experiment Results

This document summarizes the benchmark-backed conclusions currently reflected in the repository.

## Benchmark Scope

- Domain: Chinese legal and policy documents
- Benchmark size: **525** legal QA samples
- Retrieval metrics: `Recall@1`, `Recall@3`, `Recall@5`, `MRR`
- Generation metrics: `answer_correctness`, `citation_precision`, `citation_recall`, `abstain_accuracy`

## Retrieval Results

### Compared Variants

- BM25
- BM25 + Multi-Query
- BM25 + Multi-Query + HyDE
- BM25 + Multi-Query + BGE-Reranker

### Results

| Variant | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.2457 | 0.4162 | 0.4933 | 0.3396 |
| BM25 + MQ3 | 0.2457 | 0.4162 | 0.4952 | 0.3410 |
| BM25 + MQ3 + HyDE | 0.2457 | 0.3943 | 0.4848 | 0.3320 |
| **BM25 + MQ3 + BGE-Reranker** | **0.3705** | **0.5257** | **0.5781** | **0.4511** |

### Interpretation

- Multi-Query gave a mild positive improvement.
- HyDE reduced retrieval quality in this legal setting and was removed.
- The largest gain came from the BGE cross-encoder reranker.

Relative to the BM25 baseline, the final retrieval stack improved:

- `Recall@1`: `+0.1248`
- `Recall@3`: `+0.1095`
- `Recall@5`: `+0.0848`
- `MRR`: `+0.1115`

## Generation Results

All generation experiments below were run on top of the best retrieval pipeline:

- `BM25 + MQ3 + BGE-Reranker`

### Compared Variants

- Extractive Raw
- Extractive Processed
- Extractive Processed Loose
- LLM Processed
- LLM Processed Wide

### Results

| Variant | Answer Correctness | Citation Precision | Citation Recall | Abstain Accuracy |
|---|---:|---:|---:|---:|
| Extractive Raw | 0.2995 | 0.1781 | 0.5495 | 0.8114 |
| Extractive Processed | 0.3182 | 0.1935 | 0.5000 | 0.8114 |
| Extractive Processed Loose | 0.3049 | 0.1934 | 0.5000 | 0.8114 |
| LLM Processed | 0.3647 | 0.3286 | 0.3629 | 0.7790 |
| **LLM Processed Wide** | **0.3679** | **0.3354** | **0.3686** | **0.7886** |

### Interpretation

- Context processing improved the extractive generator.
- Simply loosening the extractive setup did not provide further gains.
- Grounded LLM generation clearly outperformed extractive generation on answer correctness and citation precision.
- A wider prompt context window produced the best final generation configuration.

## Final Recommended Stack

### Retrieval

- BM25
- DeepSeek Multi-Query
- BGE-Reranker v2

### Generation

- Processed contexts
- LLM grounded generation
- Wide prompt context window

## Engineering Notes

- The current dense retriever is still a research-style in-memory baseline rather than a production ANN index, so the main large-scale benchmark narrative is intentionally centered on the BM25 pipeline.
- HyDE was tested and removed based on measured degradation instead of intuition.
- The strongest claims in this repository should be made around retrieval gains, reranking gains, grounded generation correctness, and citation precision improvements.
