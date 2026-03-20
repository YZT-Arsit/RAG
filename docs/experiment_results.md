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

### Full Benchmark (strict)

| Variant | Recall@5 | MRR | Precision@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.4933 | 0.3396 | 0.0998 | 0.3778 |
| BM25 + MQ3 | 0.4952 | 0.3410 | 0.1002 | 0.3794 |
| **BM25 + MQ3 + BGE-Reranker** | **0.5781** | **0.4511** | **0.1170** | **0.4827** |

### Answerable-Only Benchmark

| Variant | Recall@5 | MRR | Precision@5 | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.5570 | 0.3834 | 0.1127 | 0.4265 |
| BM25 + MQ3 | 0.5591 | 0.3850 | 0.1131 | 0.4284 |
| **BM25 + MQ3 + BGE-Reranker** | **0.6527** | **0.5094** | **0.1320** | **0.5449** |

### Retrieval by Question Type

| Question Type | Count | Recall@5 | MRR | Precision@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| Definition | 120 | 0.9250 | 0.8540 | 0.1850 | 0.8718 |
| Comparison | 100 | 0.8350 | 0.6690 | 0.1740 | 0.7095 |
| Condition | 100 | 0.6800 | 0.4303 | 0.1360 | 0.4927 |
| Procedure | 70 | 0.3429 | 0.2076 | 0.0686 | 0.2414 |
| Responsibility | 75 | 0.2267 | 0.1320 | 0.0453 | 0.1555 |
| Unanswerable | 60 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### Interpretation

- Multi-Query gave a mild positive improvement.
- HyDE reduced retrieval quality in this legal setting and was removed.
- The largest gain came from the BGE cross-encoder reranker.
- On the `answerable` subset, the final retrieval stack improved:
  - `Recall@5`: `+0.0957`
  - `MRR`: `+0.1260`
  - `Precision@5`: `+0.0193`
  - `nDCG@5`: `+0.1184`
- Performance is strongest on `definition` and `comparison` questions, while `procedure` and `responsibility` remain the main hard cases.

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
