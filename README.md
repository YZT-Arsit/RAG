# legal-rag-zh

Chinese legal RAG research repository focused on reproducible retrieval, reranking, grounded generation, and benchmark-driven evaluation.

## What This Repo Shows

- A full offline pipeline from audit and cleaning to chunking, retrieval, reranking, generation, and evaluation
- A formal benchmark generation and validation workflow for Chinese legal/policy corpora
- Retrieval ablations over BM25, query transformation, and reranking
- Grounded generation with citations, abstention, and lightweight guardrails
- Config-driven experiments that are easy to rerun and extend

## Final Experimental Takeaways

### Retrieval

Best retrieval stack:

- `BM25`
- `DeepSeek Multi-Query`
- `BGE Reranker`

Key benchmark results on the medium legal benchmark:

| Variant | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.2457 | 0.4162 | 0.4933 | 0.3396 |
| BM25 + MQ3 | 0.2457 | 0.4162 | 0.4952 | 0.3410 |
| BM25 + MQ3 + HyDE | 0.2457 | 0.3943 | 0.4848 | 0.3320 |
| BM25 + MQ3 + BGE Reranker | 0.3705 | 0.5257 | 0.5781 | 0.4511 |

Interpretation:

- Multi-Query provided small but positive gains.
- HyDE underperformed in this legal setting and was removed.
- The largest improvement came from reranking with a cross-encoder.

### Generation

Best generation stack:

- Processed contexts
- LLM grounded generation
- Wider prompt context window

Key benchmark results:

| Variant | Answer Correctness | Citation Precision | Citation Recall | Abstain Accuracy |
|---|---:|---:|---:|---:|
| Extractive Raw | 0.2995 | 0.1781 | 0.5495 | 0.8114 |
| Extractive Processed | 0.3182 | 0.1935 | 0.5000 | 0.8114 |
| LLM Processed | 0.3647 | 0.3286 | 0.3629 | 0.7790 |
| LLM Processed Wide | 0.3679 | 0.3354 | 0.3686 | 0.7886 |

Interpretation:

- Context processing helped the extractive baseline.
- Grounded LLM generation outperformed extractive generation on correctness and citation precision.
- A wider context window improved LLM generation further.

## Repository Structure

- [src/legal_rag](/Users/Hoshino/Documents/RAG/src/legal_rag): implementation
- [configs](/Users/Hoshino/Documents/RAG/configs): runnable experiment configs
- [tests/unit](/Users/Hoshino/Documents/RAG/tests/unit): unit tests
- [docs/experiment_results.md](/Users/Hoshino/Documents/RAG/docs/experiment_results.md): experiment summary for interview/demo use
- [docs/retrieval.md](/Users/Hoshino/Documents/RAG/docs/retrieval.md): retrieval design notes
- [docs/generation.md](/Users/Hoshino/Documents/RAG/docs/generation.md): generation design notes

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m legal_rag.cli.main audit --config configs/audit/base.yaml
python -m legal_rag.cli.main clean --config configs/cleaning/base.yaml
python -m legal_rag.cli.main chunk --config configs/chunking/base.yaml
python -m legal_rag.cli.main retrieve --config configs/retrieval/base.yaml
python -m legal_rag.cli.main process-contexts --config configs/context/base.yaml
python -m legal_rag.cli.main generate --config configs/generation/base.yaml
python -m legal_rag.cli.main eval-retrieval --config configs/eval/retrieval.yaml
python -m legal_rag.cli.main eval-generation --config configs/eval/generation.yaml
```

## Not Included In Git

Large processed corpora, benchmark artifacts, model caches, and experiment report dumps are intentionally excluded from version control. The repository is meant to expose the code, configs, tests, and reproducible methodology rather than raw large files.
