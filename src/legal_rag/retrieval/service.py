from __future__ import annotations

from collections import Counter

from legal_rag.config.schema import RetrievalConfig
from legal_rag.retrieval.bm25 import BM25Retriever
from legal_rag.retrieval.dense import DenseBaselineRetriever
from legal_rag.retrieval.pipeline import RetrievalPipeline
from legal_rag.retrieval.query_transform import build_query_transformer
from legal_rag.retrieval.io import iter_queries, load_chunks, write_results
from legal_rag.reranking.service import build_reranker
from legal_rag.schemas.retrieval import RetrievalResult


def run_retrieval(config: RetrievalConfig) -> None:
    chunks = load_chunks(config.chunk_jsonl)
    queries = list(iter_queries(config.query_jsonl))

    pipeline = build_retrieval_pipeline(config, chunks)
    results: list[RetrievalResult] = []
    for query in queries:
        results.append(pipeline.retrieve(query))

    write_results(config.output_jsonl, results)
    _write_retrieval_report(config.report_path, results, config)


def build_retrieval_pipeline(config: RetrievalConfig, chunks) -> RetrievalPipeline:
    bm25_retrievers = _build_corpus_bm25_retrievers(
        chunks,
        k1=config.bm25_k1,
        b=config.bm25_b,
    )
    dense_retrievers = _build_corpus_dense_retrievers(
        chunks,
        ngram=config.dense_ngram,
    )
    reranker = build_reranker(config)
    query_transformer = build_query_transformer(config)
    return RetrievalPipeline(
        config=config,
        bm25_retrievers=bm25_retrievers,
        dense_retrievers=dense_retrievers,
        reranker=reranker,
        query_transformer=query_transformer,
    )


def _build_corpus_bm25_retrievers(
    chunks,
    *,
    k1: float,
    b: float,
) -> list[tuple[str, BM25Retriever]]:
    corpora = _partition_chunks(chunks)
    return [
        (name, BM25Retriever(corpus_chunks, k1=k1, b=b))
        for name, corpus_chunks in corpora
        if corpus_chunks
    ]


def _build_corpus_dense_retrievers(
    chunks,
    *,
    ngram: int,
) -> list[tuple[str, DenseBaselineRetriever]]:
    corpora = _partition_chunks(chunks)
    return [
        (name, DenseBaselineRetriever(corpus_chunks, ngram=ngram))
        for name, corpus_chunks in corpora
        if corpus_chunks
    ]


def _partition_chunks(chunks) -> list[tuple[str, list]]:
    structure_chunks = [chunk for chunk in chunks if chunk.chunk_method == "structure"]
    fixed_chunks = [chunk for chunk in chunks if chunk.chunk_method == "fixed"]
    other_chunks = [
        chunk for chunk in chunks if chunk.chunk_method not in {"structure", "fixed"}
    ]
    corpora: list[tuple[str, list]] = []
    if structure_chunks:
        corpora.append(("structure", structure_chunks))
    if fixed_chunks:
        corpora.append(("fixed", fixed_chunks))
    if other_chunks:
        corpora.append(("other", other_chunks))
    if not corpora:
        corpora.append(("default", list(chunks)))
    return corpora


def _write_retrieval_report(
    path, results: list[RetrievalResult], config: RetrievalConfig
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hit_counter: Counter[str] = Counter()
    for result in results:
        for hit in result.hits:
            hit_counter.update([hit.retrieval_method])
    lines = [
        "# Retrieval Report",
        "",
        "## Status Labels",
        "",
        "- Implemented: BM25 retrieval, dense baseline retrieval, hybrid score fusion, hybrid RRF fusion, multi-stage retrieval pipeline, heuristic reranker baseline.",
        "- Experimental: dense baseline based on character n-gram TF-IDF similarity, heuristic reranker, title-bonus fusion.",
        "- Planned: real embedding retriever, ANN indexing, cross-encoder reranker, retrieval evaluation against qrels.",
        "",
        "## Summary",
        "",
        f"- Queries processed: {len(results)}",
        f"- Total hits exported: {sum(len(result.hits) for result in results)}",
        f"- First-stage method: {config.method}",
        f"- Hybrid fusion: {config.hybrid_fusion if config.method == 'hybrid' else 'N/A'}",
        f"- Reranker enabled: {config.reranker_enabled}",
        f"- Retrieve top-k: {config.retrieve_top_k}",
        f"- Final top-k: {config.top_k}",
        "",
        "## Retrieval Methods",
        "",
    ]
    if hit_counter:
        for method, count in hit_counter.most_common():
            lines.append(f"- `{method}`: {count}")
    else:
        lines.append("- No hits exported.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
