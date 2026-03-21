from __future__ import annotations

from collections import Counter
from pathlib import Path

from legal_rag.config.schema import DenseIndexBuildConfig, RetrievalConfig
from legal_rag.retrieval.bm25 import BM25Retriever
from legal_rag.retrieval.dense import DenseBaselineRetriever, FaissDenseRetriever
from legal_rag.retrieval.embeddings import (
    BGEEmbeddingEncoder,
    DEFAULT_DENSE_EMBEDDING_MODEL,
)
from legal_rag.retrieval.faiss_index import save_chunk_metadata
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
        config,
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
    config: RetrievalConfig,
) -> list[tuple[str, DenseBaselineRetriever | FaissDenseRetriever]]:
    if config.dense_backend == "faiss":
        if config.dense_index_dir is None:
            msg = "dense_backend=faiss requires dense_index_dir in retrieval config."
            raise ValueError(msg)
        return _load_faiss_retrievers(config)

    corpora = _partition_chunks(chunks)
    return [
        (name, DenseBaselineRetriever(corpus_chunks, ngram=config.dense_ngram))
        for name, corpus_chunks in corpora
        if corpus_chunks
    ]


def _load_faiss_retrievers(
    config: RetrievalConfig,
) -> list[tuple[str, FaissDenseRetriever]]:
    index_dir = config.dense_index_dir
    assert index_dir is not None
    retrievers: list[tuple[str, FaissDenseRetriever]] = []
    for index_path in sorted(index_dir.glob("*.faiss")):
        corpus_name = index_path.stem
        metadata_path = index_dir / f"{corpus_name}.meta.jsonl"
        if not metadata_path.exists():
            continue
        retrievers.append(
            (
                corpus_name,
                FaissDenseRetriever.from_disk(
                    index_path=index_path,
                    metadata_path=metadata_path,
                    model_name=config.dense_model_name,
                    modelscope_model_id=config.dense_modelscope_model_id,
                    local_model_dir=config.dense_local_model_dir,
                    use_modelscope_download=config.dense_use_modelscope_download,
                    device=config.dense_device,
                    batch_size=config.dense_batch_size,
                    max_length=config.dense_max_length,
                ),
            )
        )
    return retrievers


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


def build_dense_index(config: DenseIndexBuildConfig) -> None:
    chunks = load_chunks(config.chunk_jsonl)
    corpora = _partition_chunks(chunks) if config.partition_by_chunk_method else [("default", chunks)]
    encoder = BGEEmbeddingEncoder(
        model_name=config.dense_model_name or DEFAULT_DENSE_EMBEDDING_MODEL,
        modelscope_model_id=config.dense_modelscope_model_id,
        local_model_dir=config.dense_local_model_dir,
        use_modelscope_download=config.dense_use_modelscope_download,
        device=config.dense_device,
        batch_size=config.dense_batch_size,
        max_length=config.dense_max_length,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    built: list[tuple[str, int, int]] = []
    for corpus_name, corpus_chunks in corpora:
        if not corpus_chunks:
            continue
        vectors = encoder.encode([_dense_retrieval_text(chunk) for chunk in corpus_chunks])
        _write_faiss_index(config.output_dir / f"{corpus_name}.faiss", vectors)
        save_chunk_metadata(config.output_dir / f"{corpus_name}.meta.jsonl", corpus_chunks)
        built.append((corpus_name, len(corpus_chunks), vectors.shape[1]))

    _write_dense_index_report(config.report_path, config.output_dir, built)


def _write_faiss_index(path: Path, vectors) -> None:
    try:
        import faiss
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
        msg = "faiss is required to build dense vector indexes."
        raise RuntimeError(msg) from exc

    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    faiss.write_index(index, str(path))


def _dense_retrieval_text(chunk) -> str:
    heading_prefix = str(chunk.metadata.get("heading_prefix", "")).strip()
    sub_title = str(chunk.metadata.get("sub_title", "")).strip()
    intro_title = str(chunk.metadata.get("intro_title", "")).strip()
    article_label = str(chunk.metadata.get("article_label", "")).strip()
    return " ".join(
        part
        for part in [
            chunk.title,
            sub_title,
            intro_title,
            heading_prefix,
            article_label,
            chunk.text,
        ]
        if part
    )


def _write_dense_index_report(path: Path, output_dir: Path, built: list[tuple[str, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dense Index Report",
        "",
        "## Summary",
        "",
        f"- Output dir: {output_dir}",
        f"- Corpus partitions built: {len(built)}",
        "",
        "## Partitions",
        "",
    ]
    if built:
        for corpus_name, count, dim in built:
            lines.append(f"- `{corpus_name}`: chunks={count}, dimension={dim}")
    else:
        lines.append("- No partitions built.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "- Planned: larger-scale ANN retrieval variants, stronger production indexing, retrieval evaluation against qrels.",
        "",
        "## Summary",
        "",
        f"- Queries processed: {len(results)}",
        f"- Total hits exported: {sum(len(result.hits) for result in results)}",
        f"- First-stage method: {config.method}",
        f"- Hybrid fusion: {config.hybrid_fusion if config.method == 'hybrid' else 'N/A'}",
        f"- Reranker enabled: {config.reranker_enabled}",
        f"- Dense backend: {config.dense_backend}",
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
