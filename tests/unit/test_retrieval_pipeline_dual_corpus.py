from __future__ import annotations

from pathlib import Path

from legal_rag.config.schema import RetrievalConfig
from legal_rag.retrieval.bm25 import BM25Retriever
from legal_rag.retrieval.dense import DenseBaselineRetriever
from legal_rag.retrieval.pipeline import RetrievalPipeline
from legal_rag.schemas.chunk import Chunk
from legal_rag.schemas.retrieval import QueryRecord


def make_chunk(chunk_id: str, method: str, text: str, title: str = "标题") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        chunk_index=0,
        chunk_method=method,
        text=text,
        text_length=len(text),
        start_char=0,
        end_char=len(text),
        title=title,
        source_file="source",
        publish_source="来源",
        canonical_source="来源",
        published_year=2024,
        metadata={"heading_prefix": "第一章 > 第一条" if method == "structure" else ""},
    )


def make_config(method: str) -> RetrievalConfig:
    return RetrievalConfig(
        chunk_jsonl=Path("chunks.jsonl"),
        query_jsonl=Path("queries.jsonl"),
        output_jsonl=Path("out.jsonl"),
        report_path=Path("report.md"),
        method=method,
        top_k=3,
        retrieve_top_k=3,
        hybrid_fusion="rrf",
    )


def test_pipeline_merges_structure_and_fixed_corpora_for_bm25() -> None:
    structure_chunks = [
        make_chunk("doc-1::structure::0", "structure", "第一条 夫妻共同财产分割规则")
    ]
    fixed_chunks = [
        make_chunk("doc-1::fixed::0", "fixed", "夫妻共同财产分割规则与补充说明")
    ]
    pipeline = RetrievalPipeline(
        config=make_config("bm25"),
        bm25_retrievers=[
            ("structure", BM25Retriever(structure_chunks, k1=1.5, b=0.75)),
            ("fixed", BM25Retriever(fixed_chunks, k1=1.5, b=0.75)),
        ],
        dense_retrievers=[],
        reranker=None,
    )

    result = pipeline.retrieve(QueryRecord(query_id="q1", query_text="夫妻共同财产"))
    assert len(result.hits) == 2
    assert result.hits[0].metadata["chunk_method"] == "structure"
    assert {hit.metadata["retrieval_corpus"] for hit in result.hits} == {
        "structure",
        "fixed",
    }


def test_pipeline_hybrid_uses_dual_corpus_candidates() -> None:
    structure_chunks = [
        make_chunk("doc-1::structure::0", "structure", "第一条 通用机场管理规定适用范围")
    ]
    fixed_chunks = [
        make_chunk("doc-1::fixed::0", "fixed", "通用机场管理规定适用范围与解释")
    ]
    pipeline = RetrievalPipeline(
        config=make_config("hybrid"),
        bm25_retrievers=[
            ("structure", BM25Retriever(structure_chunks, k1=1.5, b=0.75)),
            ("fixed", BM25Retriever(fixed_chunks, k1=1.5, b=0.75)),
        ],
        dense_retrievers=[
            ("structure", DenseBaselineRetriever(structure_chunks, ngram=2)),
            ("fixed", DenseBaselineRetriever(fixed_chunks, ngram=2)),
        ],
        reranker=None,
    )

    result = pipeline.retrieve(QueryRecord(query_id="q1", query_text="通用机场适用范围"))
    assert len(result.hits) >= 1
    assert result.hits[0].retrieval_method.startswith("hybrid_")
