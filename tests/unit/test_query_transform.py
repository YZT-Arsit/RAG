from __future__ import annotations

from pathlib import Path

from legal_rag.config.schema import RetrievalConfig
from legal_rag.retrieval.bm25 import BM25Retriever
from legal_rag.retrieval.dense import DenseBaselineRetriever
from legal_rag.retrieval.pipeline import RetrievalPipeline
from legal_rag.retrieval.query_transform import TransformedQuery
from legal_rag.schemas.chunk import Chunk
from legal_rag.schemas.retrieval import QueryRecord


class FakeQueryTransformer:
    def expand_query(self, query: QueryRecord) -> list[TransformedQuery]:
        return [
            TransformedQuery(text=query.query_text, source="original"),
            TransformedQuery(text="人身损害赔偿标准", source="multi_query"),
            TransformedQuery(text="人身损害赔偿责任与计算规则。", source="hyde"),
        ]


def make_chunk(chunk_id: str, method: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        chunk_index=0,
        chunk_method=method,
        text=text,
        text_length=len(text),
        start_char=0,
        end_char=len(text),
        title="标题",
        source_file="source",
        publish_source="来源",
        canonical_source="来源",
        published_year=2024,
    )


def test_retrieval_pipeline_uses_query_transformation_variants() -> None:
    chunks = [make_chunk("c1", "structure", "人身损害赔偿标准包括医疗费等项目")]
    config = RetrievalConfig(
        chunk_jsonl=Path("chunks.jsonl"),
        query_jsonl=Path("queries.jsonl"),
        output_jsonl=Path("out.jsonl"),
        report_path=Path("report.md"),
        method="hybrid",
        top_k=3,
        retrieve_top_k=3,
        hybrid_fusion="rrf",
    )
    pipeline = RetrievalPipeline(
        config=config,
        bm25_retrievers=[("structure", BM25Retriever(chunks, k1=1.5, b=0.75))],
        dense_retrievers=[("structure", DenseBaselineRetriever(chunks, ngram=2))],
        reranker=None,
        query_transformer=FakeQueryTransformer(),
    )
    result = pipeline.retrieve(QueryRecord(query_id="q1", query_text="被打伤了怎么赔"))
    assert len(result.hits) >= 1
    assert result.hits[0].metadata["query_variant_source"] in {
        "original",
        "multi_query",
        "hyde",
    }
