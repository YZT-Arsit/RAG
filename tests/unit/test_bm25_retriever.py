from __future__ import annotations

from legal_rag.retrieval.bm25 import BM25Retriever
from legal_rag.schemas.chunk import Chunk
from legal_rag.schemas.retrieval import QueryRecord


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        chunk_index=0,
        chunk_method="fixed",
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


def test_bm25_retriever_returns_relevant_chunk_first() -> None:
    chunks = [
        make_chunk("c1", "通用机场管理规定适用于机场建设和运营管理"),
        make_chunk("c2", "夫妻共同财产赠与第三者返还纠纷案例"),
    ]
    retriever = BM25Retriever(chunks, k1=1.5, b=0.75)
    result = retriever.retrieve(
        QueryRecord(query_id="q1", query_text="通用机场适用范围"), top_k=2
    )
    assert result.hits[0].chunk_id == "c1"
