from __future__ import annotations

from legal_rag.generation.citation import build_citation
from legal_rag.schemas.retrieval import RetrievalHit


def test_build_citation_preserves_chunk_identity() -> None:
    hit = RetrievalHit(
        query_id="q1",
        query_text="查询",
        chunk_id="chunk-1",
        doc_id="doc-1",
        rank=1,
        score=1.0,
        retrieval_method="hybrid",
        chunk_text="这是用于引用的正文内容。",
        title="标题",
    )
    citation = build_citation(hit, max_span_chars=8)
    assert citation.chunk_id == "chunk-1"
    assert citation.doc_id == "doc-1"
    assert citation.span_text == "这是用于引用的正"
