from __future__ import annotations

from legal_rag.retrieval.hybrid import fuse_results
from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit


def make_hit(chunk_id: str, score: float, method: str) -> RetrievalHit:
    return RetrievalHit(
        query_id="q1",
        query_text="查询",
        chunk_id=chunk_id,
        doc_id=chunk_id,
        rank=1,
        score=score,
        retrieval_method=method,
        chunk_text="正文",
        title="标题",
    )


def test_hybrid_fusion_combines_scores() -> None:
    query = QueryRecord(query_id="q1", query_text="查询")
    result = fuse_results(
        query,
        bm25_hits=[make_hit("c1", 2.0, "bm25"), make_hit("c2", 1.0, "bm25")],
        dense_hits=[make_hit("c2", 0.9, "dense"), make_hit("c1", 0.1, "dense")],
        fusion_type="score",
        alpha=0.5,
        rrf_k=60,
        top_k=2,
    )
    assert len(result.hits) == 2
    assert {hit.chunk_id for hit in result.hits} == {"c1", "c2"}
