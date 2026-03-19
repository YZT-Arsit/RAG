from __future__ import annotations

from legal_rag.reranking.heuristic import HeuristicReranker
from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit


def test_heuristic_reranker_promotes_title_overlap() -> None:
    reranker = HeuristicReranker(
        title_overlap_weight=1.0,
        body_overlap_weight=0.0,
        structure_overlap_weight=0.0,
    )
    query = QueryRecord(query_id="q1", query_text="通用机场适用范围")
    hits = [
        RetrievalHit(
            query_id="q1",
            query_text=query.query_text,
            chunk_id="c1",
            doc_id="d1",
            rank=1,
            score=0.1,
            retrieval_method="hybrid_rrf",
            chunk_text="无关正文",
            title="无关标题",
        ),
        RetrievalHit(
            query_id="q1",
            query_text=query.query_text,
            chunk_id="c2",
            doc_id="d2",
            rank=2,
            score=0.05,
            retrieval_method="hybrid_rrf",
            chunk_text="正文",
            title="通用机场管理规定适用范围",
        ),
    ]
    reranked = reranker.rerank(query, hits, top_k=2)
    assert reranked[0].chunk_id == "c2"
    assert reranked[0].retrieval_method.endswith("+heuristic_rerank")
