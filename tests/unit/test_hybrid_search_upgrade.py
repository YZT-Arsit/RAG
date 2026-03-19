from __future__ import annotations

from legal_rag.evaluation.hybrid_search import evaluate_hybrid_search_modes
from legal_rag.retrieval.bm25 import BM25Retriever
from legal_rag.retrieval.dense import DenseBaselineRetriever
from legal_rag.retrieval.hybrid import rrf_fusion
from legal_rag.schemas.chunk import Chunk
from legal_rag.schemas.retrieval import RetrievalHit


def make_chunk(chunk_id: str, text: str, title: str = "标题") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        chunk_index=0,
        chunk_method="structure",
        text=text,
        text_length=len(text),
        start_char=0,
        end_char=len(text),
        title=title,
        source_file="source",
        publish_source="来源",
        canonical_source="来源",
        published_year=2024,
    )


def make_hit(chunk_id: str, rank: int, score: float, method: str) -> RetrievalHit:
    return RetrievalHit(
        query_id="q1",
        query_text="夫妻共同财产",
        chunk_id=chunk_id,
        doc_id=chunk_id,
        rank=rank,
        score=score,
        retrieval_method=method,
        chunk_text=f"{chunk_id} 正文",
        title="标题",
    )


class ReverseReranker:
    def rerank_documents(self, query: str, candidate_docs: list[RetrievalHit]) -> list[RetrievalHit]:
        return list(reversed(candidate_docs))


def test_get_bm25_scores_returns_original_index() -> None:
    chunks = [
        make_chunk("c1", "夫妻共同财产分割规则"),
        make_chunk("c2", "通用机场建设管理办法"),
    ]
    retriever = BM25Retriever(chunks, k1=1.5, b=0.75)
    hits = retriever.get_bm25_scores("夫妻共同财产", top_k=2)
    assert hits[0].chunk_id == "c1"
    assert hits[0].metadata["original_index"] == 0


def test_rrf_fusion_dedupes_duplicate_hits() -> None:
    fused = rrf_fusion(
        [
            make_hit("c1", 1, 0.9, "dense"),
            make_hit("c1", 2, 0.8, "dense"),
        ],
        [
            make_hit("c1", 1, 10.0, "bm25"),
            make_hit("c2", 2, 9.0, "bm25"),
        ],
    )
    assert [hit.chunk_id for hit in fused] == ["c1", "c2"]


def test_evaluate_hybrid_search_modes_reports_three_modes() -> None:
    chunks = [
        make_chunk("c1", "夫妻共同财产是婚姻关系存续期间所得财产"),
        make_chunk("c2", "通用机场管理规定适用于机场建设"),
    ]
    dense = DenseBaselineRetriever(chunks, ngram=2)
    bm25 = BM25Retriever(chunks, k1=1.5, b=0.75)
    summary = evaluate_hybrid_search_modes(
        [{"query_id": "q1", "query": "夫妻共同财产", "ground_truth_id": "c1"}],
        vector_retriever=dense,
        bm25_retriever=bm25,
        reranker=ReverseReranker(),
        retrieve_top_k=2,
        final_top_k=2,
        print_table=False,
    )
    assert set(summary) == {"vector", "hybrid", "hybrid_rerank"}
    assert "Recall@3" in summary["hybrid"]
