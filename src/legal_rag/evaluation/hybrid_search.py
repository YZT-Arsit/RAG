from __future__ import annotations

from collections.abc import Sequence

from legal_rag.evaluation.retrieval_metrics import evaluate_retrieval
from legal_rag.retrieval.hybrid import rrf_fusion
from legal_rag.schemas.evaluation import RetrievalGoldRecord
from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit, RetrievalResult


def evaluate_hybrid_search_modes(
    test_set: Sequence[dict],
    *,
    vector_retriever,
    bm25_retriever,
    reranker=None,
    retrieve_top_k: int = 10,
    final_top_k: int = 5,
    rrf_k: int = 60,
    print_table: bool = True,
) -> dict[str, dict[str, float]]:
    mode_results = {
        "vector": [],
        "hybrid": [],
        "hybrid_rerank": [],
    }
    gold_records: list[RetrievalGoldRecord] = []

    for row in test_set:
        query_id = str(row["query_id"]) if "query_id" in row else str(row["query"])
        query_text = row["query"]
        ground_truth_id = row["ground_truth_id"]
        query = QueryRecord(query_id=query_id, query_text=query_text)

        vector_result = vector_retriever.retrieve(query, top_k=final_top_k)
        bm25_hits = bm25_retriever.get_bm25_scores(
            query_text,
            top_k=retrieve_top_k,
            query_record=query,
        )
        vector_candidates = vector_retriever.retrieve(query, top_k=retrieve_top_k).hits
        hybrid_hits = rrf_fusion(vector_candidates, bm25_hits, k=rrf_k)[:final_top_k]

        mode_results["vector"].append(vector_result)
        mode_results["hybrid"].append(RetrievalResult(query=query, hits=hybrid_hits))
        if reranker is None:
            reranked_hits = hybrid_hits
        else:
            reranked_hits = reranker.rerank_documents(query_text, hybrid_hits)[:final_top_k]
        mode_results["hybrid_rerank"].append(
            RetrievalResult(query=query, hits=_reset_ranks(reranked_hits))
        )
        gold_records.append(
            RetrievalGoldRecord(
                query_id=query_id,
                relevant_chunk_ids=[str(ground_truth_id)],
            )
        )

    summary: dict[str, dict[str, float]] = {}
    for mode, results in mode_results.items():
        _, metrics = evaluate_retrieval(results, gold_records, ks=[3, 5])
        summary[mode] = {
            "Recall@3": metrics.get("recall@3", 0.0),
            "Recall@5": metrics.get("recall@5", 0.0),
        }

    if print_table:
        print(format_recall_table(summary))
    return summary


def format_recall_table(summary: dict[str, dict[str, float]]) -> str:
    lines = [
        "mode                 Recall@3  Recall@5",
        "---------------------------------------",
    ]
    for mode in ("vector", "hybrid", "hybrid_rerank"):
        metrics = summary.get(mode, {})
        lines.append(
            f"{mode:<20} {metrics.get('Recall@3', 0.0):<8.4f}  {metrics.get('Recall@5', 0.0):.4f}"
        )
    return "\n".join(lines)


def _reset_ranks(hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
    normalized: list[RetrievalHit] = []
    for rank, hit in enumerate(hits, start=1):
        normalized.append(
            RetrievalHit(
                query_id=hit.query_id,
                query_text=hit.query_text,
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                rank=rank,
                score=hit.score,
                retrieval_method=hit.retrieval_method,
                chunk_text=hit.chunk_text,
                title=hit.title,
                section_path=hit.section_path,
                metadata=hit.metadata,
            )
        )
    return normalized
