from __future__ import annotations

from legal_rag.retrieval.tokenize import tokenize_for_bm25
from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit


class HeuristicReranker:
    def __init__(
        self,
        *,
        title_overlap_weight: float,
        body_overlap_weight: float,
        structure_overlap_weight: float,
    ) -> None:
        self.title_overlap_weight = title_overlap_weight
        self.body_overlap_weight = body_overlap_weight
        self.structure_overlap_weight = structure_overlap_weight

    def rerank(
        self, query: QueryRecord, hits: list[RetrievalHit], *, top_k: int
    ) -> list[RetrievalHit]:
        query_tokens = set(tokenize_for_bm25(query.query_text))
        rescored: list[tuple[RetrievalHit, float]] = []
        for hit in hits:
            title_overlap = _overlap_ratio(
                query_tokens, set(tokenize_for_bm25(hit.title))
            )
            body_overlap = _overlap_ratio(
                query_tokens, set(tokenize_for_bm25(hit.chunk_text[:500]))
            )
            structure_overlap = _overlap_ratio(
                query_tokens, set(tokenize_for_bm25(" ".join(hit.section_path)))
            )
            rerank_score = (
                self.title_overlap_weight * title_overlap
                + self.body_overlap_weight * body_overlap
                + self.structure_overlap_weight * structure_overlap
            )
            combined_score = hit.score + rerank_score
            updated_metadata = {
                **hit.metadata,
                "rerank_score": rerank_score,
                "pre_rerank_score": hit.score,
            }
            rescored.append(
                (
                    RetrievalHit(
                        query_id=hit.query_id,
                        query_text=hit.query_text,
                        chunk_id=hit.chunk_id,
                        doc_id=hit.doc_id,
                        rank=hit.rank,
                        score=combined_score,
                        retrieval_method=f"{hit.retrieval_method}+heuristic_rerank",
                        chunk_text=hit.chunk_text,
                        title=hit.title,
                        section_path=hit.section_path,
                        metadata=updated_metadata,
                    ),
                    combined_score,
                )
            )

        ranked = sorted(rescored, key=lambda item: item[1], reverse=True)[:top_k]
        output: list[RetrievalHit] = []
        for rank, (hit, score) in enumerate(ranked, start=1):
            output.append(
                RetrievalHit(
                    query_id=hit.query_id,
                    query_text=hit.query_text,
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    rank=rank,
                    score=score,
                    retrieval_method=hit.retrieval_method,
                    chunk_text=hit.chunk_text,
                    title=hit.title,
                    section_path=hit.section_path,
                    metadata=hit.metadata,
                )
            )
        return output


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)
