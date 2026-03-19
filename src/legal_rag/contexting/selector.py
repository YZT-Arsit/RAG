from __future__ import annotations

from collections import defaultdict

from legal_rag.schemas.retrieval import RetrievalHit


def select_hits(
    hits: list[RetrievalHit], *, max_chunks: int, max_per_doc: int
) -> list[RetrievalHit]:
    selected: list[RetrievalHit] = []
    per_doc_counter: defaultdict[str, int] = defaultdict(int)
    for hit in hits:
        if len(selected) >= max_chunks:
            break
        if per_doc_counter[hit.doc_id] >= max_per_doc:
            continue
        selected.append(hit)
        per_doc_counter[hit.doc_id] += 1
    return _rerank_sequentially(selected)


def _rerank_sequentially(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    output: list[RetrievalHit] = []
    for rank, hit in enumerate(hits, start=1):
        output.append(
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
    return output
