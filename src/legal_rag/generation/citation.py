from __future__ import annotations

from legal_rag.schemas.generation import Citation
from legal_rag.schemas.retrieval import RetrievalHit


def build_citation(hit: RetrievalHit, *, max_span_chars: int) -> Citation:
    source_parts = [
        str(hit.metadata.get("source_file", "")).strip(),
        str(hit.metadata.get("article_label", "")).strip(),
        str(hit.metadata.get("page", "")).strip(),
    ]
    return Citation(
        label=str(hit.rank),
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        title=hit.title,
        span_text=hit.chunk_text[:max_span_chars].strip(),
        rank=hit.rank,
        source_ref=" | ".join(part for part in source_parts if part) or None,
    )
