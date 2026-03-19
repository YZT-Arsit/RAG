from __future__ import annotations

from legal_rag.schemas.retrieval import RetrievalHit


def dedupe_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    deduped: list[RetrievalHit] = []
    for hit in hits:
        prefix = hit.chunk_text[:120].strip()
        if _is_duplicate(prefix, hit.doc_id, deduped):
            continue
        deduped.append(hit)
    return deduped


def _is_duplicate(prefix: str, doc_id: str, existing_hits: list[RetrievalHit]) -> bool:
    for existing in existing_hits:
        if existing.doc_id != doc_id:
            continue
        existing_prefix = existing.chunk_text[:120].strip()
        if (
            prefix == existing_prefix
            or prefix.startswith(existing_prefix)
            or existing_prefix.startswith(prefix)
        ):
            return True
    return False
