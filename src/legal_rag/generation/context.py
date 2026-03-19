from __future__ import annotations

from legal_rag.schemas.retrieval import RetrievalResult


def build_context_block(
    result: RetrievalResult, *, max_contexts: int, max_chars_per_context: int
) -> str:
    parts: list[str] = []
    for hit in result.hits[:max_contexts]:
        snippet = hit.chunk_text[:max_chars_per_context].strip()
        parts.append(f"[{hit.rank}] {hit.title} ({hit.chunk_id})\n{snippet}")
    return "\n\n".join(parts)
