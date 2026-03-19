from __future__ import annotations

from typing import Protocol

from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit


class BaseReranker(Protocol):
    def rerank(
        self, query: QueryRecord, hits: list[RetrievalHit], *, top_k: int
    ) -> list[RetrievalHit]:
        """Return reranked hits."""
