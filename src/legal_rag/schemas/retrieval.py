from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class QueryRecord:
    query_id: str
    query_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalHit:
    query_id: str
    query_text: str
    chunk_id: str
    doc_id: str
    rank: int
    score: float
    retrieval_method: str
    chunk_text: str
    title: str
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    query: QueryRecord
    hits: list[RetrievalHit]
