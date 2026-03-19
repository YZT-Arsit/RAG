from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Citation:
    chunk_id: str
    doc_id: str
    title: str
    span_text: str
    rank: int
    label: str | None = None
    source_ref: str | None = None


@dataclass(slots=True)
class ContextItem:
    chunk_id: str
    doc_id: str
    title: str
    text: str
    rank: int
    score: float
    retrieval_method: str
    section_path: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationInput:
    query_id: str
    query_text: str
    question_type: str | None
    answerable: bool | None
    contexts: list[ContextItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GroundedAnswer:
    query_id: str
    query_text: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    used_context_ids: list[str] = field(default_factory=list)
    generation_method: str = "extractive_grounded"
    abstained: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
