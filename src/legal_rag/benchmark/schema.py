from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class BenchmarkCandidate:
    query_id: str
    question: str
    question_type: str
    answerable: bool
    gold_answer: str
    gold_evidence_chunk_ids: list[str] = field(default_factory=list)
    source_doc_id: str | None = None
    source_chunk_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GoldEvidence:
    chunk_id: str
    doc_id: str | None = None
    evidence_text: str | None = None


@dataclass(slots=True)
class BenchmarkRecord:
    query_id: str
    question: str
    question_type: str
    answerable: bool
    gold_answer: str
    gold_evidence: list[GoldEvidence] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
