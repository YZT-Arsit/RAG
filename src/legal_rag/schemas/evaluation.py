from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RetrievalGoldRecord:
    query_id: str
    relevant_chunk_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationGoldRecord:
    query_id: str
    reference_answer: str
    supporting_chunk_ids: list[str] = field(default_factory=list)
    question_type: str = "other"
    answerable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetricRecord:
    query_id: str
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AutoEvalSample:
    query_id: str
    query: str
    ground_truth_answer: str
    ground_truth_chunk_id: str
    ground_truth_doc_id: str
    evidence_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
