from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ErrorRecord:
    query_id: str
    question: str
    error_labels: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
