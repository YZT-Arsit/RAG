from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    doc_id: str
    chunk_index: int
    chunk_method: str
    text: str
    text_length: int
    start_char: int
    end_char: int
    title: str
    source_file: str
    publish_source: str | None
    canonical_source: str | None
    published_year: int | None
    section_path: list[str] = field(default_factory=list)
    structure_labels: list[str] = field(default_factory=list)
    quality_flags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
