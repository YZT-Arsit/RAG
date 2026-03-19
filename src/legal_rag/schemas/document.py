from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class RawRecord:
    source_file: str
    record_index: int
    title: str | None
    sub_title: str | None
    data_time: str | None
    publish_source: str | None
    content_text: str | None
    intro_title: str | None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Document:
    doc_id: str
    source_file: str
    source_record_index: int
    title: str
    cleaned_title: str | None
    sub_title: str | None
    intro_title: str | None
    publish_source: str | None
    canonical_source: str | None
    published_at_raw: str | None
    published_year: int | None
    content_text: str
    normalized_text: str
    cleaned_text: str | None
    language: str
    text_length: int
    char_count_no_space: int
    structure_hints: list[str]
    quality_flags: list[str]
    cleaning_actions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
