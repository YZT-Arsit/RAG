from __future__ import annotations

from dataclasses import dataclass, field

from legal_rag.schemas.document import Document


@dataclass(slots=True)
class FieldMissingStats:
    title_missing: int = 0
    sub_title_missing: int = 0
    data_time_missing: int = 0
    publish_source_missing: int = 0
    content_text_missing: int = 0
    intro_title_missing: int = 0


@dataclass(slots=True)
class LengthStats:
    min_length: int | None = None
    max_length: int = 0
    total_length: int = 0
    buckets: dict[str, int] = field(
        default_factory=lambda: {
            "0": 0,
            "1-30": 0,
            "31-100": 0,
            "101-300": 0,
            "301-1000": 0,
            "1001-3000": 0,
            "3001-10000": 0,
            "10001+": 0,
        }
    )


@dataclass(slots=True)
class DuplicateDetail:
    duplicate_type: str
    fingerprint: str
    doc_ids: list[str]
    note: str


@dataclass(slots=True)
class AnomalyDetail:
    doc_id: str
    anomaly_type: str
    severity: str
    note: str


@dataclass(slots=True)
class AuditSummary:
    total_records: int = 0
    parsed_records: int = 0
    invalid_records: int = 0
    field_missing: FieldMissingStats = field(default_factory=FieldMissingStats)
    length_stats: LengthStats = field(default_factory=LengthStats)
    source_distribution: dict[str, int] = field(default_factory=dict)
    year_distribution: dict[str, int] = field(default_factory=dict)
    exact_title_duplicates: list[DuplicateDetail] = field(default_factory=list)
    exact_body_duplicates: list[DuplicateDetail] = field(default_factory=list)
    near_duplicates: list[DuplicateDetail] = field(default_factory=list)
    anomalies: list[AnomalyDetail] = field(default_factory=list)
    normalized_documents: list[Document] = field(default_factory=list)
