from __future__ import annotations

from legal_rag.audit.models import AuditSummary, LengthStats
from legal_rag.schemas.document import Document, RawRecord


def update_missing_stats(summary: AuditSummary, record: RawRecord) -> None:
    summary.total_records += 1
    if not _present(record.title):
        summary.field_missing.title_missing += 1
    if not _present(record.sub_title):
        summary.field_missing.sub_title_missing += 1
    if not _present(record.data_time):
        summary.field_missing.data_time_missing += 1
    if not _present(record.publish_source):
        summary.field_missing.publish_source_missing += 1
    if not _present(record.content_text):
        summary.field_missing.content_text_missing += 1
    if not _present(record.intro_title):
        summary.field_missing.intro_title_missing += 1


def update_document_stats(summary: AuditSummary, document: Document) -> None:
    summary.parsed_records += 1
    _update_length_stats(summary.length_stats, document.text_length)
    source = document.publish_source or "UNKNOWN"
    summary.source_distribution[source] = summary.source_distribution.get(source, 0) + 1
    year_key = (
        str(document.published_year)
        if document.published_year is not None
        else "UNKNOWN"
    )
    summary.year_distribution[year_key] = summary.year_distribution.get(year_key, 0) + 1


def _update_length_stats(stats: LengthStats, length: int) -> None:
    stats.total_length += length
    stats.max_length = max(stats.max_length, length)
    stats.min_length = (
        length if stats.min_length is None else min(stats.min_length, length)
    )

    if length == 0:
        stats.buckets["0"] += 1
    elif length <= 30:
        stats.buckets["1-30"] += 1
    elif length <= 100:
        stats.buckets["31-100"] += 1
    elif length <= 300:
        stats.buckets["101-300"] += 1
    elif length <= 1000:
        stats.buckets["301-1000"] += 1
    elif length <= 3000:
        stats.buckets["1001-3000"] += 1
    elif length <= 10000:
        stats.buckets["3001-10000"] += 1
    else:
        stats.buckets["10001+"] += 1


def mean_length(summary: AuditSummary) -> float:
    if summary.parsed_records == 0:
        return 0.0
    return summary.length_stats.total_length / summary.parsed_records


def _present(value: str | None) -> bool:
    return bool(value and value.strip())
