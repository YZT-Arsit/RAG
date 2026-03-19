from __future__ import annotations

import csv
import json
from pathlib import Path

from legal_rag.audit.models import AnomalyDetail, AuditSummary, DuplicateDetail
from legal_rag.audit.stats import mean_length
from legal_rag.schemas.document import Document


def write_detail_csv(
    path: Path,
    *,
    duplicate_details: list[DuplicateDetail],
    anomaly_details: list[AnomalyDetail],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["record_type", "key", "doc_ids", "severity", "note"],
        )
        writer.writeheader()
        for duplicate in duplicate_details:
            writer.writerow(
                {
                    "record_type": duplicate.duplicate_type,
                    "key": duplicate.fingerprint,
                    "doc_ids": "|".join(duplicate.doc_ids),
                    "severity": "",
                    "note": duplicate.note,
                }
            )
        for anomaly in anomaly_details:
            writer.writerow(
                {
                    "record_type": anomaly.anomaly_type,
                    "key": anomaly.doc_id,
                    "doc_ids": anomaly.doc_id,
                    "severity": anomaly.severity,
                    "note": anomaly.note,
                }
            )


def write_normalized_jsonl(path: Path, documents: list[Document]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            payload = {
                "doc_id": document.doc_id,
                "source_file": document.source_file,
                "source_record_index": document.source_record_index,
                "title": document.title,
                "cleaned_title": document.cleaned_title,
                "sub_title": document.sub_title,
                "intro_title": document.intro_title,
                "publish_source": document.publish_source,
                "canonical_source": document.canonical_source,
                "published_at_raw": document.published_at_raw,
                "published_year": document.published_year,
                "content_text": document.content_text,
                "normalized_text": document.normalized_text,
                "cleaned_text": document.cleaned_text,
                "language": document.language,
                "text_length": document.text_length,
                "char_count_no_space": document.char_count_no_space,
                "structure_hints": document.structure_hints,
                "quality_flags": document.quality_flags,
                "cleaning_actions": document.cleaning_actions,
                "metadata": document.metadata,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_markdown_report(path: Path, summary: AuditSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total = max(summary.total_records, 1)
    average_length = mean_length(summary)
    duplicate_count = (
        len(summary.exact_title_duplicates)
        + len(summary.exact_body_duplicates)
        + len(summary.near_duplicates)
    )

    lines = [
        "# Dataset Audit Report",
        "",
        "## Status Labels",
        "",
        "- Implemented: stream-friendly JSON/JSONL loading, missing-field audit, length/source/year distributions, exact duplicate detection, approximate near-duplicate detection, anomaly detection, normalized document export.",
        "- Experimental: near-duplicate detection via SimHash, garbled-text heuristic, template-repetition heuristic, low-information-density heuristic.",
        "- Planned: dataset-specific cleaning rules, stronger semantic near-duplicate detection, PDF/HTML parsing enrichment, chunking-aware structural validation.",
        "",
        "## Input Summary",
        "",
        f"- Total records seen: {summary.total_records}",
        f"- Parsed records: {summary.parsed_records}",
        f"- Invalid records: {summary.invalid_records}",
        "",
        "## Missing Field Rates",
        "",
        f"- `title`: {summary.field_missing.title_missing} ({summary.field_missing.title_missing / total:.2%})",
        f"- `subTitle`: {summary.field_missing.sub_title_missing} ({summary.field_missing.sub_title_missing / total:.2%})",
        f"- `dataTime`: {summary.field_missing.data_time_missing} ({summary.field_missing.data_time_missing / total:.2%})",
        f"- `publishSource`: {summary.field_missing.publish_source_missing} ({summary.field_missing.publish_source_missing / total:.2%})",
        f"- `contentText`: {summary.field_missing.content_text_missing} ({summary.field_missing.content_text_missing / total:.2%})",
        f"- `introTitle`: {summary.field_missing.intro_title_missing} ({summary.field_missing.intro_title_missing / total:.2%})",
        "",
        "## Body Length Distribution",
        "",
        f"- Min length: {summary.length_stats.min_length}",
        f"- Max length: {summary.length_stats.max_length}",
        f"- Mean length: {average_length:.2f}",
    ]

    for bucket, count in summary.length_stats.buckets.items():
        lines.append(f"- `{bucket}`: {count}")

    lines.extend(
        [
            "",
            "## Source Distribution (Top 20)",
            "",
        ]
    )
    for source, count in sorted(
        summary.source_distribution.items(), key=lambda item: (-item[1], item[0])
    )[:20]:
        lines.append(f"- `{source}`: {count}")

    lines.extend(
        [
            "",
            "## Year Distribution (Top 20)",
            "",
        ]
    )
    for year, count in sorted(
        summary.year_distribution.items(), key=lambda item: item[0]
    )[:20]:
        lines.append(f"- `{year}`: {count}")

    lines.extend(
        [
            "",
            "## Duplicate Detection",
            "",
            f"- Exact title duplicate groups: {len(summary.exact_title_duplicates)}",
            f"- Exact body duplicate groups: {len(summary.exact_body_duplicates)}",
            f"- Near-duplicate groups: {len(summary.near_duplicates)}",
            f"- Total duplicate-related groups: {duplicate_count}",
            "",
            "## Anomaly Detection",
            "",
            f"- Total anomaly rows: {len(summary.anomalies)}",
        ]
    )

    anomaly_counts: dict[str, int] = {}
    for anomaly in summary.anomalies:
        anomaly_counts[anomaly.anomaly_type] = (
            anomaly_counts.get(anomaly.anomaly_type, 0) + 1
        )
    for anomaly_type, count in sorted(
        anomaly_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        lines.append(f"- `{anomaly_type}`: {count}")

    lines.extend(
        [
            "",
            "## Standardized Document Schema",
            "",
            "| Field | Type | Description |",
            "|---|---|---|",
            "| `doc_id` | `str` | Stable id derived from source file and record index. |",
            "| `source_file` | `str` | Original input file path. |",
            "| `source_record_index` | `int` | Original row index in source file. |",
            "| `title` | `str` | Normalized main title. |",
            "| `sub_title` | `str | None` | Optional subtitle. |",
            "| `intro_title` | `str | None` | Optional introduction heading. |",
            "| `publish_source` | `str | None` | Source organization or website. |",
            "| `published_at_raw` | `str | None` | Raw publish date text before canonical parsing. |",
            "| `published_year` | `int | None` | Extracted year for distribution and filtering. |",
            "| `content_text` | `str` | Original body text. |",
            "| `normalized_text` | `str` | Cleaned text for later chunking and retrieval. |",
            "| `language` | `str` | Language tag, currently `zh`. |",
            "| `text_length` | `int` | Length of normalized body text. |",
            "| `char_count_no_space` | `int` | Character count without spaces. |",
            "| `structure_hints` | `list[str]` | Detected legal structure markers for later chunking. |",
            "| `quality_flags` | `list[str]` | Audit-stage flags such as `empty_body` or `short_body`. |",
            "| `metadata` | `dict[str, Any]` | Reserved extra fields from raw input. |",
            "",
            "## Suggested Next Cleaning and Chunking Steps",
            "",
            "- Verify duplicate policies before deletion: exact-title duplicates may still differ in body or publication metadata.",
            "- Build deterministic cleaning rules for boilerplate headers, footer disclaimers, and website navigation residues after reviewing anomaly CSV samples.",
            "- Normalize date and source names into controlled vocabularies before retrieval indexing.",
            "- Preserve structural markers such as `第一条`, `一、`, `（一）` for future structure-aware chunking rather than stripping them too early.",
            "- Split future chunking experiments into at least two baselines: fixed-size and structure-aware; evaluate both on the same evidence-labeled benchmark.",
            "- Review near-duplicate groups manually before automated removal because SimHash is heuristic and may merge legally similar but not identical texts.",
        ]
    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
