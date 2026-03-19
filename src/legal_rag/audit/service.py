from __future__ import annotations

import re

from legal_rag.audit.anomalies import detect_anomalies
from legal_rag.audit.duplicates import find_exact_duplicates, find_near_duplicates
from legal_rag.audit.models import AuditSummary
from legal_rag.audit.normalize import build_document
from legal_rag.audit.reader import iter_records
from legal_rag.audit.report import (
    write_detail_csv,
    write_markdown_report,
    write_normalized_jsonl,
)
from legal_rag.audit.stats import update_document_stats, update_missing_stats
from legal_rag.config.schema import AuditConfig


def run_audit(config: AuditConfig) -> AuditSummary:
    summary = AuditSummary()
    year_pattern = re.compile(config.year_regex)

    for input_path in config.input_paths:
        for record in iter_records(input_path):
            update_missing_stats(summary, record)
            document = build_document(record, year_pattern)
            update_document_stats(summary, document)
            summary.normalized_documents.append(document)
            summary.anomalies.extend(
                detect_anomalies(
                    document,
                    short_body_threshold=config.short_body_threshold,
                    template_similarity_line_threshold=config.template_similarity_line_threshold,
                )
            )
            if (
                config.sample_limit is not None
                and summary.parsed_records >= config.sample_limit
            ):
                break
        if (
            config.sample_limit is not None
            and summary.parsed_records >= config.sample_limit
        ):
            break

    summary.exact_title_duplicates = find_exact_duplicates(
        summary.normalized_documents, field="title"
    )
    summary.exact_body_duplicates = find_exact_duplicates(
        summary.normalized_documents, field="normalized_text"
    )
    summary.near_duplicates = find_near_duplicates(
        summary.normalized_documents,
        hamming_threshold=config.near_duplicate_hamming_threshold,
        prefix_bits=config.near_duplicate_prefix_bits,
    )

    all_duplicates = (
        summary.exact_title_duplicates
        + summary.exact_body_duplicates
        + summary.near_duplicates
    )
    write_normalized_jsonl(config.normalized_output_path, summary.normalized_documents)
    write_detail_csv(
        config.detail_csv_path,
        duplicate_details=all_duplicates,
        anomaly_details=summary.anomalies,
    )
    write_markdown_report(config.report_path, summary)
    return summary
