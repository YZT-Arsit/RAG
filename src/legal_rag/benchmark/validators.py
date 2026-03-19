from __future__ import annotations

from collections import Counter

from legal_rag.benchmark.schema import BenchmarkRecord


def validate_benchmark(records: list[BenchmarkRecord]) -> dict[str, object]:
    query_ids = [record.query_id for record in records]
    duplicate_query_ids = [
        query_id for query_id, count in Counter(query_ids).items() if count > 1
    ]
    invalid_answerable = [
        record.query_id
        for record in records
        if not record.answerable and record.gold_answer.strip()
    ]
    missing_evidence_for_answerable = [
        record.query_id
        for record in records
        if record.answerable and not record.gold_evidence
    ]
    return {
        "record_count": len(records),
        "duplicate_query_ids": duplicate_query_ids,
        "invalid_answerable_gold_answer": invalid_answerable,
        "missing_evidence_for_answerable": missing_evidence_for_answerable,
        "is_valid": not duplicate_query_ids
        and not invalid_answerable
        and not missing_evidence_for_answerable,
    }
