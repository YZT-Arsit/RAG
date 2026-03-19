from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from legal_rag.benchmark.schema import BenchmarkRecord, GoldEvidence


def iter_benchmark_records(path: Path) -> Iterator[BenchmarkRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            yield _parse_record(payload, line_number=line_number)


def _parse_record(payload: dict[str, object], *, line_number: int) -> BenchmarkRecord:
    required = ["query_id", "question", "question_type", "answerable", "gold_answer"]
    missing = [key for key in required if key not in payload]
    if missing:
        msg = f"Benchmark line {line_number} missing required fields: {missing}"
        raise ValueError(msg)
    evidence = [
        GoldEvidence(
            chunk_id=item["chunk_id"],
            doc_id=item.get("doc_id"),
            evidence_text=item.get("evidence_text"),
        )
        for item in payload.get("gold_evidence", [])
    ]
    return BenchmarkRecord(
        query_id=str(payload["query_id"]),
        question=str(payload["question"]),
        question_type=str(payload["question_type"]),
        answerable=bool(payload["answerable"]),
        gold_answer=str(payload["gold_answer"]),
        gold_evidence=evidence,
        metadata=payload.get("metadata", {})
        if isinstance(payload.get("metadata", {}), dict)
        else {},
    )
