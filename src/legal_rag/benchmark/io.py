from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from legal_rag.benchmark.schema import BenchmarkCandidate, BenchmarkRecord, GoldEvidence


def write_candidates(path: Path, candidates: list[BenchmarkCandidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            payload = {
                "query_id": candidate.query_id,
                "question": candidate.question,
                "question_type": candidate.question_type,
                "answerable": candidate.answerable,
                "gold_answer": candidate.gold_answer,
                "gold_evidence_chunk_ids": candidate.gold_evidence_chunk_ids,
                "source_doc_id": candidate.source_doc_id,
                "source_chunk_id": candidate.source_chunk_id,
                "metadata": candidate.metadata,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def iter_candidates(path: Path) -> Iterator[BenchmarkCandidate]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            yield BenchmarkCandidate(
                query_id=payload["query_id"],
                question=payload["question"],
                question_type=payload["question_type"],
                answerable=payload["answerable"],
                gold_answer=payload.get("gold_answer", ""),
                gold_evidence_chunk_ids=payload.get("gold_evidence_chunk_ids", []),
                source_doc_id=payload.get("source_doc_id"),
                source_chunk_id=payload.get("source_chunk_id"),
                metadata=payload.get("metadata", {}),
            )


def write_benchmark_records(path: Path, records: list[BenchmarkRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {
                "query_id": record.query_id,
                "question": record.question,
                "question_type": record.question_type,
                "answerable": record.answerable,
                "gold_answer": record.gold_answer,
                "gold_evidence_chunk_ids": [
                    evidence.chunk_id for evidence in record.gold_evidence
                ],
                "gold_evidence": [
                    {
                        "chunk_id": evidence.chunk_id,
                        "doc_id": evidence.doc_id,
                        "evidence_text": evidence.evidence_text,
                    }
                    for evidence in record.gold_evidence
                ],
                "metadata": record.metadata,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def candidate_to_record(
    candidate: BenchmarkCandidate,
    *,
    evidence_text_by_chunk_id: dict[str, str],
) -> BenchmarkRecord:
    evidence = [
        GoldEvidence(
            chunk_id=chunk_id,
            doc_id=candidate.source_doc_id,
            evidence_text=evidence_text_by_chunk_id.get(chunk_id),
        )
        for chunk_id in candidate.gold_evidence_chunk_ids
    ]
    metadata = dict(candidate.metadata)
    metadata["source_doc_id"] = candidate.source_doc_id
    metadata["source_chunk_id"] = candidate.source_chunk_id
    return BenchmarkRecord(
        query_id=candidate.query_id,
        question=candidate.question,
        question_type=candidate.question_type,
        answerable=candidate.answerable,
        gold_answer=candidate.gold_answer,
        gold_evidence=evidence,
        metadata=metadata,
    )
