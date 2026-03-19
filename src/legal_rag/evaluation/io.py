from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from legal_rag.benchmark.loader import iter_benchmark_records
from legal_rag.generation.io import iter_answers
from legal_rag.schemas.evaluation import GenerationGoldRecord, RetrievalGoldRecord
from legal_rag.schemas.generation import GroundedAnswer
from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit, RetrievalResult


def load_retrieval_results(path: Path) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            results.append(
                RetrievalResult(
                    query=QueryRecord(
                        query_id=payload["query_id"],
                        query_text=payload["query_text"],
                        metadata=payload.get("metadata", {}),
                    ),
                    hits=[
                        RetrievalHit(
                            query_id=item["query_id"],
                            query_text=item["query_text"],
                            chunk_id=item["chunk_id"],
                            doc_id=item["doc_id"],
                            rank=item["rank"],
                            score=item["score"],
                            retrieval_method=item["retrieval_method"],
                            chunk_text=item["chunk_text"],
                            title=item["title"],
                            section_path=item.get("section_path", []),
                            metadata=item.get("metadata", {}),
                        )
                        for item in payload.get("hits", [])
                    ],
                )
            )
    return results


def load_answers(path: Path) -> list[GroundedAnswer]:
    return list(iter_answers(path))


def iter_retrieval_gold(path: Path) -> Iterator[RetrievalGoldRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            yield RetrievalGoldRecord(
                query_id=payload["query_id"],
                relevant_chunk_ids=payload.get("relevant_chunk_ids", []),
                metadata=payload.get("metadata", {}),
            )


def iter_generation_gold(path: Path) -> Iterator[GenerationGoldRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            yield GenerationGoldRecord(
                query_id=payload["query_id"],
                reference_answer=payload["reference_answer"],
                supporting_chunk_ids=payload.get("supporting_chunk_ids", []),
                question_type=payload.get("question_type", "other"),
                answerable=payload.get("answerable", True),
                metadata=payload.get("metadata", {}),
            )


def load_generation_gold(
    path: Path, *, benchmark_mode: bool
) -> list[GenerationGoldRecord]:
    if benchmark_mode:
        return [
            GenerationGoldRecord(
                query_id=record.query_id,
                reference_answer=record.gold_answer,
                supporting_chunk_ids=[
                    evidence.chunk_id for evidence in record.gold_evidence
                ],
                question_type=record.question_type,
                answerable=record.answerable,
                metadata=record.metadata,
            )
            for record in iter_benchmark_records(path)
        ]
    return list(iter_generation_gold(path))


def load_retrieval_gold(
    path: Path, *, benchmark_mode: bool
) -> list[RetrievalGoldRecord]:
    if benchmark_mode:
        return [
            RetrievalGoldRecord(
                query_id=record.query_id,
                relevant_chunk_ids=[
                    evidence.chunk_id for evidence in record.gold_evidence
                ],
                metadata=record.metadata,
            )
            for record in iter_benchmark_records(path)
        ]
    return list(iter_retrieval_gold(path))
