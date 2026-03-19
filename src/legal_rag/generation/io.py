from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from legal_rag.schemas.generation import Citation, GroundedAnswer


def write_answers(path: Path, answers: list[GroundedAnswer]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for answer in answers:
            payload = {
                "query_id": answer.query_id,
                "query_text": answer.query_text,
                "answer": answer.answer,
                "citations": [
                    {
                        "label": citation.label,
                        "chunk_id": citation.chunk_id,
                        "doc_id": citation.doc_id,
                        "title": citation.title,
                        "span_text": citation.span_text,
                        "rank": citation.rank,
                        "source_ref": citation.source_ref,
                    }
                    for citation in answer.citations
                ],
                "used_context_ids": answer.used_context_ids,
                "generation_method": answer.generation_method,
                "abstained": answer.abstained,
                "metadata": answer.metadata,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def iter_answers(path: Path) -> Iterator[GroundedAnswer]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            yield GroundedAnswer(
                query_id=payload["query_id"],
                query_text=payload["query_text"],
                answer=payload["answer"],
                citations=[
                    Citation(
                        label=item.get("label"),
                        chunk_id=item["chunk_id"],
                        doc_id=item["doc_id"],
                        title=item["title"],
                        span_text=item["span_text"],
                        rank=item["rank"],
                        source_ref=item.get("source_ref"),
                    )
                    for item in payload.get("citations", [])
                ],
                used_context_ids=payload.get("used_context_ids", []),
                generation_method=payload.get(
                    "generation_method", "extractive_grounded"
                ),
                abstained=payload.get("abstained", False),
                metadata=payload.get("metadata", {}),
            )
