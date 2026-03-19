from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from legal_rag.chunking.io import iter_chunks
from legal_rag.schemas.chunk import Chunk
from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit, RetrievalResult


def load_chunks(path: Path) -> list[Chunk]:
    return list(iter_chunks(path))


def iter_queries(path: Path) -> Iterator[QueryRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            yield QueryRecord(
                query_id=payload["query_id"],
                query_text=payload["query_text"],
                metadata=payload.get("metadata", {}),
            )


def write_results(path: Path, results: list[RetrievalResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            payload = {
                "query_id": result.query.query_id,
                "query_text": result.query.query_text,
                "metadata": result.query.metadata,
                "hits": [
                    {
                        "query_id": hit.query_id,
                        "query_text": hit.query_text,
                        "chunk_id": hit.chunk_id,
                        "doc_id": hit.doc_id,
                        "rank": hit.rank,
                        "score": hit.score,
                        "retrieval_method": hit.retrieval_method,
                        "chunk_text": hit.chunk_text,
                        "title": hit.title,
                        "section_path": hit.section_path,
                        "metadata": hit.metadata,
                    }
                    for hit in result.hits
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
