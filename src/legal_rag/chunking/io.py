from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from legal_rag.schemas.chunk import Chunk


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            payload = {
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "chunk_index": chunk.chunk_index,
                "chunk_method": chunk.chunk_method,
                "text": chunk.text,
                "text_length": chunk.text_length,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "title": chunk.title,
                "source_file": chunk.source_file,
                "publish_source": chunk.publish_source,
                "canonical_source": chunk.canonical_source,
                "published_year": chunk.published_year,
                "section_path": chunk.section_path,
                "structure_labels": chunk.structure_labels,
                "quality_flags": chunk.quality_flags,
                "metadata": chunk.metadata,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def iter_chunks(path: Path) -> Iterator[Chunk]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            yield Chunk(
                chunk_id=payload["chunk_id"],
                doc_id=payload["doc_id"],
                chunk_index=payload["chunk_index"],
                chunk_method=payload["chunk_method"],
                text=payload["text"],
                text_length=payload["text_length"],
                start_char=payload["start_char"],
                end_char=payload["end_char"],
                title=payload["title"],
                source_file=payload["source_file"],
                publish_source=payload.get("publish_source"),
                canonical_source=payload.get("canonical_source"),
                published_year=payload.get("published_year"),
                section_path=payload.get("section_path", []),
                structure_labels=payload.get("structure_labels", []),
                quality_flags=payload.get("quality_flags", []),
                metadata=payload.get("metadata", {}),
            )
