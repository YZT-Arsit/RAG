from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from legal_rag.schemas.document import Document


def iter_documents(path: Path) -> Iterator[Document]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            yield Document(
                doc_id=payload["doc_id"],
                source_file=payload["source_file"],
                source_record_index=payload["source_record_index"],
                title=payload["title"],
                cleaned_title=payload.get("cleaned_title"),
                sub_title=payload.get("sub_title"),
                intro_title=payload.get("intro_title"),
                publish_source=payload.get("publish_source"),
                canonical_source=payload.get("canonical_source"),
                published_at_raw=payload.get("published_at_raw"),
                published_year=payload.get("published_year"),
                content_text=payload["content_text"],
                normalized_text=payload["normalized_text"],
                cleaned_text=payload.get("cleaned_text"),
                language=payload["language"],
                text_length=payload["text_length"],
                char_count_no_space=payload["char_count_no_space"],
                structure_hints=payload.get("structure_hints", []),
                quality_flags=payload.get("quality_flags", []),
                cleaning_actions=payload.get("cleaning_actions", []),
                metadata=payload.get("metadata", {}),
            )


def write_documents(path: Path, documents: list[Document]) -> None:
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
