from __future__ import annotations

import re

from legal_rag.schemas.chunk import Chunk
from legal_rag.schemas.document import Document

SENTENCE_RE = re.compile(r"[^。！？；\n]+[。！？；\n]?")


def chunk_document_fixed(
    document: Document,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    text = document.cleaned_text or document.normalized_text
    if not text:
        return []
    if chunk_overlap >= chunk_size:
        msg = "chunk_overlap must be smaller than chunk_size"
        raise ValueError(msg)

    sentences = _split_sentences(text)
    if sentences:
        return _chunk_by_sentences(
            document,
            sentences,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    chunks: list[Chunk] = []
    start = 0
    step = chunk_size - chunk_overlap
    chunk_index = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                _build_chunk(
                    document=document,
                    chunk_index=chunk_index,
                    method="fixed",
                    text=chunk_text,
                    start_char=start,
                    end_char=end,
                    section_path=[],
                    structure_labels=list(document.structure_hints),
                )
            )
            chunk_index += 1
        if end >= len(text):
            break
        start += step
    return chunks


def _chunk_by_sentences(
    document: Document,
    sentences: list[tuple[int, int, str]],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    sentence_index = 0
    chunk_index = 0

    while sentence_index < len(sentences):
        start_char = sentences[sentence_index][0]
        end_char = start_char
        collected: list[tuple[int, int, str]] = []
        while sentence_index < len(sentences):
            sent_start, sent_end, sent_text = sentences[sentence_index]
            candidate_end = sent_end
            candidate_len = candidate_end - start_char
            if collected and candidate_len > chunk_size:
                break
            if not collected and candidate_len > chunk_size:
                chunks.extend(
                    _split_long_sentence(
                        document=document,
                        chunk_index_start=chunk_index,
                        start_char=sent_start,
                        text=sent_text,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
                chunk_index += len(chunks) - chunk_index
                sentence_index += 1
                collected = []
                break
            collected.append((sent_start, sent_end, sent_text))
            end_char = sent_end
            sentence_index += 1

        if not collected:
            continue

        chunk_text = "".join(sentence[2] for sentence in collected).strip()
        if chunk_text:
            chunks.append(
                _build_chunk(
                    document=document,
                    chunk_index=chunk_index,
                    method="fixed",
                    text=chunk_text,
                    start_char=start_char,
                    end_char=end_char,
                    section_path=[],
                    structure_labels=list(document.structure_hints),
                )
            )
            chunk_index += 1

        sentence_index = _rewind_sentence_index(
            sentences,
            current_index=sentence_index,
            chunk_end_char=end_char,
            overlap_chars=chunk_overlap,
        )

    return chunks


def _split_long_sentence(
    *,
    document: Document,
    chunk_index_start: int,
    start_char: int,
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = max(chunk_size - chunk_overlap, 1)
    local_start = 0
    chunk_index = chunk_index_start
    while local_start < len(text):
        local_end = min(len(text), local_start + chunk_size)
        piece = text[local_start:local_end].strip()
        if piece:
            absolute_start = start_char + local_start
            absolute_end = absolute_start + len(piece)
            chunks.append(
                _build_chunk(
                    document=document,
                    chunk_index=chunk_index,
                    method="fixed",
                    text=piece,
                    start_char=absolute_start,
                    end_char=absolute_end,
                    section_path=[],
                    structure_labels=list(document.structure_hints),
                )
            )
            chunk_index += 1
        if local_end >= len(text):
            break
        local_start += step
    return chunks


def _rewind_sentence_index(
    sentences: list[tuple[int, int, str]],
    *,
    current_index: int,
    chunk_end_char: int,
    overlap_chars: int,
) -> int:
    if overlap_chars <= 0:
        return current_index
    overlap_start = max(0, chunk_end_char - overlap_chars)
    for index, (sent_start, _sent_end, _sent_text) in enumerate(sentences):
        if sent_start >= overlap_start:
            return min(index, current_index)
    return current_index


def _split_sentences(text: str) -> list[tuple[int, int, str]]:
    sentences: list[tuple[int, int, str]] = []
    for match in SENTENCE_RE.finditer(text):
        sentence = match.group(0)
        if not sentence.strip():
            continue
        sentences.append((match.start(), match.end(), sentence))
    return sentences


def _build_chunk(
    *,
    document: Document,
    chunk_index: int,
    method: str,
    text: str,
    start_char: int,
    end_char: int,
    section_path: list[str],
    structure_labels: list[str],
) -> Chunk:
    return Chunk(
        chunk_id=f"{document.doc_id}::{method}::{chunk_index}",
        doc_id=document.doc_id,
        chunk_index=chunk_index,
        chunk_method=method,
        text=text,
        text_length=len(text),
        start_char=start_char,
        end_char=end_char,
        title=document.cleaned_title or document.title,
        source_file=document.source_file,
        publish_source=document.publish_source,
        canonical_source=document.canonical_source,
        published_year=document.published_year,
        section_path=section_path,
        structure_labels=structure_labels,
        quality_flags=list(document.quality_flags),
        metadata={
            "source_record_index": document.source_record_index,
            "sub_title": document.sub_title,
            "intro_title": document.intro_title,
            "heading_prefix": " > ".join(structure_labels),
        },
    )
