from __future__ import annotations

import re
from dataclasses import dataclass

from legal_rag.schemas.chunk import Chunk
from legal_rag.schemas.document import Document


HEADING_RE = re.compile(
    r"(第[一二三四五六七八九十百千0-9]+编|第[一二三四五六七八九十百千0-9]+章|第[一二三四五六七八九十百千0-9]+节|第[一二三四五六七八九十百千0-9]+条|第[一二三四五六七八九十百千0-9]+款|第[一二三四五六七八九十百千0-9]+项|[一二三四五六七八九十]+、|（[一二三四五六七八九十0-9]+）)"
)
SENTENCE_RE = re.compile(r"[^。！？；\n]+[。！？；\n]?")
SECONDARY_BOUNDARY_RE = re.compile(r"(?<=[。！？；])|(?<=，)|(?<=:)|(?<=：)")


@dataclass(slots=True)
class Segment:
    label: str | None
    start_char: int
    end_char: int
    text: str
    level: int | None
    path: list[str]


def chunk_document_structure_aware(
    document: Document,
    *,
    max_chunk_size: int,
    min_chunk_size: int,
    sentence_overlap: int = 1,
) -> list[Chunk]:
    text = document.cleaned_text or document.normalized_text
    if not text:
        return []

    segments = _split_with_headings(text)
    if not segments:
        return []

    chunks: list[Chunk] = []
    current_segments: list[Segment] = []
    chunk_index = 0

    for segment in segments:
        candidate = "".join(item.text for item in current_segments) + segment.text
        if current_segments and (
            len(candidate) > max_chunk_size
            or _should_flush_before_segment(current_segments, segment)
        ):
            built_chunks = _finalize_segments(
                document=document,
                segments=current_segments,
                chunk_index_start=chunk_index,
                max_chunk_size=max_chunk_size,
                min_chunk_size=min_chunk_size,
                sentence_overlap=sentence_overlap,
            )
            chunks.extend(built_chunks)
            chunk_index += len(built_chunks)
            current_segments = [segment]
            continue

        current_segments.append(segment)

    if current_segments:
        built_chunks = _finalize_segments(
            document=document,
            segments=current_segments,
            chunk_index_start=chunk_index,
            max_chunk_size=max_chunk_size,
            min_chunk_size=min_chunk_size,
            sentence_overlap=sentence_overlap,
        )
        chunks.extend(built_chunks)

    return _merge_short_tail_chunks(chunks, min_chunk_size=min_chunk_size)


def _split_with_headings(text: str) -> list[Segment]:
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [Segment(None, 0, len(text), text, None, [])]

    segments: list[Segment] = []
    path_stack: list[tuple[int, str]] = []
    if matches[0].start() > 0:
        segments.append(
            Segment(None, 0, matches[0].start(), text[: matches[0].start()], None, [])
        )

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        label = match.group(0)
        level = _heading_level(label)
        if level is not None:
            while path_stack and path_stack[-1][0] >= level:
                path_stack.pop()
            path_stack.append((level, label))
        segments.append(
            Segment(label, start, end, text[start:end], level, [item[1] for item in path_stack])
        )
    return segments


def _finalize_segments(
    *,
    document: Document,
    segments: list[Segment],
    chunk_index_start: int,
    max_chunk_size: int,
    min_chunk_size: int,
    sentence_overlap: int,
) -> list[Chunk]:
    candidate_parts = _split_segments_recursively(segments, max_chunk_size=max_chunk_size)
    chunks: list[Chunk] = []
    overlap_suffix = ""
    for offset, part in enumerate(candidate_parts):
        chunk_text = (overlap_suffix + part.text).strip()
        if not chunk_text:
            continue
        section_path = part.path or _resolve_path(segments)
        chunks.append(
            _build_structure_chunk(
                document=document,
                chunk_index=chunk_index_start + len(chunks),
                text=chunk_text,
                start_char=part.start_char,
                end_char=part.end_char,
                section_path=section_path,
            )
        )
        overlap_suffix = _extract_overlap_text(part.text, sentence_overlap)

    return chunks


def _split_segments_recursively(
    segments: list[Segment],
    *,
    max_chunk_size: int,
) -> list[Segment]:
    text = "".join(segment.text for segment in segments).strip()
    if not text:
        return []
    start_char = segments[0].start_char
    end_char = segments[-1].end_char
    path = _resolve_path(segments)
    if len(text) <= max_chunk_size:
        return [Segment(segments[0].label, start_char, end_char, text, segments[0].level, path)]

    pieces = _split_text_by_boundaries(text, max_chunk_size=max_chunk_size)
    if len(pieces) == 1:
        return [Segment(segments[0].label, start_char, end_char, text, segments[0].level, path)]

    split_segments: list[Segment] = []
    cursor = start_char
    for piece in pieces:
        piece_text = piece.strip()
        if not piece_text:
            cursor += len(piece)
            continue
        piece_start = text.find(piece_text, max(cursor - start_char, 0))
        absolute_start = start_char + max(piece_start, 0)
        absolute_end = absolute_start + len(piece_text)
        split_segments.append(
            Segment(segments[0].label, absolute_start, absolute_end, piece_text, segments[0].level, path)
        )
        cursor = absolute_end
    return split_segments


def _split_text_by_boundaries(text: str, *, max_chunk_size: int) -> list[str]:
    sentences = [match.group(0) for match in SENTENCE_RE.finditer(text) if match.group(0).strip()]
    if len(sentences) > 1:
        parts = _group_units(sentences, max_chunk_size=max_chunk_size)
        if len(parts) > 1:
            return parts
    secondary_units = [piece for piece in SECONDARY_BOUNDARY_RE.split(text) if piece.strip()]
    grouped = _group_units(secondary_units, max_chunk_size=max_chunk_size)
    if len(grouped) > 1:
        return grouped
    return [text[i : i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]


def _group_units(units: list[str], *, max_chunk_size: int) -> list[str]:
    grouped: list[str] = []
    current = ""
    for unit in units:
        candidate = current + unit
        if current and len(candidate) > max_chunk_size:
            grouped.append(current)
            current = unit
            continue
        current = candidate
    if current:
        grouped.append(current)
    return grouped


def _merge_short_tail_chunks(chunks: list[Chunk], *, min_chunk_size: int) -> list[Chunk]:
    if len(chunks) < 2:
        return chunks
    merged: list[Chunk] = []
    for chunk in chunks:
        if merged and chunk.text_length < min_chunk_size:
            previous = merged.pop()
            merged_text = previous.text + chunk.text
            merged.append(
                Chunk(
                    chunk_id=previous.chunk_id,
                    doc_id=previous.doc_id,
                    chunk_index=previous.chunk_index,
                    chunk_method=previous.chunk_method,
                    text=merged_text,
                    text_length=len(merged_text),
                    start_char=previous.start_char,
                    end_char=chunk.end_char,
                    title=previous.title,
                    source_file=previous.source_file,
                    publish_source=previous.publish_source,
                    canonical_source=previous.canonical_source,
                    published_year=previous.published_year,
                    section_path=_merge_paths(previous.section_path, chunk.section_path),
                    structure_labels=sorted(
                        set(previous.structure_labels + chunk.structure_labels)
                    ),
                    quality_flags=previous.quality_flags,
                    metadata={
                        **previous.metadata,
                        "end_heading": chunk.metadata.get("end_heading"),
                    },
                )
            )
            continue
        merged.append(chunk)
    return _reset_chunk_indexes(merged)


def _reset_chunk_indexes(chunks: list[Chunk]) -> list[Chunk]:
    normalized: list[Chunk] = []
    for index, chunk in enumerate(chunks):
        normalized.append(
            Chunk(
                chunk_id=f"{chunk.doc_id}::structure::{index}",
                doc_id=chunk.doc_id,
                chunk_index=index,
                chunk_method=chunk.chunk_method,
                text=chunk.text,
                text_length=chunk.text_length,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                title=chunk.title,
                source_file=chunk.source_file,
                publish_source=chunk.publish_source,
                canonical_source=chunk.canonical_source,
                published_year=chunk.published_year,
                section_path=chunk.section_path,
                structure_labels=chunk.structure_labels,
                quality_flags=chunk.quality_flags,
                metadata=chunk.metadata,
            )
        )
    return normalized


def _resolve_path(segments: list[Segment]) -> list[str]:
    for segment in reversed(segments):
        if segment.path:
            return segment.path
    return []


def _extract_overlap_text(text: str, sentence_overlap: int) -> str:
    if sentence_overlap <= 0:
        return ""
    sentences = [match.group(0).strip() for match in SENTENCE_RE.finditer(text) if match.group(0).strip()]
    if not sentences:
        return ""
    return "".join(sentences[-sentence_overlap:])


def _heading_level(label: str) -> int | None:
    if "编" in label:
        return 0
    if "章" in label:
        return 1
    if "节" in label:
        return 2
    if "条" in label:
        return 3
    if "款" in label:
        return 4
    if "项" in label or label.endswith("、") or label.startswith("（"):
        return 5
    return None


def _should_flush_before_segment(
    current_segments: list[Segment],
    next_segment: Segment,
) -> bool:
    if next_segment.level is None or next_segment.level > 3:
        return False
    current_path = _resolve_path(current_segments)
    if not current_path or current_path == next_segment.path:
        return False
    return any(segment.level is not None and segment.level >= 3 for segment in current_segments)


def _merge_paths(left: list[str], right: list[str]) -> list[str]:
    merged: list[str] = []
    for label in [*left, *right]:
        if label and label not in merged:
            merged.append(label)
    return merged


def _build_structure_chunk(
    *,
    document: Document,
    chunk_index: int,
    text: str,
    start_char: int,
    end_char: int,
    section_path: list[str],
) -> Chunk:
    structure_labels = sorted(set(label for label in section_path if label))
    article_label = next((label for label in reversed(section_path) if "条" in label), None)
    return Chunk(
        chunk_id=f"{document.doc_id}::structure::{chunk_index}",
        doc_id=document.doc_id,
        chunk_index=chunk_index,
        chunk_method="structure",
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
            "article_label": article_label,
            "heading_depth": len(section_path),
            "sub_title": document.sub_title,
            "intro_title": document.intro_title,
            "end_heading": section_path[-1] if section_path else None,
            "heading_prefix": " > ".join(section_path),
        },
    )
