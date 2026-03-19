from __future__ import annotations

import re

from legal_rag.audit.text_utils import extract_year, normalize_text, normalize_title
from legal_rag.schemas.document import Document, RawRecord


ARTICLE_HINT_RE = re.compile(
    r"(第[一二三四五六七八九十百千0-9]+[条章节编款项]|一、|二、|三、|\([一二三四五六七八九十]\))"
)


def build_document(raw: RawRecord, year_pattern: re.Pattern[str]) -> Document:
    title = normalize_title(raw.title)
    body = normalize_text(raw.content_text or "")
    structure_hints = _extract_structure_hints(body)
    quality_flags: list[str] = []

    if not body:
        quality_flags.append("empty_body")
    if len(body) < 30 and body:
        quality_flags.append("short_body")

    published_year = extract_year(raw.data_time, year_pattern)
    doc_id = f"{raw.source_file}:{raw.record_index}"
    return Document(
        doc_id=doc_id,
        source_file=raw.source_file,
        source_record_index=raw.record_index,
        title=title,
        cleaned_title=None,
        sub_title=normalize_title(raw.sub_title) or None,
        intro_title=normalize_title(raw.intro_title) or None,
        publish_source=normalize_title(raw.publish_source) or None,
        canonical_source=None,
        published_at_raw=normalize_title(raw.data_time) or None,
        published_year=published_year,
        content_text=raw.content_text or "",
        normalized_text=body,
        cleaned_text=None,
        language="zh",
        text_length=len(body),
        char_count_no_space=len(body.replace(" ", "")),
        structure_hints=structure_hints,
        quality_flags=quality_flags,
        metadata=raw.extra.copy(),
    )


def _extract_structure_hints(text: str) -> list[str]:
    hints = sorted(set(match.group(0) for match in ARTICLE_HINT_RE.finditer(text)))
    return hints[:20]
