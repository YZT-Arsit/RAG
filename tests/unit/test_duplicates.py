from __future__ import annotations

from legal_rag.audit.duplicates import find_exact_duplicates
from legal_rag.schemas.document import Document


def make_doc(doc_id: str, title: str, text: str) -> Document:
    return Document(
        doc_id=doc_id,
        source_file="file",
        source_record_index=0,
        title=title,
        cleaned_title=None,
        sub_title=None,
        intro_title=None,
        publish_source=None,
        canonical_source=None,
        published_at_raw=None,
        published_year=None,
        content_text=text,
        normalized_text=text,
        cleaned_text=None,
        language="zh",
        text_length=len(text),
        char_count_no_space=len(text),
        structure_hints=[],
        quality_flags=[],
        cleaning_actions=[],
    )


def test_find_exact_duplicates_by_title() -> None:
    docs = [
        make_doc("a", "同一标题", "正文1"),
        make_doc("b", "同一标题", "正文2"),
        make_doc("c", "不同标题", "正文3"),
    ]
    duplicates = find_exact_duplicates(docs, field="title")
    assert len(duplicates) == 1
    assert duplicates[0].doc_ids == ["a", "b"]
