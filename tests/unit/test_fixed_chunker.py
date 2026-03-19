from __future__ import annotations

from legal_rag.chunking.fixed import chunk_document_fixed
from legal_rag.schemas.document import Document


def make_document(text: str) -> Document:
    return Document(
        doc_id="doc-1",
        source_file="source.jsonl",
        source_record_index=0,
        title="标题",
        cleaned_title="标题",
        sub_title=None,
        intro_title=None,
        publish_source="来源",
        canonical_source="来源",
        published_at_raw="2024-01-01",
        published_year=2024,
        content_text=text,
        normalized_text=text,
        cleaned_text=text,
        language="zh",
        text_length=len(text),
        char_count_no_space=len(text),
        structure_hints=["第一条"],
        quality_flags=[],
    )


def test_fixed_chunker_creates_overlapping_chunks() -> None:
    text = "甲" * 700
    chunks = chunk_document_fixed(make_document(text), chunk_size=300, chunk_overlap=50)
    assert len(chunks) == 3
    assert chunks[0].text_length == 300
    assert chunks[1].start_char == 250
    assert chunks[0].chunk_method == "fixed"


def test_fixed_chunker_prefers_sentence_boundaries() -> None:
    text = "第一句说明规则。第二句说明例外。第三句说明责任。"
    chunks = chunk_document_fixed(make_document(text), chunk_size=14, chunk_overlap=4)
    assert len(chunks) >= 2
    assert chunks[0].text.endswith("。")
    assert "第二句说明例外。" in chunks[1].text
