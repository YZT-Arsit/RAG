from __future__ import annotations

from legal_rag.chunking.structure_aware import chunk_document_structure_aware
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
        structure_hints=["第一章", "第一条"],
        quality_flags=[],
    )


def test_structure_chunker_uses_heading_boundaries() -> None:
    text = "第一章 总则第一条 为了规范行为。第二条 明确职责。第二章 附则第三条 自2024年施行。"
    chunks = chunk_document_structure_aware(
        make_document(text), max_chunk_size=20, min_chunk_size=5
    )
    assert len(chunks) >= 2
    assert chunks[0].chunk_method == "structure"
    assert any(
        label.startswith("第一章") or label.startswith("第一条")
        for label in chunks[0].structure_labels
    )


def test_structure_chunker_resets_hierarchy_between_chapters() -> None:
    text = "第一章 总则第一条 甲。第二章 附则第三条 乙。"
    chunks = chunk_document_structure_aware(
        make_document(text), max_chunk_size=12, min_chunk_size=2
    )
    assert any(chunk.section_path == ["第二章", "第三条"] for chunk in chunks)


def test_structure_chunker_splits_long_article_with_sentence_overlap() -> None:
    text = (
        "第一条 为了规范行为，制定本规定。"
        "本条适用于机场建设管理。"
        "违反规定的，应当依法处理。"
    )
    chunks = chunk_document_structure_aware(
        make_document(text),
        max_chunk_size=18,
        min_chunk_size=5,
        sentence_overlap=1,
    )
    assert len(chunks) >= 2
    assert "本条适用于机场建设管理。" in chunks[1].text
    assert chunks[0].metadata["article_label"] == "第一条"
