from __future__ import annotations

import re

from legal_rag.audit.normalize import build_document
from legal_rag.schemas.document import RawRecord


def test_build_document_extracts_structure_hint() -> None:
    record = RawRecord(
        source_file="sample.jsonl",
        record_index=0,
        title=" 测试标题 ",
        sub_title=None,
        data_time="2024-01-02",
        publish_source="机构A",
        content_text="第一条 为了规范行为。\n第二条 继续说明。",
        intro_title=None,
    )
    doc = build_document(record, re.compile(r"(19|20)\d{2}"))
    assert doc.title == "测试标题"
    assert doc.published_year == 2024
    assert "第一条" in doc.structure_hints


def test_build_document_strips_html_and_parses_timestamp_year() -> None:
    record = RawRecord(
        source_file="sample.jsonl",
        record_index=1,
        title="测试",
        sub_title=None,
        data_time="1734962960000",
        publish_source="机构A",
        content_text="<p>第一条 测试内容。</p>",
        intro_title=None,
    )
    doc = build_document(record, re.compile(r"(19|20)\d{2}"))
    assert doc.published_year == 2024
    assert doc.normalized_text == "第一条 测试内容。"
