from __future__ import annotations

from pathlib import Path

from legal_rag.cleaning.rules import apply_cleaning
from legal_rag.config.schema import CleaningConfig
from legal_rag.schemas.document import Document


def test_apply_cleaning_removes_prefix_and_maps_source(tmp_path: Path) -> None:
    config = CleaningConfig(
        input_jsonl=tmp_path / "in.jsonl",
        output_jsonl=tmp_path / "out.jsonl",
        report_path=tmp_path / "report.md",
        source_aliases={"法治日报-法治网": "法治网"},
        body_leading_patterns=[r"^\s*本文转自[:：]\s*\S+\s+"],
        body_global_patterns=[r"\s+"],
        min_cleaned_length=10,
    )
    doc = Document(
        doc_id="x",
        source_file="f",
        source_record_index=0,
        title="标题",
        cleaned_title=None,
        sub_title=None,
        intro_title=None,
        publish_source="法治日报-法治网",
        canonical_source=None,
        published_at_raw=None,
        published_year=2024,
        content_text="本文转自：法治网 正文内容",
        normalized_text="本文转自：法治网 正文内容",
        cleaned_text=None,
        language="zh",
        text_length=12,
        char_count_no_space=12,
        structure_hints=[],
        quality_flags=[],
        cleaning_actions=[],
    )
    cleaned = apply_cleaning(doc, config)
    assert cleaned.canonical_source == "法治网"
    assert cleaned.cleaned_text == "正文内容"
    assert "source_alias_applied" in cleaned.cleaning_actions
