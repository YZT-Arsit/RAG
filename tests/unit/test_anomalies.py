from __future__ import annotations

from legal_rag.audit.anomalies import detect_anomalies
from legal_rag.schemas.document import Document


def test_detects_empty_body_anomaly() -> None:
    doc = Document(
        doc_id="x",
        source_file="f",
        source_record_index=0,
        title="t",
        cleaned_title=None,
        sub_title=None,
        intro_title=None,
        publish_source=None,
        canonical_source=None,
        published_at_raw=None,
        published_year=None,
        content_text="",
        normalized_text="",
        cleaned_text=None,
        language="zh",
        text_length=0,
        char_count_no_space=0,
        structure_hints=[],
        quality_flags=[],
        cleaning_actions=[],
    )
    anomalies = detect_anomalies(
        doc, short_body_threshold=30, template_similarity_line_threshold=0.6
    )
    assert anomalies[0].anomaly_type == "empty_body"
