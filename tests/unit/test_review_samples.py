from __future__ import annotations

import csv
import json
from pathlib import Path

from legal_rag.cleaning.review import export_review_samples
from legal_rag.config.schema import ReviewSampleConfig


def test_export_review_samples(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "docs.jsonl"
    input_jsonl.write_text(
        json.dumps(
            {
                "doc_id": "doc-1",
                "source_file": "f",
                "source_record_index": 0,
                "title": "标题",
                "cleaned_title": "标题",
                "sub_title": None,
                "intro_title": None,
                "publish_source": "来源",
                "canonical_source": "来源",
                "published_at_raw": "2024-01-01",
                "published_year": 2024,
                "content_text": "正文",
                "normalized_text": "正文",
                "cleaned_text": "正文",
                "language": "zh",
                "text_length": 2,
                "char_count_no_space": 2,
                "structure_hints": [],
                "quality_flags": [],
                "cleaning_actions": [],
                "metadata": {},
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    detail_csv = tmp_path / "detail.csv"
    with detail_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["record_type", "key", "doc_ids", "severity", "note"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "record_type": "exact_title_duplicate",
                "key": "k",
                "doc_ids": "doc-1",
                "severity": "",
                "note": "",
            }
        )

    output_csv = tmp_path / "samples.csv"
    config = ReviewSampleConfig(
        input_jsonl=input_jsonl,
        detail_csv=detail_csv,
        output_csv=output_csv,
        random_sample_size=1,
        anomaly_sample_size=0,
        duplicate_sample_size=1,
    )
    export_review_samples(config)

    content = output_csv.read_text(encoding="utf-8")
    assert "sample_type" in content
    assert "doc-1" in content
