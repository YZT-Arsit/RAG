from __future__ import annotations

import json
from pathlib import Path

from legal_rag.audit.reader import iter_records


def test_iter_records_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text(
        json.dumps({"title": "A", "contentText": "正文"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    records = list(iter_records(path))
    assert len(records) == 1
    assert records[0].title == "A"


def test_iter_records_json_array(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps([{"title": "A", "contentText": "正文"}], ensure_ascii=False),
        encoding="utf-8",
    )
    records = list(iter_records(path))
    assert len(records) == 1
    assert records[0].content_text == "正文"


def test_iter_records_json_suffix_can_fallback_to_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "sample.json"
    path.write_text(
        json.dumps({"title": "A", "content": "<p>正文</p>"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    records = list(iter_records(path))
    assert len(records) == 1
    assert records[0].content_text == "<p>正文</p>"
