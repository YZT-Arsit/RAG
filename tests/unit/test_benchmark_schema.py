from __future__ import annotations

import json
from pathlib import Path

from legal_rag.benchmark.loader import iter_benchmark_records
from legal_rag.benchmark.validators import validate_benchmark


def test_load_and_validate_benchmark(tmp_path: Path) -> None:
    path = tmp_path / "benchmark.jsonl"
    path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "question": "问题",
                "question_type": "definition",
                "answerable": True,
                "gold_answer": "答案",
                "gold_evidence": [{"chunk_id": "c1", "doc_id": "d1"}],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    records = list(iter_benchmark_records(path))
    summary = validate_benchmark(records)
    assert len(records) == 1
    assert summary["is_valid"] is True
