from __future__ import annotations

from pathlib import Path

from legal_rag.audit.models import AuditSummary
from legal_rag.audit.report import write_markdown_report


def test_write_markdown_report(tmp_path: Path) -> None:
    path = tmp_path / "report.md"
    summary = AuditSummary(total_records=1, parsed_records=1)
    write_markdown_report(path, summary)
    content = path.read_text(encoding="utf-8")
    assert "# Dataset Audit Report" in content
