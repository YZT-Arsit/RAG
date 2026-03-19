from __future__ import annotations

from pathlib import Path

from legal_rag.config.loader import load_yaml_config
from legal_rag.config.schema import AuditConfig


def test_load_yaml_config(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "input_paths: []\noutput_dir: out\nnormalized_output_path: norm.jsonl\ndetail_csv_path: details.csv\nreport_path: report.md\n",
        encoding="utf-8",
    )
    raw = load_yaml_config(path)
    config = AuditConfig.model_validate(raw)

    assert config.output_dir == Path("out")
