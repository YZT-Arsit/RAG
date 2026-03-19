from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from legal_rag.config.schema import ExperimentMatrixConfig
from legal_rag.orchestration.matrix import expand_default_matrix
from legal_rag.orchestration import runner
from legal_rag.orchestration.runner import (
    _validate_matrix_config,
    run_experiment_matrix,
)


def test_expand_default_matrix_crosses_generation_dims() -> None:
    config = ExperimentMatrixConfig(
        experiment_name="t",
        cleaned_input_jsonl=Path("cleaned.jsonl"),
        query_jsonl=Path("queries.jsonl"),
        benchmark_jsonl=Path("benchmark.jsonl"),
        output_root=Path("out"),
        generation_methods=["extractive"],
        generation_context_sources=["raw", "processed"],
    )
    variants = expand_default_matrix(config)
    assert len(variants) == 12
    assert variants[0].name == "fixed_dense__gen_extractive__ctx_raw"
    assert variants[-1].name == "structure_hybrid_rerank__gen_extractive__ctx_processed"


def test_validate_matrix_config_requires_llm_api_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    config = ExperimentMatrixConfig(
        experiment_name="t",
        cleaned_input_jsonl=Path("cleaned.jsonl"),
        query_jsonl=Path("queries.jsonl"),
        benchmark_jsonl=Path("benchmark.jsonl"),
        output_root=Path("out"),
        generation_methods=["llm"],
        generation_llm_base_url="https://open.bigmodel.cn/api/paas/v4",
        generation_llm_model_name="glm-4.7-flash",
        generation_llm_api_key_env="ZAI_API_KEY",
    )
    with pytest.raises(ValueError, match="ZAI_API_KEY"):
        _validate_matrix_config(config)


def test_validate_matrix_config_accepts_local_transformers_without_api_env() -> None:
    config = ExperimentMatrixConfig(
        experiment_name="t",
        cleaned_input_jsonl=Path("cleaned.jsonl"),
        query_jsonl=Path("queries.jsonl"),
        benchmark_jsonl=Path("benchmark.jsonl"),
        output_root=Path("out"),
        generation_methods=["llm"],
        generation_llm_backend="local_transformers",
        generation_llm_modelscope_model_id="ZhipuAI/glm-edge-1.5b-chat",
        generation_llm_use_modelscope_download=True,
    )
    _validate_matrix_config(config)


def test_run_experiment_matrix_writes_bundle_and_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = ExperimentMatrixConfig(
        experiment_name="bundle_test",
        cleaned_input_jsonl=Path("cleaned.jsonl"),
        query_jsonl=Path("queries.jsonl"),
        benchmark_jsonl=Path("benchmark.jsonl"),
        output_root=tmp_path,
        matrix_scope="generation_only",
        generation_methods=["extractive"],
        generation_context_sources=["processed"],
    )

    def fake_run_variant(
        cfg: ExperimentMatrixConfig, variant, variant_dir: Path
    ) -> None:
        (variant_dir / "generation_eval_report.md").write_text(
            "\n".join(
                [
                    "# Generation Evaluation Report",
                    "",
                    "## Summary Metrics",
                    "",
                    "- `token_f1`: 0.5000",
                    "- `citation_precision`: 0.6000",
                    "- `citation_recall`: 0.7000",
                    "- `abstained`: 0.0000",
                    "",
                    "## Grouped Metrics",
                    "",
                    "### definition",
                    "- `token_f1`: 0.5000",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (variant_dir / "error_analysis_report.md").write_text(
            "\n".join(
                [
                    "# Error Analysis Report",
                    "",
                    "## Error Counts",
                    "",
                    "- `generation_hallucination`: 1",
                    "- `ranking_miss`: 2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(runner, "_run_variant", fake_run_variant)
    run_experiment_matrix(config)

    experiment_root = tmp_path / "bundle_test"
    summary_csv = experiment_root / "experiment_summary.csv"
    summary_md = experiment_root / "experiment_summary.md"
    bundle_path = experiment_root / "artifacts_bundle.zip"
    assert summary_csv.exists()
    assert summary_md.exists()
    assert bundle_path.exists()
    csv_text = summary_csv.read_text(encoding="utf-8")
    assert "status" in csv_text
    assert "completed" in csv_text
    with zipfile.ZipFile(bundle_path) as handle:
        names = set(handle.namelist())
    assert "experiment_summary.csv" in names
    assert "experiment_summary.md" in names
    assert (
        "structure_hybrid_rerank__gen_extractive__ctx_processed/config_snapshot.json"
        in names
    )


def test_run_experiment_matrix_records_variant_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = ExperimentMatrixConfig(
        experiment_name="failure_test",
        cleaned_input_jsonl=Path("cleaned.jsonl"),
        query_jsonl=Path("queries.jsonl"),
        benchmark_jsonl=Path("benchmark.jsonl"),
        output_root=tmp_path,
        matrix_scope="generation_only",
        generation_methods=["extractive", "llm"],
        generation_context_sources=["processed"],
        generation_llm_backend="local_transformers",
        generation_llm_modelscope_model_id="Qwen/Qwen3-8B",
    )

    def fake_run_variant(
        cfg: ExperimentMatrixConfig, variant, variant_dir: Path
    ) -> None:
        if variant.generation_method == "llm":
            raise RuntimeError("mock llm timeout")
        (variant_dir / "generation_eval_report.md").write_text(
            "# Generation Evaluation Report\n\n## Summary Metrics\n\n- `token_f1`: 0.4000\n",
            encoding="utf-8",
        )
        (variant_dir / "error_analysis_report.md").write_text(
            "# Error Analysis Report\n\n## Error Counts\n\n- `generation_hallucination`: 0\n- `ranking_miss`: 0\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(runner, "_run_variant", fake_run_variant)
    run_experiment_matrix(config)

    experiment_root = tmp_path / "failure_test"
    summary_csv = experiment_root / "experiment_summary.csv"
    failure_report = (
        experiment_root
        / "structure_hybrid_rerank__gen_llm__ctx_processed"
        / "failure_report.md"
    )
    assert failure_report.exists()
    csv_text = summary_csv.read_text(encoding="utf-8")
    assert "mock llm timeout" in csv_text
    assert "failed" in csv_text
