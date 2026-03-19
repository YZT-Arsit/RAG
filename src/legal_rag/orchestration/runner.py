from __future__ import annotations

import csv
import json
import os
import traceback
import zipfile
from pathlib import Path

from legal_rag.chunking.service import run_chunking
from legal_rag.config.schema import (
    ChunkingConfig,
    ContextProcessingConfig,
    ErrorAnalysisConfig,
    ExperimentMatrixConfig,
    GenerationConfig,
    GenerationEvalConfig,
    RetrievalConfig,
    RetrievalEvalConfig,
)
from legal_rag.contexting.service import run_context_processing
from legal_rag.error_analysis.service import run_error_analysis
from legal_rag.evaluation.service import (
    run_generation_evaluation,
    run_retrieval_evaluation,
)
from legal_rag.generation.service import run_generation
from legal_rag.orchestration.matrix import ExperimentVariant, expand_default_matrix
from legal_rag.retrieval.service import run_retrieval


def run_experiment_matrix(config: ExperimentMatrixConfig) -> None:
    experiment_root = config.output_root / config.experiment_name
    experiment_root.mkdir(parents=True, exist_ok=True)
    _validate_matrix_config(config)

    summary_rows: list[dict[str, str]] = []
    for variant in expand_default_matrix(config):
        variant_dir = experiment_root / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        _write_variant_snapshot(config, variant, variant_dir / "config_snapshot.json")
        failure = _run_variant_with_failure_capture(config, variant, variant_dir)
        summary_rows.append(_build_summary_row(variant, variant_dir, failure))

    _write_experiment_summary_json(
        experiment_root / "experiment_summary.json", summary_rows
    )
    _write_experiment_summary_csv(
        experiment_root / "experiment_summary.csv", summary_rows
    )
    _write_experiment_summary_markdown(
        experiment_root / "experiment_summary.md", summary_rows
    )
    _write_artifact_bundle(experiment_root, summary_rows, config.bundle_filename)


def _run_variant_with_failure_capture(
    config: ExperimentMatrixConfig, variant: ExperimentVariant, variant_dir: Path
) -> dict[str, str] | None:
    try:
        _run_variant(config, variant, variant_dir)
    except Exception as exc:
        failure = {
            "status": "failed",
            "failure_stage": _infer_failure_stage(exc),
            "failure_type": exc.__class__.__name__,
            "failure_reason": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_variant_failure(variant_dir / "failure.json", failure)
        _write_variant_failure_markdown(
            variant_dir / "failure_report.md", variant.name, failure
        )
        return failure
    return None


def _run_variant(
    config: ExperimentMatrixConfig, variant: ExperimentVariant, variant_dir: Path
) -> None:
    chunk_config = ChunkingConfig(
        input_jsonl=config.cleaned_input_jsonl,
        output_jsonl=variant_dir / "chunks.jsonl",
        report_path=variant_dir / "chunking_report.md",
        method=variant.chunk_method,
        fixed_chunk_size=config.chunk_fixed_chunk_size,
        fixed_chunk_overlap=config.chunk_fixed_overlap,
        structure_max_chunk_size=config.chunk_structure_max_size,
        structure_min_chunk_size=config.chunk_structure_min_size,
    )
    run_chunking(chunk_config)

    retrieval_config = RetrievalConfig(
        chunk_jsonl=chunk_config.output_jsonl,
        query_jsonl=config.query_jsonl,
        output_jsonl=variant_dir / "retrieval_results.jsonl",
        report_path=variant_dir / "retrieval_report.md",
        method=variant.retrieval_method,
        top_k=config.retrieval_top_k,
        retrieve_top_k=config.retrieval_first_stage_top_k,
        hybrid_fusion=config.hybrid_fusion,
        hybrid_alpha=config.hybrid_alpha,
        rrf_k=config.rrf_k,
        reranker_enabled=variant.reranker_enabled,
    )
    run_retrieval(retrieval_config)

    context_config = ContextProcessingConfig(
        input_retrieval_results_jsonl=retrieval_config.output_jsonl,
        output_jsonl=variant_dir / "processed_contexts.jsonl",
        report_path=variant_dir / "context_report.md",
        max_chunks=config.context_max_chunks,
        max_per_doc=config.context_max_per_doc,
        max_sentences_total=config.context_max_sentences_total,
        max_sentences_per_chunk=config.context_max_sentences_per_chunk,
    )
    run_context_processing(context_config)

    generation_config = GenerationConfig(
        retrieval_results_jsonl=context_config.output_jsonl
        if variant.context_source == "processed"
        else retrieval_config.output_jsonl,
        output_jsonl=variant_dir / "grounded_answers.jsonl",
        report_path=variant_dir / "generation_report.md",
        method=variant.generation_method,
        context_source=variant.context_source,
        max_contexts=config.generation_max_contexts,
        max_answer_sentences=config.generation_max_answer_sentences,
        max_citation_chars=config.generation_max_citation_chars,
        max_prompt_context_chars=config.generation_max_prompt_context_chars,
        min_hit_score=config.generation_min_hit_score,
        llm_backend=config.generation_llm_backend,
        llm_base_url=config.generation_llm_base_url,
        llm_api_key_env=config.generation_llm_api_key_env,
        llm_model_name=config.generation_llm_model_name,
        llm_modelscope_model_id=config.generation_llm_modelscope_model_id,
        llm_local_model_dir=config.generation_llm_local_model_dir,
        llm_use_modelscope_download=config.generation_llm_use_modelscope_download,
        llm_device=config.generation_llm_device,
        llm_temperature=config.generation_llm_temperature,
        llm_timeout_seconds=config.generation_llm_timeout_seconds,
        llm_max_new_tokens=config.generation_llm_max_new_tokens,
        llm_prompt_version=config.generation_llm_prompt_version,
        llm_require_context_ids=config.generation_llm_require_context_ids,
        llm_abstain_when_insufficient=config.generation_llm_abstain_when_insufficient,
    )
    run_generation(generation_config)

    retrieval_eval_config = RetrievalEvalConfig(
        retrieval_results_jsonl=retrieval_config.output_jsonl,
        gold_jsonl=config.benchmark_jsonl,
        detail_csv_path=variant_dir / "retrieval_eval_details.csv",
        report_path=variant_dir / "retrieval_eval_report.md",
        benchmark_mode=True,
        ks=config.eval_ks,
    )
    run_retrieval_evaluation(retrieval_eval_config)

    generation_eval_config = GenerationEvalConfig(
        answers_jsonl=generation_config.output_jsonl,
        gold_jsonl=config.benchmark_jsonl,
        detail_csv_path=variant_dir / "generation_eval_details.csv",
        report_path=variant_dir / "generation_eval_report.md",
        benchmark_mode=True,
    )
    run_generation_evaluation(generation_eval_config)

    error_config = ErrorAnalysisConfig(
        benchmark_jsonl=config.benchmark_jsonl,
        retrieval_results_jsonl=retrieval_config.output_jsonl,
        answers_jsonl=generation_config.output_jsonl,
        detail_csv_path=variant_dir / "error_analysis_details.csv",
        report_path=variant_dir / "error_analysis_report.md",
    )
    run_error_analysis(error_config)


def _build_summary_row(
    variant: ExperimentVariant, variant_dir: Path, failure: dict[str, str] | None
) -> dict[str, str]:
    generation_summary = _parse_summary_metrics(
        variant_dir / "generation_eval_report.md"
    )
    error_counts = _parse_error_counts(variant_dir / "error_analysis_report.md")
    question_type_summary = _parse_grouped_metrics(
        variant_dir / "generation_eval_report.md"
    )
    return {
        "variant": variant.name,
        "chunk_method": variant.chunk_method,
        "retrieval_method": variant.retrieval_method,
        "reranker_enabled": str(variant.reranker_enabled),
        "generation_method": variant.generation_method,
        "context_source": variant.context_source,
        "status": "failed" if failure else "completed",
        "failure_stage": failure["failure_stage"] if failure else "",
        "failure_reason": failure["failure_reason"] if failure else "",
        "token_f1": _format_metric(generation_summary.get("token_f1")),
        "citation_precision": _format_metric(
            generation_summary.get("citation_precision")
        ),
        "citation_recall": _format_metric(generation_summary.get("citation_recall")),
        "abstain_rate": _format_metric(generation_summary.get("abstained")),
        "generation_hallucination_count": str(
            error_counts.get("generation_hallucination", 0)
        ),
        "ranking_miss_count": str(error_counts.get("ranking_miss", 0)),
        "question_type_summary": json.dumps(question_type_summary, ensure_ascii=False),
        "config_snapshot": str(variant_dir / "config_snapshot.json"),
        "generation_report": str(variant_dir / "generation_eval_report.md"),
        "error_analysis_report": str(variant_dir / "error_analysis_report.md"),
        "failure_report": str(variant_dir / "failure_report.md"),
    }


def _write_variant_snapshot(
    config: ExperimentMatrixConfig, variant: ExperimentVariant, path: Path
) -> None:
    payload = {
        "experiment": config.model_dump(mode="json"),
        "variant": {
            "name": variant.name,
            "chunk_method": variant.chunk_method,
            "retrieval_method": variant.retrieval_method,
            "reranker_enabled": variant.reranker_enabled,
            "generation_method": variant.generation_method,
            "context_source": variant.context_source,
        },
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_variant_failure(path: Path, failure: dict[str, str]) -> None:
    path.write_text(json.dumps(failure, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_variant_failure_markdown(
    path: Path, variant_name: str, failure: dict[str, str]
) -> None:
    lines = [
        "# Variant Failure Report",
        "",
        f"- `variant`: {variant_name}",
        f"- `status`: {failure['status']}",
        f"- `failure_stage`: {failure['failure_stage']}",
        f"- `failure_type`: {failure['failure_type']}",
        f"- `failure_reason`: {failure['failure_reason']}",
        "",
        "## Traceback",
        "",
        "```text",
        failure["traceback"].rstrip(),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_experiment_summary_json(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def _validate_matrix_config(config: ExperimentMatrixConfig) -> None:
    generation_methods = config.generation_methods or [config.generation_method]
    if "llm" not in generation_methods:
        return
    if config.generation_llm_backend == "openai_compatible":
        if not config.generation_llm_base_url or not config.generation_llm_model_name:
            msg = (
                "Experiment matrix includes llm generation but generation_llm_base_url "
                "or generation_llm_model_name is missing."
            )
            raise ValueError(msg)
        if not os.getenv(config.generation_llm_api_key_env):
            msg = (
                "Experiment matrix includes llm generation but the required API key "
                f"environment variable {config.generation_llm_api_key_env} is not set."
            )
            raise ValueError(msg)
        return
    if (
        not config.generation_llm_local_model_dir
        and not config.generation_llm_modelscope_model_id
    ):
        msg = (
            "Experiment matrix includes local llm generation but neither "
            "generation_llm_local_model_dir nor generation_llm_modelscope_model_id is configured."
        )
        raise ValueError(msg)


def _parse_summary_metrics(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics
    in_summary = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "## Summary Metrics":
            in_summary = True
            continue
        if in_summary and stripped.startswith("## "):
            break
        if in_summary and stripped.startswith("- `") and ": " in stripped:
            left, right = stripped.split(": ", maxsplit=1)
            key = left.replace("- `", "").replace("`", "")
            try:
                metrics[key] = float(right)
            except ValueError:
                continue
    return metrics


def _parse_error_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not path.exists():
        return counts
    in_error_counts = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "## Error Counts":
            in_error_counts = True
            continue
        if in_error_counts and stripped.startswith("## "):
            break
        if in_error_counts and stripped.startswith("- `") and ": " in stripped:
            left, right = stripped.split(": ", maxsplit=1)
            key = left.replace("- `", "").replace("`", "")
            try:
                counts[key] = int(right)
            except ValueError:
                continue
    return counts


def _parse_grouped_metrics(path: Path) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    if not path.exists():
        return grouped
    current_group: str | None = None
    in_grouped = False
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "## Grouped Metrics":
            in_grouped = True
            continue
        if (
            in_grouped
            and stripped.startswith("## ")
            and stripped != "## Grouped Metrics"
        ):
            break
        if in_grouped and stripped.startswith("### "):
            current_group = stripped.replace("### ", "", 1)
            grouped[current_group] = {}
            continue
        if (
            in_grouped
            and current_group
            and stripped.startswith("- `")
            and ": " in stripped
        ):
            left, right = stripped.split(": ", maxsplit=1)
            key = left.replace("- `", "").replace("`", "")
            try:
                grouped[current_group][key] = float(right)
            except ValueError:
                continue
    return grouped


def _format_metric(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}"


def _write_experiment_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant_name",
        "generation_method",
        "context_source",
        "status",
        "failure_stage",
        "failure_reason",
        "token_f1",
        "citation_precision",
        "citation_recall",
        "abstain_rate",
        "generation_hallucination_count",
        "ranking_miss_count",
        "question_type_summary",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "variant_name": row["variant"],
                    "generation_method": row["generation_method"],
                    "context_source": row["context_source"],
                    "status": row.get("status", ""),
                    "failure_stage": row.get("failure_stage", ""),
                    "failure_reason": row.get("failure_reason", ""),
                    "token_f1": row.get("token_f1", ""),
                    "citation_precision": row.get("citation_precision", ""),
                    "citation_recall": row.get("citation_recall", ""),
                    "abstain_rate": row.get("abstain_rate", ""),
                    "generation_hallucination_count": row.get(
                        "generation_hallucination_count", ""
                    ),
                    "ranking_miss_count": row.get("ranking_miss_count", ""),
                    "question_type_summary": row.get("question_type_summary", ""),
                }
            )


def _write_experiment_summary_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Experiment Summary",
        "",
        "## Status Labels",
        "",
        "- Implemented: one-shot server-side experiment execution, per-variant failure capture, summary outputs, and final artifact bundle export.",
        "- Experimental: current summary metrics rely on benchmark-scale evaluation and rule-based error labels.",
        "- Planned: resume support, parallel execution, and richer leaderboard reporting.",
        "",
        "## Variants",
        "",
        "| Variant | Status | Generator | Contexts | token_f1 | citation_precision | citation_recall | abstain_rate | hallucination | ranking_miss | Failure |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row.get('status', '')} | {row['generation_method']} | {row['context_source']} | "
            f"{row.get('token_f1', '')} | {row.get('citation_precision', '')} | {row.get('citation_recall', '')} | "
            f"{row.get('abstain_rate', '')} | {row.get('generation_hallucination_count', '')} | {row.get('ranking_miss_count', '')} | {row.get('failure_reason', '')} |"
        )
        lines.append(f"Question types: {row.get('question_type_summary', '{}')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_artifact_bundle(
    experiment_root: Path, rows: list[dict[str, str]], bundle_filename: str
) -> None:
    bundle_path = experiment_root / bundle_filename
    include_paths = [
        experiment_root / "experiment_summary.json",
        experiment_root / "experiment_summary.csv",
        experiment_root / "experiment_summary.md",
    ]
    for row in rows:
        include_paths.extend(
            [
                Path(row["config_snapshot"]),
                Path(row["generation_report"]),
                Path(row["error_analysis_report"]),
                Path(row["failure_report"]),
            ]
        )

    with zipfile.ZipFile(
        bundle_path, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as handle:
        for path in include_paths:
            if not path.exists():
                continue
            handle.write(path, arcname=path.relative_to(experiment_root))


def _infer_failure_stage(exc: Exception) -> str:
    tb = exc.__traceback__
    last_module = ""
    while tb is not None:
        last_module = tb.tb_frame.f_globals.get("__name__", "")
        tb = tb.tb_next
    if "chunk" in last_module:
        return "chunking"
    if "retrieval" in last_module:
        return "retrieval"
    if "context" in last_module:
        return "context_processing"
    if "generation" in last_module:
        return "generation"
    if "evaluation" in last_module:
        return "evaluation"
    if "error_analysis" in last_module:
        return "error_analysis"
    return "unknown"
