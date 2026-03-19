from __future__ import annotations

from legal_rag.benchmark.loader import iter_benchmark_records
from legal_rag.benchmark.validators import validate_benchmark
from legal_rag.config.schema import (
    AblationConfig,
    AutoEvalConfig,
    AutoTestsetConfig,
    BenchmarkValidateConfig,
    GenerationEvalConfig,
    RetrievalEvalConfig,
)
from legal_rag.evaluation.auto_eval import run_auto_eval, run_auto_testset_generation
from legal_rag.evaluation.generation_metrics import (
    aggregate_generation_by_question_type,
    evaluate_generation,
)
from legal_rag.evaluation.io import (
    load_answers,
    load_generation_gold,
    load_retrieval_gold,
    load_retrieval_results,
)
from legal_rag.evaluation.report import write_markdown_summary, write_metric_csv


def run_retrieval_evaluation(config: RetrievalEvalConfig) -> None:
    results = load_retrieval_results(config.retrieval_results_jsonl)
    gold = load_retrieval_gold(config.gold_jsonl, benchmark_mode=config.benchmark_mode)
    per_query, summary = evaluate_retrieval(results, gold, ks=config.ks)
    write_metric_csv(config.detail_csv_path, per_query)
    write_markdown_summary(
        config.report_path,
        title="Retrieval Evaluation Report",
        implemented="Recall@k, Hit@k, MRR over labeled relevant chunk ids.",
        experimental="Current benchmark is still small-scale and manually curated.",
        planned="nDCG, graded relevance, larger benchmark coverage.",
        summary_metrics=summary,
        notes=[
            f"Queries evaluated: {len(per_query)}",
            f"Configured ks: {config.ks}",
            "Do not claim retrieval gains without comparing on the same gold set.",
        ],
    )


def run_generation_evaluation(config: GenerationEvalConfig) -> None:
    answers = load_answers(config.answers_jsonl)
    gold = load_generation_gold(config.gold_jsonl, benchmark_mode=config.benchmark_mode)
    per_query, summary = evaluate_generation(answers, gold)
    grouped_metrics = aggregate_generation_by_question_type(per_query)
    write_metric_csv(config.detail_csv_path, per_query)
    write_markdown_summary(
        config.report_path,
        title="Generation Evaluation Report",
        implemented="Answer correctness baseline, faithfulness proxy, citation precision/recall, and abstain accuracy.",
        experimental="Correctness currently uses token-overlap baseline; faithfulness uses citation alignment proxy.",
        planned="LLM-as-judge, stronger groundedness checks, richer abstain analysis.",
        summary_metrics=summary,
        notes=[
            f"Queries evaluated: {len(per_query)}",
            "Current generation metrics are baseline approximations and should not be overstated as full semantic evaluation.",
        ],
        grouped_metrics=grouped_metrics,
    )


def run_benchmark_validation(config: BenchmarkValidateConfig) -> None:
    records = list(iter_benchmark_records(config.benchmark_jsonl))
    summary = validate_benchmark(records)
    write_markdown_summary(
        config.report_path,
        title="Benchmark Validation Report",
        implemented="Formal benchmark schema validation for query ids, answerable labels, and evidence presence.",
        experimental="Validation currently checks structural consistency rather than annotation quality.",
        planned="Split validation, annotation agreement checks, evidence-text normalization.",
        summary_metrics={},
        notes=[
            f"Records validated: {summary['record_count']}",
            f"Is valid: {summary['is_valid']}",
            f"Duplicate query ids: {summary['duplicate_query_ids']}",
            f"Invalid answerable/gold answer pairs: {summary['invalid_answerable_gold_answer']}",
            f"Missing evidence for answerable items: {summary['missing_evidence_for_answerable']}",
        ],
    )


def run_ablation(config: AblationConfig) -> None:
    retrieval_rows = []
    generation_rows = []
    for variant in config.variants:
        retrieval_summary = _parse_summary_metrics(variant.retrieval_report_path)
        generation_summary = _parse_summary_metrics(variant.generation_report_path)
        retrieval_rows.append((variant.name, retrieval_summary))
        generation_rows.append((variant.name, generation_summary))

    lines = [
        "# Ablation Report",
        "",
        "## Status Labels",
        "",
        "- Implemented: Markdown comparison of variant-level summary metrics.",
        "- Experimental: Manual assembly of variant reports from existing pipeline outputs.",
        "- Planned: automated run orchestration, parameter sweep tracking, experiment registry integration.",
        "",
        "## Retrieval Variants",
        "",
    ]
    if retrieval_rows:
        for name, metrics in retrieval_rows:
            metric_text = (
                ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
                if metrics
                else "no metrics"
            )
            lines.append(f"- `{name}`: {metric_text}")
    else:
        lines.append("- No retrieval variants supplied.")

    lines.extend(["", "## Generation Variants", ""])
    if generation_rows:
        for name, metrics in generation_rows:
            metric_text = (
                ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
                if metrics
                else "no metrics"
            )
            lines.append(f"- `{name}`: {metric_text}")
    else:
        lines.append("- No generation variants supplied.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This report compares supplied summary files only. It does not assert causality or improvement beyond the recorded metrics.",
        ]
    )
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_summary_metrics(path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("- `") or ": " not in stripped:
            continue
        left, right = stripped.split(": ", maxsplit=1)
        key = left.replace("- `", "").replace("`", "")
        try:
            metrics[key] = float(right)
        except ValueError:
            continue
    return metrics


from legal_rag.evaluation.retrieval_metrics import evaluate_retrieval
