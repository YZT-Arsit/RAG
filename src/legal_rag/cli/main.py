from __future__ import annotations

import argparse
from pathlib import Path

from legal_rag.audit.service import run_audit
from legal_rag.chunking.service import run_chunking
from legal_rag.cleaning.review import export_review_samples
from legal_rag.cleaning.service import run_cleaning
from legal_rag.config.loader import load_yaml_config
from legal_rag.config.schema import (
    AblationConfig,
    AuditConfig,
    AutoEvalConfig,
    AutoTestsetConfig,
    BenchmarkGenerationConfig,
    BenchmarkValidateConfig,
    ChunkingConfig,
    CleaningConfig,
    ContextProcessingConfig,
    ErrorAnalysisConfig,
    ExperimentMatrixConfig,
    GenerationConfig,
    GenerationEvalConfig,
    LLMGenerationDebugConfig,
    RetrievalConfig,
    RetrievalEvalConfig,
    ReviewSampleConfig,
)
from legal_rag.contexting.service import run_context_processing
from legal_rag.error_analysis.service import run_error_analysis
from legal_rag.evaluation.service import (
    run_ablation,
    run_auto_eval,
    run_auto_testset_generation,
    run_benchmark_validation,
    run_generation_evaluation,
    run_retrieval_evaluation,
)
from legal_rag.benchmark.generation import run_benchmark_generation
from legal_rag.generation.service import run_generation, run_llm_generation_debug
from legal_rag.orchestration.runner import run_experiment_matrix
from legal_rag.retrieval.service import run_retrieval


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="legal-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser(
        "audit", help="Run dataset audit on JSON/JSONL files."
    )
    audit_parser.add_argument(
        "--config", required=True, help="Path to audit YAML config."
    )

    clean_parser = subparsers.add_parser(
        "clean", help="Run cleaning pipeline on normalized JSONL."
    )
    clean_parser.add_argument(
        "--config", required=True, help="Path to cleaning YAML config."
    )

    review_parser = subparsers.add_parser(
        "review-sample",
        help="Export review samples from cleaned JSONL and audit detail CSV.",
    )
    review_parser.add_argument(
        "--config", required=True, help="Path to review sample YAML config."
    )

    chunk_parser = subparsers.add_parser("chunk", help="Run chunking on cleaned JSONL.")
    chunk_parser.add_argument(
        "--config", required=True, help="Path to chunking YAML config."
    )

    retrieve_parser = subparsers.add_parser(
        "retrieve", help="Run retrieval over chunk JSONL."
    )
    retrieve_parser.add_argument(
        "--config", required=True, help="Path to retrieval YAML config."
    )

    generate_parser = subparsers.add_parser(
        "generate", help="Run grounded generation over retrieval results."
    )
    generate_parser.add_argument(
        "--config", required=True, help="Path to generation YAML config."
    )

    debug_llm_parser = subparsers.add_parser(
        "debug-llm-generation",
        help="Run LLM grounded generation for a single query and write detailed debug output.",
    )
    debug_llm_parser.add_argument(
        "--config", required=True, help="Path to LLM generation debug YAML config."
    )

    context_parser = subparsers.add_parser(
        "process-contexts",
        help="Select and compress retrieved contexts before generation.",
    )
    context_parser.add_argument(
        "--config", required=True, help="Path to context processing YAML config."
    )

    eval_retrieval_parser = subparsers.add_parser(
        "eval-retrieval", help="Evaluate retrieval results against gold chunk ids."
    )
    eval_retrieval_parser.add_argument(
        "--config", required=True, help="Path to retrieval evaluation YAML config."
    )

    eval_generation_parser = subparsers.add_parser(
        "eval-generation", help="Evaluate grounded answers against gold answers."
    )
    eval_generation_parser.add_argument(
        "--config", required=True, help="Path to generation evaluation YAML config."
    )

    auto_testset_parser = subparsers.add_parser(
        "generate-auto-testset",
        help="Generate an LLM-based legal QA testset from chunked corpus data.",
    )
    auto_testset_parser.add_argument(
        "--config", required=True, help="Path to automated testset YAML config."
    )

    auto_eval_parser = subparsers.add_parser(
        "run-auto-eval",
        help="Run automated RAG evaluation for one or more configured variants.",
    )
    auto_eval_parser.add_argument(
        "--config", required=True, help="Path to automated evaluation YAML config."
    )

    ablation_parser = subparsers.add_parser(
        "run-ablation", help="Compare evaluation summaries across variants."
    )
    ablation_parser.add_argument(
        "--config", required=True, help="Path to ablation YAML config."
    )

    benchmark_parser = subparsers.add_parser(
        "validate-benchmark", help="Validate formal benchmark JSONL schema."
    )
    benchmark_parser.add_argument(
        "--config", required=True, help="Path to benchmark validation YAML config."
    )

    generate_benchmark_parser = subparsers.add_parser(
        "generate-benchmark",
        help="Generate benchmark candidates, dedupe them, and export benchmark v1.",
    )
    generate_benchmark_parser.add_argument(
        "--config", required=True, help="Path to benchmark generation YAML config."
    )

    experiment_parser = subparsers.add_parser(
        "run-experiments", help="Run the fixed experiment matrix end-to-end."
    )
    experiment_parser.add_argument(
        "--config", required=True, help="Path to experiment matrix YAML config."
    )

    error_parser = subparsers.add_parser(
        "analyze-errors",
        help="Run rule-based error analysis for a benchmark/result pair.",
    )
    error_parser.add_argument(
        "--config", required=True, help="Path to error analysis YAML config."
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "audit":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = AuditConfig.model_validate(raw)
        run_audit(config)
    elif args.command == "clean":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = CleaningConfig.model_validate(raw)
        run_cleaning(config)
    elif args.command == "review-sample":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = ReviewSampleConfig.model_validate(raw)
        export_review_samples(config)
    elif args.command == "chunk":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = ChunkingConfig.model_validate(raw)
        run_chunking(config)
    elif args.command == "retrieve":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = RetrievalConfig.model_validate(raw)
        run_retrieval(config)
    elif args.command == "generate":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = GenerationConfig.model_validate(raw)
        run_generation(config)
    elif args.command == "debug-llm-generation":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = LLMGenerationDebugConfig.model_validate(raw)
        run_llm_generation_debug(config)
    elif args.command == "process-contexts":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = ContextProcessingConfig.model_validate(raw)
        run_context_processing(config)
    elif args.command == "eval-retrieval":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = RetrievalEvalConfig.model_validate(raw)
        run_retrieval_evaluation(config)
    elif args.command == "eval-generation":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = GenerationEvalConfig.model_validate(raw)
        run_generation_evaluation(config)
    elif args.command == "generate-auto-testset":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = AutoTestsetConfig.model_validate(raw)
        run_auto_testset_generation(config)
    elif args.command == "run-auto-eval":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = AutoEvalConfig.model_validate(raw)
        run_auto_eval(config)
    elif args.command == "run-ablation":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = AblationConfig.model_validate(raw)
        run_ablation(config)
    elif args.command == "validate-benchmark":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = BenchmarkValidateConfig.model_validate(raw)
        run_benchmark_validation(config)
    elif args.command == "generate-benchmark":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = BenchmarkGenerationConfig.model_validate(raw)
        run_benchmark_generation(config)
    elif args.command == "run-experiments":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = ExperimentMatrixConfig.model_validate(raw)
        run_experiment_matrix(config)
    elif args.command == "analyze-errors":
        config_path = Path(args.config)
        raw = load_yaml_config(config_path)
        config = ErrorAnalysisConfig.model_validate(raw)
        run_error_analysis(config)


if __name__ == "__main__":
    main()
