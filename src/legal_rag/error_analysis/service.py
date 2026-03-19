from __future__ import annotations

from legal_rag.benchmark.loader import iter_benchmark_records
from legal_rag.config.schema import ErrorAnalysisConfig
from legal_rag.error_analysis.classifier import classify_error
from legal_rag.error_analysis.reporter import write_error_csv, write_error_markdown
from legal_rag.evaluation.io import load_answers, load_retrieval_results


def run_error_analysis(config: ErrorAnalysisConfig) -> None:
    benchmark_records = {
        record.query_id: record
        for record in iter_benchmark_records(config.benchmark_jsonl)
    }
    retrieval_results = {
        result.query.query_id: result
        for result in load_retrieval_results(config.retrieval_results_jsonl)
    }
    answers = {answer.query_id: answer for answer in load_answers(config.answers_jsonl)}

    records = []
    for query_id, benchmark in benchmark_records.items():
        retrieval = retrieval_results.get(query_id)
        answer = answers.get(query_id)
        if retrieval is None or answer is None:
            continue
        records.append(classify_error(benchmark, retrieval, answer))

    write_error_csv(config.detail_csv_path, records)
    write_error_markdown(config.report_path, records)
