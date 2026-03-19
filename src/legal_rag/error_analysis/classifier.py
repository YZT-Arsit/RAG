from __future__ import annotations

from legal_rag.benchmark.schema import BenchmarkRecord
from legal_rag.error_analysis.models import ErrorRecord
from legal_rag.schemas.generation import GroundedAnswer
from legal_rag.schemas.retrieval import RetrievalResult


def classify_error(
    benchmark: BenchmarkRecord,
    retrieval: RetrievalResult,
    answer: GroundedAnswer,
) -> ErrorRecord:
    gold_chunk_ids = {evidence.chunk_id for evidence in benchmark.gold_evidence}
    retrieved_chunk_ids = [hit.chunk_id for hit in retrieval.hits]
    used_context_ids = set(answer.used_context_ids)

    labels: list[str] = []
    notes: list[str] = []

    if benchmark.answerable and not any(
        chunk_id in gold_chunk_ids for chunk_id in retrieved_chunk_ids
    ):
        labels.append("retrieval_miss")
        notes.append("No gold evidence chunk was retrieved.")
    elif (
        benchmark.answerable
        and any(chunk_id in gold_chunk_ids for chunk_id in retrieved_chunk_ids)
        and not (used_context_ids & gold_chunk_ids)
    ):
        labels.append("ranking_miss")
        notes.append("Gold evidence was retrieved but not selected into used contexts.")

    alignment = answer.metadata.get("citation_alignment", {})
    if (
        isinstance(alignment, dict)
        and float(alignment.get("citation_support_ratio", 1.0)) < 1.0
    ):
        labels.append("citation_mismatch")
        notes.append("At least one citation was unsupported or misaligned.")

    if benchmark.answerable and answer.abstained:
        labels.append("wrong_abstain")
        notes.append("Question is answerable but system abstained.")
    if (not benchmark.answerable) and (not answer.abstained):
        labels.append("wrong_abstain")
        notes.append("Question is unanswerable but system did not abstain.")

    if benchmark.answerable and not answer.abstained:
        correctness = _float_metric(answer.metadata, "answer_correctness")
        if correctness < 0.5:
            labels.append("generation_hallucination")
            notes.append("Answer correctness baseline score is below threshold 0.5.")

    return ErrorRecord(
        query_id=benchmark.query_id,
        question=benchmark.question,
        error_labels=labels,
        notes=notes,
    )


def _float_metric(metadata: dict, key: str) -> float:
    value = metadata.get(key, 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0
