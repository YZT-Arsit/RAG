from __future__ import annotations

from statistics import mean

from legal_rag.retrieval.tokenize import tokenize_for_bm25
from legal_rag.schemas.evaluation import GenerationGoldRecord, MetricRecord
from legal_rag.schemas.generation import GroundedAnswer


def evaluate_generation(
    answers: list[GroundedAnswer],
    gold_records: list[GenerationGoldRecord],
) -> tuple[list[MetricRecord], dict[str, float]]:
    gold_map = {record.query_id: record for record in gold_records}
    per_query: list[MetricRecord] = []

    for answer in answers:
        gold = gold_map.get(answer.query_id)
        if gold is None:
            continue
        ref_tokens = set(tokenize_for_bm25(gold.reference_answer))
        ans_tokens = set(tokenize_for_bm25(answer.answer))
        overlap = len(ref_tokens & ans_tokens)
        precision = overlap / len(ans_tokens) if ans_tokens else 0.0
        recall = overlap / len(ref_tokens) if ref_tokens else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0
        )

        expected_contexts = set(gold.supporting_chunk_ids)
        used_contexts = set(answer.used_context_ids)
        citation_precision = (
            len(expected_contexts & used_contexts) / len(used_contexts)
            if used_contexts
            else 0.0
        )
        citation_recall = (
            len(expected_contexts & used_contexts) / len(expected_contexts)
            if expected_contexts
            else 0.0
        )
        faithfulness = _faithfulness_score(answer)
        correctness = f1
        abstain_accuracy = _abstain_accuracy(
            answerable=gold.answerable, abstained=answer.abstained
        )
        answer.metadata["answer_correctness"] = correctness
        answer.metadata["faithfulness"] = faithfulness
        answer.metadata["abstain_accuracy"] = abstain_accuracy

        per_query.append(
            MetricRecord(
                query_id=answer.query_id,
                metrics={
                    "answer_correctness": correctness,
                    "faithfulness": faithfulness,
                    "token_f1": f1,
                    "token_precision": precision,
                    "token_recall": recall,
                    "citation_precision": citation_precision,
                    "citation_recall": citation_recall,
                    "abstained": 1.0 if answer.abstained else 0.0,
                    "abstain_accuracy": abstain_accuracy,
                    "wrong_abstain": 1.0 if abstain_accuracy == 0.0 else 0.0,
                },
                metadata={
                    "question_type": gold.question_type,
                    "answerable": gold.answerable,
                },
            )
        )

    summary = _aggregate_metric_records(per_query)
    return per_query, summary


def aggregate_generation_by_question_type(
    records: list[MetricRecord],
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[MetricRecord]] = {}
    for record in records:
        question_type = record.metadata.get("question_type", "unknown")
        grouped.setdefault(str(question_type), []).append(record)
    return {
        group_name: _aggregate_metric_records(group_records)
        for group_name, group_records in grouped.items()
    }


def _aggregate_metric_records(records: list[MetricRecord]) -> dict[str, float]:
    if not records:
        return {}
    keys = records[0].metrics.keys()
    return {key: mean(record.metrics[key] for record in records) for key in keys}


def _faithfulness_score(answer: GroundedAnswer) -> float:
    alignment = answer.metadata.get("citation_alignment", {})
    if isinstance(alignment, dict):
        value = alignment.get("citation_support_ratio")
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _abstain_accuracy(*, answerable: bool, abstained: bool) -> float:
    if answerable and not abstained:
        return 1.0
    if (not answerable) and abstained:
        return 1.0
    return 0.0
