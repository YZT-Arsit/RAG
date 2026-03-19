from __future__ import annotations

from statistics import mean

from legal_rag.schemas.evaluation import MetricRecord, RetrievalGoldRecord
from legal_rag.schemas.retrieval import RetrievalResult


def evaluate_retrieval(
    results: list[RetrievalResult],
    gold_records: list[RetrievalGoldRecord],
    *,
    ks: list[int],
) -> tuple[list[MetricRecord], dict[str, float]]:
    gold_map = {
        record.query_id: set(record.relevant_chunk_ids) for record in gold_records
    }
    per_query: list[MetricRecord] = []

    for result in results:
        relevant = gold_map.get(result.query.query_id, set())
        retrieved_ids = [hit.chunk_id for hit in result.hits]
        metrics: dict[str, float] = {}
        for k in ks:
            topk = retrieved_ids[:k]
            hits = len([chunk_id for chunk_id in topk if chunk_id in relevant])
            metrics[f"recall@{k}"] = hits / len(relevant) if relevant else 0.0
            metrics[f"hit@{k}"] = 1.0 if hits > 0 else 0.0
        metrics["mrr"] = _mrr(retrieved_ids, relevant)
        per_query.append(MetricRecord(query_id=result.query.query_id, metrics=metrics))

    summary = _aggregate_metric_records(per_query)
    return per_query, summary


def _mrr(retrieved_ids: list[str], relevant: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def _aggregate_metric_records(records: list[MetricRecord]) -> dict[str, float]:
    if not records:
        return {}
    keys = records[0].metrics.keys()
    return {key: mean(record.metrics[key] for record in records) for key in keys}
