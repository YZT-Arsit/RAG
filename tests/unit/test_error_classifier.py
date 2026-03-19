from __future__ import annotations

from legal_rag.benchmark.schema import BenchmarkRecord, GoldEvidence
from legal_rag.error_analysis.classifier import classify_error
from legal_rag.schemas.generation import GroundedAnswer
from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit, RetrievalResult


def test_classify_error_detects_retrieval_miss() -> None:
    benchmark = BenchmarkRecord(
        query_id="q1",
        question="问题",
        question_type="definition",
        answerable=True,
        gold_answer="答案",
        gold_evidence=[GoldEvidence(chunk_id="gold-1")],
    )
    retrieval = RetrievalResult(
        query=QueryRecord(query_id="q1", query_text="问题"),
        hits=[
            RetrievalHit(
                query_id="q1",
                query_text="问题",
                chunk_id="other",
                doc_id="d1",
                rank=1,
                score=1.0,
                retrieval_method="hybrid",
                chunk_text="正文",
                title="标题",
            )
        ],
    )
    answer = GroundedAnswer(
        query_id="q1",
        query_text="问题",
        answer="答案",
        used_context_ids=["other"],
        metadata={"answer_correctness": 1.0},
    )
    record = classify_error(benchmark, retrieval, answer)
    assert "retrieval_miss" in record.error_labels
