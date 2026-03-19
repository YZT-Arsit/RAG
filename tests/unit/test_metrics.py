from __future__ import annotations

from legal_rag.evaluation.generation_metrics import evaluate_generation
from legal_rag.evaluation.retrieval_metrics import evaluate_retrieval
from legal_rag.schemas.evaluation import GenerationGoldRecord, RetrievalGoldRecord
from legal_rag.schemas.generation import GroundedAnswer
from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit, RetrievalResult


def test_evaluate_retrieval_computes_recall_and_mrr() -> None:
    results = [
        RetrievalResult(
            query=QueryRecord(query_id="q1", query_text="查询"),
            hits=[
                RetrievalHit(
                    query_id="q1",
                    query_text="查询",
                    chunk_id="c1",
                    doc_id="d1",
                    rank=1,
                    score=1.0,
                    retrieval_method="hybrid",
                    chunk_text="正文",
                    title="标题",
                )
            ],
        )
    ]
    gold = [RetrievalGoldRecord(query_id="q1", relevant_chunk_ids=["c1", "c2"])]
    _, summary = evaluate_retrieval(results, gold, ks=[1])
    assert summary["recall@1"] == 0.5
    assert summary["mrr"] == 1.0


def test_evaluate_generation_computes_token_overlap_and_citation_metrics() -> None:
    answers = [
        GroundedAnswer(
            query_id="q1",
            query_text="查询",
            answer="夫妻一方可以请求返还共同财产。",
            used_context_ids=["c1"],
        )
    ]
    gold = [
        GenerationGoldRecord(
            query_id="q1",
            reference_answer="夫妻一方可以请求返还共同财产。",
            supporting_chunk_ids=["c1", "c2"],
            question_type="case_judgment",
            answerable=True,
        )
    ]
    _, summary = evaluate_generation(answers, gold)
    assert summary["token_f1"] == 1.0
    assert summary["answer_correctness"] == 1.0
    assert summary["faithfulness"] == 0.0
    assert summary["citation_precision"] == 1.0
    assert summary["citation_recall"] == 0.5
    assert summary["abstain_accuracy"] == 1.0
