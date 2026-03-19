from __future__ import annotations

from legal_rag.generation.extractive import generate_grounded_answer
from legal_rag.schemas.generation import ContextItem, GenerationInput
from legal_rag.schemas.retrieval import RetrievalHit


def make_hit(rank: int, score: float, text: str) -> RetrievalHit:
    return RetrievalHit(
        query_id="q1",
        query_text="通用机场适用范围",
        chunk_id=f"chunk-{rank}",
        doc_id="doc-1",
        rank=rank,
        score=score,
        retrieval_method="hybrid",
        chunk_text=text,
        title="通用机场管理规定",
    )


def test_generate_grounded_answer_returns_context_ids() -> None:
    generation_input = GenerationInput(
        query_id="q1",
        query_text="通用机场适用范围",
        question_type="definition",
        answerable=True,
        contexts=[
            ContextItem(
                chunk_id="chunk-1",
                doc_id="doc-1",
                title="通用机场管理规定",
                text="第一条 为了规范通用机场管理，制定本规定。第二条 中华人民共和国境内通用机场的建设、使用、运营管理应当遵守本规定。",
                rank=1,
                score=1.0,
                retrieval_method="hybrid",
            ),
            ContextItem(
                chunk_id="chunk-2",
                doc_id="doc-1",
                title="通用机场管理规定",
                text="第三条 通用机场按照飞行场地的物理特性分为跑道型机场、水上机场和直升机场。",
                rank=2,
                score=0.8,
                retrieval_method="hybrid",
            ),
        ],
    )
    answer = generate_grounded_answer(
        generation_input,
        max_contexts=2,
        max_sentences=2,
        max_span_chars=40,
        min_score=0.0,
    )
    assert answer.abstained is False
    assert answer.used_context_ids == ["chunk-1", "chunk-2"]
    assert len(answer.citations) == 2
    assert "第一条" in answer.answer
