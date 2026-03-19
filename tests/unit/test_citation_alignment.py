from __future__ import annotations

from legal_rag.generation.alignment import run_citation_alignment_check
from legal_rag.schemas.generation import (
    Citation,
    ContextItem,
    GenerationInput,
    GroundedAnswer,
)


def test_citation_alignment_detects_valid_support() -> None:
    generation_input = GenerationInput(
        query_id="q1",
        query_text="查询",
        question_type=None,
        answerable=True,
        contexts=[
            ContextItem(
                chunk_id="c1",
                doc_id="d1",
                title="通用机场管理规定",
                text="中华人民共和国境内通用机场的建设、使用、运营管理及其相关活动应当遵守本规定。",
                rank=1,
                score=1.0,
                retrieval_method="hybrid",
            )
        ],
    )
    answer = GroundedAnswer(
        query_id="q1",
        query_text="查询",
        answer="应当遵守本规定。",
        citations=[
            Citation(
                chunk_id="c1",
                doc_id="d1",
                title="通用机场管理规定",
                span_text="应当遵守本规定。",
                rank=1,
            )
        ],
        used_context_ids=["c1"],
    )
    alignment = run_citation_alignment_check(answer, generation_input)
    assert alignment["alignment_all_valid"] is True
    assert alignment["citation_support_ratio"] == 1.0
