from __future__ import annotations

from legal_rag.generation.guardrails import apply_guardrails, run_consistency_check
from legal_rag.schemas.generation import ContextItem, GenerationInput, GroundedAnswer


def make_generation_input(score: float = 0.9) -> GenerationInput:
    return GenerationInput(
        query_id="q1",
        query_text="赔偿标准是什么",
        question_type="definition",
        answerable=True,
        contexts=[
            ContextItem(
                chunk_id="c1",
                doc_id="d1",
                title="人身损害赔偿",
                text="人身损害赔偿包括医疗费、护理费、交通费等合理费用。",
                rank=1,
                score=score,
                retrieval_method="hybrid",
            )
        ],
    )


def test_guardrails_block_answer_without_citation_brackets() -> None:
    generation_input = make_generation_input()
    answer = GroundedAnswer(
        query_id="q1",
        query_text="赔偿标准是什么",
        answer="人身损害赔偿包括医疗费和护理费。",
        used_context_ids=["c1"],
    )
    guarded = apply_guardrails(
        answer,
        generation_input,
        min_top_score=0.2,
        nli_enabled=True,
        nli_threshold=0.2,
        require_citation_brackets=True,
        fail_message="抱歉，根据现有法律库无法给出确切回答，请咨询专业律师。",
    )
    assert guarded.abstained is True
    assert guarded.metadata["guardrail_status"] == "blocked"


def test_consistency_check_scores_supported_answer() -> None:
    generation_input = make_generation_input()
    consistency = run_consistency_check(
        "人身损害赔偿包括医疗费和护理费[1]。", generation_input
    )
    assert consistency["passed"] is True
    assert consistency["support_ratio"] > 0.5
