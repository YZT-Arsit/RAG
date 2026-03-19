from __future__ import annotations

import re

from legal_rag.retrieval.tokenize import tokenize_for_bm25
from legal_rag.schemas.generation import GroundedAnswer, GenerationInput


def apply_guardrails(
    answer: GroundedAnswer,
    generation_input: GenerationInput,
    *,
    min_top_score: float,
    nli_enabled: bool,
    nli_threshold: float,
    require_citation_brackets: bool,
    fail_message: str,
) -> GroundedAnswer:
    top_score = generation_input.contexts[0].score if generation_input.contexts else 0.0
    failure_reasons: list[str] = []

    if top_score < min_top_score:
        failure_reasons.append("top_score_below_threshold")
    if require_citation_brackets and not answer.abstained and not _has_citation_brackets(answer.answer):
        failure_reasons.append("missing_citation_brackets")

    consistency = run_consistency_check(answer.answer, generation_input)
    if nli_enabled and consistency["support_ratio"] < nli_threshold:
        failure_reasons.append("nli_guardrail_failed")

    answer.metadata["guardrail_top_score"] = top_score
    answer.metadata["guardrail_consistency"] = consistency

    if not failure_reasons:
        answer.metadata["guardrail_status"] = "passed"
        return answer

    answer.metadata["guardrail_status"] = "blocked"
    answer.metadata["guardrail_failure_reasons"] = failure_reasons
    answer.answer = fail_message
    answer.abstained = True
    answer.used_context_ids = []
    answer.citations = []
    return answer


def run_consistency_check(
    answer_text: str, generation_input: GenerationInput
) -> dict[str, float | int | bool]:
    clean_answer = re.sub(r"\[\d+\]", "", answer_text)
    sentences = [item.strip() for item in re.split(r"[。！？；\n]+", clean_answer) if item.strip()]
    if not sentences:
        return {"support_ratio": 0.0, "supported_sentence_count": 0, "sentence_count": 0, "passed": False}
    support_scores = [_sentence_support_ratio(sentence, generation_input) for sentence in sentences]
    supported_sentence_count = sum(1 for score in support_scores if score >= 0.4)
    support_ratio = supported_sentence_count / len(sentences)
    return {
        "support_ratio": support_ratio,
        "supported_sentence_count": supported_sentence_count,
        "sentence_count": len(sentences),
        "passed": support_ratio >= 0.5,
    }


def _sentence_support_ratio(sentence: str, generation_input: GenerationInput) -> float:
    sentence_tokens = set(tokenize_for_bm25(sentence))
    if not sentence_tokens:
        return 0.0
    context_scores: list[float] = []
    for context in generation_input.contexts:
        context_tokens = set(tokenize_for_bm25(f"{context.title} {context.text}"))
        if not context_tokens:
            continue
        overlap = len(sentence_tokens & context_tokens) / len(sentence_tokens)
        context_scores.append(overlap)
    return max(context_scores, default=0.0)


def _has_citation_brackets(answer_text: str) -> bool:
    return bool(re.search(r"\[\d+\]", answer_text))
