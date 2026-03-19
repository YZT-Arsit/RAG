from __future__ import annotations

from legal_rag.retrieval.tokenize import tokenize_for_bm25
from legal_rag.schemas.generation import GenerationInput, GroundedAnswer


def run_citation_alignment_check(
    answer: GroundedAnswer, generation_input: GenerationInput
) -> dict[str, float | bool | int]:
    hit_map = {context.chunk_id: context for context in generation_input.contexts}
    used_id_valid_count = sum(
        1 for chunk_id in answer.used_context_ids if chunk_id in hit_map
    )
    citation_id_valid_count = sum(
        1 for citation in answer.citations if citation.chunk_id in hit_map
    )

    supported_citations = 0
    for citation in answer.citations:
        hit = hit_map.get(citation.chunk_id)
        if hit is None:
            continue
        if citation.span_text and citation.span_text in hit.text:
            supported_citations += 1
            continue
        if _token_overlap_ratio(citation.span_text, hit.text) >= 0.6:
            supported_citations += 1

    citation_count = len(answer.citations)
    used_count = len(answer.used_context_ids)
    return {
        "used_context_id_valid_ratio": used_id_valid_count / used_count
        if used_count
        else 1.0,
        "citation_id_valid_ratio": citation_id_valid_count / citation_count
        if citation_count
        else 1.0,
        "citation_support_ratio": supported_citations / citation_count
        if citation_count
        else 1.0,
        "alignment_all_valid": (
            (used_id_valid_count == used_count)
            and (citation_id_valid_count == citation_count)
            and (supported_citations == citation_count)
        ),
        "unsupported_citation_count": citation_count - supported_citations,
    }


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(tokenize_for_bm25(left))
    right_tokens = set(tokenize_for_bm25(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)
