from __future__ import annotations

import re

from legal_rag.generation.citation import build_citation
from legal_rag.schemas.generation import GenerationInput, GroundedAnswer
from legal_rag.schemas.retrieval import RetrievalHit


SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；])")


def generate_grounded_answer(
    generation_input: GenerationInput,
    *,
    max_contexts: int,
    max_sentences: int,
    max_span_chars: int,
    min_score: float,
) -> GroundedAnswer:
    strong_hits = [
        _context_to_hit(generation_input, context)
        for context in generation_input.contexts[:max_contexts]
        if context.score >= min_score
    ]
    if not strong_hits:
        return GroundedAnswer(
            query_id=generation_input.query_id,
            query_text=generation_input.query_text,
            answer="未能基于当前检索证据生成可靠答案。",
            citations=[],
            used_context_ids=[],
            abstained=True,
            metadata={"reason": "no_hit_above_threshold"},
        )

    sentences = _collect_candidate_sentences(strong_hits, max_sentences=max_sentences)
    answer_text = " ".join(sentences).strip()
    citations = [
        build_citation(hit, max_span_chars=max_span_chars) for hit in strong_hits
    ]
    used_context_ids = [hit.chunk_id for hit in strong_hits]
    return GroundedAnswer(
        query_id=generation_input.query_id,
        query_text=generation_input.query_text,
        answer=answer_text or "已检索到相关证据，但暂未抽取出可直接回答的句子。",
        citations=citations,
        used_context_ids=used_context_ids,
        metadata={
            "context_count": len(strong_hits),
        },
    )


def _collect_candidate_sentences(
    hits: list[RetrievalHit], *, max_sentences: int
) -> list[str]:
    seen: set[str] = set()
    selected: list[str] = []
    for hit in hits:
        for sentence in _split_sentences(hit.chunk_text):
            normalized = sentence.strip()
            if not normalized or normalized in seen:
                continue
            selected.append(normalized)
            seen.add(normalized)
            if len(selected) >= max_sentences:
                return selected
    return selected


def _split_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_RE.split(text)
    return [part.strip() for part in parts if part.strip()]


def _context_to_hit(generation_input: GenerationInput, context) -> RetrievalHit:
    return RetrievalHit(
        query_id=generation_input.query_id,
        query_text=generation_input.query_text,
        chunk_id=context.chunk_id,
        doc_id=context.doc_id,
        rank=context.rank,
        score=context.score,
        retrieval_method=context.retrieval_method,
        chunk_text=context.text,
        title=context.title,
        section_path=context.section_path,
        metadata=context.metadata,
    )
