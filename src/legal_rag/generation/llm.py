from __future__ import annotations

import json
import re

from legal_rag.generation.base import BaseGenerator
from legal_rag.generation.citation import build_citation
from legal_rag.generation.llm_client import LLMClient
from legal_rag.generation.prompt_builder import build_grounded_prompt
from legal_rag.schemas.generation import GenerationInput, GroundedAnswer


class LLMGroundedGenerator(BaseGenerator):
    def __init__(
        self,
        *,
        client: LLMClient,
        prompt_version: str,
        max_contexts: int,
        max_chars_per_context: int,
        max_citation_chars: int,
        require_context_ids: bool,
        abstain_when_insufficient: bool,
    ) -> None:
        self.client = client
        self.prompt_version = prompt_version
        self.max_contexts = max_contexts
        self.max_chars_per_context = max_chars_per_context
        self.max_citation_chars = max_citation_chars
        self.require_context_ids = require_context_ids
        self.abstain_when_insufficient = abstain_when_insufficient

    def generate(self, generation_input: GenerationInput) -> GroundedAnswer:
        prompt = build_grounded_prompt(
            generation_input,
            prompt_version=self.prompt_version,
            max_contexts=self.max_contexts,
            max_chars_per_context=self.max_chars_per_context,
        )
        response = self.client.complete(prompt)
        payload, parse_status, parse_error = _parse_json_payload(response.content)
        normalized = _normalize_payload(
            payload,
            generation_input=generation_input,
            require_context_ids=self.require_context_ids,
            abstain_when_insufficient=self.abstain_when_insufficient,
        )
        used_context_ids = normalized["used_context_ids"]
        selected_contexts = [
            context
            for context in generation_input.contexts
            if context.chunk_id in used_context_ids
        ]
        citations = [
            build_citation(
                _context_to_hit(generation_input, context),
                max_span_chars=self.max_citation_chars,
            )
            for context in selected_contexts
        ]
        return GroundedAnswer(
            query_id=generation_input.query_id,
            query_text=generation_input.query_text,
            answer=normalized["answer"],
            citations=citations,
            used_context_ids=used_context_ids,
            generation_method="llm_grounded",
            abstained=normalized["abstained"],
            metadata={
                "prompt_chars": len(prompt),
                "raw_response": response.content,
                "prompt_version": self.prompt_version,
                "parse_status": parse_status,
                "parse_error": parse_error,
                "parse_warning": normalized["parse_warning"]
                or _default_parse_warning(parse_status),
                "error_message": normalized["error_message"],
                "schema_status": normalized["schema_status"],
                "valid_used_context_ids": normalized["valid_used_context_ids"],
                "invalid_used_context_ids": normalized["invalid_used_context_ids"],
            },
        )


def _parse_json_payload(content: str) -> tuple[dict[str, object], str, str | None]:
    cleaned_content = _strip_reasoning_segments(content)
    strict_error: str | None = None
    try:
        payload = json.loads(cleaned_content)
        if isinstance(payload, dict):
            return payload, "strict_json", None
    except json.JSONDecodeError as exc:
        strict_error = str(exc)
    try:
        start = cleaned_content.find("{")
        end = cleaned_content.rfind("}")
        if start != -1 and end != -1 and end > start:
            payload = json.loads(cleaned_content[start : end + 1])
            if isinstance(payload, dict):
                return payload, "loose_json", strict_error
    except json.JSONDecodeError:
        pass
    extracted = _extract_relaxed_payload(cleaned_content)
    if extracted is not None:
        return extracted, "relaxed_extraction", strict_error
    fallback_answer = cleaned_content.strip() or "未能生成可靠答案。"
    return (
        {
            "answer": fallback_answer,
            "used_context_ids": [],
            "abstained": True,
        },
        "minimal_fallback",
        strict_error,
    )


def _extract_relaxed_payload(content: str) -> dict[str, object] | None:
    answer_match = re.search(r'"answer"\s*:\s*"(?P<value>(?:[^"\\]|\\.)*)"', content)
    abstained_match = re.search(
        r'"abstained"\s*:\s*(true|false)', content, re.IGNORECASE
    )
    used_ids_match = re.search(
        r'"used_context_ids"\s*:\s*\[(?P<value>[^\]]*)\]', content
    )

    if not answer_match and not abstained_match and not used_ids_match:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
        if fenced:
            try:
                payload = json.loads(fenced.group(1))
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                return None
        return None

    answer = ""
    if answer_match:
        raw_answer = answer_match.group("value")
        try:
            answer = json.loads(f'"{raw_answer}"')
        except json.JSONDecodeError:
            answer = raw_answer
    abstained = False
    if abstained_match:
        abstained = abstained_match.group(1).lower() == "true"
    used_context_ids: list[str] = []
    if used_ids_match:
        used_context_ids = re.findall(r'"([^"]+)"', used_ids_match.group("value"))
    return {
        "answer": answer,
        "used_context_ids": used_context_ids,
        "abstained": abstained,
    }


def _strip_reasoning_segments(content: str) -> str:
    stripped = re.sub(
        r"<think>.*?</think>\s*", "", content, flags=re.DOTALL | re.IGNORECASE
    ).strip()
    if stripped:
        return stripped
    if "</think>" in content:
        tail = content.split("</think>")[-1].strip()
        if tail:
            return tail
    return content.strip()


def _normalize_payload(
    payload: dict[str, object],
    *,
    generation_input: GenerationInput,
    require_context_ids: bool,
    abstain_when_insufficient: bool,
) -> dict[str, object]:
    answer = str(payload.get("answer", "")).strip() or "未能生成可靠答案。"
    abstained = bool(payload.get("abstained", False))
    raw_used_context_ids = payload.get("used_context_ids", [])
    candidate_ids = (
        [item for item in raw_used_context_ids if isinstance(item, str)]
        if isinstance(raw_used_context_ids, list)
        else []
    )
    valid_context_ids = {context.chunk_id for context in generation_input.contexts}
    used_context_ids = [
        chunk_id for chunk_id in candidate_ids if chunk_id in valid_context_ids
    ]
    invalid_context_ids = [
        chunk_id for chunk_id in candidate_ids if chunk_id not in valid_context_ids
    ]

    schema_status = "valid"
    parse_warnings: list[str] = []
    error_message: str | None = None
    if invalid_context_ids:
        parse_warnings.append("filtered_invalid_context_ids")
    if require_context_ids and not used_context_ids and not abstained:
        citation_inferred_ids = _infer_context_ids_from_citation_labels(
            answer, generation_input
        )
        if citation_inferred_ids:
            used_context_ids = citation_inferred_ids
            schema_status = "inferred_context_ids_from_citations"
            parse_warnings.append("used_context_ids_inferred_from_citations")
        else:
            inferred_context_ids = _infer_context_ids_from_answer(answer, generation_input)
            if inferred_context_ids:
                used_context_ids = inferred_context_ids
                schema_status = "inferred_context_ids_from_answer"
                parse_warnings.append("used_context_ids_inferred_from_answer")
            else:
                abstained = True
                schema_status = "forced_abstain_missing_context_ids"
                error_message = "LLM output did not provide usable context ids and no grounded context ids could be inferred from the answer."
    elif invalid_context_ids:
        schema_status = "filtered_invalid_context_ids"

    if abstain_when_insufficient and not used_context_ids and not abstained:
        abstained = True
        if schema_status == "valid":
            schema_status = "forced_abstain_no_context_ids"
        error_message = (
            "LLM output did not contain grounded context ids after normalization."
        )

    if answer == "未能生成可靠答案。":
        parse_warnings.append("fallback_answer_used")

    return {
        "answer": answer,
        "abstained": abstained,
        "used_context_ids": used_context_ids,
        "valid_used_context_ids": used_context_ids,
        "invalid_used_context_ids": invalid_context_ids,
        "schema_status": schema_status,
        "parse_warning": "; ".join(dict.fromkeys(parse_warnings)) or None,
        "error_message": error_message,
    }


def _default_parse_warning(parse_status: str) -> str | None:
    if parse_status == "minimal_fallback":
        return "fallback_answer_used"
    if parse_status == "relaxed_extraction":
        return "relaxed_json_extraction_used"
    if parse_status == "loose_json":
        return "non_strict_json_wrapper_removed"
    return None


def _infer_context_ids_from_answer(
    answer: str, generation_input: GenerationInput
) -> list[str]:
    normalized_answer = re.sub(r"\s+", "", answer)
    if len(normalized_answer) < 4:
        return []

    scored: list[tuple[int, str]] = []
    for context in generation_input.contexts:
        haystack = re.sub(r"\s+", "", f"{context.title}{context.text}")
        overlap = _longest_substring_overlap(normalized_answer, haystack)
        if overlap > 0:
            scored.append((overlap, context.chunk_id))

    scored.sort(reverse=True)
    return [chunk_id for overlap, chunk_id in scored[:2] if overlap >= 4]


def _infer_context_ids_from_citation_labels(
    answer: str, generation_input: GenerationInput
) -> list[str]:
    labels = re.findall(r"\[(\d+)\]", answer)
    used_ids: list[str] = []
    for label in labels:
        index = int(label) - 1
        if 0 <= index < len(generation_input.contexts):
            chunk_id = generation_input.contexts[index].chunk_id
            if chunk_id not in used_ids:
                used_ids.append(chunk_id)
    return used_ids


def _longest_substring_overlap(left: str, right: str) -> int:
    max_len = min(len(left), 24)
    for size in range(max_len, 2, -1):
        for start in range(0, len(left) - size + 1):
            piece = left[start : start + size]
            if piece in right:
                return size
    return 0


def _context_to_hit(generation_input: GenerationInput, context):
    from legal_rag.schemas.retrieval import RetrievalHit

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
