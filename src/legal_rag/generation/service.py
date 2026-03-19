from __future__ import annotations

from collections import Counter
import json

import traceback
from pathlib import Path

from legal_rag.config.schema import GenerationConfig, LLMGenerationDebugConfig
from legal_rag.generation.alignment import run_citation_alignment_check
from legal_rag.generation.base import BaseGenerator
from legal_rag.generation.context import build_context_block
from legal_rag.generation.extractive import generate_grounded_answer
from legal_rag.generation.guardrails import apply_guardrails
from legal_rag.generation.io import write_answers
from legal_rag.generation.llm import LLMGroundedGenerator
from legal_rag.generation.llm_client import (
    LocalTransformersClient,
    OpenAICompatibleClient,
)
from legal_rag.schemas.generation import ContextItem, GenerationInput
from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit, RetrievalResult


def run_generation(config: GenerationConfig) -> None:
    results = load_retrieval_results(config.retrieval_results_jsonl)
    generator = build_generator(config)
    answers = []
    for result in results:
        generation_input = build_generation_input(
            result, context_source=config.context_source
        )
        answer = generator.generate(generation_input)
        answer.metadata["context_source"] = config.context_source
        answer.metadata["citation_alignment"] = run_citation_alignment_check(
            answer, generation_input
        )
        if config.guardrail_enabled:
            answer = apply_guardrails(
                answer,
                generation_input,
                min_top_score=config.guardrail_min_top_score,
                nli_enabled=config.guardrail_nli_enabled,
                nli_threshold=config.guardrail_nli_threshold,
                require_citation_brackets=config.guardrail_require_citation_brackets,
                fail_message=config.guardrail_fail_message,
            )
        answers.append(answer)
    write_answers(config.output_jsonl, answers)
    _write_generation_report(config.report_path, results, answers, config)


def run_llm_generation_debug(config: LLMGenerationDebugConfig) -> None:
    results = _load_retrieval_results(config.retrieval_results_jsonl)
    matched = next(
        (result for result in results if result.query.query_id == config.query_id), None
    )
    if matched is None:
        payload = {
            "success": False,
            "query_id": config.query_id,
            "failure_stage": "input_lookup",
            "failure_reason": f"Query id {config.query_id} not found in retrieval results.",
            "answer": "",
            "used_context_ids": [],
            "abstained": True,
            "raw_response": "",
            "parse_status": "not_started",
            "parse_warning": None,
            "error_message": None,
        }
        _write_debug_output(config.output_path, payload)
        raise ValueError(payload["failure_reason"])

    generation_input = build_generation_input(
        matched, context_source=config.context_source
    )
    generator = _build_llm_generator_from_debug_config(config)
    try:
        answer = generator.generate(generation_input)
        if config.llm_prompt_version:
            answer.metadata["citation_alignment"] = run_citation_alignment_check(
                answer, generation_input
            )
        payload = {
            "success": True,
            "query_id": generation_input.query_id,
            "query_text": generation_input.query_text,
            "answer": answer.answer,
            "used_context_ids": answer.used_context_ids,
            "abstained": answer.abstained,
            "raw_response": answer.metadata.get("raw_response", ""),
            "parse_status": answer.metadata.get("parse_status", "unknown"),
            "parse_error": answer.metadata.get("parse_error"),
            "parse_warning": answer.metadata.get("parse_warning"),
            "error_message": answer.metadata.get("error_message"),
            "schema_status": answer.metadata.get("schema_status", "unknown"),
            "context_source": config.context_source,
        }
        _write_debug_output(config.output_path, payload)
    except Exception as exc:
        payload = {
            "success": False,
            "query_id": generation_input.query_id,
            "query_text": generation_input.query_text,
            "failure_stage": _infer_generation_failure_stage(exc),
            "failure_type": exc.__class__.__name__,
            "failure_reason": str(exc),
            "answer": "",
            "used_context_ids": [],
            "abstained": True,
            "raw_response": "",
            "parse_status": "failed_before_parse",
            "parse_warning": None,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "context_source": config.context_source,
        }
        _write_debug_output(config.output_path, payload)
        raise


class ExtractiveGenerator(BaseGenerator):
    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    def generate(self, generation_input: GenerationInput):
        return generate_grounded_answer(
            generation_input,
            max_contexts=self.config.max_contexts,
            max_sentences=self.config.max_answer_sentences,
            max_span_chars=self.config.max_citation_chars,
            min_score=self.config.min_hit_score,
        )


def _build_generator(config: GenerationConfig) -> BaseGenerator:
    return build_generator(config)


def build_generator(config: GenerationConfig) -> BaseGenerator:
    if config.method == "extractive":
        return ExtractiveGenerator(config)
    return _build_llm_generator(
        client=build_llm_client(
            llm_backend=config.llm_backend,
            llm_base_url=config.llm_base_url,
            llm_api_key_env=config.llm_api_key_env,
            llm_model_name=config.llm_model_name,
            llm_modelscope_model_id=config.llm_modelscope_model_id,
            llm_local_model_dir=config.llm_local_model_dir,
            llm_use_modelscope_download=config.llm_use_modelscope_download,
            llm_device=config.llm_device,
            llm_temperature=config.llm_temperature,
            llm_timeout_seconds=config.llm_timeout_seconds,
            llm_max_new_tokens=config.llm_max_new_tokens,
        ),
        llm_prompt_version=config.llm_prompt_version,
        max_contexts=config.max_contexts,
        max_prompt_context_chars=config.max_prompt_context_chars,
        max_citation_chars=config.max_citation_chars,
        llm_require_context_ids=config.llm_require_context_ids,
        llm_abstain_when_insufficient=config.llm_abstain_when_insufficient,
    )


def _build_llm_generator_from_debug_config(
    config: LLMGenerationDebugConfig,
) -> LLMGroundedGenerator:
    return _build_llm_generator(
        client=build_llm_client(
            llm_backend=config.llm_backend,
            llm_base_url=config.llm_base_url,
            llm_api_key_env=config.llm_api_key_env,
            llm_model_name=config.llm_model_name,
            llm_modelscope_model_id=config.llm_modelscope_model_id,
            llm_local_model_dir=config.llm_local_model_dir,
            llm_use_modelscope_download=config.llm_use_modelscope_download,
            llm_device=config.llm_device,
            llm_temperature=config.llm_temperature,
            llm_timeout_seconds=config.llm_timeout_seconds,
            llm_max_new_tokens=config.llm_max_new_tokens,
        ),
        llm_prompt_version=config.llm_prompt_version,
        max_contexts=config.max_contexts,
        max_prompt_context_chars=config.max_prompt_context_chars,
        max_citation_chars=config.max_citation_chars,
        llm_require_context_ids=config.llm_require_context_ids,
        llm_abstain_when_insufficient=config.llm_abstain_when_insufficient,
    )


def _build_llm_generator(
    *,
    client,
    llm_prompt_version: str,
    max_contexts: int,
    max_prompt_context_chars: int,
    max_citation_chars: int,
    llm_require_context_ids: bool,
    llm_abstain_when_insufficient: bool,
) -> LLMGroundedGenerator:
    return LLMGroundedGenerator(
        client=client,
        prompt_version=llm_prompt_version,
        max_contexts=max_contexts,
        max_chars_per_context=max_prompt_context_chars,
        max_citation_chars=max_citation_chars,
        require_context_ids=llm_require_context_ids,
        abstain_when_insufficient=llm_abstain_when_insufficient,
    )


def build_llm_client(
    *,
    llm_backend: str,
    llm_base_url: str | None,
    llm_api_key_env: str,
    llm_model_name: str | None,
    llm_modelscope_model_id: str | None,
    llm_local_model_dir: Path | None,
    llm_use_modelscope_download: bool,
    llm_device: str,
    llm_temperature: float,
    llm_timeout_seconds: int,
    llm_max_new_tokens: int,
):
    if llm_backend == "openai_compatible":
        if not llm_base_url or not llm_model_name:
            msg = "OpenAI-compatible LLM generation requires llm_base_url and llm_model_name in config."
            raise ValueError(msg)
        return OpenAICompatibleClient(
            base_url=llm_base_url,
            api_key_env=llm_api_key_env,
            model_name=llm_model_name,
            temperature=llm_temperature,
            timeout_seconds=llm_timeout_seconds,
        )
    return LocalTransformersClient(
        model_name=llm_model_name,
        modelscope_model_id=llm_modelscope_model_id,
        local_model_dir=llm_local_model_dir,
        use_modelscope_download=llm_use_modelscope_download,
        device=llm_device,
        temperature=llm_temperature,
        max_new_tokens=llm_max_new_tokens,
        timeout_seconds=llm_timeout_seconds,
    )


def _write_debug_output(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _infer_generation_failure_stage(exc: Exception) -> str:
    message = str(exc).lower()
    if "timeout" in message:
        return "timeout"
    if "json" in exc.__class__.__name__.lower() or "parse" in message:
        return "output_parsing"
    if "used_context_ids" in message or "schema" in message:
        return "schema_validation"
    if "citation" in message:
        return "citation_postprocess"
    if "cuda" in message or "mps" in message or "model" in message:
        return "model_loading_or_inference"
    return "generation"


def _load_retrieval_results(path) -> list[RetrievalResult]:
    return load_retrieval_results(path)


def load_retrieval_results(path) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            query = QueryRecord(
                query_id=payload["query_id"],
                query_text=payload["query_text"],
                metadata=payload.get("metadata", {}),
            )
            hits = [
                RetrievalHit(
                    query_id=item["query_id"],
                    query_text=item["query_text"],
                    chunk_id=item["chunk_id"],
                    doc_id=item["doc_id"],
                    rank=item["rank"],
                    score=item["score"],
                    retrieval_method=item["retrieval_method"],
                    chunk_text=item["chunk_text"],
                    title=item["title"],
                    section_path=item.get("section_path", []),
                    metadata=item.get("metadata", {}),
                )
                for item in payload.get("hits", [])
            ]
            results.append(RetrievalResult(query=query, hits=hits))
    return results


def _write_generation_report(
    path, results: list[RetrievalResult], answers, config: GenerationConfig
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    abstained = sum(1 for answer in answers if answer.abstained)
    method_counter: Counter[str] = Counter(
        answer.generation_method for answer in answers
    )
    alignment_valid = sum(
        1
        for answer in answers
        if bool(
            answer.metadata.get("citation_alignment", {}).get(
                "alignment_all_valid", False
            )
        )
    )
    lines = [
        "# Generation Report",
        "",
        "## Status Labels",
        "",
        "- Implemented: extractive grounded answer baseline, LLM grounded generation interface, citations, used_context_ids export, citation alignment checks, generation report export.",
        "- Experimental: sentence extraction baseline, OpenAI-compatible LLM backend, and local transformers/ModelScope backend.",
        "- Planned: stronger prompt strategies, answer validation, abstention calibration, model-based citation grounding checks.",
        "",
        "## Summary",
        "",
        f"- Queries processed: {len(results)}",
        f"- Answers exported: {len(answers)}",
        f"- Abstained answers: {abstained}",
        f"- Citation alignment all-valid count: {alignment_valid}",
        f"- Configured generation method: {config.method}",
        f"- Context source: {config.context_source}",
        f"- Max contexts per answer: {config.max_contexts}",
        "",
        "## Generation Methods",
        "",
    ]
    for method, count in method_counter.most_common():
        lines.append(f"- `{method}`: {count}")

    lines.extend(
        [
            "",
            "## Context Formatting Note",
            "",
            "The extractive baseline remains the default reproducible control. The LLM path is an interface-level integration that requires explicit model configuration and should be validated against the same benchmark before any performance claims.",
            "",
            "## Example Context Block Template",
            "",
        ]
    )
    if results:
        lines.append("```text")
        lines.append(
            build_context_block(
                results[0], max_contexts=config.max_contexts, max_chars_per_context=120
            )
        )
        lines.append("```")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_generation_input(
    result: RetrievalResult, *, context_source: str
) -> GenerationInput:
    return build_generation_input(result, context_source=context_source)


def build_generation_input(
    result: RetrievalResult, *, context_source: str
) -> GenerationInput:
    question_type = result.query.metadata.get("question_type")
    answerable = result.query.metadata.get("answerable")
    contexts = [
        ContextItem(
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            title=hit.title,
            text=hit.chunk_text,
            rank=hit.rank,
            score=hit.score,
            retrieval_method=hit.retrieval_method,
            section_path=hit.section_path,
            metadata=hit.metadata,
        )
        for hit in result.hits
    ]
    return GenerationInput(
        query_id=result.query.query_id,
        query_text=result.query.query_text,
        question_type=question_type if isinstance(question_type, str) else None,
        answerable=answerable if isinstance(answerable, bool) else None,
        contexts=contexts,
        metadata={
            **result.query.metadata,
            "context_source": context_source,
        },
    )
