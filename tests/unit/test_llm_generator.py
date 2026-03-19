from __future__ import annotations

import json
from pathlib import Path

import pytest

from legal_rag.config.schema import LLMGenerationDebugConfig
from legal_rag.generation.llm import LLMGroundedGenerator
from legal_rag.generation.prompt_builder import build_grounded_prompt
from legal_rag.generation.service import run_llm_generation_debug
from legal_rag.schemas.generation import ContextItem, GenerationInput


class FakeClient:
    def complete(self, prompt: str):
        class Response:
            content = '{"answer":"根据证据，应当遵守本规定。","used_context_ids":["c1"],"abstained":false}'

        return Response()


def test_llm_generator_parses_json_response() -> None:
    generator = LLMGroundedGenerator(
        client=FakeClient(),
        prompt_version="strict_grounded_v1",
        max_contexts=2,
        max_chars_per_context=200,
        max_citation_chars=50,
        require_context_ids=True,
        abstain_when_insufficient=True,
    )
    generation_input = GenerationInput(
        query_id="q1",
        query_text="适用范围是什么",
        question_type="definition",
        answerable=True,
        contexts=[
            ContextItem(
                chunk_id="c1",
                doc_id="d1",
                rank=1,
                score=1.0,
                retrieval_method="hybrid",
                title="通用机场管理规定",
                text="中华人民共和国境内通用机场的建设、使用、运营管理及其相关活动应当遵守本规定。",
            )
        ],
    )
    answer = generator.generate(generation_input)
    assert answer.generation_method == "llm_grounded"
    assert answer.used_context_ids == ["c1"]
    assert len(answer.citations) == 1


class NonJsonClient:
    def complete(self, prompt: str):
        class Response:
            content = "根据给定证据可以回答，但我暂时没有按 JSON 输出。"

        return Response()


class MissingIdsClient:
    def complete(self, prompt: str):
        class Response:
            content = '{"answer":"可以回答。","abstained":false}'

        return Response()


class ThinkingTruncatedClient:
    def complete(self, prompt: str):
        class Response:
            content = """<think>
这是思考过程。
</think>

{
  "answer": "通用机场管理规定适用于中华人民共和国境内通用机场的建设、使用、运营管理及相关活动",
  "abstained": false
"""

        return Response()


def _build_generation_input() -> GenerationInput:
    return GenerationInput(
        query_id="q1",
        query_text="适用范围是什么",
        question_type="definition",
        answerable=True,
        contexts=[
            ContextItem(
                chunk_id="c1",
                doc_id="d1",
                rank=1,
                score=1.0,
                retrieval_method="hybrid",
                title="通用机场管理规定",
                text="中华人民共和国境内通用机场的建设、使用、运营管理及其相关活动应当遵守本规定。",
            )
        ],
    )


def test_llm_generator_falls_back_to_minimal_schema() -> None:
    generator = LLMGroundedGenerator(
        client=NonJsonClient(),
        prompt_version="strict_grounded_v1",
        max_contexts=2,
        max_chars_per_context=200,
        max_citation_chars=50,
        require_context_ids=True,
        abstain_when_insufficient=True,
    )
    answer = generator.generate(_build_generation_input())
    assert answer.abstained is True
    assert answer.used_context_ids == []
    assert answer.metadata["parse_status"] == "minimal_fallback"
    assert answer.metadata["parse_warning"] == "fallback_answer_used"


def test_llm_generator_forces_abstain_when_context_ids_missing() -> None:
    generator = LLMGroundedGenerator(
        client=MissingIdsClient(),
        prompt_version="strict_grounded_v1",
        max_contexts=2,
        max_chars_per_context=200,
        max_citation_chars=50,
        require_context_ids=True,
        abstain_when_insufficient=True,
    )
    answer = generator.generate(_build_generation_input())
    assert answer.abstained is True
    assert answer.used_context_ids == []
    assert answer.metadata["schema_status"] == "forced_abstain_missing_context_ids"
    assert "usable context ids" in answer.metadata["error_message"]


def test_llm_generator_recovers_from_thinking_and_truncated_json() -> None:
    generator = LLMGroundedGenerator(
        client=ThinkingTruncatedClient(),
        prompt_version="strict_grounded_v1",
        max_contexts=2,
        max_chars_per_context=200,
        max_citation_chars=50,
        require_context_ids=True,
        abstain_when_insufficient=True,
    )
    answer = generator.generate(_build_generation_input())
    assert answer.answer.startswith("通用机场管理规定适用于")
    assert answer.abstained is False
    assert answer.used_context_ids == ["c1"]
    assert answer.metadata["parse_status"] == "relaxed_extraction"
    assert answer.metadata["schema_status"] == "inferred_context_ids_from_answer"
    assert answer.metadata["parse_warning"] == "used_context_ids_inferred_from_answer"


def test_prompt_builder_supports_answer_first_variant() -> None:
    prompt = build_grounded_prompt(
        _build_generation_input(),
        prompt_version="grounded_answer_first_v1",
        max_contexts=1,
        max_chars_per_context=120,
    )
    assert "如果上下文已经足以支持问题的主要结论，应直接回答" in prompt
    assert "PromptVersion：grounded_answer_first_v1" in prompt


def test_llm_generation_debug_writes_failure_payload(tmp_path: Path) -> None:
    retrieval_results_path = tmp_path / "retrieval_results.jsonl"
    retrieval_results_path.write_text(
        json.dumps(
            {
                "query_id": "q1",
                "query_text": "适用范围是什么",
                "metadata": {"question_type": "definition", "answerable": True},
                "hits": [
                    {
                        "query_id": "q1",
                        "query_text": "适用范围是什么",
                        "chunk_id": "c1",
                        "doc_id": "d1",
                        "rank": 1,
                        "score": 1.0,
                        "retrieval_method": "hybrid",
                        "chunk_text": "中华人民共和国境内通用机场的建设、使用、运营管理及其相关活动应当遵守本规定。",
                        "title": "通用机场管理规定",
                        "section_path": [],
                        "metadata": {},
                    }
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "llm_debug.json"
    config = LLMGenerationDebugConfig(
        retrieval_results_jsonl=retrieval_results_path,
        output_path=output_path,
        query_id="missing",
        llm_backend="local_transformers",
        llm_modelscope_model_id="Qwen/Qwen3-8B",
    )
    with pytest.raises(ValueError, match="not found"):
        run_llm_generation_debug(config)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["success"] is False
    assert payload["failure_stage"] == "input_lookup"
    assert payload["parse_warning"] is None
    assert payload["error_message"] is None
