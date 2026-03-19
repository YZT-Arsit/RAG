from __future__ import annotations

import json
import random
from pathlib import Path

from legal_rag.chunking.io import iter_chunks
from legal_rag.config.loader import load_yaml_config
from legal_rag.config.schema import (
    AutoEvalConfig,
    AutoTestsetConfig,
    ContextProcessingConfig,
    GenerationConfig,
    RetrievalConfig,
)
from legal_rag.contexting.compressor import compress_hits
from legal_rag.contexting.dedupe import dedupe_hits
from legal_rag.contexting.selector import select_hits
from legal_rag.evaluation.io import load_answers
from legal_rag.generation.alignment import run_citation_alignment_check
from legal_rag.generation.guardrails import apply_guardrails
from legal_rag.generation.io import write_answers
from legal_rag.generation.service import build_generation_input, build_generator, build_llm_client
from legal_rag.retrieval.service import build_retrieval_pipeline
from legal_rag.schemas.evaluation import AutoEvalSample
from legal_rag.schemas.generation import GroundedAnswer
from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit, RetrievalResult


def run_auto_testset_generation(config: AutoTestsetConfig) -> None:
    chunks = list(iter_chunks(config.chunk_jsonl))
    random_generator = random.Random(config.random_seed)
    sample_size = min(config.sample_size, len(chunks))
    sampled_chunks = random_generator.sample(chunks, sample_size)
    client = build_llm_client(
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
    )

    samples: list[AutoEvalSample] = []
    for index, chunk in enumerate(sampled_chunks, start=1):
        prompt = _build_testset_prompt(
            chunk_text=chunk.text,
            title=chunk.title,
            difficulty=config.difficulty,
        )
        response = client.complete(prompt)
        payload = _parse_json_object(response.content)
        question = str(payload.get("question", "")).strip()
        answer = str(payload.get("answer", "")).strip()
        if not question or not answer:
            continue
        samples.append(
            AutoEvalSample(
                query_id=f"auto-{index}",
                query=question,
                ground_truth_answer=answer,
                ground_truth_chunk_id=chunk.chunk_id,
                ground_truth_doc_id=chunk.doc_id,
                evidence_text=chunk.text,
                metadata={
                    "title": chunk.title,
                    "difficulty": config.difficulty,
                },
            )
        )
    _write_auto_testset(config.output_jsonl, samples)
    _write_testset_report(config.report_path, samples)


def run_auto_eval(config: AutoEvalConfig) -> None:
    testset = _load_auto_testset(config.testset_jsonl)
    variant_reports: list[dict[str, object]] = []
    for variant in config.variants:
        retrieval_config = RetrievalConfig.model_validate(
            load_yaml_config(variant.retrieval_config_path)
        )
        generation_config = GenerationConfig.model_validate(
            load_yaml_config(variant.generation_config_path)
        )
        context_config = None
        if variant.context_config_path is not None:
            context_config = ContextProcessingConfig.model_validate(
                load_yaml_config(variant.context_config_path)
            )
        pipeline_report = _run_variant_eval(
            variant_name=variant.name,
            testset=testset,
            retrieval_config=retrieval_config,
            generation_config=generation_config,
            context_config=context_config,
            include_contexts=config.include_contexts_in_report,
        )
        variant_reports.append(pipeline_report)

    output_payload = {
        "testset_size": len(testset),
        "variants": variant_reports,
    }
    config.output_json_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_json_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_auto_eval_report(config.report_path, output_payload)


def _run_variant_eval(
    *,
    variant_name: str,
    testset: list[AutoEvalSample],
    retrieval_config: RetrievalConfig,
    generation_config: GenerationConfig,
    context_config: ContextProcessingConfig | None,
    include_contexts: bool,
) -> dict[str, object]:
    chunks = list(iter_chunks(retrieval_config.chunk_jsonl))
    retrieval_pipeline = build_retrieval_pipeline(retrieval_config, chunks)
    generator = build_generator(generation_config)
    answers: list[GroundedAnswer] = []
    per_query: list[dict[str, object]] = []

    for sample in testset:
        query = QueryRecord(query_id=sample.query_id, query_text=sample.query)
        retrieval_result = retrieval_pipeline.retrieve(query)
        if context_config is not None:
            retrieval_result = _process_retrieval_result(retrieval_result, context_config)
        generation_input = build_generation_input(
            retrieval_result, context_source=generation_config.context_source
        )
        answer = generator.generate(generation_input)
        answer.metadata["citation_alignment"] = run_citation_alignment_check(
            answer, generation_input
        )
        if generation_config.guardrail_enabled:
            answer = apply_guardrails(
                answer,
                generation_input,
                min_top_score=generation_config.guardrail_min_top_score,
                nli_enabled=generation_config.guardrail_nli_enabled,
                nli_threshold=generation_config.guardrail_nli_threshold,
                require_citation_brackets=generation_config.guardrail_require_citation_brackets,
                fail_message=generation_config.guardrail_fail_message,
            )
        answers.append(answer)
        metrics = _score_auto_eval_sample(
            sample=sample,
            answer=answer,
            retrieval_result=retrieval_result,
        )
        row: dict[str, object] = {
            "query_id": sample.query_id,
            "query": sample.query,
            "ground_truth_chunk_id": sample.ground_truth_chunk_id,
            "metrics": metrics,
            "answer": answer.answer,
        }
        if include_contexts:
            row["retrieved_contexts"] = [
                {
                    "chunk_id": hit.chunk_id,
                    "score": hit.score,
                    "rank": hit.rank,
                    "title": hit.title,
                }
                for hit in retrieval_result.hits[:5]
            ]
        per_query.append(row)

    averages = _aggregate_auto_eval_rows(per_query)
    return {
        "name": variant_name,
        "aggregate_metrics": averages,
        "per_query": per_query,
    }


def _score_auto_eval_sample(
    *,
    sample: AutoEvalSample,
    answer: GroundedAnswer,
    retrieval_result: RetrievalResult,
) -> dict[str, float]:
    retrieved_ids = [hit.chunk_id for hit in retrieval_result.hits]
    answer_tokens = set(_normalize_tokens(answer.answer))
    question_tokens = set(_normalize_tokens(sample.query))
    truth_tokens = set(_normalize_tokens(sample.ground_truth_answer))
    context_precision = _context_precision(retrieval_result.hits, sample.ground_truth_chunk_id)
    answer_relevance = (
        len(answer_tokens & question_tokens) / len(question_tokens)
        if question_tokens
        else 0.0
    )
    faithfulness = (
        len(answer_tokens & set(_normalize_tokens(sample.evidence_text))) / len(answer_tokens)
        if answer_tokens
        else 0.0
    )
    answer_correctness = (
        len(answer_tokens & truth_tokens) / len(truth_tokens)
        if truth_tokens
        else 0.0
    )
    recall_at_3 = 1.0 if sample.ground_truth_chunk_id in retrieved_ids[:3] else 0.0
    recall_at_5 = 1.0 if sample.ground_truth_chunk_id in retrieved_ids[:5] else 0.0
    return {
        "faithfulness": faithfulness,
        "answer_relevance": answer_relevance,
        "context_precision": context_precision,
        "answer_correctness": answer_correctness,
        "recall@3": recall_at_3,
        "recall@5": recall_at_5,
    }


def _context_precision(hits: list[RetrievalHit], ground_truth_chunk_id: str) -> float:
    if not hits:
        return 0.0
    precision_sum = 0.0
    hit_count = 0
    for index, hit in enumerate(hits, start=1):
        if hit.chunk_id != ground_truth_chunk_id:
            continue
        hit_count += 1
        precision_sum += hit_count / index
    return precision_sum / hit_count if hit_count else 0.0


def _process_retrieval_result(
    result: RetrievalResult, config: ContextProcessingConfig
) -> RetrievalResult:
    hits = result.hits
    if config.dedupe_enabled:
        hits = dedupe_hits(hits)
    hits = select_hits(hits, max_chunks=config.max_chunks, max_per_doc=config.max_per_doc)
    if config.compression_enabled:
        hits = compress_hits(
            result.query,
            hits,
            max_sentences_total=config.max_sentences_total,
            max_sentences_per_chunk=config.max_sentences_per_chunk,
        )
    return RetrievalResult(query=result.query, hits=hits)


def _aggregate_auto_eval_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {}
    metric_names = list(rows[0]["metrics"].keys())
    return {
        metric: sum(float(row["metrics"][metric]) for row in rows) / len(rows)
        for metric in metric_names
    }


def _build_testset_prompt(*, chunk_text: str, title: str, difficulty: str) -> str:
    return (
        "你是法律数据集构建助手。请基于给定法律文本生成一个有一定难度但答案可以直接从文本中得到的问题。\n"
        "必须只输出 JSON，不要解释。\n"
        f'JSON 格式：{{"question":"...","answer":"..."}}\n'
        f"难度要求：{difficulty}\n"
        f"标题：{title}\n"
        f"文本：{chunk_text}\n"
    )


def _normalize_tokens(text: str) -> list[str]:
    from legal_rag.retrieval.tokenize import tokenize_for_bm25

    return tokenize_for_bm25(text)


def _parse_json_object(content: str) -> dict[str, object]:
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            payload = json.loads(content[start : end + 1])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    return {}


def _write_auto_testset(path: Path, samples: list[AutoEvalSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            payload = {
                "query_id": sample.query_id,
                "query": sample.query,
                "ground_truth_answer": sample.ground_truth_answer,
                "ground_truth_chunk_id": sample.ground_truth_chunk_id,
                "ground_truth_doc_id": sample.ground_truth_doc_id,
                "evidence_text": sample.evidence_text,
                "metadata": sample.metadata,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_auto_testset(path: Path) -> list[AutoEvalSample]:
    samples: list[AutoEvalSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            samples.append(
                AutoEvalSample(
                    query_id=payload["query_id"],
                    query=payload["query"],
                    ground_truth_answer=payload["ground_truth_answer"],
                    ground_truth_chunk_id=payload["ground_truth_chunk_id"],
                    ground_truth_doc_id=payload["ground_truth_doc_id"],
                    evidence_text=payload["evidence_text"],
                    metadata=payload.get("metadata", {}),
                )
            )
    return samples


def _write_testset_report(path: Path, samples: list[AutoEvalSample]) -> None:
    lines = [
        "# Auto Testset Report",
        "",
        f"- Samples generated: {len(samples)}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_auto_eval_report(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Automated Evaluation Report",
        "",
        f"- Testset size: {payload['testset_size']}",
        "",
    ]
    for variant in payload["variants"]:
        lines.append(f"## {variant['name']}")
        for metric, value in variant["aggregate_metrics"].items():
            lines.append(f"- `{metric}`: {value:.4f}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
