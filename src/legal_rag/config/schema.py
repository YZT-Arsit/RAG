from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class AuditConfig(BaseModel):
    input_paths: list[Path]
    output_dir: Path
    normalized_output_path: Path
    detail_csv_path: Path
    report_path: Path
    sample_limit: int | None = None
    min_body_length: int = Field(default=50, ge=0)
    near_duplicate_hamming_threshold: int = Field(default=3, ge=0, le=64)
    near_duplicate_prefix_bits: int = Field(default=16, ge=1, le=32)
    short_body_threshold: int = Field(default=30, ge=0)
    template_similarity_line_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    year_regex: str = r"(19|20)\d{2}"


class CleaningConfig(BaseModel):
    input_jsonl: Path
    output_jsonl: Path
    report_path: Path
    source_aliases: dict[str, str] = Field(default_factory=dict)
    title_strip_patterns: list[str] = Field(default_factory=list)
    body_leading_patterns: list[str] = Field(default_factory=list)
    body_global_patterns: list[str] = Field(default_factory=list)
    min_cleaned_length: int = Field(default=30, ge=0)


class ReviewSampleConfig(BaseModel):
    input_jsonl: Path
    detail_csv: Path
    output_csv: Path
    random_sample_size: int = Field(default=20, ge=0)
    anomaly_sample_size: int = Field(default=20, ge=0)
    duplicate_sample_size: int = Field(default=20, ge=0)
    random_seed: int = 42


class ChunkingConfig(BaseModel):
    input_jsonl: Path
    output_jsonl: Path
    report_path: Path
    method: str = Field(default="both", pattern="^(fixed|structure|both)$")
    fixed_chunk_size: int = Field(default=500, ge=1)
    fixed_chunk_overlap: int = Field(default=100, ge=0)
    structure_max_chunk_size: int = Field(default=800, ge=1)
    structure_min_chunk_size: int = Field(default=120, ge=0)
    structure_sentence_overlap: int = Field(default=1, ge=0, le=3)


class RetrievalConfig(BaseModel):
    chunk_jsonl: Path
    query_jsonl: Path
    output_jsonl: Path
    report_path: Path
    method: str = Field(default="hybrid", pattern="^(bm25|dense|hybrid)$")
    top_k: int = Field(default=5, ge=1)
    retrieve_top_k: int = Field(default=20, ge=1)
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)
    dense_ngram: int = Field(default=3, ge=1)
    hybrid_fusion: str = Field(default="score", pattern="^(score|rrf)$")
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)
    query_transform_enabled: bool = False
    query_transform_multi_query_count: int = Field(default=0, ge=0, le=8)
    query_transform_use_hyde: bool = False
    query_transform_include_original: bool = True
    query_transform_llm_backend: str = Field(
        default="openai_compatible", pattern="^(openai_compatible|local_transformers)$"
    )
    query_transform_llm_base_url: str | None = None
    query_transform_llm_api_key_env: str = "OPENAI_API_KEY"
    query_transform_llm_model_name: str | None = None
    query_transform_llm_modelscope_model_id: str | None = None
    query_transform_llm_local_model_dir: Path | None = None
    query_transform_llm_use_modelscope_download: bool = False
    query_transform_llm_device: str = Field(
        default="auto", pattern="^(auto|cuda|mps|cpu)$"
    )
    query_transform_llm_temperature: float = Field(default=0.0, ge=0.0)
    query_transform_llm_timeout_seconds: int = Field(default=60, ge=1)
    query_transform_llm_max_new_tokens: int = Field(default=256, ge=1)
    reranker_enabled: bool = False
    reranker_type: str = Field(default="heuristic", pattern="^(heuristic|bge)$")
    reranker_title_overlap_weight: float = Field(default=0.4, ge=0.0)
    reranker_body_overlap_weight: float = Field(default=0.4, ge=0.0)
    reranker_structure_overlap_weight: float = Field(default=0.2, ge=0.0)
    reranker_model_name: str | None = None
    reranker_local_model_dir: Path | None = None
    reranker_use_modelscope_download: bool = False
    reranker_device: str = Field(default="auto", pattern="^(auto|cuda|mps|cpu)$")
    reranker_batch_size: int = Field(default=8, ge=1)


class GenerationConfig(BaseModel):
    retrieval_results_jsonl: Path
    output_jsonl: Path
    report_path: Path
    method: str = Field(default="extractive", pattern="^(extractive|llm)$")
    context_source: str = Field(default="processed", pattern="^(raw|processed)$")
    max_contexts: int = Field(default=3, ge=1)
    max_answer_sentences: int = Field(default=3, ge=1)
    max_citation_chars: int = Field(default=160, ge=1)
    max_prompt_context_chars: int = Field(default=300, ge=1)
    min_hit_score: float = Field(default=0.0, ge=0.0)
    llm_backend: str = Field(
        default="openai_compatible", pattern="^(openai_compatible|local_transformers)$"
    )
    llm_base_url: str | None = None
    llm_api_key_env: str = "OPENAI_API_KEY"
    llm_model_name: str | None = None
    llm_modelscope_model_id: str | None = None
    llm_local_model_dir: Path | None = None
    llm_use_modelscope_download: bool = False
    llm_device: str = Field(default="auto", pattern="^(auto|cuda|mps|cpu)$")
    llm_temperature: float = Field(default=0.0, ge=0.0)
    llm_timeout_seconds: int = Field(default=60, ge=1)
    llm_max_new_tokens: int = Field(default=256, ge=1)
    llm_prompt_version: str = "strict_grounded_v1"
    llm_require_context_ids: bool = True
    llm_abstain_when_insufficient: bool = True
    guardrail_enabled: bool = True
    guardrail_min_top_score: float = Field(default=0.0, ge=0.0)
    guardrail_nli_enabled: bool = True
    guardrail_nli_method: str = Field(default="heuristic", pattern="^(heuristic|llm)$")
    guardrail_nli_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    guardrail_require_citation_brackets: bool = True
    guardrail_fail_message: str = (
        "抱歉，根据现有法律库无法给出确切回答，请咨询专业律师。"
    )


class LLMGenerationDebugConfig(BaseModel):
    retrieval_results_jsonl: Path
    output_path: Path
    query_id: str
    context_source: str = Field(default="processed", pattern="^(raw|processed)$")
    max_contexts: int = Field(default=3, ge=1)
    max_citation_chars: int = Field(default=160, ge=1)
    max_prompt_context_chars: int = Field(default=300, ge=1)
    llm_backend: str = Field(
        default="openai_compatible", pattern="^(openai_compatible|local_transformers)$"
    )
    llm_base_url: str | None = None
    llm_api_key_env: str = "OPENAI_API_KEY"
    llm_model_name: str | None = None
    llm_modelscope_model_id: str | None = None
    llm_local_model_dir: Path | None = None
    llm_use_modelscope_download: bool = False
    llm_device: str = Field(default="auto", pattern="^(auto|cuda|mps|cpu)$")
    llm_temperature: float = Field(default=0.0, ge=0.0)
    llm_timeout_seconds: int = Field(default=60, ge=1)
    llm_max_new_tokens: int = Field(default=256, ge=1)
    llm_prompt_version: str = "strict_grounded_v1"
    llm_require_context_ids: bool = True
    llm_abstain_when_insufficient: bool = True


class AutoTestsetConfig(BaseModel):
    chunk_jsonl: Path
    output_jsonl: Path
    report_path: Path
    sample_size: int = Field(default=100, ge=1)
    random_seed: int = 42
    difficulty: str = "hard"
    llm_backend: str = Field(
        default="openai_compatible", pattern="^(openai_compatible|local_transformers)$"
    )
    llm_base_url: str | None = None
    llm_api_key_env: str = "OPENAI_API_KEY"
    llm_model_name: str | None = None
    llm_modelscope_model_id: str | None = None
    llm_local_model_dir: Path | None = None
    llm_use_modelscope_download: bool = False
    llm_device: str = Field(default="auto", pattern="^(auto|cuda|mps|cpu)$")
    llm_temperature: float = Field(default=0.2, ge=0.0)
    llm_timeout_seconds: int = Field(default=60, ge=1)
    llm_max_new_tokens: int = Field(default=512, ge=1)


class AutoEvalVariantConfig(BaseModel):
    name: str
    retrieval_config_path: Path
    generation_config_path: Path
    context_config_path: Path | None = None


class AutoEvalConfig(BaseModel):
    testset_jsonl: Path
    output_json_path: Path
    report_path: Path
    variants: list[AutoEvalVariantConfig]
    include_contexts_in_report: bool = True


class ContextProcessingConfig(BaseModel):
    input_retrieval_results_jsonl: Path
    output_jsonl: Path
    report_path: Path
    dedupe_enabled: bool = True
    max_chunks: int = Field(default=6, ge=1)
    max_per_doc: int = Field(default=3, ge=1)
    compression_enabled: bool = True
    max_sentences_total: int = Field(default=8, ge=1)
    max_sentences_per_chunk: int = Field(default=2, ge=1)


class RetrievalEvalConfig(BaseModel):
    retrieval_results_jsonl: Path
    gold_jsonl: Path
    detail_csv_path: Path
    report_path: Path
    benchmark_mode: bool = False
    ks: list[int] = Field(default_factory=lambda: [1, 3, 5])


class GenerationEvalConfig(BaseModel):
    answers_jsonl: Path
    gold_jsonl: Path
    detail_csv_path: Path
    report_path: Path
    benchmark_mode: bool = False


class BenchmarkValidateConfig(BaseModel):
    benchmark_jsonl: Path
    report_path: Path


class BenchmarkGenerationConfig(BaseModel):
    input_chunk_jsonl: Path
    candidates_output_jsonl: Path
    deduped_output_jsonl: Path
    benchmark_output_jsonl: Path
    report_path: Path
    preferred_chunk_method: str = Field(
        default="structure", pattern="^(fixed|structure|both)$"
    )
    target_candidate_count: int = Field(default=1000, ge=1)
    final_target_counts: dict[str, int] = Field(
        default_factory=lambda: {
            "definition": 50,
            "condition": 50,
            "procedure": 50,
            "responsibility": 35,
            "comparison": 35,
            "unanswerable": 20,
        }
    )
    candidate_target_counts: dict[str, int] = Field(
        default_factory=lambda: {
            "definition": 210,
            "condition": 210,
            "procedure": 210,
            "responsibility": 150,
            "comparison": 140,
            "unanswerable": 80,
        }
    )
    min_chunk_text_length: int = Field(default=80, ge=1)
    min_question_chars: int = Field(default=8, ge=1)
    max_question_chars: int = Field(default=80, ge=1)
    semantic_dedup_threshold: float = Field(default=0.82, ge=0.0, le=1.0)
    min_question_evidence_overlap: float = Field(default=0.12, ge=0.0, le=1.0)
    max_candidates_per_doc_per_type: int = Field(default=6, ge=1)
    random_seed: int = 42


class AblationVariantConfig(BaseModel):
    name: str
    retrieval_report_path: Path
    generation_report_path: Path


class AblationConfig(BaseModel):
    report_path: Path
    variants: list[AblationVariantConfig]


class ExperimentMatrixConfig(BaseModel):
    experiment_name: str
    cleaned_input_jsonl: Path
    query_jsonl: Path
    benchmark_jsonl: Path
    output_root: Path
    matrix_scope: str = Field(default="full", pattern="^(full|generation_only)$")
    base_chunk_method: str = Field(default="structure", pattern="^(fixed|structure)$")
    base_retrieval_method: str = Field(
        default="hybrid", pattern="^(dense|hybrid|bm25)$"
    )
    base_reranker_enabled: bool = True
    chunk_fixed_chunk_size: int = Field(default=500, ge=1)
    chunk_fixed_overlap: int = Field(default=100, ge=0)
    chunk_structure_max_size: int = Field(default=800, ge=1)
    chunk_structure_min_size: int = Field(default=120, ge=0)
    retrieval_top_k: int = Field(default=5, ge=1)
    retrieval_first_stage_top_k: int = Field(default=20, ge=1)
    hybrid_fusion: str = Field(default="rrf", pattern="^(score|rrf)$")
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1)
    context_max_chunks: int = Field(default=4, ge=1)
    context_max_per_doc: int = Field(default=2, ge=1)
    context_max_sentences_total: int = Field(default=6, ge=1)
    context_max_sentences_per_chunk: int = Field(default=2, ge=1)
    generation_method: str = Field(default="extractive", pattern="^(extractive|llm)$")
    generation_context_source: str = Field(
        default="processed", pattern="^(raw|processed)$"
    )
    generation_methods: list[str] | None = None
    generation_context_sources: list[str] | None = None
    generation_max_contexts: int = Field(default=3, ge=1)
    generation_max_answer_sentences: int = Field(default=3, ge=1)
    generation_max_citation_chars: int = Field(default=160, ge=1)
    generation_min_hit_score: float = Field(default=0.0, ge=0.0)
    generation_max_prompt_context_chars: int = Field(default=300, ge=1)
    generation_llm_backend: str = Field(
        default="openai_compatible", pattern="^(openai_compatible|local_transformers)$"
    )
    generation_llm_base_url: str | None = None
    generation_llm_api_key_env: str = "OPENAI_API_KEY"
    generation_llm_model_name: str | None = None
    generation_llm_modelscope_model_id: str | None = None
    generation_llm_local_model_dir: Path | None = None
    generation_llm_use_modelscope_download: bool = False
    generation_llm_device: str = Field(default="auto", pattern="^(auto|cuda|mps|cpu)$")
    generation_llm_temperature: float = Field(default=0.0, ge=0.0)
    generation_llm_timeout_seconds: int = Field(default=60, ge=1)
    generation_llm_max_new_tokens: int = Field(default=256, ge=1)
    generation_llm_prompt_version: str = "strict_grounded_v1"
    generation_llm_require_context_ids: bool = True
    generation_llm_abstain_when_insufficient: bool = True
    eval_ks: list[int] = Field(default_factory=lambda: [1, 3, 5])
    bundle_filename: str = "artifacts_bundle.zip"


class ErrorAnalysisConfig(BaseModel):
    benchmark_jsonl: Path
    retrieval_results_jsonl: Path
    answers_jsonl: Path
    detail_csv_path: Path
    report_path: Path
