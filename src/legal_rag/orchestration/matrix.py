from __future__ import annotations

from dataclasses import dataclass

from legal_rag.config.schema import ExperimentMatrixConfig


@dataclass(slots=True)
class ExperimentVariant:
    name: str
    chunk_method: str
    retrieval_method: str
    reranker_enabled: bool
    generation_method: str
    context_source: str


def expand_default_matrix(config: ExperimentMatrixConfig) -> list[ExperimentVariant]:
    if config.matrix_scope == "generation_only":
        rerank_suffix = "rerank" if config.base_reranker_enabled else "no_rerank"
        base_variants = [
            (
                f"{config.base_chunk_method}_{config.base_retrieval_method}_{rerank_suffix}",
                config.base_chunk_method,
                config.base_retrieval_method,
                config.base_reranker_enabled,
            )
        ]
    else:
        base_variants = [
            ("fixed_dense", "fixed", "dense", False),
            ("fixed_hybrid", "fixed", "hybrid", False),
            ("fixed_hybrid_rerank", "fixed", "hybrid", True),
            ("structure_dense", "structure", "dense", False),
            ("structure_hybrid", "structure", "hybrid", False),
            ("structure_hybrid_rerank", "structure", "hybrid", True),
        ]
    generation_methods = config.generation_methods or [config.generation_method]
    context_sources = config.generation_context_sources or [
        config.generation_context_source
    ]

    variants: list[ExperimentVariant] = []
    for name, chunk_method, retrieval_method, reranker_enabled in base_variants:
        for generation_method in generation_methods:
            for context_source in context_sources:
                variant_name = f"{name}__gen_{generation_method}__ctx_{context_source}"
                variants.append(
                    ExperimentVariant(
                        name=variant_name,
                        chunk_method=chunk_method,
                        retrieval_method=retrieval_method,
                        reranker_enabled=reranker_enabled,
                        generation_method=generation_method,
                        context_source=context_source,
                    )
                )
    return variants
