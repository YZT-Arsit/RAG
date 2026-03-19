from __future__ import annotations

from legal_rag.config.schema import RetrievalConfig
from legal_rag.reranking.bge import BGEReranker, DEFAULT_BGE_RERANKER
from legal_rag.reranking.heuristic import HeuristicReranker


def build_reranker(config: RetrievalConfig) -> HeuristicReranker | BGEReranker | None:
    if not config.reranker_enabled:
        return None
    if config.reranker_type == "heuristic":
        return HeuristicReranker(
            title_overlap_weight=config.reranker_title_overlap_weight,
            body_overlap_weight=config.reranker_body_overlap_weight,
            structure_overlap_weight=config.reranker_structure_overlap_weight,
        )
    if config.reranker_type == "bge":
        return BGEReranker(
            model_name=config.reranker_model_name or DEFAULT_BGE_RERANKER,
            device=config.reranker_device,
            batch_size=config.reranker_batch_size,
            local_model_dir=config.reranker_local_model_dir,
            use_modelscope_download=config.reranker_use_modelscope_download,
        )
    msg = f"Unsupported reranker type: {config.reranker_type}"
    raise ValueError(msg)
