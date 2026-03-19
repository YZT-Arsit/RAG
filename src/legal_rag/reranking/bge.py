from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit

DEFAULT_BGE_RERANKER = "BAAI/bge-reranker-v2-m3"


class BGEReranker:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_BGE_RERANKER,
        device: str = "cpu",
        batch_size: int = 8,
        max_length: int = 1024,
        local_model_dir: str | Path | None = None,
        use_modelscope_download: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.local_model_dir = (
            Path(local_model_dir) if local_model_dir is not None else None
        )
        self.use_modelscope_download = use_modelscope_download
        self._tokenizer = None
        self._model = None
        self._torch = None

    def rerank_documents(
        self, query: str, candidate_docs: Sequence[RetrievalHit]
    ) -> list[RetrievalHit]:
        if not candidate_docs:
            return []
        scores = self._score_pairs(query, [hit.chunk_text for hit in candidate_docs])
        rescored = sorted(
            zip(candidate_docs, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        reranked: list[RetrievalHit] = []
        for rank, (hit, score) in enumerate(rescored, start=1):
            reranked.append(
                RetrievalHit(
                    query_id=hit.query_id,
                    query_text=hit.query_text,
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    rank=rank,
                    score=float(score),
                    retrieval_method=f"{hit.retrieval_method}+bge_rerank",
                    chunk_text=hit.chunk_text,
                    title=hit.title,
                    section_path=hit.section_path,
                    metadata={
                        **hit.metadata,
                        "rerank_model": self.model_name,
                        "pre_rerank_score": hit.score,
                        "rerank_score": float(score),
                    },
                )
            )
        return reranked

    def rerank(
        self, query: QueryRecord, hits: list[RetrievalHit], *, top_k: int
    ) -> list[RetrievalHit]:
        return self.rerank_documents(query.query_text, hits)[:top_k]

    def _score_pairs(self, query: str, documents: Sequence[str]) -> list[float]:
        tokenizer, model, torch = self._load_backend()
        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch_docs = list(documents[start : start + self.batch_size])
            pairs = [[query, doc] for doc in batch_docs]
            encoded = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits
            batch_scores = logits.view(-1).detach().cpu().tolist()
            scores.extend(float(score) for score in batch_scores)
        return scores

    def _load_backend(self) -> tuple[Any, Any, Any]:
        if self._tokenizer is not None and self._model is not None and self._torch is not None:
            return self._tokenizer, self._model, self._torch

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_path = self._resolve_model_path()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        resolved_device = _resolve_device(self.device, torch)
        model.to(resolved_device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        return tokenizer, model, torch

    def _resolve_model_path(self) -> str:
        if self.local_model_dir is not None:
            return str(self.local_model_dir)
        if self.use_modelscope_download:
            from modelscope import snapshot_download

            return snapshot_download(self.model_name)
        return self.model_name


def _resolve_device(device: str, torch: Any) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device
