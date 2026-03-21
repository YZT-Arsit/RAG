from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

from legal_rag.retrieval.bm25 import _build_hits
from legal_rag.retrieval.embeddings import (
    BGEEmbeddingEncoder,
    DEFAULT_DENSE_EMBEDDING_MODEL,
)
from legal_rag.retrieval.faiss_index import load_chunk_metadata
from legal_rag.retrieval.tokenize import char_ngrams
from legal_rag.schemas.chunk import Chunk
from legal_rag.schemas.retrieval import QueryRecord, RetrievalResult


class DenseBaselineRetriever:
    def __init__(self, chunks: list[Chunk], *, ngram: int) -> None:
        self.chunks = chunks
        self.ngram = ngram
        self.doc_vectors = [
            Counter(char_ngrams(_retrieval_text(chunk), ngram)) for chunk in chunks
        ]
        self.doc_freqs = self._compute_doc_freqs()
        self.doc_count = len(chunks)
        self.doc_norms = [
            self._norm(self._tfidf_weights(vector)) for vector in self.doc_vectors
        ]

    def retrieve(self, query: QueryRecord, *, top_k: int) -> RetrievalResult:
        query_vector = Counter(char_ngrams(query.query_text, self.ngram))
        query_weights = self._tfidf_weights(query_vector)
        query_norm = self._norm(query_weights)

        scored: list[tuple[Chunk, float]] = []
        for chunk, doc_vector, doc_norm in zip(
            self.chunks, self.doc_vectors, self.doc_norms, strict=True
        ):
            if doc_norm == 0 or query_norm == 0:
                continue
            score = self._cosine_similarity(
                query_weights, query_norm, doc_vector, doc_norm
            )
            if score > 0:
                scored.append((chunk, score))

        hits = _build_hits(query, scored, method="dense", top_k=top_k)
        return RetrievalResult(query=query, hits=hits)

    def _compute_doc_freqs(self) -> Counter[str]:
        doc_freqs: Counter[str] = Counter()
        for vector in self.doc_vectors:
            doc_freqs.update(vector.keys())
        return doc_freqs

    def _tfidf_weights(self, vector: Counter[str]) -> dict[str, float]:
        weights: dict[str, float] = {}
        for token, tf in vector.items():
            doc_freq = self.doc_freqs.get(token, 0)
            idf = math.log((1 + self.doc_count) / (1 + doc_freq)) + 1
            weights[token] = tf * idf
        return weights

    def _norm(self, weights: dict[str, float]) -> float:
        return math.sqrt(sum(value * value for value in weights.values()))

    def _cosine_similarity(
        self,
        query_weights: dict[str, float],
        query_norm: float,
        doc_vector: Counter[str],
        doc_norm: float,
    ) -> float:
        doc_weights = self._tfidf_weights(doc_vector)
        numerator = 0.0
        for token, weight in query_weights.items():
            numerator += weight * doc_weights.get(token, 0.0)
        return numerator / (query_norm * doc_norm)


class FaissDenseRetriever:
    def __init__(
        self,
        *,
        index,
        chunks: list[Chunk],
        encoder: BGEEmbeddingEncoder,
    ) -> None:
        self.index = index
        self.chunks = chunks
        self.encoder = encoder

    @classmethod
    def from_disk(
        cls,
        *,
        index_path: Path,
        metadata_path: Path,
        model_name: str | None,
        modelscope_model_id: str | None,
        local_model_dir: Path | None,
        use_modelscope_download: bool,
        device: str,
        batch_size: int,
        max_length: int,
    ) -> "FaissDenseRetriever":
        try:
            import faiss
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dep
            msg = "faiss is required for dense_backend=faiss."
            raise RuntimeError(msg) from exc

        index = faiss.read_index(str(index_path))
        chunks = load_chunk_metadata(metadata_path)
        encoder = BGEEmbeddingEncoder(
            model_name=model_name or DEFAULT_DENSE_EMBEDDING_MODEL,
            modelscope_model_id=modelscope_model_id,
            local_model_dir=local_model_dir,
            use_modelscope_download=use_modelscope_download,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
        )
        return cls(index=index, chunks=chunks, encoder=encoder)

    def retrieve(self, query: QueryRecord, *, top_k: int) -> RetrievalResult:
        if not self.chunks:
            return RetrievalResult(query=query, hits=[])
        query_matrix = self.encoder.encode([query.query_text])
        scores, indices = self.index.search(query_matrix, top_k)
        scored: list[tuple[Chunk, float, int]] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0 or idx >= len(self.chunks):
                continue
            scored.append((self.chunks[idx], float(score), int(idx)))
        hits = _build_hits(query, scored, method="dense_faiss", top_k=top_k)
        return RetrievalResult(query=query, hits=hits)


def _retrieval_text(chunk: Chunk) -> str:
    heading_prefix = str(chunk.metadata.get("heading_prefix", "")).strip()
    sub_title = str(chunk.metadata.get("sub_title", "")).strip()
    intro_title = str(chunk.metadata.get("intro_title", "")).strip()
    article_label = str(chunk.metadata.get("article_label", "")).strip()
    return " ".join(
        part
        for part in [chunk.title, sub_title, intro_title, heading_prefix, article_label, chunk.text]
        if part
    )
