from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    BM25Okapi = None

from legal_rag.schemas.chunk import Chunk
from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit, RetrievalResult
from legal_rag.retrieval.tokenize import tokenize_for_bm25


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], *, k1: float, b: float) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.documents = [_retrieval_text(chunk) for chunk in chunks]
        self.doc_tokens = [tokenize_for_bm25(text) for text in self.documents]
        self.index = (
            BM25Okapi(self.doc_tokens, k1=k1, b=b)
            if BM25Okapi is not None and self.doc_tokens
            else None
        )
        self.doc_term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.doc_freqs = self._compute_doc_freqs()
        self.doc_count = len(chunks)

    def retrieve(self, query: QueryRecord, *, top_k: int) -> RetrievalResult:
        hits = self.get_bm25_scores(query.query_text, top_k=top_k, query_record=query)
        return RetrievalResult(query=query, hits=hits)

    def get_bm25_scores(
        self,
        query: str,
        top_k: int,
        *,
        query_record: QueryRecord | None = None,
        query_id: str = "bm25_query",
        query_text: str | None = None,
    ) -> list[RetrievalHit]:
        retrieval_query = query_record or QueryRecord(
            query_id=query_id,
            query_text=query_text or query,
        )
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []
        scores = (
            self.index.get_scores(query_tokens)
            if self.index is not None
            else self._fallback_scores(query_tokens)
        )
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )
        scored_chunks = [
            (self.chunks[index], float(score), index)
            for index, score in ranked[:top_k]
            if score > 0
        ]
        return _build_hits(retrieval_query, scored_chunks, method="bm25", top_k=top_k)

    def _compute_doc_freqs(self) -> Counter[str]:
        doc_freqs: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            doc_freqs.update(set(tokens))
        return doc_freqs

    def _fallback_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for term_freqs, doc_length in zip(
            self.doc_term_freqs, self.doc_lengths, strict=True
        ):
            score = 0.0
            for token in query_tokens:
                freq = term_freqs.get(token, 0)
                if freq == 0:
                    continue
                doc_freq = self.doc_freqs.get(token, 0)
                idf = math.log(1 + (self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5))
                denom = freq + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1e-9)
                )
                score += idf * ((freq * (self.k1 + 1)) / denom)
            scores.append(score)
        return scores


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


def _build_hits(
    query: QueryRecord,
    scored_chunks: Sequence[tuple[Chunk, float] | tuple[Chunk, float, int]],
    *,
    method: str,
    top_k: int,
) -> list[RetrievalHit]:
    hits: list[RetrievalHit] = []
    ranked = sorted(scored_chunks, key=lambda item: item[1], reverse=True)[:top_k]
    for rank, item in enumerate(ranked, start=1):
        if len(item) == 3:
            chunk, score, original_index = item
        else:
            chunk, score = item
            original_index = None
        hits.append(
            RetrievalHit(
                query_id=query.query_id,
                query_text=query.query_text,
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                rank=rank,
                score=float(score),
                retrieval_method=method,
                chunk_text=chunk.text,
                title=chunk.title,
                section_path=chunk.section_path,
                metadata={
                    "published_year": chunk.published_year,
                    "canonical_source": chunk.canonical_source,
                    "original_index": original_index,
                    "chunk_method": chunk.chunk_method,
                },
            )
        )
    return hits
