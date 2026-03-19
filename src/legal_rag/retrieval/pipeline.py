from __future__ import annotations

from collections.abc import Sequence

from legal_rag.config.schema import RetrievalConfig
from legal_rag.reranking.base import BaseReranker
from legal_rag.retrieval.bm25 import BM25Retriever
from legal_rag.retrieval.dense import DenseBaselineRetriever
from legal_rag.retrieval.hybrid import fuse_results
from legal_rag.retrieval.query_transform import LLMQueryTransformer, TransformedQuery
from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit, RetrievalResult


class RetrievalPipeline:
    def __init__(
        self,
        *,
        config: RetrievalConfig,
        bm25_retrievers: Sequence[tuple[str, BM25Retriever]],
        dense_retrievers: Sequence[tuple[str, DenseBaselineRetriever]],
        reranker: BaseReranker | None,
        query_transformer: LLMQueryTransformer | None = None,
    ) -> None:
        self.config = config
        self.bm25_retrievers = list(bm25_retrievers)
        self.dense_retrievers = list(dense_retrievers)
        self.reranker = reranker
        self.query_transformer = query_transformer

    def retrieve(self, query: QueryRecord) -> RetrievalResult:
        transformed_queries = self._expand_query(query)
        if self.config.method == "bm25":
            result = RetrievalResult(
                query=query,
                hits=self._retrieve_multi_corpus(
                    transformed_queries,
                    retrievers=self.bm25_retrievers,
                    top_k=self.config.retrieve_top_k,
                    method_name="bm25",
                ),
            )
        elif self.config.method == "dense":
            result = RetrievalResult(
                query=query,
                hits=self._retrieve_multi_corpus(
                    transformed_queries,
                    retrievers=self.dense_retrievers,
                    top_k=self.config.retrieve_top_k,
                    method_name="dense",
                ),
            )
        else:
            bm25_result = RetrievalResult(
                query=query,
                hits=self._retrieve_multi_corpus(
                    [item for item in transformed_queries if item.source != "hyde"],
                    retrievers=self.bm25_retrievers,
                    top_k=self.config.retrieve_top_k,
                    method_name="bm25",
                ),
            )
            dense_result = RetrievalResult(
                query=query,
                hits=self._retrieve_multi_corpus(
                    transformed_queries,
                    retrievers=self.dense_retrievers,
                    top_k=self.config.retrieve_top_k,
                    method_name="dense",
                ),
            )
            result = fuse_results(
                query,
                bm25_hits=bm25_result.hits,
                dense_hits=dense_result.hits,
                fusion_type=self.config.hybrid_fusion,
                alpha=self.config.hybrid_alpha,
                rrf_k=self.config.rrf_k,
                top_k=self.config.retrieve_top_k,
            )

        if self.reranker is None:
            return _truncate_result(result, top_k=self.config.top_k)

        reranked_hits = self.reranker.rerank(
            query, result.hits, top_k=self.config.top_k
        )
        return RetrievalResult(query=result.query, hits=reranked_hits)

    def _expand_query(self, query: QueryRecord) -> list[TransformedQuery]:
        if self.query_transformer is None:
            return [TransformedQuery(text=query.query_text, source="original")]
        transformed = self.query_transformer.expand_query(query)
        return transformed or [TransformedQuery(text=query.query_text, source="original")]

    def _retrieve_multi_corpus(
        self,
        transformed_queries: Sequence[TransformedQuery],
        *,
        retrievers: Sequence[tuple[str, BM25Retriever | DenseBaselineRetriever]],
        top_k: int,
        method_name: str,
    ) -> list[RetrievalHit]:
        if not retrievers:
            return []
        hits_by_source: list[tuple[str, str, list[RetrievalHit]]] = []
        for variant in transformed_queries:
            for corpus_name, retriever in retrievers:
                query = QueryRecord(
                    query_id=f"{variant.source}:{variant.text}",
                    query_text=variant.text,
                )
                result = retriever.retrieve(query, top_k=top_k)
                hits_by_source.append(
                    (
                        corpus_name,
                        variant.source,
                        [
                            _copy_hit(
                                hit,
                                rank=hit.rank,
                                score=hit.score,
                                retrieval_method=method_name,
                                metadata={
                                    **hit.metadata,
                                    "retrieval_corpus": corpus_name,
                                    "query_variant_source": variant.source,
                                    "query_variant_text": variant.text,
                                },
                            )
                            for hit in result.hits
                        ],
                    )
                )
        return _merge_corpus_hits(
            hits_by_source, top_k=top_k, method_name=method_name
        )


def _truncate_result(result: RetrievalResult, *, top_k: int) -> RetrievalResult:
    hits = []
    for rank, hit in enumerate(result.hits[:top_k], start=1):
        hits.append(
            type(hit)(
                query_id=hit.query_id,
                query_text=hit.query_text,
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                rank=rank,
                score=hit.score,
                retrieval_method=hit.retrieval_method,
                chunk_text=hit.chunk_text,
                title=hit.title,
                section_path=hit.section_path,
                metadata=hit.metadata,
            )
        )
    return RetrievalResult(query=result.query, hits=hits)


def _merge_corpus_hits(
    hits_by_corpus: Sequence[tuple[str, str, list[RetrievalHit]]],
    *,
    top_k: int,
    method_name: str,
) -> list[RetrievalHit]:
    normalized_by_corpus = {
        (corpus, source): _normalize_scores(hits)
        for corpus, source, hits in hits_by_corpus
    }
    combined: dict[str, tuple[RetrievalHit, float]] = {}

    for corpus_name, query_source, hits in hits_by_corpus:
        normalized_scores = normalized_by_corpus[(corpus_name, query_source)]
        for hit in hits:
            key = hit.chunk_id or hit.doc_id
            score = (
                0.65 * normalized_scores.get(key, 0.0)
                + 0.30 * (1.0 / (60 + hit.rank))
                + _chunk_method_prior(hit)
                + _query_source_prior(query_source)
            )
            existing = combined.get(key)
            candidate_hit = _copy_hit(
                hit,
                rank=hit.rank,
                score=score,
                retrieval_method=method_name,
                metadata={
                    **hit.metadata,
                    "corpus_fusion_score": score,
                },
            )
            if existing is None or score > existing[1]:
                combined[key] = (candidate_hit, score)

    ranked = sorted(combined.values(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        _copy_hit(
            hit,
            rank=rank,
            score=score,
            retrieval_method=method_name,
            metadata=hit.metadata,
        )
        for rank, (hit, score) in enumerate(ranked, start=1)
    ]


def _normalize_scores(hits: Sequence[RetrievalHit]) -> dict[str, float]:
    if not hits:
        return {}
    raw_scores = {hit.chunk_id or hit.doc_id: hit.score for hit in hits}
    max_score = max(raw_scores.values())
    min_score = min(raw_scores.values())
    if max_score == min_score:
        return {key: 1.0 for key in raw_scores}
    return {
        key: (value - min_score) / (max_score - min_score)
        for key, value in raw_scores.items()
    }


def _chunk_method_prior(hit: RetrievalHit) -> float:
    chunk_method = str(hit.metadata.get("chunk_method", ""))
    if chunk_method == "structure":
        return 0.08
    if chunk_method == "fixed":
        return 0.02
    if "::structure::" in hit.chunk_id:
        return 0.08
    if "::fixed::" in hit.chunk_id:
        return 0.02
    return 0.0


def _query_source_prior(query_source: str) -> float:
    if query_source == "original":
        return 0.08
    if query_source == "multi_query":
        return 0.05
    if query_source == "hyde":
        return 0.03
    return 0.0


def _copy_hit(
    hit: RetrievalHit,
    *,
    rank: int,
    score: float,
    retrieval_method: str,
    metadata: dict,
) -> RetrievalHit:
    return RetrievalHit(
        query_id=hit.query_id,
        query_text=hit.query_text,
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        rank=rank,
        score=score,
        retrieval_method=retrieval_method,
        chunk_text=hit.chunk_text,
        title=hit.title,
        section_path=hit.section_path,
        metadata=metadata,
    )
