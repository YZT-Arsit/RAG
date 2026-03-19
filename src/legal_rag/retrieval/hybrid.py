from __future__ import annotations

from legal_rag.retrieval.tokenize import tokenize_for_bm25
from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit, RetrievalResult


def fuse_results(
    query: QueryRecord,
    *,
    bm25_hits: list[RetrievalHit],
    dense_hits: list[RetrievalHit],
    fusion_type: str,
    alpha: float,
    rrf_k: int,
    top_k: int,
) -> RetrievalResult:
    deduped_bm25_hits = _dedupe_hits(bm25_hits)
    deduped_dense_hits = _dedupe_hits(dense_hits)
    merged_hits: dict[str, RetrievalHit] = {
        _hit_key(hit): hit for hit in [*deduped_bm25_hits, *deduped_dense_hits]
    }

    scored: list[tuple[RetrievalHit, float]] = []
    if fusion_type == "rrf":
        fused_hits = rrf_fusion(deduped_dense_hits, deduped_bm25_hits, k=rrf_k)
        fused_scores = {_hit_key(hit): hit.score for hit in fused_hits}
    else:
        fused_scores = {}

    for chunk_id, hit in merged_hits.items():
        title_bonus = 0.15 * _title_overlap_score(query.query_text, hit.title)
        if fusion_type == "rrf":
            score = fused_scores.get(chunk_id, 0.0) + title_bonus
            retrieval_method = "hybrid_rrf"
        else:
            bm25_scores = _normalize_scores(
                {_hit_key(item): item.score for item in deduped_bm25_hits}
            )
            dense_scores = _normalize_scores(
                {_hit_key(item): item.score for item in deduped_dense_hits}
            )
            score = (
                alpha * bm25_scores.get(chunk_id, 0.0)
                + (1 - alpha) * dense_scores.get(chunk_id, 0.0)
                + title_bonus
            )
            retrieval_method = "hybrid_score"
        scored.append((_copy_hit(hit, retrieval_method=retrieval_method), score))

    ranked = sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]
    fused_hits: list[RetrievalHit] = []
    for rank, (hit, score) in enumerate(ranked, start=1):
        fused_hits.append(
            RetrievalHit(
                query_id=hit.query_id,
                query_text=hit.query_text,
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                rank=rank,
                score=score,
                retrieval_method=hit.retrieval_method,
                chunk_text=hit.chunk_text,
                title=hit.title,
                section_path=hit.section_path,
                metadata=hit.metadata,
            )
        )
    return RetrievalResult(query=query, hits=fused_hits)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    min_score = min(scores.values())
    if max_score == min_score:
        return {key: 1.0 for key in scores}
    return {
        key: (value - min_score) / (max_score - min_score)
        for key, value in scores.items()
    }


def _title_overlap_score(query_text: str, title: str) -> float:
    query_tokens = set(tokenize_for_bm25(query_text))
    title_tokens = set(tokenize_for_bm25(title))
    if not query_tokens or not title_tokens:
        return 0.0
    overlap = len(query_tokens & title_tokens)
    return overlap / len(query_tokens)


def _rrf_score(
    chunk_id: str,
    bm25_hits: list[RetrievalHit],
    dense_hits: list[RetrievalHit],
    *,
    rrf_k: int,
) -> float:
    score = 0.0
    for hits in (bm25_hits, dense_hits):
        for rank, hit in enumerate(hits, start=1):
            if hit.chunk_id == chunk_id:
                score += 1.0 / (rrf_k + rank)
                break
    return score


def rrf_fusion(
    vector_results: list[RetrievalHit],
    bm25_results: list[RetrievalHit],
    k: int = 60,
) -> list[RetrievalHit]:
    all_hits = _dedupe_hits([*vector_results, *bm25_results])
    hit_lookup = {_hit_key(hit): hit for hit in all_hits}
    score_map: dict[str, float] = {key: 0.0 for key in hit_lookup}

    for hits in (_dedupe_hits(vector_results), _dedupe_hits(bm25_results)):
        for rank, hit in enumerate(hits, start=1):
            score_map[_hit_key(hit)] += 1.0 / (k + rank)

    ranked = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
    fused_hits: list[RetrievalHit] = []
    for rank, (key, score) in enumerate(ranked, start=1):
        source_hit = hit_lookup[key]
        fused_hits.append(
            RetrievalHit(
                query_id=source_hit.query_id,
                query_text=source_hit.query_text,
                chunk_id=source_hit.chunk_id,
                doc_id=source_hit.doc_id,
                rank=rank,
                score=score,
                retrieval_method="hybrid_rrf",
                chunk_text=source_hit.chunk_text,
                title=source_hit.title,
                section_path=source_hit.section_path,
                metadata={
                    **source_hit.metadata,
                    "rrf_score": score,
                },
            )
        )
    return fused_hits


def _copy_hit(hit: RetrievalHit, *, retrieval_method: str) -> RetrievalHit:
    return RetrievalHit(
        query_id=hit.query_id,
        query_text=hit.query_text,
        chunk_id=hit.chunk_id,
        doc_id=hit.doc_id,
        rank=hit.rank,
        score=hit.score,
        retrieval_method=retrieval_method,
        chunk_text=hit.chunk_text,
        title=hit.title,
        section_path=hit.section_path,
        metadata=hit.metadata,
    )


def _dedupe_hits(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    deduped: dict[str, RetrievalHit] = {}
    for hit in hits:
        key = _hit_key(hit)
        existing = deduped.get(key)
        if existing is None or hit.rank < existing.rank or hit.score > existing.score:
            deduped[key] = hit
    return sorted(deduped.values(), key=lambda item: (item.rank, -item.score))


def _hit_key(hit: RetrievalHit) -> str:
    if hit.chunk_id:
        return f"chunk:{hit.chunk_id}"
    if hit.doc_id:
        return f"doc:{hit.doc_id}"
    normalized_text = " ".join(hit.chunk_text.split())[:120]
    return f"text:{hit.title}|{normalized_text}"
