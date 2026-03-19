from __future__ import annotations

from collections import defaultdict

from legal_rag.contexting.sentences import split_sentences
from legal_rag.retrieval.tokenize import tokenize_for_bm25
from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit


def compress_hits(
    query: QueryRecord,
    hits: list[RetrievalHit],
    *,
    max_sentences_total: int,
    max_sentences_per_chunk: int,
) -> list[RetrievalHit]:
    remaining_total = max_sentences_total
    compressed: list[RetrievalHit] = []
    query_tokens = set(tokenize_for_bm25(query.query_text))

    for hit in hits:
        if remaining_total <= 0:
            break
        sentences = split_sentences(hit.chunk_text)
        ranked_sentences = sorted(
            sentences,
            key=lambda sentence: _sentence_score(query_tokens, sentence),
            reverse=True,
        )
        chosen = [
            sentence
            for sentence in ranked_sentences[:max_sentences_per_chunk]
            if sentence.strip()
        ]
        chosen = chosen[:remaining_total]
        remaining_total -= len(chosen)
        if not chosen:
            continue

        updated_metadata = dict(hit.metadata)
        updated_metadata["compression_applied"] = True
        updated_metadata["original_text_length"] = len(hit.chunk_text)
        compressed.append(
            RetrievalHit(
                query_id=hit.query_id,
                query_text=hit.query_text,
                chunk_id=hit.chunk_id,
                doc_id=hit.doc_id,
                rank=hit.rank,
                score=hit.score,
                retrieval_method=hit.retrieval_method,
                chunk_text=" ".join(chosen),
                title=hit.title,
                section_path=hit.section_path,
                metadata=updated_metadata,
            )
        )
    return _rerank_sequentially(compressed)


def _sentence_score(query_tokens: set[str], sentence: str) -> float:
    sentence_tokens = set(tokenize_for_bm25(sentence))
    if not query_tokens or not sentence_tokens:
        return 0.0
    overlap = len(query_tokens & sentence_tokens)
    return overlap / len(query_tokens)


def _rerank_sequentially(hits: list[RetrievalHit]) -> list[RetrievalHit]:
    output: list[RetrievalHit] = []
    for rank, hit in enumerate(hits, start=1):
        output.append(
            RetrievalHit(
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
    return output
