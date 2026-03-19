from __future__ import annotations

from legal_rag.contexting.compressor import compress_hits
from legal_rag.contexting.dedupe import dedupe_hits
from legal_rag.contexting.selector import select_hits
from legal_rag.schemas.retrieval import QueryRecord, RetrievalHit


def make_hit(chunk_id: str, doc_id: str, text: str, rank: int = 1) -> RetrievalHit:
    return RetrievalHit(
        query_id="q1",
        query_text="通用机场适用范围",
        chunk_id=chunk_id,
        doc_id=doc_id,
        rank=rank,
        score=1.0 / rank,
        retrieval_method="hybrid_rrf",
        chunk_text=text,
        title="通用机场管理规定",
        section_path=["第一条"],
    )


def test_dedupe_hits_removes_same_doc_same_prefix() -> None:
    hits = [
        make_hit(
            "c1", "d1", "第一条 中华人民共和国境内通用机场的建设、使用应当遵守本规定。"
        ),
        make_hit(
            "c2",
            "d1",
            "第一条 中华人民共和国境内通用机场的建设、使用应当遵守本规定。后续。",
        ),
    ]
    deduped = dedupe_hits(hits)
    assert len(deduped) == 1


def test_select_hits_limits_per_doc() -> None:
    hits = [
        make_hit("c1", "d1", "a", rank=1),
        make_hit("c2", "d1", "b", rank=2),
        make_hit("c3", "d1", "c", rank=3),
        make_hit("c4", "d2", "d", rank=4),
    ]
    selected = select_hits(hits, max_chunks=3, max_per_doc=2)
    assert [hit.chunk_id for hit in selected] == ["c1", "c2", "c4"]


def test_compress_hits_keeps_query_relevant_sentences() -> None:
    query = QueryRecord(query_id="q1", query_text="通用机场适用范围")
    hits = [
        make_hit(
            "c1",
            "d1",
            "第一条 无关内容。第二条 中华人民共和国境内通用机场的建设、使用、运营管理应当遵守本规定。第三条 其他内容。",
        ),
    ]
    compressed = compress_hits(
        query, hits, max_sentences_total=2, max_sentences_per_chunk=1
    )
    assert len(compressed) == 1
    assert "通用机场" in compressed[0].chunk_text
    assert compressed[0].metadata["compression_applied"] is True
