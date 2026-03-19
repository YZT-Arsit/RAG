from __future__ import annotations

from collections import Counter

from legal_rag.chunking.fixed import chunk_document_fixed
from legal_rag.chunking.io import write_chunks
from legal_rag.chunking.structure_aware import chunk_document_structure_aware
from legal_rag.cleaning.io import iter_documents
from legal_rag.config.schema import ChunkingConfig


def run_chunking(config: ChunkingConfig) -> None:
    all_chunks = []
    for document in iter_documents(config.input_jsonl):
        if config.method in {"fixed", "both"}:
            all_chunks.extend(
                chunk_document_fixed(
                    document,
                    chunk_size=config.fixed_chunk_size,
                    chunk_overlap=config.fixed_chunk_overlap,
                )
            )
        if config.method in {"structure", "both"}:
            all_chunks.extend(
                chunk_document_structure_aware(
                    document,
                    max_chunk_size=config.structure_max_chunk_size,
                    min_chunk_size=config.structure_min_chunk_size,
                    sentence_overlap=config.structure_sentence_overlap,
                )
            )

    write_chunks(config.output_jsonl, all_chunks)
    _write_chunking_report(config.report_path, all_chunks)


def _write_chunking_report(path, chunks) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    method_counter: Counter[str] = Counter(chunk.chunk_method for chunk in chunks)
    lines = [
        "# Chunking Report",
        "",
        "## Status Labels",
        "",
        "- Implemented: fixed-size chunking, structure-aware chunking, chunk JSONL export, chunk report export.",
        "- Experimental: regex-based legal structure splitting on mixed legal/news/policy corpora.",
        "- Planned: recursive splitting, sentence-aware overlap, richer hierarchy tracking, chunk quality validation.",
        "",
        "## Summary",
        "",
        f"- Total chunks: {len(chunks)}",
        "",
        "## Chunk Methods",
        "",
    ]
    if method_counter:
        for method, count in method_counter.most_common():
            lines.append(f"- `{method}`: {count}")
    else:
        lines.append("- No chunks generated.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
