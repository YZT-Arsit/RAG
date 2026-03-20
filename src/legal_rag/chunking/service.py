from __future__ import annotations

from collections import Counter

from legal_rag.chunking.fixed import chunk_document_fixed
from legal_rag.chunking.io import append_chunks
from legal_rag.chunking.structure_aware import chunk_document_structure_aware
from legal_rag.cleaning.io import iter_documents
from legal_rag.config.schema import ChunkingConfig


def run_chunking(config: ChunkingConfig) -> None:
    if config.output_jsonl.exists():
        config.output_jsonl.unlink()

    total_chunks = 0
    method_counter: Counter[str] = Counter()
    all_chunks = []

    for document in iter_documents(config.input_jsonl):
        document_chunks = []

        if config.method in {"fixed", "both"}:
            fixed_chunks = chunk_document_fixed(
                document,
                chunk_size=config.fixed_chunk_size,
                chunk_overlap=config.fixed_chunk_overlap,
            )
            document_chunks.extend(fixed_chunks)
            method_counter.update(chunk.chunk_method for chunk in fixed_chunks)

        if config.method in {"structure", "both"}:
            structure_chunks = chunk_document_structure_aware(
                document,
                max_chunk_size=config.structure_max_chunk_size,
                min_chunk_size=config.structure_min_chunk_size,
                sentence_overlap=config.structure_sentence_overlap,
            )
            document_chunks.extend(structure_chunks)
            method_counter.update(chunk.chunk_method for chunk in structure_chunks)

        if document_chunks:
            append_chunks(config.output_jsonl, document_chunks)
            all_chunks.extend(document_chunks)
            total_chunks += len(document_chunks)

    _write_chunking_report(
        config.report_path,
        total_chunks=total_chunks,
        method_counter=method_counter,
    )


def _write_chunking_report(path, *, total_chunks: int, method_counter: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        f"- Total chunks: {total_chunks}",
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
