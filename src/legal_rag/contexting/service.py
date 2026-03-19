from __future__ import annotations

from collections import Counter

from legal_rag.config.schema import ContextProcessingConfig
from legal_rag.contexting.compressor import compress_hits
from legal_rag.contexting.dedupe import dedupe_hits
from legal_rag.contexting.selector import select_hits
from legal_rag.evaluation.io import load_retrieval_results
from legal_rag.retrieval.io import write_results
from legal_rag.schemas.retrieval import RetrievalResult


def run_context_processing(config: ContextProcessingConfig) -> None:
    results = load_retrieval_results(config.input_retrieval_results_jsonl)
    processed: list[RetrievalResult] = []
    for result in results:
        hits = result.hits
        original_count = len(hits)
        if config.dedupe_enabled:
            hits = dedupe_hits(hits)
        hits = select_hits(
            hits, max_chunks=config.max_chunks, max_per_doc=config.max_per_doc
        )
        if config.compression_enabled:
            hits = compress_hits(
                result.query,
                hits,
                max_sentences_total=config.max_sentences_total,
                max_sentences_per_chunk=config.max_sentences_per_chunk,
            )
        processed.append(RetrievalResult(query=result.query, hits=hits))
        result.query.metadata["original_hit_count"] = original_count
        result.query.metadata["processed_hit_count"] = len(hits)

    write_results(config.output_jsonl, processed)
    _write_context_report(config.report_path, results=processed, config=config)


def _write_context_report(
    path, *, results: list[RetrievalResult], config: ContextProcessingConfig
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hit_counts = [len(result.hits) for result in results]
    method_counter: Counter[str] = Counter()
    for result in results:
        for hit in result.hits:
            if hit.metadata.get("compression_applied"):
                method_counter.update(["compressed"])
    average_hits = sum(hit_counts) / len(hit_counts) if hit_counts else 0.0
    lines = [
        "# Context Processing Report",
        "",
        "## Status Labels",
        "",
        "- Implemented: context dedupe, per-document selection, sentence-level compression, processed context export.",
        "- Experimental: lexical sentence scoring and prefix-based dedupe heuristics.",
        "- Planned: semantic dedupe, cross-chunk merging, query-aware compression with LLM or encoder support.",
        "",
        "## Summary",
        "",
        f"- Queries processed: {len(results)}",
        f"- Average selected contexts per query: {average_hits:.2f}",
        f"- Dedupe enabled: {config.dedupe_enabled}",
        f"- Compression enabled: {config.compression_enabled}",
        "",
        "## Compression Flags",
        "",
    ]
    if method_counter:
        for key, count in method_counter.items():
            lines.append(f"- `{key}`: {count}")
    else:
        lines.append("- No compression flags recorded.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
