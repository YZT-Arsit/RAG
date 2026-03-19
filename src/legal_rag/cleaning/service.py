from __future__ import annotations

from collections import Counter

from legal_rag.cleaning.io import iter_documents, write_documents
from legal_rag.cleaning.rules import apply_cleaning
from legal_rag.config.schema import CleaningConfig


def run_cleaning(config: CleaningConfig) -> None:
    cleaned_documents = [
        apply_cleaning(document, config)
        for document in iter_documents(config.input_jsonl)
    ]
    write_documents(config.output_jsonl, cleaned_documents)
    _write_cleaning_report(config.report_path, cleaned_documents)


def _write_cleaning_report(path, documents) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    action_counter: Counter[str] = Counter()
    flag_counter: Counter[str] = Counter()
    for document in documents:
        action_counter.update(document.cleaning_actions)
        flag_counter.update(document.quality_flags)

    lines = [
        "# Cleaning Report",
        "",
        "## Status Labels",
        "",
        "- Implemented: regex-based title/body cleaning, source alias normalization, cleaned text export, review-sample preparation.",
        "- Experimental: generic regex cleaning defaults for mixed policy/news/legal corpora.",
        "- Planned: site-specific boilerplate rules, date canonicalization, section-aware cleaning, incremental cleaning diffs.",
        "",
        "## Summary",
        "",
        f"- Documents processed: {len(documents)}",
        "",
        "## Cleaning Actions",
        "",
    ]
    if action_counter:
        for action, count in action_counter.most_common():
            lines.append(f"- `{action}`: {count}")
    else:
        lines.append("- No cleaning actions were applied.")

    lines.extend(
        [
            "",
            "## Quality Flags After Cleaning",
            "",
        ]
    )
    if flag_counter:
        for flag, count in flag_counter.most_common():
            lines.append(f"- `{flag}`: {count}")
    else:
        lines.append("- No quality flags present.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
