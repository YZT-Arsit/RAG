from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from legal_rag.error_analysis.models import ErrorRecord


def write_error_csv(path: Path, records: list[ErrorRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["query_id", "question", "error_labels", "notes"]
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "query_id": record.query_id,
                    "question": record.question,
                    "error_labels": "|".join(record.error_labels),
                    "notes": " | ".join(record.notes),
                }
            )


def write_error_markdown(path: Path, records: list[ErrorRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(record.error_labels)
    lines = [
        "# Error Analysis Report",
        "",
        "## Status Labels",
        "",
        "- Implemented: rule-based error categorization across retrieval, ranking, generation, citation, and abstention failures.",
        "- Experimental: labels are driven by current benchmark/eval proxies and should be interpreted as diagnostic hints rather than absolute truth.",
        "- Planned: richer claim-level diagnosis and model-assisted error explanations.",
        "",
        "## Error Counts",
        "",
    ]
    if counter:
        for label, count in counter.most_common():
            lines.append(f"- `{label}`: {count}")
    else:
        lines.append("- No error labels were assigned.")

    lines.extend(["", "## Query Notes", ""])
    for record in records:
        label_text = (
            ", ".join(record.error_labels) if record.error_labels else "no_error_label"
        )
        note_text = (
            " ".join(record.notes)
            if record.notes
            else "No issues detected by current rules."
        )
        lines.append(f"- `{record.query_id}` [{label_text}]: {note_text}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
