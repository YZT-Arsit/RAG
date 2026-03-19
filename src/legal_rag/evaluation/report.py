from __future__ import annotations

import csv
from pathlib import Path

from legal_rag.schemas.evaluation import MetricRecord


def write_metric_csv(path: Path, records: list[MetricRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    all_metric_keys = sorted({key for record in records for key in record.metrics})
    all_metadata_keys = sorted({key for record in records for key in record.metadata})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["query_id", *all_metadata_keys, *all_metric_keys]
        )
        writer.writeheader()
        for record in records:
            row = {"query_id": record.query_id}
            row.update(record.metadata)
            row.update(record.metrics)
            writer.writerow(row)


def write_markdown_summary(
    path: Path,
    *,
    title: str,
    implemented: str,
    experimental: str,
    planned: str,
    summary_metrics: dict[str, float],
    notes: list[str],
    grouped_metrics: dict[str, dict[str, float]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {title}",
        "",
        "## Status Labels",
        "",
        f"- Implemented: {implemented}",
        f"- Experimental: {experimental}",
        f"- Planned: {planned}",
        "",
        "## Summary Metrics",
        "",
    ]
    if summary_metrics:
        for key, value in summary_metrics.items():
            lines.append(f"- `{key}`: {value:.4f}")
    else:
        lines.append("- No metrics available.")
    lines.extend(["", "## Notes", ""])
    for note in notes:
        lines.append(f"- {note}")
    if grouped_metrics:
        lines.extend(["", "## Grouped Metrics", ""])
        for group_name, metrics in grouped_metrics.items():
            lines.append(f"### {group_name}")
            if metrics:
                for key, value in metrics.items():
                    lines.append(f"- `{key}`: {value:.4f}")
            else:
                lines.append("- No metrics available.")
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
