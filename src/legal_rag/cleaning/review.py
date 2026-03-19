from __future__ import annotations

import csv
import random
from pathlib import Path

from legal_rag.cleaning.io import iter_documents
from legal_rag.config.schema import ReviewSampleConfig
from legal_rag.schemas.document import Document


def export_review_samples(config: ReviewSampleConfig) -> None:
    documents = {
        document.doc_id: document for document in iter_documents(config.input_jsonl)
    }
    duplicate_rows, anomaly_rows = _load_detail_rows(config.detail_csv)
    rng = random.Random(config.random_seed)

    random_docs = rng.sample(
        list(documents.values()), min(config.random_sample_size, len(documents))
    )
    anomaly_docs = _pick_docs(documents, anomaly_rows, config.anomaly_sample_size)
    duplicate_docs = _pick_docs(documents, duplicate_rows, config.duplicate_sample_size)

    rows = []
    rows.extend(_to_rows("random_sample", random_docs))
    rows.extend(_to_rows("anomaly_sample", anomaly_docs))
    rows.extend(_to_rows("duplicate_sample", duplicate_docs))

    _write_rows(config.output_csv, rows)


def _load_detail_rows(path: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    duplicate_rows: list[dict[str, str]] = []
    anomaly_rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            record_type = row["record_type"]
            if "duplicate" in record_type:
                duplicate_rows.append(row)
            else:
                anomaly_rows.append(row)
    return duplicate_rows, anomaly_rows


def _pick_docs(
    documents: dict[str, Document], rows: list[dict[str, str]], limit: int
) -> list[Document]:
    picked: list[Document] = []
    seen: set[str] = set()
    for row in rows:
        for doc_id in row["doc_ids"].split("|"):
            if doc_id in seen or doc_id not in documents:
                continue
            picked.append(documents[doc_id])
            seen.add(doc_id)
            if len(picked) >= limit:
                return picked
    return picked


def _to_rows(sample_type: str, documents: list[Document]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for document in documents:
        rows.append(
            {
                "sample_type": sample_type,
                "doc_id": document.doc_id,
                "title": document.cleaned_title or document.title,
                "publish_source": document.canonical_source
                or document.publish_source
                or "",
                "published_year": document.published_year or "",
                "quality_flags": "|".join(document.quality_flags),
                "cleaning_actions": "|".join(document.cleaning_actions),
                "text_preview": (document.cleaned_text or document.normalized_text)[
                    :300
                ],
            }
        )
    return rows


def _write_rows(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_type",
                "doc_id",
                "title",
                "publish_source",
                "published_year",
                "quality_flags",
                "cleaning_actions",
                "text_preview",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
