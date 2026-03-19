from __future__ import annotations

from legal_rag.audit.models import AnomalyDetail
from legal_rag.audit.text_utils import (
    is_probably_garbled,
    low_information_ratio,
    repeated_line_ratio,
)
from legal_rag.schemas.document import Document


def detect_anomalies(
    document: Document,
    *,
    short_body_threshold: int,
    template_similarity_line_threshold: float,
) -> list[AnomalyDetail]:
    text = document.normalized_text
    details: list[AnomalyDetail] = []

    if not text:
        details.append(
            AnomalyDetail(
                doc_id=document.doc_id,
                anomaly_type="empty_body",
                severity="high",
                note="Normalized body text is empty.",
            )
        )
        return details

    if len(text) < short_body_threshold:
        details.append(
            AnomalyDetail(
                doc_id=document.doc_id,
                anomaly_type="very_short_body",
                severity="medium",
                note=f"Body length {len(text)} is below threshold {short_body_threshold}.",
            )
        )

    if is_probably_garbled(text):
        details.append(
            AnomalyDetail(
                doc_id=document.doc_id,
                anomaly_type="garbled_text",
                severity="high",
                note="Text contains mojibake markers or replacement characters.",
            )
        )

    repeated_ratio = repeated_line_ratio(text)
    if repeated_ratio >= template_similarity_line_threshold:
        details.append(
            AnomalyDetail(
                doc_id=document.doc_id,
                anomaly_type="template_repetition",
                severity="medium",
                note=f"Repeated line ratio {repeated_ratio:.2f} exceeds threshold {template_similarity_line_threshold:.2f}.",
            )
        )

    info_ratio = low_information_ratio(text)
    if info_ratio > 0.85 and len(text) >= short_body_threshold:
        details.append(
            AnomalyDetail(
                doc_id=document.doc_id,
                anomaly_type="low_information_density",
                severity="low",
                note=f"Low-information heuristic ratio is {info_ratio:.2f}.",
            )
        )

    return details
