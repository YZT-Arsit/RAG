from __future__ import annotations

from collections import defaultdict
from collections import deque

from legal_rag.audit.models import DuplicateDetail
from legal_rag.audit.text_utils import content_hash, hamming_distance, simhash64
from legal_rag.schemas.document import Document


def find_exact_duplicates(
    documents: list[Document], *, field: str
) -> list[DuplicateDetail]:
    groups: dict[str, list[str]] = defaultdict(list)
    for document in documents:
        value = getattr(document, field)
        if not isinstance(value, str):
            continue
        normalized = value.strip()
        if not normalized:
            continue
        groups[content_hash(normalized)].append(document.doc_id)

    duplicate_type = (
        "exact_title_duplicate" if field == "title" else "exact_body_duplicate"
    )
    details: list[DuplicateDetail] = []
    for fingerprint, doc_ids in groups.items():
        if len(doc_ids) > 1:
            details.append(
                DuplicateDetail(
                    duplicate_type=duplicate_type,
                    fingerprint=fingerprint,
                    doc_ids=sorted(doc_ids),
                    note=f"{len(doc_ids)} documents share the same {field}.",
                )
            )
    return sorted(details, key=lambda item: (-len(item.doc_ids), item.fingerprint))


def find_near_duplicates(
    documents: list[Document],
    *,
    hamming_threshold: int,
    prefix_bits: int,
) -> list[DuplicateDetail]:
    if not documents:
        return []

    buckets: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for document in documents:
        if not document.normalized_text:
            continue
        fingerprint = simhash64(document.normalized_text)
        bucket_key = fingerprint >> (64 - prefix_bits)
        buckets[bucket_key].append((document.doc_id, fingerprint))

    details: list[DuplicateDetail] = []
    for bucket_items in buckets.values():
        sorted_items = sorted(bucket_items, key=lambda item: item[1])
        current_group: deque[tuple[str, int]] = deque()
        for doc_id, fingerprint in sorted_items:
            while current_group and hamming_distance(
                current_group[0][1], fingerprint
            ) > (hamming_threshold + 4):
                current_group.popleft()

            local_matches = [
                other_id
                for other_id, other_fp in current_group
                if hamming_distance(other_fp, fingerprint) <= hamming_threshold
            ]
            if local_matches:
                group_sorted = sorted({doc_id, *local_matches})
                details.append(
                    DuplicateDetail(
                        duplicate_type="near_body_duplicate",
                        fingerprint=str(fingerprint),
                        doc_ids=group_sorted,
                        note=f"Approximate duplicate by SimHash with Hamming distance <= {hamming_threshold}.",
                    )
                )
            current_group.append((doc_id, fingerprint))

    deduped: dict[tuple[str, ...], DuplicateDetail] = {}
    for item in details:
        key = tuple(item.doc_ids)
        deduped[key] = item
    return sorted(deduped.values(), key=lambda item: (-len(item.doc_ids), item.doc_ids))
