from __future__ import annotations

import re
from dataclasses import replace

from legal_rag.audit.text_utils import normalize_text, normalize_title
from legal_rag.config.schema import CleaningConfig
from legal_rag.schemas.document import Document


def apply_cleaning(document: Document, config: CleaningConfig) -> Document:
    actions: list[str] = []
    cleaned_title = normalize_title(document.title)
    for pattern in config.title_strip_patterns:
        updated_title, changed = _sub_once(
            pattern, "", cleaned_title, flags=re.IGNORECASE
        )
        if changed:
            cleaned_title = normalize_title(updated_title)
            actions.append(f"title_pattern:{pattern}")

    cleaned_text = document.normalized_text
    for pattern in config.body_leading_patterns:
        updated_text, changed = _sub_once(
            pattern, "", cleaned_text, flags=re.IGNORECASE
        )
        if changed:
            cleaned_text = normalize_text(updated_text)
            actions.append(f"body_leading_pattern:{pattern}")

    for pattern in config.body_global_patterns:
        updated_text, changed = _sub_all(
            pattern, " ", cleaned_text, flags=re.IGNORECASE
        )
        if changed:
            cleaned_text = normalize_text(updated_text)
            actions.append(f"body_global_pattern:{pattern}")

    canonical_source = _canonicalize_source(
        document.publish_source, config.source_aliases
    )
    if canonical_source and canonical_source != document.publish_source:
        actions.append("source_alias_applied")

    quality_flags = list(document.quality_flags)
    if cleaned_text and len(cleaned_text) < config.min_cleaned_length:
        quality_flags.append("cleaned_text_too_short")

    return replace(
        document,
        cleaned_title=cleaned_title or document.title,
        canonical_source=canonical_source,
        cleaned_text=cleaned_text,
        cleaning_actions=sorted(set([*document.cleaning_actions, *actions])),
        quality_flags=sorted(set(quality_flags)),
    )


def _sub_once(pattern: str, repl: str, text: str, *, flags: int) -> tuple[str, bool]:
    compiled = re.compile(pattern, flags=flags)
    updated, count = compiled.subn(repl, text, count=1)
    return updated, count > 0


def _sub_all(pattern: str, repl: str, text: str, *, flags: int) -> tuple[str, bool]:
    compiled = re.compile(pattern, flags=flags)
    updated, count = compiled.subn(repl, text)
    return updated, count > 0


def _canonicalize_source(source: str | None, aliases: dict[str, str]) -> str | None:
    if source is None:
        return None
    return aliases.get(source, source)
