from __future__ import annotations

import re


SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；])")


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_RE.split(text)
    return [part.strip() for part in parts if part.strip()]
