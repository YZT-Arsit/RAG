from __future__ import annotations

import hashlib
import re
from collections import Counter
from datetime import UTC, datetime


WHITESPACE_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
MOJIBAKE_MARKERS = ("锟斤拷", "�", "\ufffd", "Ã", "æ", "å", "ð")


def normalize_text(text: str) -> str:
    stripped = text.replace("\u3000", " ").replace("\xa0", " ")
    stripped = HTML_TAG_RE.sub(" ", stripped)
    stripped = WHITESPACE_RE.sub(" ", stripped)
    return stripped.strip()


def normalize_title(text: str | None) -> str:
    if not text:
        return ""
    return normalize_text(text)


def content_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def extract_year(raw: str | None, pattern: re.Pattern[str]) -> int | None:
    if not raw:
        return None
    if raw.isdigit() and len(raw) >= 10:
        try:
            timestamp = int(raw)
            if len(raw) >= 13:
                timestamp //= 1000
            return datetime.fromtimestamp(timestamp, tz=UTC).year
        except (OverflowError, ValueError, OSError):
            pass
    match = pattern.search(raw)
    if not match:
        return None
    year = int(match.group(0))
    if 1900 <= year <= 2100:
        return year
    return None


def iter_char_ngrams(
    text: str, n: int = 3, max_chars: int = 2000, max_samples: int = 128
) -> list[str]:
    clean = normalize_text(text)
    if len(clean) > max_chars:
        clean = clean[:max_chars]
    if len(clean) < n:
        return [clean] if clean else []
    max_index = len(clean) - n + 1
    step = max(1, max_index // max_samples)
    return [clean[idx : idx + n] for idx in range(0, max_index, step)][:max_samples]


def simhash64(text: str) -> int:
    vector = [0] * 64
    for token in iter_char_ngrams(text, n=3):
        digest = hashlib.md5(token.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], byteorder="big", signed=False)
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1

    fingerprint = 0
    for bit, score in enumerate(vector):
        if score > 0:
            fingerprint |= 1 << bit
    return fingerprint


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def is_probably_garbled(text: str) -> bool:
    if not text:
        return False
    if any(marker in text for marker in MOJIBAKE_MARKERS):
        return True
    weird_ratio = sum(1 for char in text if ord(char) == 65533) / max(len(text), 1)
    return weird_ratio > 0.02


def repeated_line_ratio(text: str) -> float:
    lines = [normalize_text(line) for line in text.splitlines() if normalize_text(line)]
    if len(lines) < 2:
        return 0.0
    counts = Counter(lines)
    repeated = sum(count for _, count in counts.items() if count > 1)
    return repeated / len(lines)


def low_information_ratio(text: str) -> float:
    clean = normalize_text(text)
    if not clean:
        return 1.0
    unique_chars = len(set(clean))
    return 1 - (unique_chars / len(clean))
