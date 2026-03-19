from __future__ import annotations

import re

try:
    import jieba
except ModuleNotFoundError:  # pragma: no cover - optional dependency fallback
    jieba = None


TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def tokenize_for_bm25(text: str) -> list[str]:
    return tokenize_zh_text(text)


def tokenize_zh_text(text: str) -> list[str]:
    if not text:
        return []
    fine_grained = TOKEN_RE.findall(text)
    if jieba is not None:
        tokens = [token.strip() for token in jieba.lcut(text, cut_all=False)]
        normalized = [token for token in tokens if token and not token.isspace()]
        if normalized:
            return normalized + fine_grained
    return fine_grained


def char_ngrams(text: str, n: int) -> list[str]:
    clean = "".join(text.split())
    if len(clean) < n:
        return [clean] if clean else []
    return [clean[i : i + n] for i in range(len(clean) - n + 1)]
