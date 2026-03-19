from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from legal_rag.schemas.document import RawRecord


def _parse_raw_record(
    item: dict[str, Any], source_file: str, record_index: int
) -> RawRecord:
    known = {
        "title",
        "subTitle",
        "dataTime",
        "publishSource",
        "contentText",
        "content",
        "introTitle",
    }
    extra = {key: value for key, value in item.items() if key not in known}
    return RawRecord(
        source_file=source_file,
        record_index=record_index,
        title=_string_or_none(item.get("title")),
        sub_title=_string_or_none(item.get("subTitle")),
        data_time=_string_or_none(item.get("dataTime")),
        publish_source=_string_or_none(item.get("publishSource")),
        content_text=_string_or_none(item.get("contentText") or item.get("content")),
        intro_title=_string_or_none(item.get("introTitle")),
        extra=extra,
    )


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def iter_records(path: Path) -> Iterator[RawRecord]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _iter_jsonl(path)
        return
    if suffix == ".json":
        yield from _iter_json_with_fallback(path)
        return
    msg = f"Unsupported input file: {path}"
    raise ValueError(msg)


def _iter_jsonl(path: Path) -> Iterator[RawRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            stripped = line.strip()
            if not stripped:
                continue
            item = json.loads(stripped)
            if not isinstance(item, dict):
                msg = f"Each JSONL line must be an object in {path}"
                raise ValueError(msg)
            yield _parse_raw_record(item, source_file=str(path), record_index=index)


def _iter_json(path: Path) -> Iterator[RawRecord]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                msg = f"Each JSON array element must be an object in {path}"
                raise ValueError(msg)
            yield _parse_raw_record(item, source_file=str(path), record_index=index)
        return
    if isinstance(data, dict):
        records = data.get("data") if isinstance(data.get("data"), list) else None
        if records is None:
            msg = (
                f"JSON file {path} must be an array or contain a top-level 'data' array"
            )
            raise ValueError(msg)
        for index, item in enumerate(records):
            if not isinstance(item, dict):
                msg = f"Each record in {path} must be an object"
                raise ValueError(msg)
            yield _parse_raw_record(item, source_file=str(path), record_index=index)
        return
    msg = f"Unsupported JSON structure in {path}"
    raise ValueError(msg)


def _iter_json_with_fallback(path: Path) -> Iterator[RawRecord]:
    try:
        yield from _iter_json(path)
    except (json.JSONDecodeError, ValueError):
        yield from _iter_jsonl(path)
