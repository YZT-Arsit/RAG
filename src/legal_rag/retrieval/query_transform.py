from __future__ import annotations

import json
import re
from dataclasses import dataclass

from legal_rag.config.schema import RetrievalConfig
from legal_rag.generation.llm_client import LLMClient
from legal_rag.generation.service import build_llm_client
from legal_rag.schemas.retrieval import QueryRecord


@dataclass(slots=True)
class TransformedQuery:
    text: str
    source: str


class LLMQueryTransformer:
    def __init__(
        self,
        *,
        client: LLMClient,
        multi_query_count: int,
        use_hyde: bool,
        include_original: bool,
    ) -> None:
        self.client = client
        self.multi_query_count = multi_query_count
        self.use_hyde = use_hyde
        self.include_original = include_original

    def expand_query(self, query: QueryRecord) -> list[TransformedQuery]:
        variants: list[TransformedQuery] = []
        if self.include_original or (
            self.multi_query_count == 0 and not self.use_hyde
        ):
            variants.append(TransformedQuery(text=query.query_text, source="original"))

        prompt = _build_query_transform_prompt(
            query.query_text,
            multi_query_count=self.multi_query_count,
            use_hyde=self.use_hyde,
        )
        response = self.client.complete(prompt)
        payload = _parse_query_transform_payload(response.content)

        for item in payload.get("multi_queries", [])[: self.multi_query_count]:
            text = str(item).strip()
            if text:
                variants.append(TransformedQuery(text=text, source="multi_query"))
        hyde_text = str(payload.get("hyde", "")).strip()
        if self.use_hyde and hyde_text:
            variants.append(TransformedQuery(text=hyde_text, source="hyde"))
        return _dedupe_variants(variants)


def build_query_transformer(config: RetrievalConfig) -> LLMQueryTransformer | None:
    if not config.query_transform_enabled:
        return None
    if config.query_transform_multi_query_count <= 0 and not config.query_transform_use_hyde:
        return None
    client = build_llm_client(
        llm_backend=config.query_transform_llm_backend,
        llm_base_url=config.query_transform_llm_base_url,
        llm_api_key_env=config.query_transform_llm_api_key_env,
        llm_model_name=config.query_transform_llm_model_name,
        llm_modelscope_model_id=config.query_transform_llm_modelscope_model_id,
        llm_local_model_dir=config.query_transform_llm_local_model_dir,
        llm_use_modelscope_download=config.query_transform_llm_use_modelscope_download,
        llm_device=config.query_transform_llm_device,
        llm_temperature=config.query_transform_llm_temperature,
        llm_timeout_seconds=config.query_transform_llm_timeout_seconds,
        llm_max_new_tokens=config.query_transform_llm_max_new_tokens,
    )
    return LLMQueryTransformer(
        client=client,
        multi_query_count=config.query_transform_multi_query_count,
        use_hyde=config.query_transform_use_hyde,
        include_original=config.query_transform_include_original,
    )


def _build_query_transform_prompt(
    query: str, *, multi_query_count: int, use_hyde: bool
) -> str:
    return (
        "你是中文法律检索专家。请把用户口语问题转换为法律检索友好的查询。\n"
        "必须只输出 JSON，不要输出解释。\n"
        f"请生成 {multi_query_count} 个专业法律检索 query，要求侧重点不同。"
        "例如可覆盖构成要件、赔偿标准、程序路径、责任承担等。\n"
        + (
            "同时请生成一段 120-220 字的假想法律依据/裁判要点摘要，用于 HyDE 检索。\n"
            if use_hyde
            else "无需生成 HyDE 文本。\n"
        )
        + 'JSON 格式：{"multi_queries":["..."],"hyde":"..."}\n'
        + f"用户问题：{query}\n"
    )


def _parse_query_transform_payload(content: str) -> dict[str, object]:
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            payload = json.loads(content[start : end + 1])
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    queries = re.findall(r'"([^"]+)"', content)
    return {
        "multi_queries": queries[:3],
        "hyde": "",
    }


def _dedupe_variants(variants: list[TransformedQuery]) -> list[TransformedQuery]:
    deduped: list[TransformedQuery] = []
    seen: set[str] = set()
    for variant in variants:
        normalized = re.sub(r"\s+", "", variant.text)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(variant)
    return deduped
