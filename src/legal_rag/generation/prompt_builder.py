from __future__ import annotations

from legal_rag.schemas.generation import GenerationInput


def build_grounded_prompt(
    generation_input: GenerationInput,
    *,
    prompt_version: str,
    max_contexts: int,
    max_chars_per_context: int,
) -> str:
    context_lines: list[str] = []
    for index, context in enumerate(generation_input.contexts[:max_contexts], start=1):
        snippet = context.text[:max_chars_per_context].strip()
        source_file = str(context.metadata.get("source_file", "")).strip()
        article_label = str(context.metadata.get("article_label", "")).strip()
        source_meta = " | ".join(part for part in [source_file, article_label, context.chunk_id] if part)
        context_lines.append(f"[{index}] {context.title} ({source_meta})\n{snippet}")
    context_block = "\n\n".join(context_lines)
    if prompt_version == "grounded_answer_first_v1":
        instruction_block = (
            "你是一个基于证据回答问题的中文法律助手。\n"
            "只能依据给定上下文回答，不要使用外部知识。\n"
            "如果上下文已经足以支持问题的主要结论，应直接回答；"
            "只有在核心结论无法从任何上下文获得支持时，才返回 abstained=true。\n"
            "不要求覆盖所有细节后才回答，但回答中的内容必须能被给定上下文支持。\n"
            "不要输出思考过程，不要输出<think>标签，不要输出解释、前后缀、Markdown 或代码块。\n"
            "必须只输出一个完整的 JSON 对象。\n"
            "字段必须包括：answer, used_context_ids, abstained。\n"
            "answer 字段中的每个关键结论后都必须追加形如 [1]、[2] 的引用编号，编号只能对应给定上下文顺序。\n"
            "used_context_ids 必须填写所引用上下文对应的 chunk_id（不是数字编号）；如果证据不足，可返回空数组并令 abstained=true。\n\n"
            "输出示例："
            '{"answer":"应当遵守本规定[1]。", "used_context_ids":["c1"], "abstained":false}\n\n'
        )
    else:
        instruction_block = (
            "你是一个严格基于证据回答问题的中文法律助手。\n"
            "只能依据给定上下文回答，不要使用外部知识。\n"
            "如果证据不足，请返回 abstained=true。\n"
            "不要输出思考过程，不要输出<think>标签，不要输出解释、前后缀、Markdown 或代码块。\n"
            "必须只输出一个完整的 JSON 对象。\n"
            "字段必须包括：answer, used_context_ids, abstained。\n"
            "answer 字段中的每个关键结论后都必须追加形如 [1]、[2] 的引用编号，编号只能对应给定上下文顺序。\n"
            "used_context_ids 必须填写所引用上下文对应的 chunk_id（不是数字编号）；如果证据不足可返回空数组并令 abstained=true。\n\n"
            "输出示例："
            '{"answer":"应当遵守本规定[1]。", "used_context_ids":["c1"], "abstained":false}\n\n'
        )
    return (
        instruction_block
        + f"PromptVersion：{prompt_version}\n"
        + f"问题：{generation_input.query_text}\n\n"
        + f"上下文：\n{context_block}\n"
    )
