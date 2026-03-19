from __future__ import annotations
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from legal_rag.benchmark.io import (
    candidate_to_record,
    write_benchmark_records,
    write_candidates,
)
from legal_rag.benchmark.schema import BenchmarkCandidate
from legal_rag.benchmark.validators import validate_benchmark
from legal_rag.chunking.io import iter_chunks
from legal_rag.config.schema import BenchmarkGenerationConfig
from legal_rag.schemas.chunk import Chunk

QUESTION_TYPES = (
    "definition",
    "condition",
    "procedure",
    "responsibility",
    "comparison",
    "unanswerable",
)

ARTICLE_PATTERN = re.compile(r"(第[一二三四五六七八九十百千万零〇两\d]+条)")
SENTENCE_SPLIT_PATTERN = re.compile(r"[。！？；]")
TERM_CLEAN_PATTERN = re.compile(r"[（(][^）)]*[）)]")


@dataclass(slots=True)
class ArticleUnit:
    doc_id: str
    chunk_id: str
    title: str
    text: str
    article_label: str | None
    chunk_method: str
    published_year: int | None
    section_path: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CandidateProcessingResult:
    deduped_candidates: list[BenchmarkCandidate]
    filter_reason_counts: Counter[str]


def run_benchmark_generation(config: BenchmarkGenerationConfig) -> None:
    candidate_target_total = sum(config.candidate_target_counts.values())
    if candidate_target_total != config.target_candidate_count:
        msg = (
            "candidate_target_counts must sum to target_candidate_count. "
            f"Got {candidate_target_total} vs {config.target_candidate_count}."
        )
        raise ValueError(msg)
    chunks = _load_chunks(
        config.input_chunk_jsonl, preferred_method=config.preferred_chunk_method
    )
    units = _build_article_units(
        chunks, min_chunk_text_length=config.min_chunk_text_length
    )
    evidence_text_by_chunk_id = {chunk.chunk_id: chunk.text for chunk in chunks}

    raw_candidates = _generate_candidates(
        units,
        candidate_target_counts=config.candidate_target_counts,
        target_candidate_count=config.target_candidate_count,
        max_candidates_per_doc_per_type=config.max_candidates_per_doc_per_type,
        random_seed=config.random_seed,
    )
    write_candidates(config.candidates_output_jsonl, raw_candidates)

    processing_result = _filter_and_dedup_candidates(
        raw_candidates,
        semantic_threshold=config.semantic_dedup_threshold,
        min_question_chars=config.min_question_chars,
        max_question_chars=config.max_question_chars,
        min_question_evidence_overlap=config.min_question_evidence_overlap,
    )
    write_candidates(config.deduped_output_jsonl, processing_result.deduped_candidates)

    final_candidates = _select_final_benchmark_candidates(
        processing_result.deduped_candidates,
        final_target_counts=config.final_target_counts,
    )
    records = [
        candidate_to_record(
            candidate,
            evidence_text_by_chunk_id=evidence_text_by_chunk_id,
        )
        for candidate in final_candidates
    ]
    write_benchmark_records(config.benchmark_output_jsonl, records)

    validation_summary = validate_benchmark(records)
    report = _build_report(
        raw_candidates=raw_candidates,
        deduped_candidates=processing_result.deduped_candidates,
        final_candidates=final_candidates,
        filter_reason_counts=processing_result.filter_reason_counts,
        validation_summary=validation_summary,
        config=config,
    )
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(report, encoding="utf-8")


def _load_chunks(path: Path, *, preferred_method: str) -> list[Chunk]:
    chunks = list(iter_chunks(path))
    if preferred_method == "both":
        return chunks
    filtered = [chunk for chunk in chunks if chunk.chunk_method == preferred_method]
    return filtered or chunks


def _build_article_units(
    chunks: Iterable[Chunk], *, min_chunk_text_length: int
) -> list[ArticleUnit]:
    units: list[ArticleUnit] = []
    seen_signatures: set[tuple[str, str]] = set()
    for chunk in chunks:
        cleaned_text = _normalize_space(chunk.text)
        if len(cleaned_text) < min_chunk_text_length:
            continue
        parts = _split_articles(cleaned_text)
        if not parts:
            parts = [(None, cleaned_text)]
        for article_label, article_text in parts:
            article_text = _normalize_space(article_text)
            if len(article_text) < min_chunk_text_length:
                continue
            signature = (chunk.doc_id, article_text[:120])
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            units.append(
                ArticleUnit(
                    doc_id=chunk.doc_id,
                    chunk_id=chunk.chunk_id,
                    title=chunk.title,
                    text=article_text,
                    article_label=article_label,
                    chunk_method=chunk.chunk_method,
                    published_year=chunk.published_year,
                    section_path=chunk.section_path,
                )
            )
    return units


def _split_articles(text: str) -> list[tuple[str | None, str]]:
    matches = list(ARTICLE_PATTERN.finditer(text))
    if not matches:
        return []
    sections: list[tuple[str | None, str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        article_label = match.group(1)
        article_text = text[start:end]
        sections.append((article_label, article_text))
    return sections


def _generate_candidates(
    units: list[ArticleUnit],
    *,
    candidate_target_counts: dict[str, int],
    target_candidate_count: int,
    max_candidates_per_doc_per_type: int,
    random_seed: int,
) -> list[BenchmarkCandidate]:
    rng = random.Random(random_seed)
    working_units = units[:]
    rng.shuffle(working_units)

    pools: dict[str, list[BenchmarkCandidate]] = {
        question_type: [] for question_type in QUESTION_TYPES
    }
    doc_type_counts: dict[tuple[str, str], int] = defaultdict(int)
    for unit in working_units:
        for candidate in _generate_candidates_for_unit(unit):
            key = (candidate.source_doc_id or "", candidate.question_type)
            if doc_type_counts[key] >= max_candidates_per_doc_per_type:
                continue
            target = candidate_target_counts.get(candidate.question_type, 0)
            if len(pools[candidate.question_type]) >= target:
                continue
            pools[candidate.question_type].append(candidate)
            doc_type_counts[key] += 1

    for candidate in _generate_document_comparison_candidates(working_units):
        key = (candidate.source_doc_id or "", candidate.question_type)
        if doc_type_counts[key] >= max_candidates_per_doc_per_type:
            continue
        target = candidate_target_counts.get(candidate.question_type, 0)
        if len(pools[candidate.question_type]) >= target:
            continue
        pools[candidate.question_type].append(candidate)
        doc_type_counts[key] += 1

    pools["unanswerable"] = _generate_unanswerable_candidates(
        answerable_pools=pools,
        units=working_units,
        target_count=candidate_target_counts.get("unanswerable", 0),
        random_seed=random_seed,
    )

    selected: list[BenchmarkCandidate] = []
    for question_type in QUESTION_TYPES:
        target = candidate_target_counts.get(question_type, 0)
        selected.extend(pools[question_type][:target])

    if len(selected) < target_candidate_count:
        msg = (
            f"Unable to generate enough candidates. Generated {len(selected)} "
            f"but target_candidate_count={target_candidate_count}."
        )
        raise ValueError(msg)
    selected = selected[:target_candidate_count]
    for index, candidate in enumerate(selected, start=1):
        candidate.query_id = f"cand_{index:04d}"
    return selected


def _generate_candidates_for_unit(unit: ArticleUnit) -> list[BenchmarkCandidate]:
    sentences = [
        sentence for sentence in _split_sentences(unit.text) if len(sentence) >= 12
    ]
    generated: list[BenchmarkCandidate] = []

    generated.extend(_generate_definition_candidates(unit, sentences))
    generated.extend(_generate_condition_candidates(unit, sentences))
    generated.extend(_generate_procedure_candidates(unit, sentences))
    generated.extend(_generate_responsibility_candidates(unit, sentences))
    generated.extend(_generate_comparison_candidates(unit, sentences))
    return generated


def _generate_definition_candidates(
    unit: ArticleUnit, sentences: list[str]
) -> list[BenchmarkCandidate]:
    candidates: list[BenchmarkCandidate] = []
    for offset, sentence in enumerate(sentences):
        match = re.search(r"(?P<term>[^，,。；]{2,24}?)(?:，|,)?是指", sentence)
        if not match:
            continue
        term = _clean_term(match.group("term"))
        if len(term) < 2:
            continue
        question = _pick_template(
            [
                f"{term}是指什么？",
                f"根据《{unit.title}》，{term}的定义是什么？",
            ],
            seed_key=f"{unit.chunk_id}:{offset}:definition",
        )
        candidates.append(
            _build_candidate(
                unit=unit,
                question_type="definition",
                question=question,
                gold_answer=sentence,
                generation_rule="definition_is_zhi",
            )
        )
    return candidates


def _generate_condition_candidates(
    unit: ArticleUnit, sentences: list[str]
) -> list[BenchmarkCandidate]:
    candidates: list[BenchmarkCandidate] = []
    for offset, sentence in enumerate(sentences):
        if not any(
            marker in sentence
            for marker in [
                "应当具备",
                "具备下列条件",
                "有下列情形之一",
                "符合下列条件",
                "满足下列条件",
                "方可",
                "可以申请",
            ]
        ):
            continue
        article_label = unit.article_label or "该条款"
        question = _pick_template(
            [
                f"根据《{unit.title}》，{article_label}规定了哪些条件？",
                f"{article_label}所列的适用条件是什么？",
            ],
            seed_key=f"{unit.chunk_id}:{offset}:condition",
        )
        candidates.append(
            _build_candidate(
                unit=unit,
                question_type="condition",
                question=question,
                gold_answer=sentence,
                generation_rule="condition_requirement",
            )
        )
    return candidates


def _generate_procedure_candidates(
    unit: ArticleUnit, sentences: list[str]
) -> list[BenchmarkCandidate]:
    candidates: list[BenchmarkCandidate] = []
    for offset, sentence in enumerate(sentences):
        if not any(
            marker in sentence
            for marker in [
                "申请",
                "提交",
                "受理",
                "审查",
                "审核",
                "补正",
                "通知",
                "办理",
            ]
        ):
            continue
        article_label = unit.article_label or "该条款"
        question = _pick_template(
            [
                f"根据《{unit.title}》，{article_label}规定的办理程序是什么？",
                f"{article_label}涉及的申请或审查流程是什么？",
            ],
            seed_key=f"{unit.chunk_id}:{offset}:procedure",
        )
        candidates.append(
            _build_candidate(
                unit=unit,
                question_type="procedure",
                question=question,
                gold_answer=sentence,
                generation_rule="procedure_application_flow",
            )
        )
    return candidates


def _generate_responsibility_candidates(
    unit: ArticleUnit, sentences: list[str]
) -> list[BenchmarkCandidate]:
    candidates: list[BenchmarkCandidate] = []
    for offset, sentence in enumerate(sentences):
        if not any(
            marker in sentence
            for marker in ["负责", "监督管理", "承担", "应当对", "由", "主管"]
        ):
            continue
        if (
            "由" not in sentence
            and "负责" not in sentence
            and "监督管理" not in sentence
        ):
            continue
        article_label = unit.article_label or "该条款"
        question = _pick_template(
            [
                f"根据《{unit.title}》，{article_label}中相关事项由谁负责？",
                f"{article_label}规定的责任主体是谁？",
            ],
            seed_key=f"{unit.chunk_id}:{offset}:responsibility",
        )
        candidates.append(
            _build_candidate(
                unit=unit,
                question_type="responsibility",
                question=question,
                gold_answer=sentence,
                generation_rule="responsibility_actor",
            )
        )
    return candidates


def _generate_comparison_candidates(
    unit: ArticleUnit, sentences: list[str]
) -> list[BenchmarkCandidate]:
    candidates: list[BenchmarkCandidate] = []
    whole_text = unit.text
    matches: list[tuple[str, str]] = []

    if "A类" in whole_text and "B类" in whole_text:
        matches.append(("A类", "B类"))
    if "一级专名" in whole_text and "二级专名" in whole_text:
        matches.append(("一级专名", "二级专名"))
    generic_match = re.search(
        r"分为(?P<a>[^，,。；]{1,12}?)(?:、|和|与)(?P<b>[^，,。；]{1,12}?)(?:两类|两种|两级|两部分)",
        whole_text,
    )
    if generic_match:
        matches.append(
            (
                _clean_term(generic_match.group("a")),
                _clean_term(generic_match.group("b")),
            )
        )

    for offset, (left, right) in enumerate(matches):
        question = _pick_template(
            [
                f"根据《{unit.title}》，{left}与{right}有什么区别？",
                f"{left}和{right}在该规定中的区别是什么？",
            ],
            seed_key=f"{unit.chunk_id}:{offset}:comparison",
        )
        candidates.append(
            _build_candidate(
                unit=unit,
                question_type="comparison",
                question=question,
                gold_answer=whole_text,
                generation_rule="comparison_dual_category",
                extra_metadata={"comparison_pair": [left, right]},
            )
        )
    return candidates


def _generate_document_comparison_candidates(
    units: list[ArticleUnit],
) -> list[BenchmarkCandidate]:
    grouped_units: dict[str, list[ArticleUnit]] = defaultdict(list)
    for unit in units:
        if not unit.article_label:
            continue
        grouped_units[unit.doc_id].append(unit)

    generated: list[BenchmarkCandidate] = []
    for doc_units in grouped_units.values():
        ordered_units = sorted(
            doc_units,
            key=lambda unit: (
                unit.chunk_id,
                unit.article_label or "",
            ),
        )
        for left, right in zip(ordered_units, ordered_units[1:]):
            if left.article_label == right.article_label:
                continue
            question = _pick_template(
                [
                    f"根据《{left.title}》，{left.article_label}与{right.article_label}的规定有什么区别？",
                    f"《{left.title}》中，{left.article_label}和{right.article_label}分别规定了什么？",
                ],
                seed_key=f"{left.chunk_id}:{right.chunk_id}:doc_comparison",
            )
            answer = f"{left.text} {right.text}"
            evidence_ids = list(dict.fromkeys([left.chunk_id, right.chunk_id]))
            generated.append(
                BenchmarkCandidate(
                    query_id="",
                    question=_normalize_space(question),
                    question_type="comparison",
                    answerable=True,
                    gold_answer=_normalize_space(answer),
                    gold_evidence_chunk_ids=evidence_ids,
                    source_doc_id=left.doc_id,
                    source_chunk_id=left.chunk_id,
                    metadata={
                        "generation_rule": "comparison_adjacent_article_units",
                        "title": left.title,
                        "left_article_label": left.article_label,
                        "right_article_label": right.article_label,
                        "chunk_method": left.chunk_method,
                        "published_year": left.published_year,
                        "comparison_pair": [left.article_label, right.article_label],
                    },
                )
            )
    return generated


def _generate_unanswerable_candidates(
    *,
    answerable_pools: dict[str, list[BenchmarkCandidate]],
    units: list[ArticleUnit],
    target_count: int,
    random_seed: int,
) -> list[BenchmarkCandidate]:
    rng = random.Random(random_seed)
    definition_terms: list[tuple[str, str]] = []
    for candidate in answerable_pools.get("definition", []):
        term = _extract_question_subject(candidate.question)
        if term:
            definition_terms.append((term, candidate.source_doc_id or ""))

    unit_pool = units[:]
    rng.shuffle(unit_pool)
    rng.shuffle(definition_terms)

    generated: list[BenchmarkCandidate] = []
    used_signatures: set[tuple[str, str]] = set()
    for offset, unit in enumerate(unit_pool):
        if len(generated) >= target_count:
            break
        for term, source_doc_id in definition_terms:
            if source_doc_id == unit.doc_id:
                continue
            if term in unit.text or term in unit.title:
                continue
            signature = (unit.doc_id, term)
            if signature in used_signatures:
                continue
            question = _pick_template(
                [
                    f"根据《{unit.title}》，{term}的定义是什么？",
                    f"{term}在《{unit.title}》中的含义是什么？",
                ],
                seed_key=f"{unit.chunk_id}:{offset}:unanswerable",
            )
            generated.append(
                BenchmarkCandidate(
                    query_id="",
                    question=question,
                    question_type="unanswerable",
                    answerable=False,
                    gold_answer="",
                    gold_evidence_chunk_ids=[],
                    source_doc_id=unit.doc_id,
                    source_chunk_id=unit.chunk_id,
                    metadata={
                        "generation_rule": "unanswerable_cross_doc_term_mismatch",
                        "title": unit.title,
                        "chunk_method": unit.chunk_method,
                        "published_year": unit.published_year,
                        "unanswerable_source_term_doc_id": source_doc_id,
                    },
                )
            )
            used_signatures.add(signature)
            break
    return generated


def _build_candidate(
    *,
    unit: ArticleUnit,
    question_type: str,
    question: str,
    gold_answer: str,
    generation_rule: str,
    extra_metadata: dict[str, object] | None = None,
) -> BenchmarkCandidate:
    metadata: dict[str, object] = {
        "generation_rule": generation_rule,
        "title": unit.title,
        "chunk_method": unit.chunk_method,
        "published_year": unit.published_year,
        "article_label": unit.article_label,
        "section_path": unit.section_path,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return BenchmarkCandidate(
        query_id="",
        question=_normalize_space(question),
        question_type=question_type,
        answerable=True,
        gold_answer=_normalize_space(gold_answer),
        gold_evidence_chunk_ids=[unit.chunk_id],
        source_doc_id=unit.doc_id,
        source_chunk_id=unit.chunk_id,
        metadata=metadata,
    )


def _filter_and_dedup_candidates(
    candidates: list[BenchmarkCandidate],
    *,
    semantic_threshold: float,
    min_question_chars: int,
    max_question_chars: int,
    min_question_evidence_overlap: float,
) -> CandidateProcessingResult:
    filter_reason_counts: Counter[str] = Counter()
    deduped: list[BenchmarkCandidate] = []
    lexical_seen: set[tuple[str, str]] = set()
    semantic_buckets: dict[str, list[BenchmarkCandidate]] = defaultdict(list)

    for candidate in candidates:
        filter_reason = _detect_filter_reason(
            candidate,
            min_question_chars=min_question_chars,
            max_question_chars=max_question_chars,
            min_question_evidence_overlap=min_question_evidence_overlap,
        )
        if filter_reason:
            filter_reason_counts[filter_reason] += 1
            continue

        lexical_signature = (
            candidate.question_type,
            _normalize_question(candidate.question),
        )
        if lexical_signature in lexical_seen:
            filter_reason_counts["lexical_duplicate"] += 1
            continue
        lexical_seen.add(lexical_signature)

        semantic_signature = _semantic_signature(candidate.question)
        is_semantic_duplicate = False
        for existing in semantic_buckets[candidate.question_type]:
            similarity = _jaccard_similarity(
                semantic_signature,
                _semantic_signature(existing.question),
            )
            if similarity >= semantic_threshold:
                filter_reason_counts["semantic_near_duplicate"] += 1
                is_semantic_duplicate = True
                break
        if is_semantic_duplicate:
            continue

        semantic_buckets[candidate.question_type].append(candidate)
        deduped.append(candidate)

    return CandidateProcessingResult(
        deduped_candidates=deduped,
        filter_reason_counts=filter_reason_counts,
    )


def _detect_filter_reason(
    candidate: BenchmarkCandidate,
    *,
    min_question_chars: int,
    max_question_chars: int,
    min_question_evidence_overlap: float,
) -> str | None:
    question = candidate.question.strip()
    if len(question) < min_question_chars:
        return "too_short_question"
    if len(question) > max_question_chars:
        return "too_long_question"
    if any(marker in question for marker in ["什么什么", "某某", "XXX", "……"]):
        return "placeholder_or_incomplete_question"
    if candidate.answerable and not candidate.gold_evidence_chunk_ids:
        return "missing_evidence"
    if candidate.question_type == "comparison":
        pair = candidate.metadata.get("comparison_pair")
        if not isinstance(pair, list) or len(pair) != 2:
            return "malformed_comparison"
    if candidate.question_type == "unanswerable":
        if candidate.gold_answer.strip() or candidate.gold_evidence_chunk_ids:
            return "malformed_unanswerable"
        return None
    overlap = _question_evidence_overlap(question, candidate.gold_answer)
    if overlap < min_question_evidence_overlap:
        return "weak_question_evidence_overlap"
    return None


def _select_final_benchmark_candidates(
    candidates: list[BenchmarkCandidate],
    *,
    final_target_counts: dict[str, int],
) -> list[BenchmarkCandidate]:
    grouped: dict[str, list[BenchmarkCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.question_type].append(candidate)

    final_candidates: list[BenchmarkCandidate] = []
    final_doc_counts: Counter[str] = Counter()
    final_chunk_counts: Counter[str] = Counter()

    for question_type in QUESTION_TYPES:
        target = final_target_counts.get(question_type, 0)
        pool = sorted(
            grouped.get(question_type, []),
            key=lambda candidate: (
                final_doc_counts[candidate.source_doc_id or ""],
                final_chunk_counts[candidate.source_chunk_id or ""],
                candidate.source_doc_id or "",
                candidate.query_id,
            ),
        )
        selected: list[BenchmarkCandidate] = []
        for candidate in pool:
            if len(selected) >= target:
                break
            if final_doc_counts[candidate.source_doc_id or ""] >= 2:
                continue
            if final_chunk_counts[candidate.source_chunk_id or ""] >= 1:
                continue
            selected.append(candidate)
            final_doc_counts[candidate.source_doc_id or ""] += 1
            final_chunk_counts[candidate.source_chunk_id or ""] += 1
        if len(selected) < target:
            for candidate in pool:
                if len(selected) >= target:
                    break
                if candidate in selected:
                    continue
                selected.append(candidate)
        if len(selected) < target:
            msg = (
                f"Insufficient deduped candidates for {question_type}: "
                f"needed {target}, found {len(selected)}."
            )
            raise ValueError(msg)
        final_candidates.extend(selected[:target])

    for index, candidate in enumerate(final_candidates, start=1):
        candidate.query_id = f"benchmark_v1_q{index:03d}"
    return final_candidates


def _build_report(
    *,
    raw_candidates: list[BenchmarkCandidate],
    deduped_candidates: list[BenchmarkCandidate],
    final_candidates: list[BenchmarkCandidate],
    filter_reason_counts: Counter[str],
    validation_summary: dict[str, object],
    config: BenchmarkGenerationConfig,
) -> str:
    raw_distribution = Counter(candidate.question_type for candidate in raw_candidates)
    deduped_distribution = Counter(
        candidate.question_type for candidate in deduped_candidates
    )
    final_distribution = Counter(
        candidate.question_type for candidate in final_candidates
    )

    lines = [
        "# Benchmark Generation Report",
        "",
        "## Status Labels",
        "",
        "- Implemented: rule-based benchmark candidate generation, lexical deduplication, semantic near-duplicate filtering, low-quality sample filtering, and quota-based benchmark assembly.",
        "- Experimental: heuristic question templates, weak semantic dedup, and automated unanswerable construction.",
        "- Planned: manual spot-checking, stronger semantic filtering, and evidence-level verification refinement.",
        "",
        "## Summary",
        "",
        f"- Raw generated candidates: {len(raw_candidates)}",
        f"- Deduped candidates: {len(deduped_candidates)}",
        f"- Final benchmark size: {len(final_candidates)}",
        f"- Benchmark validation passed: {validation_summary.get('is_valid', False)}",
        "",
        "## Question Type Distribution",
        "",
        "| Question Type | Raw | Deduped | Final |",
        "|---|---:|---:|---:|",
    ]
    for question_type in QUESTION_TYPES:
        lines.append(
            f"| {question_type} | {raw_distribution.get(question_type, 0)} | "
            f"{deduped_distribution.get(question_type, 0)} | {final_distribution.get(question_type, 0)} |"
        )

    lines.extend(
        [
            "",
            "## Filter Reason Counts",
            "",
            "| Reason | Count |",
            "|---|---:|",
        ]
    )
    for reason, count in sorted(filter_reason_counts.items()):
        lines.append(f"| {reason} | {count} |")
    if not filter_reason_counts:
        lines.append("| none | 0 |")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- Candidates: `{config.candidates_output_jsonl}`",
            f"- Deduped candidates: `{config.deduped_output_jsonl}`",
            f"- Final benchmark: `{config.benchmark_output_jsonl}`",
            "",
            "## Validation Notes",
            "",
            f"- Duplicate query ids: {len(validation_summary.get('duplicate_query_ids', []))}",
            f"- Invalid unanswerable gold answers: {len(validation_summary.get('invalid_answerable_gold_answer', []))}",
            f"- Missing evidence for answerable records: {len(validation_summary.get('missing_evidence_for_answerable', []))}",
        ]
    )
    return "\n".join(lines) + "\n"


def _split_sentences(text: str) -> list[str]:
    return [
        _normalize_space(sentence)
        for sentence in SENTENCE_SPLIT_PATTERN.split(text)
        if _normalize_space(sentence)
    ]


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_question(text: str) -> str:
    normalized = _normalize_space(text)
    normalized = re.sub(r"[《》“”\"'‘’【】\[\]（）()，,。！？；：:]", "", normalized)
    return normalized.lower()


def _semantic_signature(text: str) -> set[str]:
    normalized = _normalize_question(text)
    if len(normalized) < 3:
        return {normalized}
    return {normalized[index : index + 3] for index in range(len(normalized) - 2)}


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _question_evidence_overlap(question: str, evidence: str) -> float:
    question_terms = _extract_overlap_terms(question)
    evidence_terms = _extract_overlap_terms(evidence)
    if not question_terms:
        return 0.0
    overlap = question_terms & evidence_terms
    return len(overlap) / len(question_terms)


def _extract_overlap_terms(text: str) -> set[str]:
    normalized = _normalize_question(text)
    return {
        normalized[index : index + 2] for index in range(max(0, len(normalized) - 1))
    }


def _clean_term(term: str) -> str:
    term = TERM_CLEAN_PATTERN.sub("", term)
    term = term.strip("，,：:、 ")
    return _normalize_space(term)


def _pick_template(options: list[str], *, seed_key: str) -> str:
    index = sum(ord(char) for char in seed_key) % len(options)
    return options[index]


def _extract_question_subject(question: str) -> str:
    question = question.strip()
    patterns = [
        r"根据《[^》]+》，(?P<term>.+?)的定义是什么\？?$",
        r"(?P<term>.+?)是指什么\？?$",
        r"根据《[^》]+》，(?P<term>.+?)的含义是什么\？?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, question)
        if match:
            return _clean_term(match.group("term"))
    return ""
