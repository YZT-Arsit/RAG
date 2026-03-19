from __future__ import annotations

from pathlib import Path

from legal_rag.benchmark.io import iter_candidates
from legal_rag.benchmark.loader import iter_benchmark_records
from legal_rag.benchmark.validators import validate_benchmark
from legal_rag.benchmark.generation import run_benchmark_generation
from legal_rag.chunking.io import write_chunks
from legal_rag.config.schema import BenchmarkGenerationConfig
from legal_rag.schemas.chunk import Chunk


def test_run_benchmark_generation_pipeline(tmp_path: Path) -> None:
    input_chunk_jsonl = tmp_path / "chunks.jsonl"
    candidates_output_jsonl = tmp_path / "candidates.jsonl"
    deduped_output_jsonl = tmp_path / "deduped.jsonl"
    benchmark_output_jsonl = tmp_path / "benchmark.jsonl"
    report_path = tmp_path / "report.md"

    write_chunks(
        input_chunk_jsonl,
        [
            _chunk(
                "doc-1",
                "c-1",
                "电子证照管理办法",
                "第一条 电子证照，是指由行政机关依法制发的电子形式证照。",
            ),
            _chunk(
                "doc-2",
                "c-2",
                "备案管理规定",
                "第二条 申请备案的单位应当具备下列条件：具有固定场所、具备人员和制度。",
            ),
            _chunk(
                "doc-3",
                "c-3",
                "许可办理细则",
                "第三条 申请人申请许可，应当提交申请书。受理机关收到材料后应当进行审查，并一次性通知补正。",
            ),
            _chunk(
                "doc-4",
                "c-4",
                "监督管理条例",
                "第四条 市场监督管理部门负责本条例的监督管理工作。",
            ),
            _chunk(
                "doc-5",
                "c-5",
                "分类办法",
                "第五条 许可事项分为一般许可和特殊许可两类。一般许可适用常规程序，特殊许可适用特别程序。",
            ),
            _chunk(
                "doc-6",
                "c-6",
                "电子证照适用规则",
                "第一条 电子证照，是指通过信息系统签发的数字证明文件。",
            ),
            _chunk(
                "doc-7",
                "c-7",
                "变更登记办法",
                "第六条 办理变更登记，应当提交申请表。受理机关收到材料后应当进行审查。申请人应当符合下列条件：具备合法身份证明。",
            ),
        ],
    )

    config = BenchmarkGenerationConfig(
        input_chunk_jsonl=input_chunk_jsonl,
        candidates_output_jsonl=candidates_output_jsonl,
        deduped_output_jsonl=deduped_output_jsonl,
        benchmark_output_jsonl=benchmark_output_jsonl,
        report_path=report_path,
        target_candidate_count=9,
        candidate_target_counts={
            "definition": 2,
            "condition": 2,
            "procedure": 1,
            "responsibility": 1,
            "comparison": 1,
            "unanswerable": 2,
        },
        final_target_counts={
            "definition": 1,
            "condition": 1,
            "procedure": 1,
            "responsibility": 1,
            "comparison": 1,
            "unanswerable": 1,
        },
        min_chunk_text_length=20,
        max_candidates_per_doc_per_type=2,
    )

    run_benchmark_generation(config)

    candidates = list(iter_candidates(candidates_output_jsonl))
    deduped_candidates = list(iter_candidates(deduped_output_jsonl))
    records = list(iter_benchmark_records(benchmark_output_jsonl))
    summary = validate_benchmark(records)

    assert len(candidates) == 9
    assert len(deduped_candidates) >= 6
    assert len(records) == 6
    assert {record.question_type for record in records} == {
        "definition",
        "condition",
        "procedure",
        "responsibility",
        "comparison",
        "unanswerable",
    }
    assert summary["is_valid"] is True
    assert report_path.exists()


def test_benchmark_generation_requires_candidate_target_sum_match(
    tmp_path: Path,
) -> None:
    config = BenchmarkGenerationConfig(
        input_chunk_jsonl=tmp_path / "chunks.jsonl",
        candidates_output_jsonl=tmp_path / "candidates.jsonl",
        deduped_output_jsonl=tmp_path / "deduped.jsonl",
        benchmark_output_jsonl=tmp_path / "benchmark.jsonl",
        report_path=tmp_path / "report.md",
        target_candidate_count=10,
        candidate_target_counts={
            "definition": 1,
            "condition": 1,
            "procedure": 1,
            "responsibility": 1,
            "comparison": 1,
            "unanswerable": 1,
        },
    )

    try:
        run_benchmark_generation(config)
    except ValueError as exc:
        assert "candidate_target_counts must sum to target_candidate_count" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for mismatched candidate target totals."
        )


def _chunk(doc_id: str, chunk_id: str, title: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_index=0,
        chunk_method="structure",
        text=text,
        text_length=len(text),
        start_char=0,
        end_char=len(text),
        title=title,
        source_file=f"{doc_id}.json",
        publish_source=None,
        canonical_source=None,
        published_year=2024,
        section_path=[],
        structure_labels=[],
        quality_flags=[],
        metadata={},
    )
