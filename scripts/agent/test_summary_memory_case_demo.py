"""Focused checks for the batch ingest behavior in the case workflow."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from summary_memory_agent_schemas import normalize_typed_memory_record
from summary_memory_case_demo import (
    SummaryDemoInput,
    deduplicate_batch_typed_memory_records,
    save_batch_ingest_outputs,
)


def make_task(content: str, task: str):
    record = normalize_typed_memory_record(
        {
            "memory_type": "task",
            "content": content,
            "structured_fields": {
                "owner": "张雨晴",
                "task": task,
                "deadline": "2026-03-13 12:00",
            },
            "source": {"source_time": "2026-03-12", "evidence_span": content},
            "confidence": "high",
        }
    )
    assert record is not None
    return record


def test_batch_dedup_only_merges_chunks_from_the_same_history_source() -> None:
    detailed = make_task(
        "张雨晴负责整理两个错题提醒并发送到班级学习群，明天中午12点前完成。",
        "整理错题提醒并发送到班级学习群",
    )
    short = make_task(
        "张雨晴明天中午12点前整理两个错题提醒。",
        "整理两个错题提醒",
    )
    other_source_short = make_task(
        "张雨晴明天中午12点前整理两个错题提醒。",
        "整理两个错题提醒",
    )
    batch_results = [
        {
            "history_source_id": "pasted_history_1",
            "source_time": "2026-03-12",
            "typed_records": [detailed],
        },
        {
            "history_source_id": "pasted_history_1",
            "source_time": "2026-03-12",
            "typed_records": [short],
        },
        {
            "history_source_id": "pasted_history_2",
            "source_time": "2026-03-12",
            "typed_records": [other_source_short],
        },
    ]

    report = deduplicate_batch_typed_memory_records(batch_results)

    assert report["retained_count"] == 3
    assert report["removed_count"] == 0
    assert report["groups"][0]["judge"]["status"] == "skipped_no_llm"

    def judge_low_confidence(prompt: str, temperature: float, max_tokens: int) -> str:
        return json.dumps(
            {
                "decisions": [
                    {
                        "pair_id": "candidate_1",
                        "same_underlying_fact": True,
                        "confidence": "low",
                        "reason": "证据不足。",
                    }
                ]
            },
            ensure_ascii=False,
        )

    report = deduplicate_batch_typed_memory_records(batch_results, judge_llm_call=judge_low_confidence)
    assert report["removed_count"] == 0
    assert report["groups"][0]["candidate_pairs"][0]["outcome"] == "not_merged_judge_low_confidence"

    def judge_same_task(prompt: str, temperature: float, max_tokens: int) -> str:
        assert temperature == 0.0
        assert max_tokens == 1200
        assert "pasted_history" not in prompt
        return json.dumps(
            {
                "decisions": [
                    {
                        "pair_id": "candidate_1",
                        "same_underlying_fact": True,
                        "confidence": "high",
                        "reason": "同一待办的详细和简略表述。",
                    }
                ]
            },
            ensure_ascii=False,
        )

    report = deduplicate_batch_typed_memory_records(batch_results, judge_llm_call=judge_same_task)

    assert report["input_count"] == 3
    assert report["retained_count"] == 2
    assert report["removed_count"] == 1
    assert len(batch_results[0]["typed_records"]) == 1
    assert batch_results[1]["typed_records"] == []
    assert batch_results[2]["typed_records"] == [other_source_short]
    assert report["groups"][0]["history_source_id"] == "pasted_history_1"
    assert report["groups"][0]["semantic_candidate_count"] == 1
    assert report["groups"][0]["candidate_pairs"][0]["outcome"] == "merged_by_judge"

    for item in batch_results:
        item["source_file"] = None
        item["typed_memories"] = [record.to_dict() for record in item["typed_records"]]
        item["memories"] = [record.content for record in item["typed_records"]]
        item["add_results"] = []
        item["extraction_trace"] = {}

    demo_inputs = [
        SummaryDemoInput(
            user_id="user",
            project="project",
            source_type="classroom_transcript",
            source_time="2026-03-12",
            history_summary=f"chunk {index}",
            current_content="",
            history_source_id=str(item["history_source_id"]),
        )
        for index, item in enumerate(batch_results, start=1)
    ]
    with TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        save_batch_ingest_outputs(output_dir, demo_inputs, batch_results, typed_memory_dedup_report=report)
        saved_report = json.loads((output_dir / "typed_memory_dedup_report.json").read_text(encoding="utf-8"))
        saved_memories = json.loads((output_dir / "typed_memories.json").read_text(encoding="utf-8"))

    assert saved_report["removed_count"] == 1
    assert sum(len(item["typed_memories"]) for item in saved_memories) == 2


def test_batch_dedup_requires_a_source_time_anchor() -> None:
    first = make_task(
        "张雨晴负责整理两个错题提醒并发送到班级学习群。",
        "整理错题提醒并发送到班级学习群",
    )
    second = make_task(
        "张雨晴整理两个错题提醒并发送到班级学习群。",
        "整理错题提醒并发送到班级学习群",
    )
    batch_results = [
        {"history_source_id": "pasted_history_1", "source_time": "", "typed_records": [first]},
        {"history_source_id": "pasted_history_1", "source_time": "", "typed_records": [second]},
    ]

    report = deduplicate_batch_typed_memory_records(batch_results)

    assert report["removed_count"] == 0
    assert report["groups"] == []
    assert batch_results[0]["typed_records"] == [first]
    assert batch_results[1]["typed_records"] == [second]


def test_batch_dedup_fails_closed_when_judge_errors() -> None:
    first = make_task(
        "张雨晴负责整理两个错题提醒并发送到班级学习群。",
        "整理错题提醒并发送到班级学习群",
    )
    second = make_task(
        "张雨晴明天中午12点前整理两个错题提醒。",
        "整理两个错题提醒",
    )
    batch_results = [
        {"history_source_id": "pasted_history_1", "source_time": "2026-03-12", "typed_records": [first]},
        {"history_source_id": "pasted_history_1", "source_time": "2026-03-12", "typed_records": [second]},
    ]

    def failing_judge(*args, **kwargs):
        raise RuntimeError("gateway unavailable")

    report = deduplicate_batch_typed_memory_records(batch_results, judge_llm_call=failing_judge)

    assert report["removed_count"] == 0
    assert report["groups"][0]["judge"]["status"] == "failed"
    assert report["groups"][0]["candidate_pairs"][0]["outcome"] == "not_merged_judge_error"
    assert batch_results[0]["typed_records"] == [first]
    assert batch_results[1]["typed_records"] == [second]


if __name__ == "__main__":
    test_batch_dedup_only_merges_chunks_from_the_same_history_source()
    test_batch_dedup_requires_a_source_time_anchor()
    test_batch_dedup_fails_closed_when_judge_errors()
    print("summary_memory_case_demo tests passed")
