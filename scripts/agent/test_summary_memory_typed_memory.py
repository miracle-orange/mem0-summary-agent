"""Focused checks for typed-memory extraction tracing."""

from __future__ import annotations

import json

from summary_memory_agent_schemas import normalize_typed_memory_record
from summary_memory_typed_memory import (
    apply_cross_chunk_dedup_judgements,
    build_cross_chunk_dedup_candidates,
    canonicalize_deadline_for_dedup,
    deduplicate_cross_chunk_typed_memory_records,
    extract_typed_memories_from_summary,
    parse_cross_chunk_dedup_judge_output,
)


def test_extract_typed_memories_records_trace() -> None:
    trace: dict[str, object] = {}

    raw_output = json.dumps(
        [
            {"memory_type": "time", "content": "明天中午前完成提醒整理"},
            {"memory_type": "unsupported_type", "content": "应被 schema 过滤"},
        ],
        ensure_ascii=False,
    )
    records = extract_typed_memories_from_summary(
        "历史材料。",
        lambda prompt, temperature, max_tokens: raw_output,
        source_time="2026-03-12",
        max_tokens=3000,
        trace=trace,
    )

    assert len(records) == 1
    assert records[0].memory_type == "time"
    assert trace["raw_output"] == raw_output
    assert trace["max_tokens"] == 3000
    assert trace["raw_record_count"] == 2
    assert trace["accepted_record_count"] == 1
    assert trace["rejected_record_count"] == 1
    assert "材料 source_time：2026-03-12" in str(trace["prompt"])
    assert "历史材料：\n历史材料。" in str(trace["prompt"])


def make_typed_record(memory_type: str, content: str, structured_fields: dict[str, object]):
    record = normalize_typed_memory_record(
        {
            "memory_type": memory_type,
            "content": content,
            "structured_fields": structured_fields,
            "source": {"source_time": "2026-03-12", "evidence_span": content},
            "confidence": "high",
        }
    )
    assert record is not None
    return record


def test_cross_chunk_dedup_judge_merges_real_detail_and_reminder_pairs() -> None:
    detailed_task = make_typed_record(
        "task",
        "张雨晴负责整理两个错题提醒并发送到班级学习群，明天中午12点前完成。",
        {
            "owner": "张雨晴",
            "task": "整理两个错题提醒并发布到班级学习群",
            "deadline": "2026-03-13 12:00:00",
            "status": "pending",
        },
    )
    time_record = make_typed_record(
        "time",
        "张雨晴需在明天中午12点前将课堂问题整理发到班级学习群。",
        {
            "owner": "张雨晴",
            "task": "整理两个错题提醒并发布到班级学习群",
            "normalized_time": "2026-03-13 12:00:00",
            "deadline": "2026-03-13 12:00:00",
            "completion_status": "pending",
        },
    )
    short_task = make_typed_record(
        "task",
        "张雨晴明天中午12点前整理两个错题提醒。",
        {
            "owner": "张雨晴",
            "task": "整理两个错题提醒",
            "deadline": "2026-03-13 12:00",
            "status": "pending",
        },
    )
    short_time = make_typed_record(
        "time",
        "张雨晴明天中午12点前整理两个错题提醒。",
        {
            "owner": "张雨晴",
            "task": "整理两个错题提醒",
            "normalized_time": "2026-03-13 12:00",
            "deadline": "2026-03-13 12:00",
            "completion_status": "pending",
        },
    )

    direct_deduplicated, direct_removed = deduplicate_cross_chunk_typed_memory_records(
        [[detailed_task, time_record], [short_task, short_time]]
    )
    candidates, total_candidate_count = build_cross_chunk_dedup_candidates(direct_deduplicated, max_candidates=20)
    decisions = parse_cross_chunk_dedup_judge_output(
        json.dumps(
            {
                "decisions": [
                    {
                        "pair_id": candidate["pair_id"],
                        "same_underlying_fact": True,
                        "confidence": "high",
                        "reason": "详细任务与结尾提醒指向同一待办。",
                    }
                    for candidate in candidates
                ]
            },
            ensure_ascii=False,
        ),
        {candidate["pair_id"] for candidate in candidates},
    )
    deduplicated, judge_removed, outcomes = apply_cross_chunk_dedup_judgements(
        direct_deduplicated,
        candidates,
        decisions,
    )

    assert deduplicated[0] == [detailed_task, time_record]
    assert deduplicated[1] == []
    assert direct_removed == []
    assert total_candidate_count == 2
    assert len(candidates) == 2
    assert len(judge_removed) == 2
    assert {item["removed_memory_id"] for item in judge_removed} == {short_task.memory_id, short_time.memory_id}
    assert {item["retained_memory_id"] for item in judge_removed} == {detailed_task.memory_id, time_record.memory_id}
    assert set(outcomes.values()) == {"merged_by_judge"}


def test_deadline_canonicalization_is_minute_safe() -> None:
    assert canonicalize_deadline_for_dedup("2026-03-13 12:00:00") == "2026-03-13T12:00"
    assert canonicalize_deadline_for_dedup("2026-03-13T12:00") == "2026-03-13T12:00"
    assert canonicalize_deadline_for_dedup("2026/03/13 12:00:30") == "2026-03-13T12:00:30"
    assert canonicalize_deadline_for_dedup("2026-03-13") == ""


if __name__ == "__main__":
    test_extract_typed_memories_records_trace()
    test_cross_chunk_dedup_judge_merges_real_detail_and_reminder_pairs()
    test_deadline_canonicalization_is_minute_safe()
    print("summary_memory_typed_memory tests passed")
