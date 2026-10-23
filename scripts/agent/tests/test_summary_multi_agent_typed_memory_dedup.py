from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from summary_memory_agent_schemas import normalize_typed_memory_record  # noqa: E402
from summary_memory_typed_memory import (  # noqa: E402
    apply_cross_chunk_dedup_judgements,
    build_cross_chunk_dedup_candidates,
    build_cross_chunk_dedup_judge_prompt,
    deduplicate_cross_chunk_typed_memory_records,
)


def make_record(
    memory_type: str,
    content: str,
    *,
    fields: dict | None = None,
    evidence: str | None = None,
    entities: list[str] | None = None,
    topics: list[str] | None = None,
    status: str = "unknown",
):
    record = normalize_typed_memory_record(
        {
            "memory_type": memory_type,
            "content": content,
            "structured_fields": fields or {},
            "source": {
                "source_time": "2026-07-21",
                "evidence_span": evidence or content,
            },
            "entities": entities or [],
            "topics": topics or [],
            "status": status,
            "confidence": "high",
        }
    )
    assert record is not None
    return record


def high_decisions(candidates):
    return {
        item["pair_id"]: {
            "pair_id": item["pair_id"],
            "same_underlying_fact": True,
            "confidence": "high",
            "reason": "固定测试裁决",
        }
        for item in candidates
    }


class TypedMemoryDedupTests(unittest.TestCase):
    def test_exact_direct_dedup_requires_equal_business_fields(self):
        left = make_record(
            "task",
            "完成接口联调",
            fields={"owner": "李明", "task": "完成接口联调"},
        )
        exact = make_record(
            "task",
            "完成接口联调",
            fields={"owner": "李明", "task": "完成接口联调"},
        )
        different_owner = make_record(
            "task",
            "完成接口联调",
            fields={"owner": "王芳", "task": "完成接口联调"},
        )

        deduped, removed = deduplicate_cross_chunk_typed_memory_records(
            [[left], [exact, different_owner]]
        )

        self.assertEqual(2, sum(len(items) for items in deduped))
        self.assertEqual(1, len(removed))
        self.assertIn(different_owner, deduped[1])

    def test_missing_owner_and_date_duplicate_reaches_judge(self):
        detailed = make_record("task", "完成接口联调并提交测试报告")
        short = make_record("task", "完成接口联调并提交报告")

        candidates, total = build_cross_chunk_dedup_candidates(
            [[detailed], [short]], max_candidates=20
        )
        deduped, removed, outcomes = apply_cross_chunk_dedup_judgements(
            [[detailed], [short]], candidates, high_decisions(candidates)
        )

        self.assertEqual(1, total)
        self.assertEqual(1, len(candidates))
        self.assertEqual(1, sum(len(items) for items in deduped))
        self.assertEqual(1, len(removed))
        self.assertEqual("merged_by_judge", next(iter(outcomes.values())))

    def test_same_owner_different_task_is_not_merged(self):
        left = make_record(
            "task",
            "李明负责完成接口A联调",
            fields={"owner": "李明", "task": "完成接口A联调"},
        )
        right = make_record(
            "task",
            "李明负责完成接口B联调",
            fields={"owner": "李明", "task": "完成接口B联调"},
        )
        candidates, _ = build_cross_chunk_dedup_candidates(
            [[left], [right]], max_candidates=20
        )
        deduped, removed, outcomes = apply_cross_chunk_dedup_judgements(
            [[left], [right]], candidates, high_decisions(candidates)
        )

        self.assertEqual(2, sum(len(items) for items in deduped))
        self.assertEqual([], removed)
        self.assertEqual(
            {"not_merged_business_field_conflict"}, set(outcomes.values())
        )

    def test_numbers_negation_status_and_date_precision_are_conflicts(self):
        pairs = [
            (
                make_record("risk", "当前时延为10ms"),
                make_record("risk", "当前时延为2ms"),
            ),
            (
                make_record("decision", "继续使用旧接口"),
                make_record("decision", "不再使用旧接口"),
            ),
            (
                make_record(
                    "task",
                    "李明完成联调",
                    fields={"task": "完成联调", "status": "completed"},
                ),
                make_record(
                    "task",
                    "李明完成联调",
                    fields={"task": "完成联调", "status": "cancelled"},
                ),
            ),
            (
                make_record(
                    "time",
                    "七月二十九日完成联调",
                    fields={"deadline": "2026-07-29"},
                ),
                make_record(
                    "time",
                    "七月二十九日上午完成联调",
                    fields={"deadline": "2026-07-29 10:00"},
                ),
            ),
        ]
        for left, right in pairs:
            with self.subTest(left=left.content, right=right.content):
                candidates, _ = build_cross_chunk_dedup_candidates(
                    [[left], [right]], max_candidates=20
                )
                self.assertTrue(candidates)
                deduped, removed, _ = apply_cross_chunk_dedup_judgements(
                    [[left], [right]], candidates, high_decisions(candidates)
                )
                self.assertEqual(2, sum(len(items) for items in deduped))
                self.assertEqual([], removed)

    def test_complementary_fields_and_cross_types_are_retained(self):
        basic = make_record(
            "task",
            "李明负责完成接口联调",
            fields={
                "owner": "李明",
                "task": "完成接口联调",
                "constraint": "必须使用测试环境",
            },
        )
        complementary = make_record(
            "task",
            "李明负责完成接口联调并通知王芳",
            fields={
                "owner": "李明",
                "task": "完成接口联调",
                "dependency": "通知王芳确认",
            },
        )
        other_type = make_record("risk", "李明负责完成接口联调")
        candidates, total = build_cross_chunk_dedup_candidates(
            [[basic, other_type], [complementary]], max_candidates=20
        )
        deduped, removed, _ = apply_cross_chunk_dedup_judgements(
            [[basic, other_type], [complementary]],
            candidates,
            high_decisions(candidates),
        )

        self.assertEqual(1, total)
        self.assertEqual(3, sum(len(items) for items in deduped))
        self.assertEqual([], removed)

    def test_low_confidence_missing_decision_and_transitive_path_retain(self):
        a = make_record("task", "完成接口联调并提交完整测试报告")
        b = make_record("task", "完成接口联调并提交测试报告")
        c = make_record("task", "完成接口联调并提交报告")
        candidates, _ = build_cross_chunk_dedup_candidates(
            [[a], [b], [c]], max_candidates=20
        )
        decisions = high_decisions(candidates)
        first_pair = candidates[0]["pair_id"]
        decisions[first_pair]["confidence"] = "low"
        decisions.pop(candidates[-1]["pair_id"])
        deduped, removed, outcomes = apply_cross_chunk_dedup_judgements(
            [[a], [b], [c]], candidates, decisions
        )

        self.assertGreaterEqual(sum(len(items) for items in deduped), 2)
        self.assertLessEqual(len(removed), 1)
        self.assertIn("not_merged_judge_low_confidence", outcomes.values())
        self.assertIn("not_merged_missing_judge_decision", outcomes.values())

    def test_direct_judgements_do_not_merge_through_transitive_path(self):
        a = make_record("task", "完成接口联调并提交完整测试报告")
        b = make_record("task", "完成接口联调并提交测试报告")
        c = make_record("task", "完成接口联调并提交报告")
        all_candidates, _ = build_cross_chunk_dedup_candidates(
            [[a], [b], [c]], max_candidates=20
        )
        candidates = [
            item
            for item in all_candidates
            if {
                item["left_locator"][1], item["right_locator"][1]
            }
            in ({a.memory_id, b.memory_id}, {b.memory_id, c.memory_id})
        ]
        deduped, removed, outcomes = apply_cross_chunk_dedup_judgements(
            [[a], [b], [c]], candidates, high_decisions(candidates)
        )

        self.assertEqual(2, sum(len(items) for items in deduped))
        self.assertEqual(1, len(removed))
        self.assertIn("not_merged_transitive_path", outcomes.values())

    def test_candidate_cap_is_three_pairs_per_record_and_prompt_is_cautious(self):
        anchor = make_record("task", "完成接口联调并提交报告")
        others = [
            make_record("task", f"完成接口联调并提交报告给小组{index}")
            for index in range(6)
        ]
        candidates, total = build_cross_chunk_dedup_candidates(
            [[anchor], *[[item] for item in others]], max_candidates=20
        )
        anchor_pairs = [
            item
            for item in candidates
            if anchor.memory_id
            in {
                item["left_locator"][1],
                item["right_locator"][1],
            }
        ]

        self.assertGreater(total, 3)
        self.assertLessEqual(len(anchor_pairs), 3)
        prompt = build_cross_chunk_dedup_judge_prompt(candidates)
        self.assertIn("候选资格当成重复结论", prompt)
        self.assertIn("互补信息", prompt)


if __name__ == "__main__":
    unittest.main()
