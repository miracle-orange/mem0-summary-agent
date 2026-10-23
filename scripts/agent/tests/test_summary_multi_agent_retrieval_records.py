from __future__ import annotations

import sys
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.retrieval_records import (  # noqa: E402
    assemble_retrieved_records,
)


def record(
    memory_id: str,
    content: str,
    *,
    score: float,
    status: str = "active",
) -> dict:
    return {
        "memory_id": memory_id,
        "memory_type": "task",
        "content": content,
        "score": score,
        "metadata": {
            "history_source_id": "history-1",
            "source_time": "2026-08-01",
            "memory_status": status,
            "structured_fields_json": {"owner": "owner-a"},
        },
    }


class RetrievalRecordAssemblyTests(unittest.TestCase):
    def test_same_id_conflicting_versions_are_all_retained(self) -> None:
        retained, merges, conflicts, stats = assemble_retrieved_records(
            [
                record("memory-1", "task is pending", score=0.9),
                record(
                    "memory-1",
                    "task was cancelled",
                    score=0.8,
                    status="cancelled",
                ),
            ]
        )

        self.assertEqual(2, len(retained))
        self.assertEqual([], merges)
        self.assertEqual(1, len(conflicts))
        self.assertEqual(2, stats["unique_record_version_count"])
        self.assertEqual(
            {"task is pending", "task was cancelled"},
            {item["content"] for item in retained},
        )

    def test_exact_same_source_different_ids_merge_directly(self) -> None:
        retained, merges, conflicts, stats = assemble_retrieved_records(
            [
                record("memory-low", "same task", score=0.4),
                record("memory-high", "same task", score=0.9),
            ]
        )

        self.assertEqual(["memory-high"], [item["memory_id"] for item in retained])
        self.assertEqual([], conflicts)
        self.assertEqual(
            [
                {
                    "removed_memory_id": "memory-low",
                    "retained_memory_id": "memory-high",
                    "reason": "same_source_and_exact_business_fields",
                }
            ],
            merges,
        )
        self.assertEqual(1, stats["exact_equivalent_merge_count"])

    def test_status_difference_prevents_cross_id_merge(self) -> None:
        retained, merges, _conflicts, _stats = assemble_retrieved_records(
            [
                record("memory-active", "same task", score=0.9),
                record(
                    "memory-cancelled",
                    "same task",
                    score=0.8,
                    status="cancelled",
                ),
            ]
        )

        self.assertEqual(2, len(retained))
        self.assertEqual([], merges)

    def test_unknown_business_field_difference_prevents_merge(self) -> None:
        left = record("memory-left", "same task", score=0.9)
        right = record("memory-right", "same task", score=0.8)
        left["metadata"]["future_business_field"] = {"value": "left"}
        right["metadata"]["future_business_field"] = {"value": "right"}

        retained, merges, _conflicts, _stats = assemble_retrieved_records(
            [left, right]
        )

        self.assertEqual(2, len(retained))
        self.assertEqual([], merges)

    def test_case_sensitive_business_text_difference_is_preserved(self) -> None:
        retained, merges, _conflicts, _stats = assemble_retrieved_records(
            [
                record("memory-upper", "Use US routing", score=0.9),
                record("memory-lower", "Use us routing", score=0.8),
            ]
        )

        self.assertEqual(2, len(retained))
        self.assertEqual([], merges)


if __name__ == "__main__":
    unittest.main()
