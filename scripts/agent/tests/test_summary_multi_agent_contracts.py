from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.schemas import (  # noqa: E402
    EvidenceRef,
    ExecutionPlan,
    PlanSource,
    SourceKind,
    SubTask,
    SummaryRunResult,
    TaskType,
    RunStatus,
)


class ExecutionPlanContractTests(unittest.TestCase):
    def test_valid_plan_is_accepted(self) -> None:
        plan = ExecutionPlan(
            tasks=(
                SubTask(
                    task_id="memory",
                    task_type=TaskType.SEMANTIC_MEMORY,
                    description="Retrieve relevant typed memories.",
                ),
                SubTask(
                    task_id="template",
                    task_type=TaskType.TEMPLATE_ADAPTATION,
                    description="Select the summary structure.",
                    dependencies=("memory",),
                ),
            )
        )
        self.assertEqual(2, len(plan.tasks))

    def test_unknown_dependency_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ExecutionPlan(
                tasks=(
                    SubTask(
                        task_id="template",
                        task_type=TaskType.TEMPLATE_ADAPTATION,
                        description="Select template.",
                        dependencies=("missing",),
                    ),
                )
            )

    def test_cycle_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ExecutionPlan(
                tasks=(
                    SubTask(
                        task_id="a",
                        task_type=TaskType.SEMANTIC_MEMORY,
                        description="Task A.",
                        dependencies=("b",),
                    ),
                    SubTask(
                        task_id="b",
                        task_type=TaskType.TEMPORAL_REASONING,
                        description="Task B.",
                        dependencies=("a",),
                    ),
                )
            )

    def test_legacy_action_item_task_type_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            SubTask(
                task_id="legacy",
                task_type="action_item",
                description="Legacy type must be mapped by an adapter.",
            )

    def test_replan_requires_complete_lineage_metadata(self) -> None:
        with self.assertRaises(ValidationError):
            ExecutionPlan(
                source=PlanSource.REPLAN,
                tasks=(
                    SubTask(
                        task_id="memory",
                        task_type=TaskType.SEMANTIC_MEMORY,
                        description="Memory.",
                    ),
                ),
            )

    def test_initial_plan_rejects_replan_metadata(self) -> None:
        with self.assertRaises(ValidationError):
            ExecutionPlan(
                source=PlanSource.MODEL,
                parent_plan_id="plan_previous",
                tasks=(
                    SubTask(
                        task_id="memory",
                        task_type=TaskType.SEMANTIC_MEMORY,
                        description="Memory.",
                    ),
                ),
            )


class EvidenceAndCompletionContractTests(unittest.TestCase):
    def test_current_evidence_requires_offsets(self) -> None:
        with self.assertRaises(ValidationError):
            EvidenceRef(
                source_kind=SourceKind.CURRENT_MATERIAL,
                source_id="source",
                segment_id="seg_0000",
            )

    def test_completed_result_requires_full_coverage(self) -> None:
        with self.assertRaises(ValidationError):
            SummaryRunResult(
                run_id="run_1",
                status=RunStatus.COMPLETED,
                final_summary="Summary.",
                coverage_complete=False,
            )


if __name__ == "__main__":
    unittest.main()
