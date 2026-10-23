from __future__ import annotations

import json
import asyncio
import sys
import tempfile
import unittest
from pathlib import Path


AGENT_DIR = Path(__file__).resolve().parents[1]
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from summary_multi_agent.history_selection import build_history_selection_report  # noqa: E402
from summary_multi_agent.adapters import (  # noqa: E402
    ContextOverflowError,
    ContextWindowModelClient,
)
from summary_multi_agent.model_input_audit import ModelInputAuditRecorder  # noqa: E402
from summary_multi_agent.raw_material import build_raw_material_bundle  # noqa: E402
from summary_multi_agent.schemas import (  # noqa: E402
    AgentResult,
    AgentStatus,
    ExecutionPlan,
    PlanSource,
    SubTask,
    SummaryRequest,
    TaskType,
)
from summary_multi_agent.working_memory import initialize_working_state  # noqa: E402


class ModelInputAuditTests(unittest.TestCase):
    def test_report_is_written_before_call_and_contains_only_lengths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = ModelInputAuditRecorder(Path(temp_dir))
            secret_prompt = "协议说明\n完整当前材料"
            call_id = recorder.start_call(
                stage="planner",
                prompt_version="planner.v-test",
                prompt=secret_prompt,
                sections={
                    "instructions": "协议说明",
                    "current_material": "完整当前材料",
                },
                max_tokens=1000,
                attempt_index=1,
                input_metrics={
                    "record_occurrence_count": 8,
                    "unique_memory_id_count": 1,
                },
            )
            started = json.loads(recorder.path.read_text("utf-8"))

            self.assertEqual("started", started["calls"][0]["status"])
            self.assertNotIn(secret_prompt, recorder.path.read_text("utf-8"))
            self.assertEqual(
                len(secret_prompt), started["calls"][0]["totals"]["chars"]
            )
            self.assertIn("token_count_measured", started["calls"][0]["totals"])
            self.assertEqual(
                8,
                started["calls"][0]["input_metrics"][
                    "record_occurrence_count"
                ],
            )
            self.assertIsNone(started["calls"][0]["message_tokens"])
            self.assertIsNone(started["calls"][0]["server_usage_tokens"])

            recorder.finish_call(
                call_id,
                status="failed",
                model_lock_wait_ms=17,
                model_call_duration_ms=31,
                error="TimeoutError",
            )
            finished = json.loads(recorder.path.read_text("utf-8"))["calls"][0]
            self.assertEqual("failed", finished["status"])
            self.assertEqual(17, finished["model_lock_wait_ms"])
            self.assertEqual(31, finished["model_call_duration_ms"])
            self.assertIn("started_at", finished)
            self.assertIn("finished_at", finished)

    def test_worker_progress_survives_without_terminal_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = ModelInputAuditRecorder(Path(temp_dir))
            recorder.record_worker_progress(
                task_id="semantic_memory",
                step_index=1,
                event="tool_observations_completed",
                observation_ids=("observation-1",),
                model_calls_used=1,
                tool_calls_used=1,
            )
            report = json.loads(recorder.path.read_text("utf-8"))
            self.assertEqual(
                "observation-1",
                report["worker_progress"][0]["observation_ids"][0],
            )

    def test_measured_context_overflow_is_rejected_before_delegate_call(self):
        class Delegate:
            def __init__(self):
                self.calls = 0

            async def complete(self, prompt, *, temperature, max_tokens):
                del prompt, temperature, max_tokens
                self.calls += 1
                return "unreachable"

        class Counter:
            @staticmethod
            def measure(prompt):
                del prompt
                return {"tokens": 4000, "token_count_measured": True}

        delegate = Delegate()
        guarded = ContextWindowModelClient(
            delegate,
            context_window_tokens=4096,
            safety_margin_tokens=128,
        )
        guarded._counter = Counter()

        with self.assertRaises(ContextOverflowError):
            asyncio.run(
                guarded.complete(
                    "prompt",
                    temperature=0.0,
                    max_tokens=256,
                )
            )
        self.assertEqual(0, delegate.calls)

    def test_estimated_count_is_audit_only_and_does_not_block_call(self):
        class Delegate:
            def __init__(self):
                self.calls = 0

            async def complete(self, prompt, *, temperature, max_tokens):
                del prompt, temperature, max_tokens
                self.calls += 1
                return "ok"

        class Counter:
            @staticmethod
            def measure(prompt):
                del prompt
                return {"tokens": 999999, "token_count_measured": False}

        delegate = Delegate()
        guarded = ContextWindowModelClient(
            delegate,
            context_window_tokens=4096,
            safety_margin_tokens=128,
        )
        guarded._counter = Counter()

        result = asyncio.run(
            guarded.complete("prompt", temperature=0.0, max_tokens=256)
        )
        self.assertEqual("ok", result)
        self.assertEqual(1, delegate.calls)

    def test_server_context_error_is_normalized(self):
        class Delegate:
            async def complete(self, prompt, *, temperature, max_tokens):
                del prompt, temperature, max_tokens
                raise RuntimeError(
                    "This model's maximum context length is 16384 tokens. "
                    "However, you requested 20000 tokens."
                )

        guarded = ContextWindowModelClient(
            Delegate(),
            context_window_tokens=None,
        )
        with self.assertRaises(ContextOverflowError) as captured:
            asyncio.run(
                guarded.complete(
                    "prompt",
                    temperature=0.0,
                    max_tokens=256,
                )
            )
        self.assertIn("context_overflow", str(captured.exception))


class HistorySelectionReportTests(unittest.TestCase):
    @staticmethod
    def make_state():
        request = SummaryRequest(current_material="当前材料")
        raw = build_raw_material_bundle(request.current_material)
        return initialize_working_state(request, raw)

    @staticmethod
    def semantic_plan() -> ExecutionPlan:
        return ExecutionPlan(
            source=PlanSource.MODEL,
            tasks=(
                SubTask(
                    task_id="semantic_memory",
                    task_type=TaskType.SEMANTIC_MEMORY,
                    description="检索相关历史",
                    allowed_tools=("typed_memory_search",),
                    inputs={"queries": ["相关历史"]},
                    max_tool_calls=1,
                ),
            ),
        )

    def test_unplanned_semantic_is_distinct_from_failure_and_empty(self):
        state = self.make_state()
        self.assertEqual(
            "not_planned", build_history_selection_report(state)["status"]
        )

        state.plan = self.semantic_plan()
        state.agent_results["semantic_memory"] = AgentResult(
            task_id="semantic_memory",
            task_type=TaskType.SEMANTIC_MEMORY,
            status=AgentStatus.FAILED,
            error="search timeout",
        )
        self.assertEqual(
            "unavailable", build_history_selection_report(state)["status"]
        )

        state.agent_results["semantic_memory"] = AgentResult(
            task_id="semantic_memory",
            task_type=TaskType.SEMANTIC_MEMORY,
            status=AgentStatus.SUCCESS,
            is_empty=True,
            payload={"records": []},
        )
        self.assertEqual("empty", build_history_selection_report(state)["status"])

    def test_selected_and_available_ids_are_reported_separately(self):
        state = self.make_state()
        state.plan = self.semantic_plan()
        state.agent_results["semantic_memory"] = AgentResult(
            task_id="semantic_memory",
            task_type=TaskType.SEMANTIC_MEMORY,
            status=AgentStatus.SUCCESS,
            payload={
                "records": [
                    {
                        "memory_id": "selected-1",
                        "memory_type": "task",
                        "content": "相关任务",
                    }
                ],
                "selection_mode": "record_ids",
                "available_record_ids": ["selected-1", "excluded-1"],
                "requested_memory_ids": ["selected-1"],
                "selected_memory_ids": ["selected-1"],
                "equivalent_record_merges": [],
            },
        )
        report = build_history_selection_report(state)

        self.assertEqual("records", report["status"])
        self.assertEqual(
            ["selected-1", "excluded-1"], report["available_memory_ids"]
        )
        self.assertEqual(["selected-1"], report["selected_memory_ids"])


if __name__ == "__main__":
    unittest.main()
