from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.schemas import RunStatus  # noqa: E402
from summary_multi_agent_web.multi_agent_runner import run_multi_agent_summary  # noqa: E402
from summary_multi_agent_web.run_manager import WebRunManager  # noqa: E402
from summary_multi_agent_web.schemas import (  # noqa: E402
    WebRunStatus,
    WebSummarizeRequest,
    build_run_config,
    build_summary_request,
)


def make_request(**overrides: object) -> WebSummarizeRequest:
    values: dict[str, object] = {
        "current_material": "原始当前材料\n第二段",
        "summary_request": "请原样保留这个请求，并突出任务。",
        "user_id": "user-a",
        "project": "project-a",
        "source_time": "2026-08-20",
        "target_entities": ["张三"],
        "summary_style": "structured",
        "planner_mode": "model",
        "worker_mode": "rule",
        "context_compression_mode": "deterministic",
        "enable_review": True,
    }
    values.update(overrides)
    return WebSummarizeRequest.model_validate(values)


class FakeMemory:
    def __init__(self) -> None:
        self.closed = False
        self.mutations: list[str] = []

    def add(self, *args, **kwargs):
        self.mutations.append("add")
        raise AssertionError("summarize must not call memory.add")

    def update(self, *args, **kwargs):
        self.mutations.append("update")
        raise AssertionError("summarize must not call memory.update")

    def delete(self, *args, **kwargs):
        self.mutations.append("delete")
        raise AssertionError("summarize must not call memory.delete")


class FakeResult:
    status = RunStatus.COMPLETED_WITH_WARNINGS
    final_summary = "正式 candidate 0"
    recommended_candidate_index = 1
    warnings = ("worker warning",)


class FakePipeline:
    def __init__(self, request_sink: list[object]) -> None:
        self.request_sink = request_sink

    async def run(self, request):
        self.request_sink.append(request)
        return FakeResult()


class WebRuntimeTests(unittest.TestCase):
    def test_request_and_config_mapping_preserve_user_request(self) -> None:
        web_request = make_request(
            worker_model_timeout_seconds=321,
            model_context_window_tokens=65536,
            model_context_safety_margin_tokens=2048,
            minutes_output_mode="rich",
            rich_summary_max_tokens=7200,
        )
        request = build_summary_request(web_request)
        config = build_run_config(web_request)

        self.assertEqual(web_request.current_material, request.current_material)
        self.assertEqual(web_request.summary_request, request.summary_request)
        self.assertEqual(["张三"], request.target_entities)
        self.assertEqual("structured", request.personalization_preferences["summary_style"])
        self.assertEqual("model", config.planner_mode.value)
        self.assertEqual("rule", config.worker_mode.value)
        self.assertTrue(config.review_enabled)
        self.assertEqual(321, config.worker_model_timeout_seconds)
        self.assertEqual(65536, config.model_context_window_tokens)
        self.assertEqual(2048, config.model_context_safety_margin_tokens)
        self.assertEqual("rich", config.minutes_output_mode.value)
        self.assertEqual(7200, config.rich_summary_max_tokens)

    def test_manager_exposes_only_explicit_public_report_artifacts(self) -> None:
        async def runner(request, output_dir, *, status_callback=None):
            (output_dir / "minutes_report.html").write_text(
                "<html>report</html>", encoding="utf-8"
            )
            (output_dir / "minutes_report_bundle.zip").write_bytes(b"zip")
            return FakeResult()

        manager = WebRunManager(runner=runner)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                snapshot = manager.start_summarize(make_request(), Path(temp_dir))
                for _ in range(100):
                    final = manager.get_snapshot(snapshot.run_id)
                    if final.finished_at:
                        break
                    threading.Event().wait(0.01)
                self.assertEqual(
                    "minutes_report.html",
                    manager.get_report_artifact(
                        snapshot.run_id, "minutes_report.html"
                    ).name,
                )
                self.assertEqual(
                    "minutes_report_bundle.zip",
                    manager.get_report_artifact(
                        snapshot.run_id, "minutes_report_bundle.zip"
                    ).name,
                )
                with self.assertRaises(ValueError):
                    manager.get_report_artifact(
                        snapshot.run_id, "multi_agent_summary_prompt.txt"
                    )
        finally:
            manager.close()

    def test_runner_calls_v2_pipeline_and_closes_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory = FakeMemory()
            received: list[object] = []
            builder = Mock(return_value=FakePipeline(received))
            closed: list[object] = []

            result = asyncio.run(
                run_multi_agent_summary(
                    make_request(),
                    Path(temp_dir),
                    memory_factory=lambda: memory,
                    memory_closer=closed.append,
                    llm_call=Mock(),
                    pipeline_builder=builder,
                )
            )

            self.assertEqual("正式 candidate 0", result.final_summary)
            self.assertEqual(1, len(received))
            request = received[0]
            self.assertEqual(make_request().summary_request, request.summary_request)
            builder.assert_called_once()
            self.assertEqual([memory], closed)
            self.assertEqual([], memory.mutations)
            payload = json.loads((Path(temp_dir) / "web_run_info.json").read_text())
            self.assertEqual("completed_with_warnings", payload["status"])
            self.assertNotIn("api_key", json.dumps(payload).lower())

    def test_manager_starts_in_background_and_allows_only_one_active_run(self) -> None:
        started = threading.Event()
        release = threading.Event()

        async def runner(request, output_dir, *, status_callback=None):
            started.set()
            release.wait(timeout=2)
            return FakeResult()

        manager = WebRunManager(runner=runner)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                first = manager.start_summarize(make_request(), Path(temp_dir))
                self.assertTrue(started.wait(timeout=1))
                self.assertIn(first.status, {WebRunStatus.QUEUED, WebRunStatus.RUNNING})
                with self.assertRaises(RuntimeError):
                    manager.start_summarize(make_request(project="project-b"), Path(temp_dir))
                release.set()
                for _ in range(100):
                    snapshot = manager.get_snapshot(first.run_id)
                    if snapshot.finished_at:
                        break
                    threading.Event().wait(0.01)
                self.assertEqual(WebRunStatus.COMPLETED_WITH_WARNINGS, snapshot.status)
                second = manager.start_summarize(make_request(project="project-c"), Path(temp_dir))
                self.assertNotEqual(first.output_dir, second.output_dir)
                release.set()
        finally:
            release.set()
            manager.close()

    def test_stop_request_is_cancelling_and_late_completion_is_not_called_cancelled(self) -> None:
        started = threading.Event()
        release = threading.Event()

        async def runner(request, output_dir, *, status_callback=None):
            started.set()
            release.wait(timeout=2)
            return FakeResult()

        manager = WebRunManager(runner=runner)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                snapshot = manager.start_summarize(make_request(), Path(temp_dir))
                self.assertTrue(started.wait(timeout=1))
                stopping = manager.request_stop(snapshot.run_id)
                self.assertEqual(WebRunStatus.CANCELLING, stopping.status)
                self.assertNotEqual("cancelled", stopping.status.value)
                release.set()
                for _ in range(100):
                    final = manager.get_snapshot(snapshot.run_id)
                    if final.finished_at:
                        break
                    threading.Event().wait(0.01)
                self.assertEqual(WebRunStatus.COMPLETED_WITH_WARNINGS, final.status)
                self.assertTrue(any("未宣称已取消" in warning for warning in final.warnings))
        finally:
            release.set()
            manager.close()


if __name__ == "__main__":
    unittest.main()
