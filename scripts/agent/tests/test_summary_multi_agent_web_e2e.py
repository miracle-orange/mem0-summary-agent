from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.schemas import RunStatus  # noqa: E402
from summary_multi_agent_web.e2e_ingest import (  # noqa: E402
    ingest_history_sources,
    run_e2e_ingest,
    run_e2e_run,
)
from summary_multi_agent_web.schemas import (  # noqa: E402
    WebE2ERequest,
    WebHistorySource,
    WebRunStatus,
    WebSummarizeRequest,
)


class RecordingMemory:
    def __init__(self) -> None:
        self.add_calls: list[dict[str, object]] = []
        self.closed = False

    def add(self, messages, *, user_id, metadata, infer):
        self.add_calls.append(
            {
                "messages": messages,
                "user_id": user_id,
                "metadata": metadata,
                "infer": infer,
            }
        )
        return {"results": [{"event": "ADD", "id": f"memory-{len(self.add_calls)}"}]}


class E2ELlm:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        self.prompts.append(prompt)
        if "typed memory extraction" in prompt:
            return json.dumps(
                [
                    {
                        "memory_type": "task",
                        "content": "张三负责在 2026-08-21 12:00 前完成原始任务。",
                        "structured_fields": {
                            "owner": "张三",
                            "task": "完成原始任务",
                            "deadline": "2026-08-21 12:00",
                            "status": "pending",
                        },
                        "source": {
                            "source_time": "2026-08-20",
                            "evidence_span": "原始任务",
                        },
                        "confidence": "high",
                    }
                ],
                ensure_ascii=False,
            )
        if "cross-chunk" in prompt or "跨 chunk" in prompt:
            return json.dumps({"decisions": []}, ensure_ascii=False)
        return "历史基础摘要，仅用于展示。"


def make_summary_request() -> WebSummarizeRequest:
    return WebSummarizeRequest(
        current_material="当前材料",
        summary_request="请突出任务",
        user_id="user-a",
        project="project-a",
        source_time="2026-08-20",
        worker_mode="rule",
        qdrant_path="/tmp/web-qdrant",
        collection="web_collection",
    )


def make_e2e_request(*, run_type: str = "e2e_ingest", summary: WebSummarizeRequest | None = None) -> WebE2ERequest:
    return WebE2ERequest(
        run_type=run_type,
        history_sources=[
            WebHistorySource(label="history-source", text="RAW-CHUNK-ONE\n\nRAW-CHUNK-TWO")
        ],
        user_id="user-a",
        project="project-a",
        source_time="2026-08-20",
        source_type="classroom_transcript",
        qdrant_path="/tmp/web-qdrant",
        collection="web_collection",
        summary=summary,
    )


class WebE2EIngestTests(unittest.TestCase):
    def test_standalone_e2e_ingest_uses_stage_directory_and_top_level_status(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = asyncio.run(
                run_e2e_ingest(
                    make_e2e_request(),
                    Path(temp_dir),
                    memory_factory=RecordingMemory,
                    memory_closer=lambda value: None,
                    llm_call=E2ELlm(),
                )
            )

            self.assertEqual(WebRunStatus.COMPLETED, result.status)
            self.assertTrue((Path(temp_dir) / "e2e_ingest" / "typed_memories.json").exists())
            self.assertTrue((Path(temp_dir) / "e2e_ingest" / "e2e_preprocess_info.json").exists())
            self.assertTrue((Path(temp_dir) / "web_run_info.json").exists())
            self.assertTrue((Path(temp_dir) / "e2e_run_result.json").exists())

    def test_ingest_extracts_from_raw_chunks_and_writes_infer_false(self) -> None:
        memory = RecordingMemory()
        llm = E2ELlm()
        with tempfile.TemporaryDirectory() as temp_dir:
            previous_limit = os.environ.get("SMW_E2E_CHUNK_CHAR_LIMIT")
            os.environ["SMW_E2E_CHUNK_CHAR_LIMIT"] = "13"
            try:
                result = ingest_history_sources(
                    make_e2e_request(),
                    Path(temp_dir),
                    memory_factory=lambda: memory,
                    memory_closer=lambda value: setattr(value, "closed", True),
                    llm_call=llm,
                )
            finally:
                if previous_limit is None:
                    os.environ.pop("SMW_E2E_CHUNK_CHAR_LIMIT", None)
                else:
                    os.environ["SMW_E2E_CHUNK_CHAR_LIMIT"] = previous_limit

            self.assertGreater(result["chunk_count"], 1)
            self.assertTrue(memory.closed)
            self.assertTrue(memory.add_calls)
            self.assertTrue(all(call["infer"] is False for call in memory.add_calls))
            extraction_prompts = [
                prompt
                for prompt in llm.prompts
                if "typed memory extraction" in prompt
            ]
            self.assertEqual(result["chunk_count"], len(extraction_prompts))
            self.assertTrue(any("RAW-CHUNK-ONE" in prompt for prompt in extraction_prompts))
            self.assertTrue(any("RAW-CHUNK-TWO" in prompt for prompt in extraction_prompts))
            self.assertTrue((Path(temp_dir) / "raw_history_inputs").is_dir())
            self.assertTrue((Path(temp_dir) / "generated_history_summaries.md").exists())
            self.assertTrue((Path(temp_dir) / "e2e_preprocess_info.json").exists())
            self.assertTrue((Path(temp_dir) / "typed_memories.json").exists())
            self.assertTrue((Path(temp_dir) / "typed_memory_dedup_report.json").exists())
            self.assertTrue((Path(temp_dir) / "add_results.json").exists())

    def test_e2e_run_does_not_summarize_after_ingest_failure(self) -> None:
        summary_request = make_summary_request()
        request = make_e2e_request(run_type="e2e_run", summary=summary_request)
        pipeline_builder = Mock()

        def failing_llm(prompt: str, *, temperature: float, max_tokens: int) -> str:
            raise RuntimeError("ingest model unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(RuntimeError):
                asyncio.run(
                    run_e2e_run(
                        request,
                        Path(temp_dir),
                        ingest_memory_factory=RecordingMemory,
                        ingest_memory_closer=lambda value: None,
                        summary_memory_factory=RecordingMemory,
                        summary_memory_closer=lambda value: None,
                        llm_call=failing_llm,
                        pipeline_builder=pipeline_builder,
                    )
                )

            pipeline_builder.assert_not_called()
            result = json.loads(
                (Path(temp_dir) / "e2e_run_result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("failed", result["status"])
            self.assertIn("未启动 Summarize", result["warnings"][0])

    def test_e2e_run_keeps_store_identity_and_separates_stage_outputs(self) -> None:
        summary_request = make_summary_request()
        request = make_e2e_request(run_type="e2e_run", summary=summary_request)
        ingest_memory = RecordingMemory()
        summary_memory = RecordingMemory()
        received_requests: list[object] = []

        class FakeResult:
            status = RunStatus.COMPLETED_WITH_WARNINGS
            final_summary = "正式 candidate 0"
            recommended_candidate_index = 1
            warnings = ("非关键 warning",)

        class FakePipeline:
            async def run(self, pipeline_request):
                received_requests.append(pipeline_request)
                return FakeResult()

        with tempfile.TemporaryDirectory() as temp_dir:
            previous_limit = os.environ.get("SMW_E2E_CHUNK_CHAR_LIMIT")
            os.environ["SMW_E2E_CHUNK_CHAR_LIMIT"] = "1000"
            try:
                result = asyncio.run(
                    run_e2e_run(
                        request,
                        Path(temp_dir),
                        ingest_memory_factory=lambda: ingest_memory,
                        ingest_memory_closer=lambda value: None,
                        summary_memory_factory=lambda: summary_memory,
                        summary_memory_closer=lambda value: None,
                        llm_call=E2ELlm(),
                        pipeline_builder=Mock(return_value=FakePipeline()),
                    )
                )
            finally:
                if previous_limit is None:
                    os.environ.pop("SMW_E2E_CHUNK_CHAR_LIMIT", None)
                else:
                    os.environ["SMW_E2E_CHUNK_CHAR_LIMIT"] = previous_limit

            self.assertEqual(WebRunStatus.COMPLETED_WITH_WARNINGS, result.status)
            self.assertEqual("正式 candidate 0", result.final_summary)
            self.assertEqual(1, result.recommended_candidate_index)
            self.assertEqual(
                "structured",
                received_requests[0].model_dump()["personalization_preferences"]["summary_style"],
            )
            self.assertEqual("user-a", request.user_id)
            self.assertEqual("project-a", request.project)
            self.assertTrue((Path(temp_dir) / "e2e_ingest" / "typed_memories.json").exists())
            self.assertTrue((Path(temp_dir) / "multi_agent_summarize" / "web_run_info.json").exists())
            top_info = json.loads((Path(temp_dir) / "web_run_info.json").read_text())
            self.assertEqual("/tmp/web-qdrant", top_info["store"]["qdrant_path"])
            self.assertEqual("web_collection", top_info["store"]["collection"])


if __name__ == "__main__":
    unittest.main()
