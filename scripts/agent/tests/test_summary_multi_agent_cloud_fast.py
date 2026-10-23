from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_memory_llm import _external_request_kwargs, _external_settings, _external_complete
from summary_memory_typed_memory import build_typed_memory_extraction_prompt, parse_typed_memory_json
from summary_multi_agent.runtime import build_multi_agent_v2_pipeline
from summary_multi_agent.adapters.model import AsyncOpenAICompatibleModelClient
from summary_multi_agent.schemas import RunConfig, RunStatus, SummaryRequest
from summary_multi_agent_web.e2e_ingest import ingest_history_sources
from summary_multi_agent_web.schemas import (
    WebHistorySource, WebIngestRequest, WebSummarizeRequest, build_run_config,
)
from summary_multi_agent_web_demo import build_evaluation_compare_request
from test_summary_multi_agent_pipeline import EmptyMemoryBackend, FakeSummaryModel
from test_summary_multi_agent_web_e2e import E2ELlm, RecordingMemory


class CloudConfigurationTests(unittest.TestCase):
    def test_external_settings_validate_without_exposing_key(self) -> None:
        with patch.dict("os.environ", {
            "OPENAI_BASE_URL": "https://api.deepseek.com",
            "OPENAI_API_KEY": "test-secret",
            "OPENAI_MODEL": "deepseek-chat",
        }):
            self.assertEqual(
                ("https://api.deepseek.com", "test-secret", "deepseek-chat"),
                _external_settings(),
            )
            request = _external_request_kwargs(
                "deepseek-chat", "{}", 0, 100, True,
                base_url="https://api.deepseek.com",
            )
            self.assertEqual({"type": "json_object"}, request["response_format"])
            self.assertEqual({"thinking": {"type": "disabled"}}, request["extra_body"])
        with patch.dict("os.environ", {
            "OPENAI_BASE_URL": "https://test-secret@example.com",
            "OPENAI_API_KEY": "test-secret",
            "OPENAI_MODEL": "model",
        }):
            with self.assertRaisesRegex(RuntimeError, "must not contain credentials") as error:
                _external_settings()
            self.assertNotIn("test-secret", str(error.exception))

    def test_old_request_remains_legacy_and_cloud_defaults_are_explicit(self) -> None:
        fields = dict(current_material="本次讨论进展。", summary_request="生成纪要",
                      user_id="u", project="p")
        old = WebSummarizeRequest(**fields)
        cloud = WebSummarizeRequest(execution_profile="cloud_fast", **fields)
        self.assertEqual("legacy", old.execution_profile)
        self.assertEqual("classic", old.minutes_output_mode)
        self.assertEqual("rich", cloud.minutes_output_mode)
        self.assertEqual("lite", cloud.evaluation_mode)
        self.assertEqual("auto", cloud.lite_evaluation.reference_chunking)
        self.assertEqual(8192, cloud.rich_summary_max_tokens)
        self.assertEqual(4, build_run_config(cloud).max_concurrent_model_calls)

    def test_cloud_comparison_uses_auto_reference_chunking(self) -> None:
        request = build_evaluation_compare_request(
            project="p", summary_request="比较纪要", target_entities="",
            current_values=("current", "当前材料", "2026-07-28", "本次内容", None),
            history_values=(),
            summary_values=(("s1", "甲", "纪要甲", None), ("s2", "乙", "纪要乙", None)),
            evaluation_mode="lite", execution_profile="cloud_fast",
        )
        self.assertEqual("auto", request.evaluation_config.lite_evaluation.reference_chunking)

    def test_cloud_ingest_json_object_contract_keeps_old_array_parser(self) -> None:
        cloud = build_typed_memory_extraction_prompt("历史材料", json_object_output=True)
        legacy = build_typed_memory_extraction_prompt("历史材料")
        self.assertIn('最外层必须是含 typed_memories 数组的对象', cloud)
        self.assertIn('{"typed_memories": [', cloud)
        self.assertIn('注意最外层必须是数组', legacy)
        self.assertEqual([], parse_typed_memory_json('{"typed_memories": []}'))
        self.assertEqual([], parse_typed_memory_json('[]'))

    def test_sync_compatible_response_is_audited_without_secrets(self) -> None:
        calls = []
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                finish_reason="stop", message=types.SimpleNamespace(content='{"ok":true}'),
            )],
            usage=types.SimpleNamespace(model_dump=lambda **_: {"total_tokens": 12}),
        )
        client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kwargs: (calls.append(kwargs), response)[1]),
        ))
        fake_openai = types.SimpleNamespace(APIStatusError=type("APIStatusError", (Exception,), {}))
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {"openai": fake_openai}), \
             patch.dict("os.environ", {"OPENAI_BASE_URL": "https://api.deepseek.com",
                                    "OPENAI_API_KEY": "test-secret", "OPENAI_MODEL": "deepseek-flash",
                                    "SMW_OPENAI_STREAM": "false"}), \
             patch("summary_memory_llm._sync_external_client", return_value=client):
            trace = Path(temporary) / "calls.jsonl"
            self.assertEqual('{"ok":true}', _external_complete("{}", 0, 100, json_output=True,
                                                             audit_path=trace))
            self.assertEqual({"type": "json_object"}, calls[0]["response_format"])
            audit = trace.read_text("utf-8")
            self.assertIn('"total_tokens": 12', audit)
            self.assertNotIn("test-secret", audit)


class CloudAsyncTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_json_transport_records_usage_and_rejects_truncation(self) -> None:
        calls = []
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(
                finish_reason="stop", message=types.SimpleNamespace(content='{"ok":true}'),
            )], usage=None,
        )

        async def create(**kwargs):
            calls.append(kwargs)
            return response

        client = types.SimpleNamespace(chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=create),
        ))
        adapter = AsyncOpenAICompatibleModelClient.__new__(AsyncOpenAICompatibleModelClient)
        adapter._client = client
        adapter._url = "https://api.deepseek.com"
        adapter._model = "deepseek-flash"
        adapter._trace_lock = asyncio.Lock()
        fake_openai = types.SimpleNamespace(APIStatusError=type("APIStatusError", (Exception,), {}))
        with tempfile.TemporaryDirectory() as temporary, patch.dict(sys.modules, {"openai": fake_openai}), \
             patch.dict("os.environ", {"SMW_OPENAI_STREAM": "false"}):
            adapter._trace_path = Path(temporary) / "model_calls.jsonl"
            self.assertEqual('{"ok":true}', await adapter.complete_json("{}", temperature=0,
                                                                        max_tokens=100))
            self.assertEqual({"type": "json_object"}, calls[0]["response_format"])
            response.choices[0].finish_reason = "length"
            with self.assertRaisesRegex(RuntimeError, "External model request failed"):
                await adapter.complete("test-secret", temperature=0, max_tokens=1)
            audit = adapter._trace_path.read_text("utf-8")
            self.assertEqual(2, len(audit.splitlines()))
            self.assertNotIn("test-secret", audit)


class CloudPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_default_cloud_generation_skips_mapper_but_review_restores_it(self) -> None:
        request = SummaryRequest(
            current_material="负责人确认项目范围。", source_time="2026-08-10",
            summary_request="生成结构化项目纪要",
            personalization_preferences={"summary_style": "structured"},
        )
        for review_enabled in (False, True):
            with self.subTest(review_enabled=review_enabled):
                model = FakeSummaryModel()
                with tempfile.TemporaryDirectory() as temporary:
                    output_dir = Path(temporary)
                    pipeline = build_multi_agent_v2_pipeline(
                        memory=EmptyMemoryBackend(), llm_call=model,
                        user_id="u", project="p", output_dir=output_dir,
                        config=RunConfig(
                            execution_profile="cloud_fast", workflow_mode="multi_agent_v2",
                            planner_mode="rule", worker_mode="rule",
                            review_enabled=review_enabled, max_review_rounds=0,
                        ),
                    )
                    result = await pipeline.run(request)
                    self.assertIn(result.status, {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_WARNINGS})
                    mapper_calls = [call for call in model.calls
                                    if "Current Material Mapper" in call["prompt"]]
                    self.assertEqual(bool(review_enabled), bool(mapper_calls))
                    ledger = json.loads((output_dir / "coverage_ledger.json").read_text("utf-8"))
                    self.assertEqual(
                        review_enabled,
                        "current_fact_map" in ledger["segments"][0]["required_stages"],
                    )
                    prompt = (output_dir / "multi_agent_summary_prompt.txt").read_text("utf-8")
                    self.assertEqual(1, prompt.count(request.current_material))


class CloudIngestTests(unittest.TestCase):
    def test_cloud_ingest_skips_foundation_summary_and_keeps_source_dates(self) -> None:
        llm = E2ELlm()
        memory = RecordingMemory()
        request = WebIngestRequest(
            execution_profile="cloud_fast", user_id="u", project="p",
            source_time="2026-07-28",
            history_sources=[
                WebHistorySource(label="a", text="原始任务在历史会议中提出。", source_time="2026-07-21"),
                WebHistorySource(label="b", text="第二次会议跟进原始任务。", source_time="2026-07-22"),
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = ingest_history_sources(
                request, Path(temporary),
                memory_factory=lambda: memory, memory_closer=lambda _: None,
                llm_call=llm,
            )
            self.assertEqual(2, result["chunk_count"])
            self.assertEqual(2, len([p for p in llm.prompts if "typed memory extraction" in p]))
            self.assertFalse(any("基础摘要" in p for p in llm.prompts))
            self.assertFalse((Path(temporary) / "generated_history_summaries.md").exists())
            dates = {call["metadata"].get("source_time") for call in memory.add_calls}
            self.assertEqual({"2026-07-21", "2026-07-22"}, dates)
