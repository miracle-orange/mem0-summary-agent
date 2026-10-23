from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.minutes_report import (  # noqa: E402
    build_report_bundle,
    persist_report_bundle,
    render_minutes_markdown,
)
from summary_multi_agent.minutes_report.models import (  # noqa: E402
    MinutesDocumentDraft,
)
from summary_multi_agent.minutes_report.sources import (  # noqa: E402
    build_history_aliases,
    normalize_document,
    resolve_sources,
)
from summary_multi_agent.schemas import (  # noqa: E402
    CurrentSegmentAnalysis,
    MinutesOutputMode,
    RunConfig,
    RunStatus,
    SummaryRequest,
    SynthesisContext,
)
from summary_multi_agent.runtime import build_multi_agent_v2_pipeline  # noqa: E402
from summary_multi_agent.synthesis import (  # noqa: E402
    SummarySynthesizer,
    build_rich_summary_synthesis_prompt,
)


CURRENT_MATERIAL = (
    "项目组确认接口联调继续推进。李明负责在周五前完成接口联调，"
    "前提是测试环境按时开放。历史方案本次调整为先灰度验证。"
)


def make_context(*, style: str = "structured") -> SynthesisContext:
    return SynthesisContext(
        request_id="rich-request",
        source_time="2026-10-02",
        summary_request="生成项目例会纪要，保留行动、风险和历史变化。",
        target_entities=("李明",),
        personalization_preferences={"summary_style": style},
        segment_analyses=(
            CurrentSegmentAnalysis(
                segment_id="segment-1",
                start_offset=0,
                end_offset=len(CURRENT_MATERIAL),
                items=(),
            ),
        ),
        merged_current_items=(),
        worker_results={},
        historical_records={
            "record-ref-1": {
                "record_ref_id": "record-ref-1",
                "memory_id": "memory-1",
                "memory_type": "decision",
                "content": "历史方案要求直接全量上线。",
                "metadata": {
                    "source_time": "2026-09-25",
                    "memory_status": "confirmed",
                },
                "score": 0.91,
            }
        },
        evidence_index={},
        template={
            "template_name": "project_sync_structured_v1",
            "meeting_scene": "project_sync",
            "format_kind": style,
            "sections": ["本期概览", "各工作项进展", "历史任务与状态变化", "后续安排"],
            "instructions": ["行动项保留负责人、时间要求和依赖。"],
        },
        auxiliary_context_info={},
        warnings=(),
    )


def valid_response(*, title: str = "项目例会纪要") -> str:
    return json.dumps(
        {
            "title": title,
            "overview": ["接口联调继续推进，历史上线策略调整为先灰度验证。"],
            "sections": [
                {
                    "heading": "接口联调与上线策略",
                    "entries": [
                        {
                            "kind": "fact",
                            "title": "当前进展",
                            "content": "项目组确认接口联调继续推进。",
                            "owner": "",
                            "time_requirement": "",
                            "dependency": "",
                            "recorded_state": "",
                            "current_state": "",
                            "relation": "",
                            "current_evidence_quotes": ["项目组确认接口联调继续推进。"],
                            "history_refs": [],
                        },
                        {
                            "kind": "action",
                            "title": "联调交付",
                            "content": "完成接口联调",
                            "owner": "李明",
                            "time_requirement": "周五前",
                            "dependency": "测试环境按时开放",
                            "recorded_state": "",
                            "current_state": "",
                            "relation": "",
                            "current_evidence_quotes": [
                                "李明负责在周五前完成接口联调，前提是测试环境按时开放。"
                            ],
                            "history_refs": [],
                        },
                        {
                            "kind": "history_update",
                            "title": "上线策略变化",
                            "content": "上线策略由直接全量上线调整为先灰度验证。",
                            "owner": "",
                            "time_requirement": "",
                            "dependency": "",
                            "recorded_state": "历史方案要求直接全量上线。",
                            "current_state": "先灰度验证",
                            "relation": "变更",
                            "current_evidence_quotes": ["历史方案本次调整为先灰度验证。"],
                            "history_refs": ["H001"],
                        },
                    ],
                }
            ],
        },
        ensure_ascii=False,
    )


class SequenceClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts: list[str] = []
        self.max_tokens: list[int] = []

    async def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        self.prompts.append(prompt)
        self.max_tokens.append(max_tokens)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


class EmptyMemory:
    def search(self, query, *, filters, top_k):
        return {"results": []}


class RichPipelineModel:
    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        if "Current Material Mapper" in prompt:
            match = re.search(
                r"<current_segment>\n(.*?)\n</current_segment>", prompt, re.DOTALL
            )
            quote = match.group(1)[:12]
            return json.dumps(
                {
                    "items": [
                        {
                            "category": "fact",
                            "statement": quote,
                            "evidence_quote": quote,
                            "entities": [],
                            "time_expression": None,
                        }
                    ]
                },
                ensure_ascii=False,
            )
        if "Rich Minutes Summarizer" in prompt:
            return json.dumps(
                {
                    "title": "项目纪要",
                    "overview": ["接口联调继续推进。"],
                    "sections": [
                        {
                            "heading": "当前进展",
                            "entries": [
                                {
                                    "kind": "fact",
                                    "content": "项目组确认接口联调继续推进。",
                                    "current_evidence_quotes": [
                                        "项目组确认接口联调继续推进。"
                                    ],
                                }
                            ],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        raise AssertionError("unexpected model prompt")


class RichMinutesTests(unittest.IsolatedAsyncioTestCase):
    async def test_rich_synthesis_builds_all_views_from_one_model_response(self) -> None:
        context = make_context()
        client = SequenceClient([valid_response()])
        synthesizer = SummarySynthesizer(
            model_client=client,
            config=RunConfig(
                minutes_output_mode=MinutesOutputMode.RICH,
                rich_summary_max_tokens=6000,
            ),
        )

        outcome = await synthesizer.generate(
            context,
            current_material=CURRENT_MATERIAL,
        )

        self.assertEqual(1, len(client.prompts))
        self.assertEqual([6000], client.max_tokens)
        self.assertEqual(1, client.prompts[0].count(CURRENT_MATERIAL))
        self.assertNotIn("memory-1", client.prompts[0])
        self.assertIn('"source_ref":"H001"', client.prompts[0])
        self.assertIsNotNone(outcome.report_bundle)
        self.assertEqual(outcome.summary, outcome.report_bundle.markdown.strip())
        self.assertEqual(1, outcome.summary.count("完成接口联调"))
        self.assertIn("2026-09-25", outcome.summary)
        history_source = outcome.report_bundle.sources.entries[2].sources[-1]
        self.assertEqual(("record-ref-1", "memory-1"), history_source.record_ids)
        self.assertEqual("history_record", history_source.status)
        self.assertEqual("rich_projected", outcome.model_input_report["mode"])

    async def test_pipeline_persists_formal_summary_and_public_report_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline = build_multi_agent_v2_pipeline(
                memory=EmptyMemory(),
                llm_call=RichPipelineModel(),
                user_id="user",
                project="project",
                output_dir=root,
                config=RunConfig(
                    planner_mode="rule",
                    worker_mode="rule",
                    minutes_output_mode="rich",
                    raw_segment_max_chars=3000,
                ),
            )
            result = await pipeline.run(
                SummaryRequest(
                    current_material=CURRENT_MATERIAL,
                    source_time="2026-10-02",
                    summary_request="生成项目纪要",
                )
            )

            self.assertIn(
                result.status, {RunStatus.COMPLETED, RunStatus.COMPLETED_WITH_WARNINGS}
            )
            for name in (
                "summary_multi_agent.md",
                "minutes_document.json",
                "minutes_sources.json",
                "minutes_report.html",
                "minutes_report_manifest.json",
                "minutes_report_bundle.zip",
            ):
                self.assertTrue((root / name).is_file(), name)
                self.assertIn(name, result.artifact_paths)
            self.assertEqual(
                result.final_summary,
                (root / "summary_multi_agent.md").read_text("utf-8").rstrip(),
            )

    async def test_invalid_json_is_repaired_once(self) -> None:
        client = SequenceClient(["not json", valid_response()])
        synthesizer = SummarySynthesizer(
            model_client=client,
            config=RunConfig(minutes_output_mode="rich"),
        )

        outcome = await synthesizer.generate(
            make_context(),
            current_material=CURRENT_MATERIAL,
        )

        self.assertIsNotNone(outcome.report_bundle)
        self.assertEqual(2, len(client.prompts))
        self.assertIn("上一次输出未通过", client.prompts[1])
        self.assertEqual(2, len(synthesizer.synthesis_trace[0]["attempts"]))

    async def test_timeout_does_not_start_schema_repair(self) -> None:
        client = SequenceClient([asyncio.TimeoutError("timeout")])
        synthesizer = SummarySynthesizer(
            model_client=client,
            config=RunConfig(minutes_output_mode="rich"),
        )

        with self.assertRaises(asyncio.TimeoutError):
            await synthesizer.generate(
                make_context(),
                current_material=CURRENT_MATERIAL,
            )
        self.assertEqual(1, len(client.prompts))


class RichMinutesRenderingTests(unittest.TestCase):
    def _bundle(self, *, style: str = "structured", title: str = "项目例会纪要"):
        draft = MinutesDocumentDraft.model_validate(json.loads(valid_response(title=title)))
        document = normalize_document(
            draft,
            source_time="2026-10-02",
            meeting_scene="project_sync",
            format_kind=style,
        )
        history_aliases = {
            "H001": {
                "record_ids": ["memory-1"],
                "content": "历史方案要求直接全量上线。",
                "source_date": "2026-09-25",
                "recorded_status": "confirmed",
            },
            "H002": {
                "record_ids": ["unused-memory"],
                "content": "未被正文引用的历史记录。",
                "source_date": "2026-09-20",
                "recorded_status": "pending",
            },
        }
        sources = resolve_sources(
            document,
            current_material=CURRENT_MATERIAL,
            history_aliases=history_aliases,
        )
        return build_report_bundle(document, sources)

    def test_natural_mode_has_no_formal_tables(self) -> None:
        bundle = self._bundle(style="natural")
        self.assertNotIn("| 事项 |", bundle.markdown)
        self.assertIn("负责人：李明", bundle.markdown)
        self.assertNotIn("<table>", bundle.html.split("mr-panels", 1)[0])

    def test_quotes_are_exact_ambiguous_or_missing_without_guessing(self) -> None:
        draft = MinutesDocumentDraft.model_validate(
            {
                "title": "来源测试",
                "overview": ["测试来源定位。"],
                "sections": [
                    {
                        "heading": "来源",
                        "entries": [
                            {
                                "kind": "fact",
                                "content": "来源定位",
                                "current_evidence_quotes": ["重复", "不存在"],
                            }
                        ],
                    }
                ],
            }
        )
        document = normalize_document(
            draft,
            source_time=None,
            meeting_scene="general",
            format_kind="structured",
        )
        sources = resolve_sources(
            document,
            current_material="重复内容，之后再次重复。",
            history_aliases={},
        )
        resolved = sources.entries[0].sources
        self.assertEqual("ambiguous", resolved[0].status)
        self.assertIsNone(resolved[0].start_offset)
        self.assertEqual("missing", resolved[1].status)

    def test_overlapping_duplicate_quotes_are_ambiguous(self) -> None:
        draft = MinutesDocumentDraft.model_validate(
            {
                "title": "重叠引文测试",
                "overview": ["测试重叠位置。"],
                "sections": [
                    {
                        "heading": "来源",
                        "entries": [
                            {
                                "kind": "fact",
                                "content": "引文有两个重叠位置。",
                                "current_evidence_quotes": ["甲甲"],
                            }
                        ],
                    }
                ],
            }
        )
        document = normalize_document(
            draft,
            source_time=None,
            meeting_scene="general",
            format_kind="structured",
        )
        sources = resolve_sources(
            document,
            current_material="甲甲甲",
            history_aliases={},
        )

        self.assertEqual("ambiguous", sources.entries[0].sources[0].status)

    def test_model_html_is_escaped_and_public_zip_omits_raw_inputs(self) -> None:
        bundle = self._bundle(title="<script>alert(1)</script>")
        self.assertEqual({"H001"}, set(bundle.sources.history_aliases))
        self.assertNotIn("<script>", bundle.html)
        self.assertIn("&lt;script&gt;", bundle.html)
        self.assertNotIn("<script>", bundle.markdown)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary_multi_agent.md").write_text(
                bundle.markdown, encoding="utf-8"
            )
            (root / "current_material.txt").write_text(
                CURRENT_MATERIAL, encoding="utf-8"
            )
            (root / "multi_agent_summary_prompt.txt").write_text(
                "private prompt", encoding="utf-8"
            )
            manifest = persist_report_bundle(root, bundle)
            self.assertEqual(
                "completed", manifest.artifact_status["minutes_report_bundle.zip"]
            )
            with zipfile.ZipFile(root / "minutes_report_bundle.zip") as archive:
                names = set(archive.namelist())
            self.assertIn("summary_multi_agent.md", names)
            self.assertIn("minutes_document.json", names)
            self.assertIn("minutes_sources.json", names)
            self.assertIn("minutes_report.html", names)
            self.assertNotIn("current_material.txt", names)
            self.assertNotIn("multi_agent_summary_prompt.txt", names)

    def test_derivative_write_failure_keeps_formal_summary_and_reports_status(self) -> None:
        bundle = self._bundle()
        original_write_text = Path.write_text

        def fail_sources(path: Path, data: str, *, encoding: str | None = None) -> int:
            if path.name == "minutes_sources.json":
                raise OSError("simulated source attachment failure")
            return original_write_text(path, data, encoding=encoding)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "summary_multi_agent.md").write_text(
                bundle.markdown, encoding="utf-8"
            )
            with patch.object(Path, "write_text", fail_sources):
                manifest = persist_report_bundle(root, bundle)

            self.assertTrue((root / "summary_multi_agent.md").is_file())
            self.assertEqual(
                "failed", manifest.artifact_status["minutes_sources.json"]
            )
            self.assertEqual(
                "completed", manifest.artifact_status["minutes_report_bundle.zip"]
            )
            self.assertTrue(
                any("minutes_sources.json write failed" in item for item in manifest.warnings)
            )
            with zipfile.ZipFile(root / "minutes_report_bundle.zip") as archive:
                self.assertNotIn("minutes_sources.json", archive.namelist())

    def test_identical_history_records_share_one_model_alias(self) -> None:
        records = (
            ("ref-1", {"memory_id": "m1", "content": "同一历史事实"}),
            ("ref-2", {"memory_id": "m2", "content": "同一历史事实"}),
        )

        visible, aliases = build_history_aliases(
            records,
            projector=lambda item: {"content": item["content"]},
        )

        self.assertEqual([{"source_ref": "H001", "content": "同一历史事实"}], visible)
        self.assertEqual(
            ["ref-1", "m1", "ref-2", "m2"],
            aliases["H001"]["record_ids"],
        )

    def test_rich_prompt_preserves_five_sections_and_aliases(self) -> None:
        prompt, report, aliases, sections = build_rich_summary_synthesis_prompt(
            make_context(),
            current_material=CURRENT_MATERIAL,
        )
        self.assertEqual(1, prompt.count(CURRENT_MATERIAL))
        self.assertEqual(
            {
                "summary_task",
                "current_material",
                "historical_context",
                "current_temporal_results",
                "output_template",
            },
            set(sections),
        )
        self.assertEqual(
            ["record-ref-1", "memory-1"], aliases["H001"]["record_ids"]
        )
        self.assertEqual("rich_projected", report["mode"])


if __name__ == "__main__":
    unittest.main()
