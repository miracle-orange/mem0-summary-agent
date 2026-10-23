from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.adapters.model import ContextOverflowError
from summary_multi_agent.evaluation.lite import (
    LiteCandidateInput,
    LiteEvaluationComparisonRunner,
    LiteEvaluationRunner,
    LiteEvidenceSource,
    write_lite_comparison_artifacts,
)
from summary_multi_agent.evaluation.lite.model_io import LiteEvaluationModelIO
from summary_multi_agent.evaluation.lite.schemas import LiteDecomposeResponse
from summary_multi_agent.evaluation.schemas import (
    EvaluationCompareConfig,
    EvaluationCompareRequest,
    EvaluationSourceDocument,
    EvaluationSummaryDocument,
)
from summary_multi_agent.runtime import build_multi_agent_v2_pipeline
from summary_multi_agent.schemas import (
    EvaluationMode,
    EvaluationStatus,
    LiteEvaluationConfig,
    RunConfig,
    SummaryRequest,
)
from summary_multi_agent_web.artifact_reader import (
    create_safe_artifact_archive,
    read_artifact_snapshot,
)
from summary_multi_agent_web.evaluation_compare import (
    run_web_evaluation_compare,
)
from summary_multi_agent_web.schemas import (
    WebSummarizeRequest,
    build_run_config,
)
from summary_multi_agent_web_demo import build_evaluation_compare_request

from scripts.agent.tests.test_summary_multi_agent_pipeline import (
    EmptyMemoryBackend,
    FakeSummaryModel,
)


class LiteModelClient:
    def __init__(self, *, fail_verify: bool = False) -> None:
        self.prompts: list[str] = []
        self.fail_verify = fail_verify

    async def complete(
        self, prompt: str, *, temperature: float, max_tokens: int
    ) -> str:
        self.prompts.append(prompt)
        if "[任务上下文]" in prompt:
            return json.dumps(
                {"key_facts": ["Alice决定周五提交报告。", "项目预算为100万元。"]},
                ensure_ascii=False,
            )
        if "[待评测纪要]" in prompt:
            facts = (
                [
                    "Alice决定周五提交报告。",
                    "项目预算为100万元。",
                    "Bob已经取消测试。",
                ]
                if "纪要A" in prompt
                else ["Alice决定周五提交报告。"]
            )
            return json.dumps({"atomic_facts": facts}, ensure_ascii=False)
        if "[参考事实]" in prompt:
            has_budget = "项目预算为100万元" in prompt.split("[候选事实]", 1)[1]
            return json.dumps(
                {
                    "matches": [
                        {"fact_idx": 0, "align_unit_idx": 0},
                        {
                            "fact_idx": 1,
                            "align_unit_idx": 1 if has_budget else None,
                        },
                    ]
                },
                ensure_ascii=False,
            )
        if "[允许证据]" in prompt:
            if self.fail_verify:
                raise RuntimeError("verification unavailable")
            claim_block = prompt.split("[候选事实]", 1)[1].split("[/候选事实]", 1)[0]
            claim_count = sum(
                1 for line in claim_block.splitlines() if line[:1].isdigit()
            )
            return json.dumps(
                {
                    "verifications": [
                        {
                            "fact_idx": index,
                            "conclusion": (
                                "not supported" if index == 2 else "supported"
                            ),
                        }
                        for index in range(claim_count)
                    ]
                },
                ensure_ascii=False,
            )
        raise AssertionError("unexpected lite evaluation prompt")


def lite_sources() -> tuple[LiteEvidenceSource, tuple[LiteEvidenceSource, ...]]:
    return (
        LiteEvidenceSource(
            source_id="current",
            label="当前材料",
            source_time="2026-10-15",
            source_kind="current",
            text="Alice决定周五提交报告。项目预算为100万元。",
        ),
        (
            LiteEvidenceSource(
                source_id="history",
                label="上次会议",
                source_time="2026-10-08",
                source_kind="historical_raw",
                text="历史记录只用于支持性判断。",
            ),
        ),
    )


class LiteEvaluationCoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_lock_wait_does_not_consume_model_call_budget(
        self,
    ) -> None:
        class BlockingModel:
            def __init__(self) -> None:
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def complete(
                self, prompt: str, *, temperature: float, max_tokens: int
            ) -> str:
                self.started.set()
                await self.release.wait()
                return '{"atomic_facts":["事实"]}'

        model = BlockingModel()
        model_io = LiteEvaluationModelIO(
            model,
            LiteEvaluationConfig(
                repair_attempts=0,
                max_concurrent_model_calls=1,
            ),
        )
        first = asyncio.create_task(
            model_io.complete_json(
                stage="lite_candidate_decompose",
                prompt="first",
                sections={"summary": "first"},
                response_model=LiteDecomposeResponse,
                max_tokens=128,
            )
        )
        await model.started.wait()
        second = asyncio.create_task(
            model_io.complete_json(
                stage="lite_candidate_decompose",
                prompt="second",
                sections={"summary": "second"},
                response_model=LiteDecomposeResponse,
                max_tokens=128,
            )
        )
        await asyncio.sleep(0)
        second.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await second
        model.release.set()
        await first

        self.assertEqual(1, model_io.model_call_count)

    async def test_shared_reference_and_three_calls_per_candidate(self) -> None:
        model = LiteModelClient()
        current, history = lite_sources()
        result = await LiteEvaluationRunner(
            model_client=model,
            config=LiteEvaluationConfig(repair_attempts=0),
        ).run(
            current_source=current,
            historical_sources=history,
            summary_request="生成项目纪要",
            candidates=(
                LiteCandidateInput(summary_id="a", display_name="纪要 A", text="纪要A"),
                LiteCandidateInput(summary_id="b", display_name="纪要 B", text="纪要B"),
            ),
        )

        self.assertEqual("completed", result.status)
        self.assertEqual(7, result.execution_stats["model_call_count"])
        self.assertEqual(2, len(result.reference.facts))
        first, second = result.candidates
        self.assertEqual(1.0, first.metrics.current_key_fact_coverage.value)
        self.assertEqual(2 / 3, first.metrics.material_support_rate.value)
        self.assertEqual(2 / 3, first.metrics.key_fact_relevance_rate.value)
        self.assertEqual(0.5, second.metrics.current_key_fact_coverage.value)
        self.assertEqual(1.0, second.metrics.material_support_rate.value)
        self.assertEqual(1.0, second.metrics.key_fact_relevance_rate.value)
        reference_prompts = [p for p in model.prompts if "[任务上下文]" in p]
        self.assertEqual(1, len(reference_prompts))
        verify_prompts = [p for p in model.prompts if "[允许证据]" in p]
        self.assertEqual(2, len(verify_prompts))
        self.assertTrue(
            all(p.count("历史记录只用于支持性判断") == 1 for p in verify_prompts)
        )

    async def test_exact_reference_duplicates_across_overlapping_chunks_are_merged(
        self,
    ) -> None:
        current, history = lite_sources()
        current = current.model_copy(update={"text": current.text * 10})
        result = await LiteEvaluationRunner(
            model_client=LiteModelClient(),
            config=LiteEvaluationConfig(
                segment_max_chars=100,
                segment_overlap_chars=10,
                repair_attempts=0,
            ),
        ).run(
            current_source=current,
            historical_sources=history,
            summary_request="生成项目纪要",
            candidates=(
                LiteCandidateInput(summary_id="a", display_name="纪要 A", text="纪要A"),
            ),
        )

        self.assertGreater(result.reference.segment_count, 1)
        self.assertEqual(2, len(result.reference.facts))
        self.assertTrue(
            all(len(item.source_segment_ids) > 1 for item in result.reference.facts)
        )

    async def test_invalid_match_contract_is_repaired_once(self) -> None:
        class RepairingModel(LiteModelClient):
            def __init__(self) -> None:
                super().__init__()
                self.match_attempts = 0

            async def complete(
                self, prompt: str, *, temperature: float, max_tokens: int
            ) -> str:
                if "[参考事实]" in prompt:
                    self.prompts.append(prompt)
                    self.match_attempts += 1
                    if self.match_attempts == 1:
                        return json.dumps(
                            {"matches": [{"fact_idx": 0, "align_unit_idx": 99}]}
                        )
                return await super().complete(
                    prompt, temperature=temperature, max_tokens=max_tokens
                )

        model = RepairingModel()
        current, history = lite_sources()
        result = await LiteEvaluationRunner(
            model_client=model,
            config=LiteEvaluationConfig(repair_attempts=1),
        ).run(
            current_source=current,
            historical_sources=history,
            summary_request="生成项目纪要",
            candidates=(
                LiteCandidateInput(summary_id="a", display_name="纪要 A", text="纪要A"),
            ),
        )
        self.assertEqual("completed", result.status)
        self.assertEqual(1, result.execution_stats["repair_call_count"])
        self.assertEqual(5, result.execution_stats["model_call_count"])

    async def test_support_failure_preserves_coverage_but_marks_support_na(
        self,
    ) -> None:
        current, history = lite_sources()
        result = await LiteEvaluationRunner(
            model_client=LiteModelClient(fail_verify=True),
            config=LiteEvaluationConfig(repair_attempts=0),
        ).run(
            current_source=current,
            historical_sources=history,
            summary_request="生成项目纪要",
            candidates=(
                LiteCandidateInput(summary_id="a", display_name="纪要 A", text="纪要A"),
            ),
        )

        candidate = result.candidates[0]
        self.assertEqual("completed_with_warnings", candidate.status)
        self.assertEqual(1.0, candidate.metrics.current_key_fact_coverage.value)
        self.assertIsNone(candidate.metrics.material_support_rate.value)
        self.assertEqual(0, candidate.metrics.material_support_rate.denominator)
        self.assertTrue(
            any("support verification failed" in w for w in candidate.warnings)
        )

    async def test_context_overflow_is_not_repaired(self) -> None:
        class OverflowModel:
            def __init__(self) -> None:
                self.calls = 0

            async def complete(
                self, prompt: str, *, temperature: float, max_tokens: int
            ) -> str:
                self.calls += 1
                raise ContextOverflowError("context_overflow: service rejected request")

        model = OverflowModel()
        current, history = lite_sources()
        runner = LiteEvaluationRunner(
            model_client=model,
            config=LiteEvaluationConfig(repair_attempts=1),
        )
        with self.assertRaises(ContextOverflowError):
            await runner.run(
                current_source=current,
                historical_sources=history,
                summary_request="生成项目纪要",
                candidates=(
                    LiteCandidateInput(summary_id="a", display_name="A", text="纪要A"),
                ),
            )
        self.assertEqual(1, model.calls)
        self.assertEqual(0, runner.execution_stats["repair_call_count"])


class LiteEvaluationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_standalone_artifacts_and_safe_archive_exclude_private_trace(
        self,
    ) -> None:
        request = EvaluationCompareRequest(
            project="demo",
            summary_request="生成项目纪要",
            current_source=EvaluationSourceDocument(
                source_id="current",
                label="当前材料",
                source_time="2026-10-15",
                text="Alice决定周五提交报告。项目预算为100万元。",
            ),
            historical_sources=(
                EvaluationSourceDocument(
                    source_id="history",
                    label="上次会议",
                    source_time="2026-10-08",
                    text="历史记录只用于支持性判断。",
                ),
            ),
            summaries=(
                EvaluationSummaryDocument(
                    summary_id="a", display_name="A", text="纪要A"
                ),
                EvaluationSummaryDocument(
                    summary_id="b", display_name="B", text="纪要B"
                ),
            ),
            target_entities=("Alice",),
            evaluation_config=EvaluationCompareConfig(
                evaluation_mode="lite",
                comparison_interpretation_enabled=True,
                lite_evaluation=LiteEvaluationConfig(repair_attempts=0),
            ),
        )
        runner = LiteEvaluationComparisonRunner(model_client=LiteModelClient())
        result = await runner.run(request)
        self.assertEqual(7, result.execution_stats["model_call_count"])
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            write_lite_comparison_artifacts(
                output_dir, request, result, model_trace=runner.model_trace
            )
            (output_dir / "web_run_info.json").write_text(
                json.dumps(
                    {
                        "run_id": "web-lite",
                        "run_type": "evaluation_compare",
                        "status": "completed",
                    }
                ),
                encoding="utf-8",
            )
            snapshot = read_artifact_snapshot(output_dir)
            self.assertEqual("lite", snapshot.evaluation["mode"])
            self.assertEqual(
                "disabled", snapshot.evaluation["personalization"]["status"]
            )
            archive = create_safe_artifact_archive(output_dir)
            import zipfile

            with zipfile.ZipFile(archive) as bundle:
                names = set(bundle.namelist())
            self.assertIn("evaluation_lite_result.json", names)
            self.assertIn("evaluation_lite_report.md", names)
            self.assertNotIn("evaluation_lite_model_trace.json", names)
            self.assertNotIn("evaluation_compare_request.json", names)

    async def test_web_comparison_entry_uses_lite_core_not_strict_artifacts(
        self,
    ) -> None:
        async_model = LiteModelClient()

        def sync_model(prompt: str, *, temperature: float, max_tokens: int) -> str:
            import asyncio

            return asyncio.run(
                async_model.complete(
                    prompt, temperature=temperature, max_tokens=max_tokens
                )
            )

        request = EvaluationCompareRequest(
            project="demo",
            summary_request="生成项目纪要",
            current_source=EvaluationSourceDocument(
                source_id="current",
                label="当前材料",
                source_time="2026-10-15",
                text="Alice决定周五提交报告。项目预算为100万元。",
            ),
            summaries=(
                EvaluationSummaryDocument(
                    summary_id="a", display_name="A", text="纪要A"
                ),
                EvaluationSummaryDocument(
                    summary_id="b", display_name="B", text="纪要B"
                ),
            ),
            evaluation_config=EvaluationCompareConfig(
                evaluation_mode="lite",
                comparison_interpretation_enabled=True,
                lite_evaluation=LiteEvaluationConfig(repair_attempts=0),
            ),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            result = await run_web_evaluation_compare(
                request, output_dir, llm_call=sync_model
            )
            self.assertEqual("lite", result.evaluation_mode)
            self.assertEqual(7, result.execution_stats["model_call_count"])
            self.assertTrue((output_dir / "evaluation_lite_result.json").is_file())
            self.assertFalse(
                (output_dir / "evaluation_comparison_metrics.json").exists()
            )
            web_info = json.loads((output_dir / "web_run_info.json").read_text("utf-8"))
            self.assertEqual("lite", web_info["config"]["evaluation_mode"])

    async def test_online_pipeline_uses_lite_sidecar_and_preserves_strict_default(
        self,
    ) -> None:
        class PipelineModel(FakeSummaryModel):
            def __call__(
                self, prompt: str, *, temperature: float, max_tokens: int
            ) -> str:
                if "[任务上下文]" in prompt:
                    return json.dumps(
                        {"key_facts": ["负责人确认项目范围。"]}, ensure_ascii=False
                    )
                if "[待评测纪要]" in prompt:
                    return json.dumps(
                        {"atomic_facts": ["负责人确认项目范围。"]}, ensure_ascii=False
                    )
                if "[参考事实]" in prompt:
                    return json.dumps(
                        {"matches": [{"fact_idx": 0, "align_unit_idx": 0}]}
                    )
                if "[允许证据]" in prompt:
                    return json.dumps(
                        {"verifications": [{"fact_idx": 0, "conclusion": "supported"}]}
                    )
                return super().__call__(
                    prompt, temperature=temperature, max_tokens=max_tokens
                )

        self.assertEqual(EvaluationMode.STRICT, RunConfig().evaluation_mode)
        request = SummaryRequest(
            current_material="负责人确认项目范围。",
            source_time="2026-10-15",
            summary_request="生成项目纪要",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            pipeline = build_multi_agent_v2_pipeline(
                memory=EmptyMemoryBackend(),
                llm_call=PipelineModel(),
                user_id="user",
                project="project",
                output_dir=output_dir,
                config=RunConfig(
                    workflow_mode="multi_agent_v2",
                    planner_mode="rule",
                    evaluation_enabled=True,
                    evaluation_mode="lite",
                    lite_evaluation=LiteEvaluationConfig(repair_attempts=0),
                ),
            )
            result = await pipeline.run(request)

            self.assertEqual(EvaluationMode.LITE, result.evaluation_mode)
            self.assertEqual(EvaluationStatus.COMPLETED, result.evaluation_status)
            self.assertTrue((output_dir / "evaluation_lite_result.json").is_file())
            self.assertFalse((output_dir / "evaluation_metrics.json").exists())
            lite = json.loads(
                (output_dir / "evaluation_lite_result.json").read_text("utf-8")
            )
            self.assertEqual(4, lite["execution_stats"]["model_call_count"])

    def test_old_web_request_defaults_to_strict_and_lite_maps_independently(
        self,
    ) -> None:
        strict_request = WebSummarizeRequest(
            current_material="当前材料",
            summary_request="生成纪要",
            user_id="user",
            project="project",
        )
        self.assertEqual(EvaluationMode.STRICT, strict_request.evaluation_mode)
        self.assertEqual(
            EvaluationMode.STRICT, build_run_config(strict_request).evaluation_mode
        )
        lite_request = strict_request.model_copy(
            update={
                "evaluation_mode": EvaluationMode.LITE,
                "lite_evaluation": LiteEvaluationConfig(
                    segment_max_chars=1800,
                    max_model_calls=12,
                ),
            }
        )
        config = build_run_config(lite_request)
        self.assertEqual(EvaluationMode.LITE, config.evaluation_mode)
        self.assertEqual(1800, config.lite_evaluation.segment_max_chars)
        self.assertEqual(12, config.lite_evaluation.max_model_calls)
        self.assertEqual(6, config.evaluation_match_batch_size)

    def test_standalone_web_maps_lite_context_budget_independently(self) -> None:
        request = build_evaluation_compare_request(
            project="project",
            summary_request="生成会议纪要",
            target_entities="",
            current_values=(
                "current",
                "当前材料",
                "2026-10-15",
                "Alice确认周五提交报告。",
                None,
            ),
            history_values=(),
            summary_values=(
                ("a", "纪要 A", "Alice确认周五提交报告。", None),
                ("b", "纪要 B", "未记录。", None),
            ),
            evaluation_mode="lite",
            lite_config_values=(
                3000,
                100,
                4096,
                4096,
                4096,
                4096,
                1,
                180,
                1200,
                64,
                2,
                65536,
                1536,
            ),
        )

        self.assertEqual(EvaluationMode.LITE, request.evaluation_config.evaluation_mode)
        self.assertEqual(
            65536,
            request.evaluation_config.lite_evaluation.model_context_window_tokens,
        )
        self.assertEqual(
            1536,
            request.evaluation_config.lite_evaluation.context_safety_margin_tokens,
        )
        self.assertEqual(
            2,
            request.evaluation_config.lite_evaluation.max_concurrent_model_calls,
        )


if __name__ == "__main__":
    unittest.main()
