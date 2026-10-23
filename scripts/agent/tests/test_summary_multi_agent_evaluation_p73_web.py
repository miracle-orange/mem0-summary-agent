from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.evaluation.comparison import (  # noqa: E402
    EvaluationComparisonResult,
    EvaluationComparisonSummaryResult,
    aggregate_comparison,
    build_comparison_report,
)
from summary_multi_agent.evaluation.comparison_artifacts import (  # noqa: E402
    write_comparison_artifacts,
    write_comparison_failure_artifacts,
)
from summary_multi_agent.evaluation.metrics import (  # noqa: E402
    calculate_evaluation_metrics,
)
from summary_multi_agent.evaluation.schemas import (  # noqa: E402
    EvaluationArtifacts,
    EvaluationClaimScope,
    EvaluationCompareRequest,
    EvaluationCompareConfig,
    EvaluationEvidence,
    EvaluationFactMatch,
    EvaluationFactState,
    EvaluationMatchStatus,
    EvaluationSalience,
    EvaluationSourceDocument,
    EvaluationSourceFact,
    EvaluationSourceScope,
    EvaluationSourceSnapshot,
    EvaluationSummaryDocument,
    EvaluationSummaryFact,
)
from summary_multi_agent.evaluation.snapshot import (  # noqa: E402
    EvaluationSourceSnapshotBundle,
)
from summary_multi_agent_web.artifact_reader import (  # noqa: E402
    create_safe_artifact_archive,
    read_artifact_snapshot,
)
from summary_multi_agent_web_demo import (  # noqa: E402
    build_evaluation_compare_request,
)


def _fixture():
    current_text = "项目范围已确认。"
    summary_text = "项目范围已确认。"
    source = EvaluationSourceFact(
        fact_id="c1",
        source_scope=EvaluationSourceScope.CURRENT,
        fact="项目范围已确认",
        state=EvaluationFactState.CONFIRMED,
        salience=EvaluationSalience.MUST_INCLUDE,
        evidence=(
            EvaluationEvidence(
                evidence_text=current_text,
                source_id="current",
                start_offset=0,
                end_offset=len(current_text),
                source_time="2026-09-03",
            ),
        ),
    )
    claim = EvaluationSummaryFact(
        summary_fact_id="s1",
        fact="项目范围已确认",
        evidence_quote=summary_text,
        start_offset=0,
        end_offset=len(summary_text),
        claimed_scope=EvaluationClaimScope.CURRENT,
    )
    match = EvaluationFactMatch(
        summary_fact_id="s1",
        match_status=EvaluationMatchStatus.SUPPORTED_FULL,
        matched_source_fact_ids=("c1",),
        primary_source_fact_id="c1",
        primary_source_scope=EvaluationSourceScope.CURRENT,
        reason_code="same_fact",
    )
    snapshot = EvaluationSourceSnapshot(
        snapshot_id="snapshot-test",
        current_source_id="current",
        current_source_time="2026-09-03",
        current_material_sha256=(
            "d79116bf56c7c5fffbf1719a7995d8e0b5d52e9666b18ef15a7c8a359f9d5b61"
        ),
        summary_request="生成项目纪要",
        source_facts=(source,),
    )
    bundle = EvaluationSourceSnapshotBundle(
        snapshot=snapshot,
        current_material=current_text,
        visible_history_text="",
    )
    metrics = calculate_evaluation_metrics(
        summary=summary_text,
        current_material=current_text,
        visible_history_text="",
        source_facts=(source,),
        summary_facts=(claim,),
        matches=(match,),
    )
    artifacts = EvaluationArtifacts(
        source_facts=(source,),
        summary_facts=(claim,),
        matches=(match,),
        metrics=metrics,
        report_markdown="# 评测\n",
    )
    request = EvaluationCompareRequest(
        project="demo",
        summary_request="生成项目纪要",
        current_source=EvaluationSourceDocument(
            source_id="current",
            label="当前会议",
            source_time="2026-09-03",
            text=current_text,
        ),
        summaries=(
            EvaluationSummaryDocument(
                summary_id="a", display_name="A", text=summary_text
            ),
            EvaluationSummaryDocument(
                summary_id="b", display_name="B", text=summary_text
            ),
        ),
    )
    results = tuple(
        EvaluationComparisonSummaryResult(
            summary_id=item.summary_id,
            display_name=item.display_name,
            status="completed",
            artifacts=artifacts,
            model_trace=({"prompt": "secret prompt", "raw_output": "secret"},),
        )
        for item in request.summaries
    )
    comparison_metrics = aggregate_comparison(
        request, bundle, results, total_model_calls=5
    )
    result = EvaluationComparisonResult(
        status="completed",
        snapshot_bundle=bundle,
        summaries=results,
        comparison_metrics=comparison_metrics,
        report_markdown=build_comparison_report(request, comparison_metrics),
    )
    return request, result


class P73WebArtifactTests(unittest.TestCase):
    def test_web_compare_maps_explicit_stable_evaluation_config(self) -> None:
        request = build_evaluation_compare_request(
            project="demo",
            summary_request="生成课堂纪要",
            target_entities="周明远",
            current_values=(
                "current",
                "当前材料",
                "2026-09-04",
                "当前事实。",
                None,
            ),
            history_values=(),
            summary_values=(
                ("a", "纪要 A", "当前事实。", None),
                ("b", "纪要 B", "当前事实。", None),
            ),
            evaluation_config_values=(
                500,
                3000,
                2800,
                2600,
                2200,
                3,
                True,
                500,
                1,
            ),
        )
        self.assertEqual(("周明远",), request.target_entities)
        self.assertEqual(500, request.evaluation_config.segment_max_chars)
        self.assertEqual(2800, request.evaluation_config.summary_fact_max_tokens)
        self.assertEqual(3, request.evaluation_config.repair_attempts)
        self.assertTrue(request.evaluation_config.adaptive_split_enabled)
        self.assertTrue(
            request.evaluation_config.comparison_interpretation_enabled
        )

    def test_snapshot_failure_artifacts_include_stage_and_source_trace(self) -> None:
        request, _ = _fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = write_comparison_failure_artifacts(
                root,
                request,
                stage="building_source_snapshot",
                error="JSONDecodeError: truncated",
                source_model_trace=(
                    {
                        "stage": "evaluation.current_source.seg_1",
                        "attempts": [
                            {
                                "attempt": 1,
                                "failure_kind": "likely_output_truncation",
                            }
                        ],
                    },
                ),
                source_execution_stats={"model_call_count": 1},
            )
            payload = json.loads(
                (root / "evaluation_compare_run_result.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual("building_source_snapshot", result["stage"])
            self.assertEqual("building_source_snapshot", payload["stage"])
            self.assertTrue((root / "evaluation_source_model_trace.json").exists())

    def test_safe_archive_excludes_raw_inputs_and_model_traces(self) -> None:
        request, result = _fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_comparison_artifacts(root, request, result)
            archive = create_safe_artifact_archive(root)
            with zipfile.ZipFile(archive) as handle:
                names = set(handle.namelist())
            self.assertIn("evaluation_comparison_metrics.json", names)
            self.assertIn("evaluation_comparison_report.md", names)
            self.assertIn("summaries/a/fact_matches.json", names)
            self.assertNotIn("evaluation_compare_request.json", names)
            self.assertNotIn("evaluation_source_snapshot.json", names)
            self.assertNotIn("summaries/a/summary_facts.json", names)
            self.assertNotIn("summaries/a/evaluation_model_trace.json", names)

    def test_comparison_inspector_uses_redacted_aggregate_artifacts(self) -> None:
        request, result = _fixture()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_comparison_artifacts(root, request, result)
            (root / "web_run_info.json").write_text(
                '{"run_type":"evaluation_compare","run_id":"web-1",'
                '"status":"completed","stage":"completed"}',
                encoding="utf-8",
            )
            snapshot = read_artifact_snapshot(root)
            self.assertTrue(snapshot.evaluation["comparison"])
            self.assertEqual("snapshot-test", snapshot.evaluation["metrics"]["snapshot_id"])
            self.assertNotIn("项目范围已确认。", str(snapshot.evaluation))

    def test_web_source_declares_four_separate_tabs(self) -> None:
        source = (AGENT_ROOT / "summary_multi_agent_web_demo.py").read_text(
            encoding="utf-8"
        )
        positions = [
            source.index(f'with gr.Tab("{label}")')
            for label in (
                "资料导入",
                "生成纪要",
                "纪要横评",
                "结果中心",
            )
        ]
        self.assertEqual(sorted(positions), positions)
        self.assertNotIn('gr.Tab("E2E Run")', source)
        self.assertIn("关注的人或团队（可选", source)
        self.assertIn("compare_segment_chars", source)


if __name__ == "__main__":
    unittest.main()
