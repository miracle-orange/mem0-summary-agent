from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.adapters.model import SyncCallableModelClient  # noqa: E402
from summary_multi_agent.evaluation.comparison_interpretation import (  # noqa: E402
    ComparisonInterpretationRunner,
)
from summary_multi_agent.evaluation.schemas import (  # noqa: E402
    EvaluationCompareConfig,
    EvaluationCompareRequest,
    EvaluationSourceDocument,
    EvaluationSummaryDocument,
)


class InterpretationModel:
    """返回可追溯定性说明，并识别一个机械伪差异。"""

    def __init__(
        self,
        *,
        invalid_unit_id: str | None = None,
        invalid_qualitative_note: bool = False,
    ) -> None:
        self.prompts: list[str] = []
        self.invalid_unit_id = invalid_unit_id
        self.invalid_qualitative_note = invalid_qualitative_note

    def __call__(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        del temperature, max_tokens
        self.prompts.append(prompt)
        if "<source_documents>" in prompt:
            source_documents = json.loads(
                re.search(
                    r"<source_documents>\n(.*?)\n</source_documents>",
                    prompt,
                    re.DOTALL,
                ).group(1)
            )
            summaries = json.loads(
                re.search(
                    r"<summaries>\n(.*?)\n</summaries>",
                    prompt,
                    re.DOTALL,
                ).group(1)
            )

            def document_span_id(
                documents: list[dict[str, object]],
                document_id_key: str,
                document_id: str,
                text: str,
            ) -> str:
                document = next(
                    item
                    for item in documents
                    if str(item[document_id_key]) == document_id
                )
                return next(
                    str(item["span_id"])
                    for item in document["spans"]
                    if text in str(item["text"])
                )

            project_source_span = document_span_id(
                source_documents,
                "source_id",
                "current",
                "项目范围已确认",
            )
            if self.invalid_qualitative_note:
                project_source_span = "D9999"
            return json.dumps(
                {
                    "overview": "B 增加预算信息，但省略项目范围。",
                    "summaries": [
                        {
                            "summary_id": "a",
                            "strengths": [
                                {
                                    "description": "保留项目范围",
                                    "source_id": "current",
                                    "source_span_ids": [project_source_span],
                                    "summary_span_ids": [
                                        document_span_id(
                                            summaries,
                                            "summary_id",
                                            "a",
                                            "项目范围已确认",
                                        )
                                    ],
                                }
                            ],
                            "limitations": [],
                            "relative_to_baseline": "作为基线。",
                        },
                        {
                            "summary_id": "b",
                            "strengths": [
                                {
                                    "description": "补充预算状态",
                                    "source_id": "current",
                                    "source_span_ids": [
                                        document_span_id(
                                            source_documents,
                                            "source_id",
                                            "current",
                                            "预算已冻结",
                                        )
                                    ],
                                    "summary_span_ids": [
                                        document_span_id(
                                            summaries,
                                            "summary_id",
                                            "b",
                                            "预算已冻结",
                                        )
                                    ],
                                }
                            ],
                            "limitations": [
                                {
                                    "description": "省略项目范围",
                                    "source_id": "current",
                                    "source_span_ids": [
                                        document_span_id(
                                            source_documents,
                                            "source_id",
                                            "current",
                                            "项目范围已确认",
                                        )
                                    ],
                                    "summary_span_ids": [],
                                }
                            ],
                            "relative_to_baseline": "补充预算，减少范围信息。",
                        },
                    ],
                },
                ensure_ascii=False,
            )
        if "<fact_units>" in prompt:
            units = json.loads(
                re.search(
                    r"<fact_units>\n(.*?)\n</fact_units>", prompt, re.DOTALL
                ).group(1)
            )
            baseline_spans = json.loads(
                re.search(
                    r"<baseline_summary_spans[^>]*>\n(.*?)\n</baseline_summary_spans>",
                    prompt,
                    re.DOTALL,
                ).group(1)
            )
            candidate_spans = json.loads(
                re.search(
                    r"<candidate_summary_spans[^>]*>\n(.*?)\n</candidate_summary_spans>",
                    prompt,
                    re.DOTALL,
                ).group(1)
            )

            def span_id(spans: list[dict[str, object]], text: str) -> str:
                return next(
                    str(item["span_id"])
                    for item in spans
                    if text in str(item["text"])
                )

            values = {
                "risk": {
                    "baseline_presence": "full",
                    "candidate_presence": "full",
                    "baseline_span_ids": [span_id(baseline_spans, "风险登记工作已经办妥")],
                    "candidate_span_ids": [span_id(candidate_spans, "风险登记已完成")],
                    "reason": "两份纪要语义相同，仅表述不同",
                },
                "budget": {
                    "baseline_presence": "absent",
                    "candidate_presence": "full",
                    "baseline_span_ids": [],
                    "candidate_span_ids": [span_id(candidate_spans, "预算已冻结")],
                    "reason": "仅候选纪要明确表达",
                },
                "scope": {
                    "baseline_presence": "full",
                    "candidate_presence": "absent",
                    "baseline_span_ids": [span_id(baseline_spans, "项目范围已确认")],
                    "candidate_span_ids": [],
                    "reason": "候选纪要没有表达",
                },
            }
            if self.invalid_unit_id:
                values[self.invalid_unit_id]["baseline_span_ids"] = ["B9999"]
            default_candidate_span = str(candidate_spans[0]["span_id"])
            return json.dumps(
                {
                    "items": [
                        {
                            "summary_id": "b",
                            "unit_id": unit["unit_id"],
                            **values.get(
                                unit["unit_id"],
                                {
                                    "baseline_presence": "absent",
                                    "candidate_presence": "full",
                                    "baseline_span_ids": [],
                                    "candidate_span_ids": [default_candidate_span],
                                    "reason": "仅候选纪要表达",
                                },
                            ),
                        }
                        for unit in units
                    ]
                },
                ensure_ascii=False,
            )
        raise AssertionError("unexpected prompt")


def _request() -> EvaluationCompareRequest:
    return EvaluationCompareRequest(
        project="demo",
        summary_request="生成项目纪要",
        current_source=EvaluationSourceDocument(
            source_id="current",
            label="当前会议",
            source_time="2026-09-09",
            text="项目范围已确认。风险登记已完成。预算已冻结。",
        ),
        historical_sources=(
            EvaluationSourceDocument(
                source_id="history",
                label="历史会议",
                source_time="2026-09-02",
                text="历史范围已经登记。",
            ),
        ),
        summaries=(
            EvaluationSummaryDocument(
                summary_id="a",
                display_name="纪要 A",
                text="**项目范围已确认**；风险登记工作已经办妥。",
            ),
            EvaluationSummaryDocument(
                summary_id="b",
                display_name="纪要 B",
                text="风险登记已完成；预算已冻结。",
            ),
        ),
        evaluation_config=EvaluationCompareConfig(
            comparison_interpretation_enabled=True,
            comparison_interpretation_max_model_calls=8,
            repair_attempts=0,
        ),
    )


def _metrics() -> dict[str, object]:
    differences = {
        "b": {
            "status": "completed",
            "baseline_summary_id": "a",
            "added_covered_count": 2,
            "reduced_covered_count": 1,
            "added_covered_units": [
                {"unit_id": "risk", "fact": "风险登记已完成", "source_scopes": ["current"]},
                {"unit_id": "budget", "fact": "预算已冻结", "source_scopes": ["current"]},
            ],
            "reduced_covered_units": [
                {"unit_id": "scope", "fact": "项目范围已确认", "source_scopes": ["current"]}
            ],
        }
    }
    return {
        "baseline_summary_id": "a",
        "scope_content_differences": {
            "all_sources": differences,
            "current_only": differences,
        },
    }


def _many_metrics(count: int = 25) -> dict[str, object]:
    differences = {
        "b": {
            "status": "completed",
            "baseline_summary_id": "a",
            "added_covered_count": count,
            "reduced_covered_count": 0,
            "added_covered_units": [
                {
                    "unit_id": f"unit_{index:02d}",
                    "fact": f"候选事实 {index}",
                    "source_scopes": ["current"],
                }
                for index in range(count)
            ],
            "reduced_covered_units": [],
        }
    }
    return {
        "baseline_summary_id": "a",
        "scope_content_differences": {
            "all_sources": differences,
            "current_only": differences,
        },
    }


class ComparisonInterpretationTests(unittest.IsolatedAsyncioTestCase):
    async def test_verifies_mechanical_differences_without_changing_metrics(self) -> None:
        model = InterpretationModel()
        metrics = _metrics()
        original = json.dumps(metrics, ensure_ascii=False, sort_keys=True)
        runner = ComparisonInterpretationRunner(
            model_client=SyncCallableModelClient(model),
            config=_request().evaluation_config,
        )

        result = await runner.run(request=_request(), metrics=metrics)

        self.assertEqual("completed", result["status"])
        for scope in ("all_sources", "current_only"):
            difference = result["verified_content_differences"][scope]["b"]
            self.assertEqual(1, difference["confirmed_added_count"])
            self.assertEqual(1, difference["confirmed_reduced_count"])
            self.assertEqual(1, difference["not_confirmed_count"])
            risk = next(item for item in difference["items"] if item["unit_id"] == "risk")
            self.assertEqual("not_confirmed", risk["verification_result"])
            self.assertIn(risk["baseline_quote"], _request().summaries[0].text)
            self.assertIn(risk["candidate_quote"], _request().summaries[1].text)
            self.assertEqual("grounded", risk["evidence_validation"]["status"])
            scope_item = next(
                item for item in difference["items"] if item["unit_id"] == "scope"
            )
            self.assertEqual("**项目范围已确认**；", scope_item["baseline_quote"])
        self.assertEqual(original, json.dumps(metrics, ensure_ascii=False, sort_keys=True))
        presence_prompts = [prompt for prompt in model.prompts if "<fact_units>" in prompt]
        self.assertTrue(presence_prompts)
        self.assertTrue(all("preliminary_kind" not in prompt for prompt in presence_prompts))
        self.assertTrue(all("added_covered" not in prompt for prompt in presence_prompts))
        self.assertTrue(all("baseline_quote" not in prompt for prompt in presence_prompts))

    async def test_qualitative_notes_use_grounded_spans(self) -> None:
        model = InterpretationModel()
        runner = ComparisonInterpretationRunner(
            model_client=SyncCallableModelClient(model),
            config=_request().evaluation_config,
        )

        result = await runner.run(request=_request(), metrics=_metrics())

        for scope in ("all_sources", "current_only"):
            comparison = result["qualitative_by_scope"][scope]
            self.assertEqual("completed", comparison["status"])
            summary_a = next(
                item for item in comparison["summaries"] if item["summary_id"] == "a"
            )
            note = summary_a["strengths"][0]
            self.assertEqual("项目范围已确认。", note["source_quote"])
            self.assertEqual("**项目范围已确认**；", note["summary_quote"])
            self.assertEqual("grounded", note["evidence_validation"]["status"])

    async def test_invalid_qualitative_span_only_drops_its_note(self) -> None:
        model = InterpretationModel(invalid_qualitative_note=True)
        runner = ComparisonInterpretationRunner(
            model_client=SyncCallableModelClient(model),
            config=_request().evaluation_config,
        )

        result = await runner.run(request=_request(), metrics=_metrics())

        self.assertEqual("completed_with_warnings", result["status"])
        for scope in ("all_sources", "current_only"):
            comparison = result["qualitative_by_scope"][scope]
            self.assertEqual("completed_with_warnings", comparison["status"])
            summary_a = next(
                item for item in comparison["summaries"] if item["summary_id"] == "a"
            )
            self.assertEqual([], summary_a["strengths"])
            self.assertEqual(1, len(summary_a["unverified_notes"]))
        self.assertTrue(
            any(
                "current_only qualitative a strengths[1] source evidence "
                "unverified: unknown_span_id:D9999" in warning
                for warning in result["warnings"]
            )
        )

    async def test_difference_verification_batches_all_selected_units(self) -> None:
        model = InterpretationModel()
        config = _request().evaluation_config.model_copy(
            update={
                "comparison_interpretation_max_model_calls": 16,
                "comparison_difference_verification_max_units_per_pair_scope": 48,
            }
        )
        runner = ComparisonInterpretationRunner(
            model_client=SyncCallableModelClient(model),
            config=config,
        )

        result = await runner.run(request=_request(), metrics=_many_metrics())

        for scope in ("all_sources", "current_only"):
            difference = result["verified_content_differences"][scope]["b"]
            self.assertEqual("completed", difference["status"])
            self.assertEqual(3, difference["batch_count"])
            self.assertEqual(25, difference["confirmed_added_count"])
            self.assertEqual(0, difference["unverified_count"])
        presence_prompts = [
            prompt for prompt in model.prompts if "<fact_units>" in prompt
        ]
        self.assertEqual(6, len(presence_prompts))
        for prompt in presence_prompts:
            units = json.loads(
                re.search(
                    r"<fact_units>\n(.*?)\n</fact_units>", prompt, re.DOTALL
                ).group(1)
            )
            self.assertLessEqual(len(units), 12)

    async def test_invalid_span_only_marks_its_fact_unverified(self) -> None:
        model = InterpretationModel(invalid_unit_id="risk")
        runner = ComparisonInterpretationRunner(
            model_client=SyncCallableModelClient(model),
            config=_request().evaluation_config,
        )

        result = await runner.run(request=_request(), metrics=_metrics())

        self.assertEqual("completed_with_warnings", result["status"])
        for scope in ("all_sources", "current_only"):
            difference = result["verified_content_differences"][scope]["b"]
            self.assertEqual("completed_with_warnings", difference["status"])
            self.assertEqual(1, difference["confirmed_added_count"])
            self.assertEqual(1, difference["confirmed_reduced_count"])
            self.assertEqual(1, difference["unverified_count"])
            risk = next(item for item in difference["items"] if item["unit_id"] == "risk")
            self.assertEqual("unverified", risk["verification_result"])
            self.assertEqual(
                "unknown_span_id:B9999",
                risk["evidence_validation"]["baseline_error"],
            )
        self.assertTrue(
            any(
                "all_sources b risk baseline evidence unverified: unknown_span_id:B9999"
                in warning
                for warning in result["warnings"]
            )
        )

    async def test_interpretation_failure_is_returned_not_raised(self) -> None:
        def failing_model(prompt: str, *, temperature: float, max_tokens: int) -> str:
            del prompt, temperature, max_tokens
            raise RuntimeError("gateway unavailable")

        config = _request().evaluation_config.model_copy(
            update={"comparison_interpretation_max_model_calls": 8}
        )
        runner = ComparisonInterpretationRunner(
            model_client=SyncCallableModelClient(failing_model),
            config=config,
        )

        result = await runner.run(request=_request(), metrics=_metrics())

        self.assertEqual("failed", result["status"])
        self.assertTrue(result["warnings"])
        self.assertGreater(result["execution_stats"]["failed_step_count"], 0)


if __name__ == "__main__":
    unittest.main()
