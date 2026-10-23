from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.current_material import (  # noqa: E402
    CurrentFactDraft,
    CurrentMaterialAnalyzer,
    CurrentSegmentDraft,
)
from summary_multi_agent.grounding import match_evidence_quote  # noqa: E402
from summary_multi_agent.raw_material import build_raw_material_bundle  # noqa: E402
from summary_multi_agent.schemas import RunConfig, SummaryRequest  # noqa: E402
from summary_multi_agent.synthesis import remove_summary_meta_preamble  # noqa: E402
from summary_multi_agent.tools.calendar import resolve_relative_date  # noqa: E402
from summary_multi_agent.workers.temporal_reasoning import (  # noqa: E402
    TemporalReasoningWorker,
)


class SummaryOutputContractTests(unittest.TestCase):
    def test_only_short_meta_preamble_before_summary_is_removed(self) -> None:
        summary = (
            "好的，这是根据您提供的 SynthesisContext 重新生成的最终中文摘要。\n\n"
            "### 本次进展\n- 项目范围已经确认。"
        )
        cleaned, removed = remove_summary_meta_preamble(summary)

        self.assertTrue(removed)
        self.assertEqual("### 本次进展\n- 项目范围已经确认。", cleaned)

        substantive = "本次摘要记录项目范围已经确认，并列出后续任务。"
        unchanged, removed = remove_summary_meta_preamble(substantive)
        self.assertFalse(removed)
        self.assertEqual(substantive, unchanged)


class GroundingTests(unittest.TestCase):
    def test_unique_format_only_difference_maps_back_to_exact_source(self) -> None:
        source = "负责人：\n请在第 3 项后提交全角Ａ版报告。"
        result = match_evidence_quote(
            source, "负责人：请在第3项后提交全角A版报告。"
        )

        self.assertIsNotNone(result.grounded_quote)
        grounded = result.grounded_quote
        assert grounded is not None
        self.assertEqual("unique_width_whitespace_normalized", grounded.strategy)
        self.assertEqual(source[grounded.start : grounded.end], grounded.evidence_text)
        self.assertIn("\n", grounded.evidence_text)

    def test_ambiguous_normalized_match_is_rejected(self) -> None:
        source = "第 3 项待确认；第　3项已经关闭。"
        result = match_evidence_quote(source, "第3 项")

        self.assertIsNone(result.grounded_quote)
        self.assertEqual("ambiguous_format_equivalent_source_span", result.reason)

    def test_semantic_similarity_is_not_accepted(self) -> None:
        result = match_evidence_quote("请提交最终报告。", "需要交付最后的文档。")

        self.assertIsNone(result.grounded_quote)
        self.assertEqual("no_format_equivalent_source_span", result.reason)

    def test_math_compatibility_characters_are_not_folded(self) -> None:
        result = match_evidence_quote("函数项为 x²。", "函数项为 x2。")

        self.assertIsNone(result.grounded_quote)

    def test_mapper_resets_only_unsupported_temporal_enrichment(self) -> None:
        source = "负责人确认，报告周五已经提交。"
        request = SummaryRequest(
            current_material=source,
            source_time="2026-03-19",
            summary_request="生成项目摘要",
        )
        segment = build_raw_material_bundle(source).segments[0]
        draft = CurrentSegmentDraft(
            items=(
                CurrentFactDraft(
                    category="fact",
                    statement="负责人确认，报告上周五已经提交。",
                    evidence_quote=source,
                    entities=("负责人",),
                    time_expression="周五",
                ),
            )
        )

        outcome = CurrentMaterialAnalyzer._ground_draft(request, segment, draft)

        self.assertEqual(source, outcome.analysis.items[0].statement)
        self.assertEqual("周五", outcome.analysis.items[0].time_expression)
        self.assertTrue(
            any("temporally enriched" in item for item in outcome.analysis.warnings)
        )


class SequenceModel:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.prompts: list[str] = []

    async def complete(
        self, prompt: str, *, temperature: float, max_tokens: int
    ) -> str:
        del temperature, max_tokens
        self.prompts.append(prompt)
        return self.outputs.pop(0)


class GroundingRepairTests(unittest.IsolatedAsyncioTestCase):
    async def test_unmatched_quote_gets_controlled_model_repair(self) -> None:
        source = "项目负责人要求周五提交最终报告。"
        invalid = json.dumps(
            {
                "items": [
                    {
                        "category": "task",
                        "statement": "周五交付最终文档。",
                        "evidence_quote": "需要在星期五交付最后的文档",
                        "entities": ["项目负责人"],
                        "time_expression": "星期五",
                    }
                ]
            },
            ensure_ascii=False,
        )
        repaired = json.dumps(
            {
                "items": [
                    {
                        "category": "task",
                        "statement": source,
                        "evidence_quote": source,
                        "entities": ["项目负责人"],
                        "time_expression": "周五",
                    }
                ]
            },
            ensure_ascii=False,
        )
        model = SequenceModel([invalid, repaired])
        analyzer = CurrentMaterialAnalyzer(
            model_client=model,
            config=RunConfig(
                workflow_mode="multi_agent_v2", schema_repair_attempts=1
            ),
        )
        request = SummaryRequest(
            current_material=source,
            source_time="2026-03-19",
            summary_request="生成项目摘要",
        )
        segment = build_raw_material_bundle(source).segments[0]

        outcome = await analyzer._map_segment(request, segment)

        self.assertIsNotNone(outcome.analysis)
        assert outcome.analysis is not None
        self.assertEqual(1, len(outcome.analysis.items))
        self.assertEqual(source, outcome.analysis.items[0].evidence_quote)
        self.assertEqual(2, len(model.prompts))
        self.assertIn("不要使用语义近似引文", model.prompts[1])


class TemporalResolutionTests(unittest.TestCase):
    def test_unmarked_weekday_without_direction_remains_ambiguous(self) -> None:
        result = resolve_relative_date("周五", "2026-03-19", direction="auto")

        self.assertIsNone(result.normalized_date)
        self.assertEqual("weekday:ambiguous_direction", result.rule)
        self.assertIsNotNone(result.warning)

    def test_past_and_future_directions_are_resolved_generally(self) -> None:
        past = resolve_relative_date("周五", "2026-03-19", direction="past")
        future = resolve_relative_date("周五", "2026-03-19", direction="future")

        self.assertEqual("2026-03-13", past.normalized_date)
        self.assertEqual("2026-03-20", future.normalized_date)

    def test_same_day_past_does_not_jump_to_previous_week(self) -> None:
        result = resolve_relative_date("周五", "2026-03-20", direction="past")

        self.assertEqual("2026-03-20", result.normalized_date)

    def test_direction_cues_use_clause_semantics_not_domain_terms(self) -> None:
        past_text = "负责人说明，报告周五已经提交。"
        completed_before_weekday = "负责人答复：发了，周五中午发到项目群了。"
        future_text = "负责人要求，请在周五提交报告。"

        self.assertEqual(
            "past",
            TemporalReasoningWorker._direction_for_context(
                past_text, past_text.index("周五")
            ),
        )
        self.assertEqual(
            "past",
            TemporalReasoningWorker._direction_for_context(
                completed_before_weekday,
                completed_before_weekday.index("周五"),
            ),
        )
        self.assertEqual(
            "future",
            TemporalReasoningWorker._direction_for_context(
                future_text, future_text.index("周五")
            ),
        )


if __name__ == "__main__":
    unittest.main()
