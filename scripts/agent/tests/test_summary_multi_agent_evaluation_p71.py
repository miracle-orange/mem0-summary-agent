from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from pydantic import BaseModel  # noqa: E402

from summary_multi_agent.evaluation.matcher import (  # noqa: E402
    EvaluationFactMatcher,
    MatchDraft,
    MatchDraftList,
)
from summary_multi_agent.evaluation.model_io import EvaluationModelIO  # noqa: E402
from summary_multi_agent.evaluation.source_fact_builder import (  # noqa: E402
    CurrentFactDraft,
    CurrentFactDraftList,
    EvaluationSourceFactBuilder,
)
from summary_multi_agent.evaluation.schemas import (  # noqa: E402
    EvaluationClaimScope,
    EvaluationEvidence,
    EvaluationFactState,
    EvaluationMatchStatus,
    EvaluationSalience,
    EvaluationSourceFact,
    EvaluationSourceScope,
    EvaluationSummaryFact,
)
from summary_multi_agent.schemas import RunConfig  # noqa: E402
from summary_multi_agent.raw_material import build_raw_material_bundle  # noqa: E402
from summary_multi_agent.schemas import SummaryRequest  # noqa: E402


def source_fact(fact_id: str, text: str) -> EvaluationSourceFact:
    return EvaluationSourceFact(
        fact_id=fact_id,
        source_scope=EvaluationSourceScope.CURRENT,
        fact=text,
        state=EvaluationFactState.CONFIRMED,
        salience=EvaluationSalience.MUST_INCLUDE,
        evidence=(EvaluationEvidence(evidence_text=text),),
    )


def summary_fact(fact_id: str, text: str) -> EvaluationSummaryFact:
    return EvaluationSummaryFact(
        summary_fact_id=fact_id,
        fact=text,
        evidence_quote=text,
        start_offset=0,
        end_offset=max(1, len(text)),
        claimed_scope=EvaluationClaimScope.CURRENT,
    )


class MatcherModelIO:
    def __init__(
        self,
        *,
        omit_last: bool = False,
        omit_all: bool = False,
        duplicate: bool = False,
        unknown: bool = False,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.omit_last = omit_last
        self.omit_all = omit_all
        self.duplicate = duplicate
        self.unknown = unknown

    async def complete_json(self, **kwargs):
        self.calls.append(kwargs)
        summary_payload = json.loads(
            re.search(
                r"<evaluation_summary_facts>\n(.*?)\n</evaluation_summary_facts>",
                kwargs["prompt"],
                re.DOTALL,
            ).group(1)
        )
        source_payload = json.loads(
            re.search(
                r"<evaluation_source_facts>\n(.*?)\n</evaluation_source_facts>",
                kwargs["prompt"],
                re.DOTALL,
            ).group(1)
        )
        if self.omit_all:
            result = MatchDraftList(matches=())
            kwargs["validator"](result)
            return result
        summary_items = summary_payload[:-1] if self.omit_last else summary_payload
        source_id = source_payload[0]["fact_id"]
        result = MatchDraftList(
            matches=tuple(
                MatchDraft(
                    summary_fact_id=item["summary_fact_id"],
                    status=EvaluationMatchStatus.SUPPORTED_FULL,
                    matched_source_fact_ids=(source_id,),
                    primary_source_fact_id=source_id,
                    reason_code="supported",
                )
                for item in summary_items
            )
        )
        if self.duplicate and result.matches:
            result = MatchDraftList(matches=(result.matches[0], result.matches[0]))
        if self.unknown:
            result = MatchDraftList(
                matches=(
                    MatchDraft(
                        summary_fact_id="unknown-summary-id",
                        status=EvaluationMatchStatus.UNSUPPORTED,
                        reason_code="unknown",
                    ),
                )
            )
        kwargs["validator"](result)
        return result


class P71MatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_small_source_set_uses_one_full_scan_call(self) -> None:
        model_io = MatcherModelIO()
        matcher = EvaluationFactMatcher(
            model_io=model_io,
            config=RunConfig(
                evaluation_match_batch_size=2,
                evaluation_match_full_scan_threshold=48,
            ),
        )
        claims = (
            summary_fact("s1", "决定保留项目范围"),
            summary_fact("s2", "负责人确认时间安排"),
            summary_fact("s3", "记录风险说明"),
        )
        sources = tuple(source_fact(f"f{i}", f"事实 {i}") for i in range(5))
        matches = await matcher.match(claims, sources, temporal_support={})

        self.assertEqual(2, len(model_io.calls))
        self.assertEqual(2, matcher.execution_stats["matcher_full_scan_batch_count"])
        self.assertEqual(0, matcher.execution_stats["matcher_candidate_batch_count"])
        self.assertEqual(3, len(matches))
        self.assertTrue(
            all(
                item["trace_metadata"]["candidate_strategy"] == "full_scan"
                for item in model_io.calls
            )
        )

    async def test_large_source_set_uses_bounded_round_robin_candidates(self) -> None:
        model_io = MatcherModelIO()
        matcher = EvaluationFactMatcher(
            model_io=model_io,
            config=RunConfig(
                evaluation_match_batch_size=6,
                evaluation_match_full_scan_threshold=4,
                evaluation_match_candidate_limit_per_fact=3,
                evaluation_match_candidate_union_limit=4,
            ),
        )
        claims = (
            summary_fact("s1", "负责人张三确认 2026-08-26 预算 100 元"),
            summary_fact("s2", "负责人李四确认 2026-08-27 预算 200 元"),
        )
        sources = tuple(
            source_fact(
                f"f{i}",
                (
                    "负责人张三确认 2026-08-26 预算 100 元"
                    if i == 37
                    else f"无关事实 {i}"
                ),
            )
            for i in range(60)
        )
        await matcher.match(claims, sources, temporal_support={})

        self.assertEqual(1, len(model_io.calls))
        metadata = model_io.calls[0]["trace_metadata"]
        self.assertEqual("candidate_top_k", metadata["candidate_strategy"])
        self.assertLessEqual(metadata["candidate_source_fact_count"], 4)
        self.assertEqual(56, metadata["candidate_source_fact_filtered_count"])

    async def test_partial_omission_is_conservative_and_audited(self) -> None:
        model_io = MatcherModelIO(omit_last=True)
        matcher = EvaluationFactMatcher(
            model_io=model_io,
            config=RunConfig(evaluation_match_batch_size=6),
        )
        matches = await matcher.match(
            (summary_fact("s1", "事实一"), summary_fact("s2", "事实二")),
            (source_fact("f1", "事实一"),),
            temporal_support={},
        )

        self.assertEqual(2, len(matches))
        self.assertEqual(EvaluationMatchStatus.UNSUPPORTED, matches[1].match_status)
        self.assertEqual("matcher_omitted_summary_fact", matches[1].reason_code)
        self.assertEqual(1, matcher.execution_stats["matcher_omitted_summary_fact_count"])
        self.assertTrue(matcher.warnings)

    async def test_all_omitted_is_invalid(self) -> None:
        model_io = MatcherModelIO(omit_all=True)
        matcher = EvaluationFactMatcher(
            model_io=model_io,
            config=RunConfig(evaluation_match_batch_size=6, evaluation_repair_attempts=0),
        )
        with self.assertRaises(ValueError):
            await matcher.match(
                (summary_fact("s1", "事实一"),),
                (source_fact("f1", "事实一"),),
                temporal_support={},
            )

    async def test_unknown_and_duplicate_ids_are_strict_protocol_errors(self) -> None:
        for mode in ("unknown", "duplicate"):
            model_io = MatcherModelIO(**{mode: True})
            matcher = EvaluationFactMatcher(
                model_io=model_io,
                config=RunConfig(
                    evaluation_match_batch_size=6,
                    evaluation_repair_attempts=0,
                ),
            )
            with self.assertRaises(ValueError):
                await matcher.match(
                    (summary_fact("s1", "事实一"), summary_fact("s2", "事实二")),
                    (source_fact("f1", "事实一"),),
                    temporal_support={},
                )


class CurrentFactModelIO:
    async def complete_json(self, **kwargs):
        segment = re.search(
            r"<evaluation_current_segment>\n(.*?)\n</evaluation_current_segment>",
            kwargs["prompt"],
            re.DOTALL,
        ).group(1)
        return CurrentFactDraftList(
            facts=(
                CurrentFactDraft(
                    fact="当前材料中的事实",
                    salience=EvaluationSalience.MUST_INCLUDE,
                    evidence_quote=segment[:8],
                ),
            )
        )


class P71SegmentationTests(unittest.IsolatedAsyncioTestCase):
    async def test_evaluation_segmentation_is_independent_and_preserves_offsets(self) -> None:
        material = "甲乙丙丁戊己庚辛壬癸" * 220
        production_bundle = build_raw_material_bundle(
            material,
            source_id="current_source",
            max_segment_chars=1500,
            overlap_ratio=0.05,
        )
        builder = EvaluationSourceFactBuilder(
            model_io=CurrentFactModelIO(),
            config=RunConfig(
                raw_segment_max_chars=1500,
                evaluation_segment_max_chars=1000,
            ),
        )
        request = SummaryRequest(
            current_material=material,
            summary_request="生成摘要",
        )
        facts = await builder.build_current_facts(request, production_bundle)

        self.assertGreater(builder.current_segment_count, len(production_bundle.segments))
        self.assertTrue(facts)
        self.assertTrue(
            all(
                item.evidence[0].source_id == "current_source"
                and item.evidence[0].start_offset is not None
                and item.evidence[0].end_offset is not None
                and item.evidence[0].end_offset <= len(material)
                for item in facts
            )
        )


class TinyResponse(BaseModel):
    value: str


class TruncatingClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        self.prompts.append(prompt)
        return '{"value": "truncated' if len(self.prompts) == 1 else '{"value": "ok"}'


class FailingGatewayClient:
    async def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        del prompt, temperature, max_tokens
        raise RuntimeError("DeepSeek WebSocket gateway returned an error: 6513")


class P71ModelIOTests(unittest.IsolatedAsyncioTestCase):
    async def test_gateway_failure_is_distinguished_from_empty_output(self) -> None:
        model_io = EvaluationModelIO(FailingGatewayClient(), max_model_calls=1)
        with self.assertRaises(RuntimeError):
            await model_io.complete_json(
                stage="evaluation.gateway",
                prompt="base prompt",
                response_model=TinyResponse,
                max_tokens=3000,
                repair_attempts=0,
                timeout_seconds=1,
            )

        self.assertEqual(
            "gateway_error",
            model_io.traces[0]["attempts"][0]["failure_kind"],
        )

    async def test_truncation_repair_does_not_reinject_raw_output(self) -> None:
        client = TruncatingClient()
        model_io = EvaluationModelIO(client, max_model_calls=2)
        result = await model_io.complete_json(
            stage="evaluation.test",
            prompt="base prompt",
            response_model=TinyResponse,
            max_tokens=3000,
            repair_attempts=1,
            timeout_seconds=1,
        )

        self.assertEqual("ok", result.value)
        self.assertEqual(2, model_io.model_call_count)
        self.assertEqual(1, model_io.repair_call_count)
        self.assertNotIn("truncated", client.prompts[1])
        self.assertEqual(
            "likely_output_truncation",
            model_io.traces[0]["attempts"][0]["failure_kind"],
        )


if __name__ == "__main__":
    unittest.main()
