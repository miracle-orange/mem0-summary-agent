from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.evaluation.evaluator import _visible_historical_records  # noqa: E402
from summary_multi_agent.evaluation.matcher import (  # noqa: E402
    EvaluationFactMatcher,
    MatchDraft,
    MatchDraftList,
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
    HistoricalDisposition,
)
from summary_multi_agent.schemas import RunConfig  # noqa: E402


class CurrentContradictionModelIO:
    def __init__(self) -> None:
        self.stages: list[str] = []

    async def complete_json(self, **kwargs):
        self.stages.append(kwargs["stage"])
        return MatchDraftList(
            matches=(
                MatchDraft(
                    summary_fact_id="q1",
                    status=EvaluationMatchStatus.CONTRADICTED,
                    matched_source_fact_ids=("c1",),
                    primary_source_fact_id="c1",
                    reason_code="current_state_conflict",
                ),
            )
        )


class EvaluationHistoryTests(unittest.IsolatedAsyncioTestCase):
    def test_visible_history_supports_deterministic_and_disabled_compression(self) -> None:
        context = SimpleNamespace(
            historical_records={
                "m1": {"memory_id": "m1", "content": "canonical"}
            },
            worker_results={
                "semantic": {
                    "payload": {
                        "records": [
                            {"memory_id": "m1", "content": "passthrough duplicate"},
                            {"memory_id": "m2", "content": "passthrough visible"},
                        ]
                    },
                    "evidence_ids": ["e1"],
                }
            },
        )
        records = _visible_historical_records(context)
        self.assertEqual({"m1", "m2"}, set(records))
        self.assertEqual("canonical", records["m1"]["content"])
        self.assertEqual(["e1"], records["m2"]["evidence_ids"])

    async def test_current_non_unsupported_match_prevents_stale_history_override(self) -> None:
        model_io = CurrentContradictionModelIO()
        matcher = EvaluationFactMatcher(
            model_io=model_io,
            config=RunConfig(evaluation_match_batch_size=12),
        )
        current = EvaluationSourceFact(
            fact_id="c1",
            source_scope=EvaluationSourceScope.CURRENT,
            fact="任务已经完成",
            state=EvaluationFactState.COMPLETED,
            salience=EvaluationSalience.MUST_INCLUDE,
            evidence=(EvaluationEvidence(evidence_text="任务已经完成"),),
        )
        history = EvaluationSourceFact(
            fact_id="h1",
            source_scope=EvaluationSourceScope.HISTORICAL,
            fact="任务仍待完成",
            state=EvaluationFactState.PENDING,
            salience=EvaluationSalience.EXCLUDE,
            evidence=(EvaluationEvidence(evidence_text="任务待完成"),),
            history_disposition=HistoricalDisposition.SUPERSEDED,
        )
        claim = EvaluationSummaryFact(
            summary_fact_id="q1",
            fact="任务仍待完成",
            evidence_quote="任务仍待完成",
            start_offset=0,
            end_offset=6,
            claimed_scope=EvaluationClaimScope.HISTORICAL,
        )
        matches = await matcher.match(
            (claim,), (current, history), temporal_support={}
        )
        self.assertEqual(EvaluationMatchStatus.CONTRADICTED, matches[0].match_status)
        self.assertEqual(EvaluationSourceScope.CURRENT, matches[0].primary_source_scope)
        self.assertEqual(1, len(model_io.stages))
        self.assertIn(".current.", model_io.stages[0])


if __name__ == "__main__":
    unittest.main()
