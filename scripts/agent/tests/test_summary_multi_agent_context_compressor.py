from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.context_builder import build_synthesis_context  # noqa: E402
from summary_multi_agent.context_compressor import (  # noqa: E402
    AuxiliaryContextCompressor,
)
from summary_multi_agent.raw_material import (  # noqa: E402
    build_raw_material_bundle,
    mark_segment_processed,
)
from summary_multi_agent.schemas import (  # noqa: E402
    AgentResult,
    AgentStatus,
    ContextCompressionAction,
    ContextCompressionMode,
    CurrentFactCategory,
    CurrentFactItem,
    CurrentSegmentAnalysis,
    EvidenceRef,
    SourceKind,
    SummaryRequest,
    TaskType,
)
from summary_multi_agent.working_memory import initialize_working_state  # noqa: E402


def _memory_record(
    memory_id: str,
    memory_type: str,
    content: str,
    score: float,
    *,
    matched_entities: list[str] | None = None,
) -> dict[str, object]:
    return {
        "memory_id": memory_id,
        "memory_type": memory_type,
        "content": content,
        "score": score,
        "metadata": {
            "typed_memory_id": f"tm_{memory_id}",
            "memory_type": memory_type,
            "source_time": "2026-08-01",
            "evidence_span": content,
            "unused_transport_field": "must not enter synthesis context",
        },
        **(
            {"matched_entities": matched_entities}
            if matched_entities is not None
            else {}
        ),
    }


def _long_term_evidence(memory_id: str, evidence_id: str) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        source_kind=SourceKind.LONG_TERM_MEMORY,
        memory_id=memory_id,
        source_time="2026-08-01",
        evidence_text=f"evidence for {memory_id}",
    )


class AuxiliaryContextCompressorTests(unittest.TestCase):
    def _state(self):
        request = SummaryRequest(
            current_material="今天确认项目范围，明天提交报告。",
            source_time="2026-08-10",
            summary_request="生成项目摘要",
            target_entities=["负责人"],
        )
        bundle = build_raw_material_bundle(
            request.current_material,
            max_segment_chars=100,
            overlap_ratio=0.0,
        )
        state = initialize_working_state(request, bundle)
        segment = bundle.segments[0]
        current_evidence = EvidenceRef(
            evidence_id="evidence_current",
            source_kind=SourceKind.CURRENT_MATERIAL,
            source_id=bundle.source_id,
            segment_id=segment.segment_id,
            start_offset=0,
            end_offset=2,
            evidence_text=request.current_material[:2],
        )
        state.evidence_index[current_evidence.evidence_id] = current_evidence
        return state

    def test_deterministic_compression_merges_cross_worker_memory_identity(self) -> None:
        state = self._state()
        semantic_records = [
            _memory_record("shared", "task", "负责人提交报告。", 0.8),
            _memory_record("time_low", "time", "明天提交。", 0.1),
            _memory_record("pref_high", "preference", "使用表格。", 0.9),
            _memory_record("pref_low", "preference", "使用短句。", 0.2),
        ]
        entity_records = [
            _memory_record(
                "shared",
                "task",
                "负责人需要提交项目报告。",
                0.95,
                matched_entities=["负责人"],
            ),
            _memory_record("task_low", "task", "负责人复核附件。", 0.05),
        ]
        semantic_evidence = tuple(
            _long_term_evidence(str(item["memory_id"]), f"ev_s_{index}")
            for index, item in enumerate(semantic_records)
        )
        entity_evidence = tuple(
            _long_term_evidence(str(item["memory_id"]), f"ev_e_{index}")
            for index, item in enumerate(entity_records)
        )
        state.agent_results = {
            "memory_search": AgentResult(
                task_id="memory_search",
                task_type=TaskType.SEMANTIC_MEMORY,
                status=AgentStatus.SUCCESS,
                payload={"queries": ["任务时间"], "records": semantic_records},
                evidence=semantic_evidence,
            ),
            "entity_search": AgentResult(
                task_id="entity_search",
                task_type=TaskType.ENTITY_PERSONALIZATION,
                status=AgentStatus.SUCCESS,
                payload={
                    "target_entities": ["负责人"],
                    "records": entity_records,
                    "processed_segment_ids": [
                        state.raw_material.segments[0].segment_id
                    ],
                },
                evidence=entity_evidence,
            ),
        }
        for evidence in (*semantic_evidence, *entity_evidence):
            state.evidence_index[evidence.evidence_id] = evidence

        raw_before = state.raw_material.model_dump(mode="json")
        ledger_before = state.coverage_ledger.model_dump(mode="json")
        evidence_before = copy.deepcopy(state.evidence_index)
        output = AuxiliaryContextCompressor(
            mode=ContextCompressionMode.DETERMINISTIC,
            top_k_per_type=1,
        ).compress(state)

        self.assertEqual(
            {"shared", "time_low", "pref_high", "task_low"},
            set(output.historical_records),
        )
        shared = output.historical_records["shared"]
        self.assertEqual(
            {"memory_search", "entity_search"}, set(shared["source_task_ids"])
        )
        self.assertEqual(["负责人"], shared["matched_entities"])
        self.assertEqual(2, len(shared["content_variants"]))
        self.assertEqual(2, len(shared["record_variants"]))
        self.assertNotIn("records", output.worker_results["memory_search"]["payload"])
        self.assertEqual(
            {"shared", "time_low", "pref_high"},
            set(output.worker_results["memory_search"]["payload"]["record_ids"]),
        )
        self.assertNotIn("queries", output.worker_results["memory_search"]["payload"])
        self.assertNotIn(
            "processed_segment_ids",
            output.worker_results["entity_search"]["payload"],
        )
        self.assertEqual(6, output.report.input_record_count)
        self.assertEqual(5, output.report.unique_record_count)
        self.assertEqual(4, output.report.retained_record_count)
        self.assertEqual(1, output.report.merged_record_count)
        self.assertEqual(1, output.report.dropped_record_count)
        self.assertEqual(3, output.report.protected_record_count)
        self.assertTrue(output.report.invariants_preserved)
        self.assertFalse(output.report.current_material_compressed)
        self.assertLess(output.report.output_chars, output.report.input_chars)
        self.assertLess(
            output.report.estimated_output_tokens,
            output.report.estimated_input_tokens,
        )
        self.assertEqual(
            output.report.current_material_sha256_before,
            output.report.current_material_sha256_after,
        )
        self.assertTrue(
            any(
                item.action == ContextCompressionAction.MERGE
                and item.retained_id == "shared"
                for item in output.report.decisions
            )
        )
        self.assertTrue(
            any(
                item.action == ContextCompressionAction.DROP
                and item.memory_type == "preference"
                for item in output.report.decisions
            )
        )
        self.assertEqual(raw_before, state.raw_material.model_dump(mode="json"))
        self.assertEqual(ledger_before, state.coverage_ledger.model_dump(mode="json"))
        self.assertEqual(evidence_before, state.evidence_index)

    def test_disabled_mode_keeps_records_in_their_worker_payloads(self) -> None:
        state = self._state()
        record = _memory_record("task_1", "task", "负责人提交报告。", 0.8)
        evidence = _long_term_evidence("task_1", "ev_task_1")
        state.agent_results["memory_search"] = AgentResult(
            task_id="memory_search",
            task_type=TaskType.SEMANTIC_MEMORY,
            status=AgentStatus.SUCCESS,
            payload={"queries": ["任务"], "records": [record]},
            evidence=(evidence,),
        )
        state.evidence_index[evidence.evidence_id] = evidence

        output = AuxiliaryContextCompressor(
            mode=ContextCompressionMode.DISABLED,
            top_k_per_type=1,
        ).compress(state)

        self.assertFalse(output.historical_records)
        self.assertEqual(
            "task_1",
            output.worker_results["memory_search"]["payload"]["records"][0][
                "memory_id"
            ],
        )
        self.assertEqual(ContextCompressionMode.DISABLED, output.report.mode)
        self.assertEqual(1, output.report.retained_record_count)
        self.assertEqual(0, output.report.dropped_record_count)

    def test_synthesis_context_preserves_current_fact_offsets(self) -> None:
        state = self._state()
        segment = state.raw_material.segments[0]
        item = CurrentFactItem(
            item_id="current_fact_1",
            category=CurrentFactCategory.FACT,
            statement="今天确认项目范围。",
            evidence_quote=state.request.current_material[:2],
            evidence_id="evidence_current",
        )
        state.current_segment_analyses = [
            CurrentSegmentAnalysis(
                segment_id=segment.segment_id,
                start_offset=segment.start_offset,
                end_offset=segment.end_offset,
                items=(item,),
            )
        ]
        state.coverage_ledger = mark_segment_processed(
            state.coverage_ledger,
            segment_id=segment.segment_id,
            stage="current_fact_map",
            evidence_ids=(item.evidence_id,),
        )

        context = build_synthesis_context(
            state,
            compression_mode=ContextCompressionMode.DETERMINISTIC,
            compression_top_k_per_type=1,
        )

        self.assertEqual((item,), context.merged_current_items)
        self.assertEqual(0, context.evidence_index["evidence_current"].start_offset)
        self.assertEqual(2, context.evidence_index["evidence_current"].end_offset)
        self.assertFalse(
            context.auxiliary_context_info["current_material_compressed"]
        )
        self.assertTrue(context.auxiliary_context_info["invariants_preserved"])
        self.assertIsNotNone(state.context_compression_report)


if __name__ == "__main__":
    unittest.main()
