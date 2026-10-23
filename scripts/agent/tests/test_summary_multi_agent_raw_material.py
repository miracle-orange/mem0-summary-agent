from __future__ import annotations

import sys
import unittest
from pathlib import Path

from pydantic import ValidationError


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent.raw_material import (  # noqa: E402
    IncompleteCoverageError,
    assert_complete_coverage,
    build_coverage_ledger,
    build_raw_material_bundle,
    mark_segment_failed,
    mark_segment_processed,
)
from summary_multi_agent.schemas import RawMaterialBundle  # noqa: E402


class RawMaterialBundleTests(unittest.TestCase):
    def test_long_material_is_fully_covered_without_gaps(self) -> None:
        raw_text = "0123456789" * 7
        bundle = build_raw_material_bundle(
            raw_text,
            source_id="current_test",
            max_segment_chars=16,
            overlap_chars=3,
        )

        self.assertGreater(len(bundle.segments), 1)
        self.assertEqual(0, bundle.segments[0].start_offset)
        self.assertEqual(len(raw_text), bundle.segments[-1].end_offset)
        covered = [False] * len(raw_text)
        for segment in bundle.segments:
            self.assertEqual(
                raw_text[segment.start_offset : segment.end_offset], segment.text
            )
            for index in range(segment.start_offset, segment.end_offset):
                covered[index] = True
        self.assertTrue(all(covered))

    def test_bundle_detects_tampered_segment_text(self) -> None:
        bundle = build_raw_material_bundle("abcdefghij", max_segment_chars=6)
        payload = bundle.model_dump(mode="python")
        payload["segments"][0]["text"] = "xxxxxx"
        with self.assertRaises(ValidationError):
            RawMaterialBundle.model_validate(payload)

    def test_empty_material_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_raw_material_bundle("   ")


class CoverageLedgerTests(unittest.TestCase):
    def test_all_segments_and_stages_must_complete(self) -> None:
        bundle = build_raw_material_bundle(
            "abcdefghij" * 4,
            max_segment_chars=12,
            overlap_chars=2,
        )
        ledger = build_coverage_ledger(
            bundle, required_stages=("current_fact_map", "synthesis_merge")
        )
        with self.assertRaises(IncompleteCoverageError):
            assert_complete_coverage(ledger)

        for segment in bundle.segments:
            ledger = mark_segment_processed(
                ledger,
                segment_id=segment.segment_id,
                stage="current_fact_map",
                evidence_ids=(f"evidence_{segment.ordinal}",),
            )
            ledger = mark_segment_processed(
                ledger,
                segment_id=segment.segment_id,
                stage="synthesis_merge",
            )

        assert_complete_coverage(ledger)
        self.assertTrue(ledger.is_complete)

    def test_failed_stage_remains_incomplete(self) -> None:
        bundle = build_raw_material_bundle("abcdefghij", max_segment_chars=10)
        ledger = build_coverage_ledger(bundle, required_stages=("current_fact_map",))
        ledger = mark_segment_failed(
            ledger,
            segment_id=bundle.segments[0].segment_id,
            stage="current_fact_map",
            error="model timeout",
        )
        self.assertFalse(ledger.is_complete)
        self.assertEqual(
            "model timeout", ledger.segments[0].failed_stages["current_fact_map"]
        )

    def test_failure_after_success_revokes_stage_completion(self) -> None:
        bundle = build_raw_material_bundle("abcdefghij", max_segment_chars=10)
        segment_id = bundle.segments[0].segment_id
        ledger = build_coverage_ledger(bundle, required_stages=("current_fact_map",))
        ledger = mark_segment_processed(
            ledger, segment_id=segment_id, stage="current_fact_map"
        )
        self.assertTrue(ledger.is_complete)

        ledger = mark_segment_failed(
            ledger,
            segment_id=segment_id,
            stage="current_fact_map",
            error="retry failed",
        )
        self.assertFalse(ledger.is_complete)
        self.assertNotIn("current_fact_map", ledger.segments[0].completed_stages)


if __name__ == "__main__":
    unittest.main()
