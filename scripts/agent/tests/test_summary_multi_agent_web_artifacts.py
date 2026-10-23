from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent_web.artifact_reader import (  # noqa: E402
    ArtifactReader,
    create_safe_artifact_archive,
    read_artifact_snapshot,
)
from summary_multi_agent_web.schemas import WebRunStatus  # noqa: E402


class WebArtifactReaderTests(unittest.TestCase):
    def test_reader_loads_optional_rich_report_without_breaking_old_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "summary_run_result.json").write_text(
                json.dumps({"run_id": "rich-1", "status": "completed"}),
                encoding="utf-8",
            )
            (root / "summary_multi_agent.md").write_text(
                "# 正式纪要", encoding="utf-8"
            )
            document = {
                "schema_version": "summary_multi_agent.minutes_document.v1",
                "title": "正式纪要",
                "source_time": "2026-10-02",
                "meeting_scene": "general",
                "format_kind": "structured",
                "overview": ["概览"],
                "sections": [
                    {
                        "section_id": "SEC001",
                        "heading": "进展",
                        "entries": [
                            {
                                "entry_id": "E0001",
                                "kind": "fact",
                                "title": "",
                                "content": "有据进展",
                                "owner": "",
                                "time_requirement": "",
                                "dependency": "",
                                "recorded_state": "",
                                "current_state": "",
                                "relation": "",
                                "current_evidence_quotes": [],
                                "history_refs": [],
                            }
                        ],
                    }
                ],
            }
            sources = {
                "schema_version": "summary_multi_agent.minutes_sources.v1",
                "history_aliases": {},
                "entries": [{"entry_id": "E0001", "sources": [], "warnings": []}],
                "warnings": [],
            }
            manifest = {
                "schema_version": "summary_multi_agent.minutes_report_manifest.v1",
                "document_schema_version": document["schema_version"],
                "selected_candidate_index": 0,
                "summary_sha256": "a" * 64,
                "output_mode": "rich",
                "artifact_status": {"minutes_report.html": "completed"},
                "warnings": [],
            }
            for name, payload in (
                ("minutes_document.json", document),
                ("minutes_sources.json", sources),
                ("minutes_report_manifest.json", manifest),
            ):
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
            (root / "minutes_report.html").write_text(
                "<!doctype html><title>正式纪要</title>", encoding="utf-8"
            )

            snapshot = read_artifact_snapshot(root)

            self.assertEqual("正式纪要", snapshot.minutes_document["title"])
            self.assertEqual(
                "summary_multi_agent.minutes_sources.v1",
                snapshot.minutes_sources["schema_version"],
            )
            self.assertEqual("rich", snapshot.minutes_manifest["output_mode"])
            self.assertNotIn("minutes_document.json", snapshot.errors)

            archive_path = create_safe_artifact_archive(root)
            import zipfile

            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())
            self.assertIn("minutes_document.json", names)
            self.assertIn("minutes_sources.json", names)
            self.assertIn("minutes_report.html", names)

    def test_reader_projects_formal_and_shadow_candidates_without_raw_model_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "summary_run_result.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-1",
                        "status": "completed_with_warnings",
                        "warnings": ["worker warning"],
                        "recommended_candidate_index": 1,
                    }
                ),
                encoding="utf-8",
            )
            (root / "summary_multi_agent.md").write_text("正式 candidate 0", encoding="utf-8")
            (root / "review_candidate_summaries.json").write_text(
                json.dumps(
                    [
                        {"candidate_index": 0, "summary": "正式 candidate 0"},
                        {"candidate_index": 1, "summary": "Shadow candidate 1"},
                    ]
                ),
                encoding="utf-8",
            )
            (root / "agent_results.json").write_text(
                json.dumps(
                    {
                        "task-1": {
                            "task_type": "semantic_memory",
                            "status": "success",
                            "tool_results": [
                                {
                                    "call_id": "tool-1",
                                    "tool_name": "typed_memory_search",
                                    "status": "success",
                                    "payload": {"secret": "do-not-show"},
                                }
                            ],
                            "worker_execution": {
                                "mode": "model",
                                "model_attempts": [{"raw_output": "hidden chain"}],
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )

            snapshot = read_artifact_snapshot(root)

            self.assertEqual(WebRunStatus.COMPLETED_WITH_WARNINGS, snapshot.status)
            self.assertEqual("正式 candidate 0", snapshot.formal_summary)
            self.assertEqual(1, len(snapshot.shadow_candidates))
            self.assertEqual("typed_memory_search", snapshot.tools[0]["tool_name"])
            encoded = json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False)
            self.assertNotIn("hidden chain", encoded)
            self.assertNotIn("do-not-show", encoded)

    def test_reader_rejects_path_traversal_and_reports_corrupt_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "run"
            root.mkdir()
            outside = Path(temp_dir) / "outside.json"
            outside.write_text('{"secret": true}', encoding="utf-8")
            (root / "summary_run_result.json").write_text("{broken", encoding="utf-8")

            reader = ArtifactReader(root)
            traversal = reader.read_json("../outside.json")
            snapshot = reader.build_snapshot()

            self.assertIn("..", traversal.error)
            self.assertIn("summary_run_result.json", snapshot.errors)
            self.assertEqual({}, snapshot.planner["plan"])

    def test_missing_artifacts_are_page_readable_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = read_artifact_snapshot(Path(temp_dir))

            self.assertIn("summary_run_result.json", snapshot.errors)
            self.assertIn("summary_multi_agent.md", snapshot.errors)
            self.assertEqual("暂无运行" if False else "", snapshot.formal_summary)

    def test_safe_archive_omits_raw_inputs_and_redacts_config_secrets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "web_run_info.json").write_text(
                json.dumps({"api_key": "secret-value", "status": "completed"}),
                encoding="utf-8",
            )
            (root / "summary_run_result.json").write_text(
                json.dumps({"status": "completed", "final_summary": "done"}),
                encoding="utf-8",
            )
            (root / "summary_multi_agent.md").write_text("done", encoding="utf-8")
            (root / "model_input_report.json").write_text(
                json.dumps(
                    {
                        "schema_version": "summary_multi_agent.model_input_report.v1",
                        "current_material_sha256": "a" * 64,
                        "history_record_ids": ["memory-1"],
                        "totals": {
                            "projected": {"chars": 100},
                            "legacy_full_context": {"chars": 200},
                        },
                        "char_reduction_ratio": 0.5,
                    }
                ),
                encoding="utf-8",
            )
            (root / "model_call_input_report.json").write_text(
                json.dumps(
                    {
                        "schema_version": "summary_multi_agent.model_call_input_report.v1",
                        "calls": [{"stage": "summarizer", "totals": {"chars": 100}}],
                    }
                ),
                encoding="utf-8",
            )
            (root / "history_selection_report.json").write_text(
                json.dumps(
                    {
                        "schema_version": "summary_multi_agent.history_selection_report.v1",
                        "status": "records",
                        "selected_memory_ids": ["memory-1"],
                    }
                ),
                encoding="utf-8",
            )
            (root / "planner_prompt.txt").write_text("api_key=secret-value", encoding="utf-8")
            (root / "current_material.txt").write_text("secret-value", encoding="utf-8")

            archive_path = create_safe_artifact_archive(root)

            import zipfile

            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                content = "\n".join(archive.read(name).decode("utf-8") for name in names)
            self.assertIn("summary_run_result.json", names)
            self.assertIn("model_input_report.json", names)
            self.assertIn("model_call_input_report.json", names)
            self.assertIn("history_selection_report.json", names)
            self.assertNotIn("planner_prompt.txt", names)
            self.assertNotIn("current_material.txt", names)
            self.assertNotIn("secret-value", content)


if __name__ == "__main__":
    unittest.main()
