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

from summary_multi_agent_web.artifact_reader import (  # noqa: E402
    create_safe_artifact_archive,
    read_artifact_snapshot,
)
from summary_multi_agent_web.schemas import (  # noqa: E402
    WebSummarizeRequest,
    build_run_config,
)


class WebEvaluationTests(unittest.TestCase):
    def test_web_request_maps_evaluation_configuration(self) -> None:
        request = WebSummarizeRequest(
            current_material="当前会议材料",
            summary_request="生成纪要",
            user_id="u",
            project="p",
            enable_evaluation=True,
            evaluation_timeout_seconds=456,
            evaluation_match_batch_size=7,
            evaluation_personalization_batch_size=11,
            evaluation_personalization_max_model_calls=13,
        )
        config = build_run_config(request)
        self.assertTrue(config.evaluation_enabled)
        self.assertEqual(456, config.evaluation_timeout_seconds)
        self.assertEqual(7, config.evaluation_match_batch_size)
        self.assertEqual(11, config.evaluation_personalization_batch_size)
        self.assertEqual(13, config.evaluation_personalization_max_model_calls)

    def test_inspector_projects_metrics_and_safe_archive_excludes_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            (run_dir / "summary_run_result.json").write_text(
                json.dumps(
                    {
                        "run_id": "run_eval",
                        "status": "completed",
                        "evaluation_enabled": True,
                        "evaluation_status": "completed",
                        "personalization_evaluation_status": "completed",
                        "warnings": [],
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "summary_multi_agent.md").write_text("正式摘要", encoding="utf-8")
            (run_dir / "evaluation_metrics.json").write_text(
                json.dumps({"status": "completed", "core_metrics": {}}),
                encoding="utf-8",
            )
            (run_dir / "evaluation_source_facts.json").write_text("[]", encoding="utf-8")
            (run_dir / "evaluation_summary_facts.json").write_text("[]", encoding="utf-8")
            (run_dir / "evaluation_fact_matches.json").write_text("[]", encoding="utf-8")
            (run_dir / "evaluation_report.md").write_text("# 评测", encoding="utf-8")
            (run_dir / "evaluation_model_trace.json").write_text(
                json.dumps({"prompt": "sensitive"}), encoding="utf-8"
            )
            (run_dir / "evaluation_personalization_metrics.json").write_text(
                json.dumps({"status": "completed", "per_target": []}),
                encoding="utf-8",
            )
            (run_dir / "evaluation_personalization_source_annotations.json").write_text(
                "[]", encoding="utf-8"
            )
            (run_dir / "evaluation_personalization_claim_annotations.json").write_text(
                "[]", encoding="utf-8"
            )
            (run_dir / "evaluation_personalization_report.md").write_text(
                "# 个性化评测", encoding="utf-8"
            )
            (run_dir / "evaluation_personalization_model_trace.json").write_text(
                json.dumps({"prompt": "sensitive target trace"}), encoding="utf-8"
            )

            snapshot = read_artifact_snapshot(run_dir)
            self.assertEqual("completed", snapshot.evaluation["status"])
            self.assertEqual("# 评测", snapshot.evaluation["report_markdown"])
            self.assertEqual(
                "completed", snapshot.evaluation["personalization"]["status"]
            )
            self.assertEqual(
                "# 个性化评测",
                snapshot.evaluation["personalization"]["report_markdown"],
            )
            archive = create_safe_artifact_archive(run_dir)
            with zipfile.ZipFile(archive) as opened:
                names = set(opened.namelist())
            self.assertIn("evaluation_metrics.json", names)
            self.assertIn("evaluation_report.md", names)
            self.assertNotIn("evaluation_model_trace.json", names)
            self.assertIn("evaluation_personalization_metrics.json", names)
            self.assertIn("evaluation_personalization_report.md", names)
            self.assertNotIn(
                "evaluation_personalization_model_trace.json", names
            )


if __name__ == "__main__":
    unittest.main()
