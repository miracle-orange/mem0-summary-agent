from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent_web.ui_presentation import (
    combine_source_dates,
    file_date_rows,
    filtered_runs,
    ready_message,
    run_heading_html,
    run_label,
    run_list_html,
)


class WorkspacePresentationTests(unittest.TestCase):
    def test_upload_dates_keep_order_and_duplicate_names(self) -> None:
        files = ["/tmp/a/meeting.txt", "/tmp/b/meeting.txt"]
        self.assertEqual(
            file_date_rows(files, [["meeting.txt", "2026-07-01"], ["meeting.txt", "2026-07-08"]]),
            [["meeting.txt", "2026-07-01"], ["meeting.txt", "2026-07-08"]],
        )
        self.assertEqual(
            combine_source_dates(
                pasted="历史文本", uploaded=files, server_paths="/tmp/third.txt",
                manual_dates="2026-06-30\n2026-07-15",
                table=[["meeting.txt", "2026-07-01"], ["meeting.txt", "2026-07-08"]],
            ),
            "2026-06-30\n2026-07-01\n2026-07-08\n2026-07-15",
        )
        upload = SimpleNamespace(name="/tmp/gradio-random.txt", orig_name="meeting.txt")
        self.assertEqual(file_date_rows([upload]), [["meeting.txt", ""]])
        self.assertEqual(
            combine_source_dates(
                pasted="", uploaded=[upload], server_paths="", manual_dates="",
                table=[["meeting.txt", "2026-07-01"]],
            ),
            "2026-07-01",
        )

    def test_file_table_mismatch_fails_instead_of_assigning_wrong_date(self) -> None:
        with self.assertRaisesRegex(ValueError, "不一致"):
            combine_source_dates(
                pasted="", uploaded=["/tmp/current.txt"], server_paths="",
                manual_dates="", table=[["other.txt", "2026-07-01"]],
            )
        with self.assertRaisesRegex(ValueError, "行数"):
            combine_source_dates(
                pasted="", uploaded=["/tmp/current.txt"], server_paths="",
                manual_dates="2026-07-01", table=[["current.txt", ""]],
            )

    def test_run_views_escape_user_supplied_metadata(self) -> None:
        entries = [{
            "run_id": "run_1", "run_type": "ingest", "status": "completed",
            "project": "<script>alert(1)</script>", "source_time": "2026-07-01",
            "output_dir": "/tmp/one", "started_at": "2026-07-01T12:00:00Z",
        }]
        self.assertEqual(filtered_runs(entries, "INGEST"), entries)
        self.assertIn("资料导入", run_label(entries[0]))
        self.assertNotIn("<script>", run_list_html(entries))
        self.assertIn("&lt;script&gt;", run_list_html(entries))
        self.assertNotIn(
            "<script>",
            run_heading_html(title="<script>", project="p", source_time="2026-07-01", status="completed"),
        )
        self.assertIn("2 份资料已就绪", ready_message(["a.txt"], "文字"))


if __name__ == "__main__":
    unittest.main()
