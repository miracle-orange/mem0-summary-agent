import json
import sys
import tempfile
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent_p66_check import evaluate_suite


MANIFEST = AGENT_ROOT / "eval_inputs" / "p6_6" / "cases.json"


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class P66CheckerTests(unittest.TestCase):
    def _build_fixture(self, root: Path):
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        ingest_runs = {}
        summary_runs = {}
        for index, case in enumerate(manifest["cases"]):
            case_id = case["case_id"]
            ingest_dir = root / f"ingest_{case_id}"
            summary_dir = root / f"summary_{case_id}"
            ingest_dir.mkdir()
            summary_dir.mkdir()
            ingest_runs[case_id] = ingest_dir
            summary_runs[case_id] = summary_dir

            write_json(
                ingest_dir / "run_info.json",
                {
                    "user_id": case["user_id"],
                    "project": case["project"],
                    "source_time": case["history_source_time"],
                },
            )
            write_json(
                ingest_dir / "typed_memories.json",
                [{"memory_id": f"memory_{case_id}", "memory_type": "task"}],
            )
            write_json(
                ingest_dir / "add_results.json",
                [{"results": [{"event": "ADD", "id": f"memory_{case_id}"}]}],
            )

            current = (MANIFEST.parent / case["current_file"]).read_text(
                encoding="utf-8"
            )
            task_types = (
                ["semantic_memory", "temporal_reasoning"]
                if index == 0
                else ["semantic_memory"]
                if index == 1
                else ["semantic_memory", "template_adaptation"]
            )
            tasks = []
            agent_results = {}
            for task_index, task_type in enumerate(task_types):
                task_id = f"{case_id}_{task_type}"
                dependencies = (
                    [f"{case_id}_semantic_memory"]
                    if index == 0 and task_index == 1
                    else []
                )
                allowed_tool = {
                    "semantic_memory": "typed_memory_search",
                    "temporal_reasoning": "calendar_resolve",
                    "template_adaptation": "template_adaptation",
                }[task_type]
                tasks.append(
                    {
                        "task_id": task_id,
                        "task_type": task_type,
                        "description": "fixture task",
                        "dependencies": dependencies,
                        "critical": True,
                        "timeout_seconds": 60,
                        "allowed_tools": [allowed_tool],
                        "inputs": {"fixture": True},
                        "max_model_calls": 2,
                        "max_tool_calls": 2,
                    }
                )
                agent_results[task_id] = {
                    "task_id": task_id,
                    "task_type": task_type,
                    "status": "success",
                    "consumed_result_ids": dependencies,
                    "tool_results": [{"tool_name": allowed_tool}],
                    "worker_execution": {
                        "mode": "model",
                        "fallback_reason": None,
                        "steps": [],
                        "consumed_prior_result_ids": dependencies,
                    },
                }
            plan = {"source": "model", "tasks": tasks}
            write_json(
                summary_dir / "summary_run_result.json",
                {
                    "status": "completed_with_warnings",
                    "coverage_complete": True,
                    "review_enabled": True,
                    "review_shadow_mode": True,
                    "review_termination": "accepted",
                    "warnings": [],
                },
            )
            write_json(summary_dir / "plan.json", plan)
            write_json(summary_dir / "plan_history.json", [plan])
            write_json(
                summary_dir / "planner_validation_report.json",
                {
                    "status": "model_plan_accepted",
                    "fallback_used": False,
                    "selected_source": "model",
                },
            )
            write_json(summary_dir / "agent_results.json", agent_results)
            write_json(summary_dir / "coverage_ledger.json", {"segments": []})
            write_json(
                summary_dir / "evidence_index.json",
                {
                    "current": {
                        "source_kind": "current_material",
                        "start_offset": 0,
                        "end_offset": 1,
                        "evidence_text": current[:1],
                    },
                    "memory": {
                        "source_kind": "long_term_memory",
                        "memory_id": f"memory_{case_id}",
                    },
                },
            )
            write_json(
                summary_dir / "context_compression_report.json",
                {
                    "mode": "deterministic",
                    "compression_ratio": 0.8,
                    "current_material_compressed": False,
                    "invariants_preserved": True,
                    "current_material_sha256_before": "a",
                    "current_material_sha256_after": "a",
                    "segment_manifest_sha256_before": "b",
                    "segment_manifest_sha256_after": "b",
                    "coverage_ledger_sha256_before": "c",
                    "coverage_ledger_sha256_after": "c",
                    "current_evidence_offsets_sha256_before": "d",
                    "current_evidence_offsets_sha256_after": "d",
                },
            )
            (summary_dir / "current_material.txt").write_text(
                current, encoding="utf-8"
            )
            (summary_dir / "summary_multi_agent.md").write_text(
                "# fixture summary\n", encoding="utf-8"
            )
        return ingest_runs, summary_runs

    def test_three_domain_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ingest_runs, summary_runs = self._build_fixture(Path(temp_dir))
            report = evaluate_suite(MANIFEST, ingest_runs, summary_runs)

        self.assertEqual("passed", report["status"])
        self.assertTrue(all(item["passed"] for item in report["suite_checks"]))
        self.assertEqual(
            "summary_multi_agent.p6_6_acceptance.v2",
            report["schema_version"],
        )
        self.assertTrue(report["suite_observations"][0]["observed"])

    def test_planner_fallback_fails_the_suite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ingest_runs, summary_runs = self._build_fixture(Path(temp_dir))
            path = summary_runs["project_meeting"] / "planner_validation_report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report.update(
                {
                    "status": "rule_fallback",
                    "fallback_used": True,
                    "selected_source": "rule_fallback",
                }
            )
            write_json(path, report)
            result = evaluate_suite(MANIFEST, ingest_runs, summary_runs)

        self.assertEqual("failed", result["status"])
        self.assertFalse(result["cases"]["project_meeting"]["passed"])

    def test_no_cross_agent_dependency_is_non_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ingest_runs, summary_runs = self._build_fixture(Path(temp_dir))
            for run_dir in summary_runs.values():
                plan_path = run_dir / "plan.json"
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                for task in plan["tasks"]:
                    task["dependencies"] = []
                write_json(plan_path, plan)
                write_json(run_dir / "plan_history.json", [plan])

                results_path = run_dir / "agent_results.json"
                results = json.loads(results_path.read_text(encoding="utf-8"))
                for result in results.values():
                    result["consumed_result_ids"] = []
                    result["worker_execution"]["consumed_prior_result_ids"] = []
                write_json(results_path, results)

            report = evaluate_suite(MANIFEST, ingest_runs, summary_runs)

        self.assertEqual("passed", report["status"])
        observation = next(
            item
            for item in report["suite_observations"]
            if item["name"] == "cross_agent_dependency_observed"
        )
        self.assertFalse(observation["observed"])
        self.assertFalse(observation["blocking"])


if __name__ == "__main__":
    unittest.main()
