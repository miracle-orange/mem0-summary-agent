"""执行 P6.6 三领域运行产物的自动化验收检查。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = Path(__file__).parent / "eval_inputs" / "p6_6" / "cases.json"
READ_ONLY_TOOLS = {
    "calendar_resolve",
    "entity_query",
    "template_adaptation",
    "typed_memory_search",
}
ACCEPTED_RUN_STATUSES = {"completed", "completed_with_warnings"}
ACCEPTED_PLAN_SOURCES = {"model", "model_repaired"}
ACCEPTED_REVIEW_TERMINATIONS = {
    "accepted",
    "converged",
    "max_rounds",
    "no_actionable_issues",
    "oscillation",
    "stop_with_warnings",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_path_mapping(values: list[str], option: str) -> dict[str, Path]:
    mappings: dict[str, Path] = {}
    for raw in values:
        case_id, separator, path_text = raw.partition("=")
        case_id = case_id.strip()
        path_text = path_text.strip()
        if not separator or not case_id or not path_text:
            raise ValueError(f"{option} expects CASE_ID=PATH, got {raw!r}")
        if case_id in mappings:
            raise ValueError(f"duplicate {option} case id: {case_id}")
        mappings[case_id] = Path(path_text).expanduser()
    return mappings


def _record_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: Any,
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _required_json(run_dir: Path, name: str) -> Any:
    path = run_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"missing required artifact: {path}")
    return _load_json(path)


def _count_ingest_operations(add_results: Any) -> tuple[int, dict[str, int]]:
    event_counts: dict[str, int] = {}
    operation_count = 0
    values = add_results if isinstance(add_results, list) else []
    for result in values:
        if not isinstance(result, dict):
            continue
        operations = result.get("results")
        if not isinstance(operations, list):
            continue
        for operation in operations:
            if not isinstance(operation, dict):
                continue
            event = str(operation.get("event") or "UNKNOWN").upper()
            event_counts[event] = event_counts.get(event, 0) + 1
            operation_count += 1
    return operation_count, dict(sorted(event_counts.items()))


def _validate_ingest(case: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        run_info = _required_json(run_dir, "run_info.json")
        typed_memories = _required_json(run_dir, "typed_memories.json")
        add_results = _required_json(run_dir, "add_results.json")
    except Exception as exc:
        _record_check(checks, "required_ingest_artifacts", False, str(exc))
        return {"run_dir": str(run_dir), "passed": False, "checks": checks}

    typed_count = len(typed_memories) if isinstance(typed_memories, list) else 0
    operation_count, event_counts = _count_ingest_operations(add_results)
    _record_check(checks, "typed_memories_extracted", typed_count > 0, typed_count)
    _record_check(checks, "memory_operations_recorded", operation_count > 0, event_counts)
    _record_check(
        checks,
        "ingest_identity_matches_case",
        run_info.get("user_id") == case["user_id"]
        and run_info.get("project") == case["project"]
        and run_info.get("source_time") == case["history_source_time"],
        {
            "user_id": run_info.get("user_id"),
            "project": run_info.get("project"),
            "source_time": run_info.get("source_time"),
        },
    )
    return {
        "run_dir": str(run_dir),
        "passed": all(item["passed"] for item in checks),
        "typed_memory_count": typed_count,
        "operation_count": operation_count,
        "event_counts": event_counts,
        "checks": checks,
    }


def _plan_signature(plan: dict[str, Any]) -> str:
    tasks = plan.get("tasks") if isinstance(plan, dict) else []
    task_types = {
        str(item.get("task_id")): str(item.get("task_type"))
        for item in tasks
        if isinstance(item, dict)
    }
    normalized = []
    for item in tasks:
        if not isinstance(item, dict):
            continue
        dependency_types = sorted(
            task_types.get(str(task_id), "unknown")
            for task_id in item.get("dependencies", [])
        )
        normalized.append(
            {
                "task_type": item.get("task_type"),
                "dependency_types": dependency_types,
                "allowed_tools": sorted(item.get("allowed_tools", [])),
                "input_keys": sorted((item.get("inputs") or {}).keys()),
            }
        )
    normalized.sort(key=lambda item: json.dumps(item, sort_keys=True))
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _plan_depth(plan: dict[str, Any]) -> int:
    tasks = {
        str(item.get("task_id")): item
        for item in plan.get("tasks", [])
        if isinstance(item, dict)
    }
    memo: dict[str, int] = {}

    def depth(task_id: str) -> int:
        if task_id in memo:
            return memo[task_id]
        dependencies = tasks.get(task_id, {}).get("dependencies", [])
        value = 1 + max((depth(str(item)) for item in dependencies), default=0)
        memo[task_id] = value
        return value

    return max((depth(task_id) for task_id in tasks), default=0)


def _tool_names(agent_results: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    for result in agent_results.values():
        if not isinstance(result, dict):
            continue
        for tool_result in result.get("tool_results", []):
            if isinstance(tool_result, dict) and tool_result.get("tool_name"):
                names.add(str(tool_result["tool_name"]))
        execution = result.get("worker_execution") or {}
        for step in execution.get("steps", []):
            if not isinstance(step, dict):
                continue
            for tool_result in step.get("tool_results", []):
                if isinstance(tool_result, dict) and tool_result.get("tool_name"):
                    names.add(str(tool_result["tool_name"]))
    return sorted(names)


def _validate_current_evidence(
    evidence_index: dict[str, Any], current_material: str
) -> tuple[int, list[str]]:
    checked = 0
    invalid: list[str] = []
    for evidence_id, item in evidence_index.items():
        if not isinstance(item, dict) or item.get("source_kind") != "current_material":
            continue
        checked += 1
        start = item.get("start_offset")
        end = item.get("end_offset")
        evidence_text = item.get("evidence_text")
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
            or end > len(current_material)
            or not isinstance(evidence_text, str)
            or current_material[start:end] != evidence_text
        ):
            invalid.append(str(evidence_id))
    return checked, invalid


def _validate_summary_run(
    case: dict[str, Any], run_dir: Path, manifest_dir: Path
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    required = (
        "summary_run_result.json",
        "plan.json",
        "plan_history.json",
        "planner_validation_report.json",
        "agent_results.json",
        "coverage_ledger.json",
        "evidence_index.json",
        "context_compression_report.json",
        "current_material.txt",
        "summary_multi_agent.md",
    )
    missing = [name for name in required if not (run_dir / name).is_file()]
    _record_check(checks, "required_summary_artifacts", not missing, missing)
    if missing:
        return {"run_dir": str(run_dir), "passed": False, "checks": checks}

    result = _load_json(run_dir / "summary_run_result.json")
    plan = _load_json(run_dir / "plan.json")
    plan_history = _load_json(run_dir / "plan_history.json")
    planner_report = _load_json(run_dir / "planner_validation_report.json")
    agent_results = _load_json(run_dir / "agent_results.json")
    evidence_index = _load_json(run_dir / "evidence_index.json")
    compression = _load_json(run_dir / "context_compression_report.json")
    current_material = (run_dir / "current_material.txt").read_text(encoding="utf-8")
    expected_material = (manifest_dir / case["current_file"]).read_text(
        encoding="utf-8"
    )

    initial_plan = plan_history[0] if isinstance(plan_history, list) and plan_history else {}
    initial_source = initial_plan.get("source")
    _record_check(
        checks,
        "run_completed",
        result.get("status") in ACCEPTED_RUN_STATUSES,
        result.get("status"),
    )
    _record_check(
        checks,
        "coverage_complete",
        result.get("coverage_complete") is True,
        result.get("coverage_complete"),
    )
    _record_check(
        checks,
        "model_plan_accepted",
        planner_report.get("status") == "model_plan_accepted"
        and planner_report.get("fallback_used") is False
        and planner_report.get("selected_source") in ACCEPTED_PLAN_SOURCES
        and initial_source in ACCEPTED_PLAN_SOURCES,
        {
            "status": planner_report.get("status"),
            "fallback_used": planner_report.get("fallback_used"),
            "selected_source": planner_report.get("selected_source"),
            "initial_source": initial_source,
        },
    )

    replanner_path = run_dir / "replanner_validation_report.json"
    if replanner_path.is_file():
        replanner_report = _load_json(replanner_path)
        _record_check(
            checks,
            "model_replan_accepted_when_present",
            replanner_report.get("status") == "model_replan_accepted"
            and replanner_report.get("fallback_used") is False,
            {
                "status": replanner_report.get("status"),
                "fallback_used": replanner_report.get("fallback_used"),
                "selected_strategy": replanner_report.get("selected_strategy"),
            },
        )

    task_by_id = {
        str(item.get("task_id")): item
        for item in plan.get("tasks", [])
        if isinstance(item, dict)
    }
    missing_results = sorted(set(task_by_id) - set(agent_results))
    unsuccessful = sorted(
        task_id
        for task_id in task_by_id
        if not isinstance(agent_results.get(task_id), dict)
        or agent_results[task_id].get("status") != "success"
    )
    _record_check(
        checks,
        "all_planned_workers_succeeded",
        not missing_results and not unsuccessful,
        {"missing_results": missing_results, "unsuccessful": unsuccessful},
    )

    worker_mode_failures: list[str] = []
    lineage_failures: list[str] = []
    dependency_edges = 0
    for task_id, task in task_by_id.items():
        worker_result = agent_results.get(task_id) or {}
        execution = worker_result.get("worker_execution") or {}
        if execution.get("mode") != "model" or execution.get("fallback_reason"):
            worker_mode_failures.append(task_id)
        dependencies = {str(item) for item in task.get("dependencies", [])}
        dependency_edges += len(dependencies)
        consumed = {str(item) for item in worker_result.get("consumed_result_ids", [])}
        execution_consumed = {
            str(item) for item in execution.get("consumed_prior_result_ids", [])
        }
        if dependencies and (consumed != dependencies or execution_consumed != dependencies):
            lineage_failures.append(task_id)
    _record_check(
        checks,
        "strict_model_workers_without_fallback",
        not worker_mode_failures,
        worker_mode_failures,
    )
    _record_check(
        checks,
        "declared_dependencies_consumed",
        not lineage_failures,
        lineage_failures,
    )

    tools = _tool_names(agent_results)
    unexpected_tools = sorted(set(tools) - READ_ONLY_TOOLS)
    _record_check(
        checks,
        "summarize_tools_are_read_only",
        not unexpected_tools,
        {"used": tools, "unexpected": unexpected_tools},
    )
    _record_check(
        checks,
        "current_material_unchanged",
        current_material == expected_material,
        {
            "expected_sha256": hashlib.sha256(expected_material.encode("utf-8")).hexdigest(),
            "actual_sha256": hashlib.sha256(current_material.encode("utf-8")).hexdigest(),
        },
    )
    _record_check(
        checks,
        "compression_invariants_preserved",
        compression.get("current_material_compressed") is False
        and compression.get("invariants_preserved") is True
        and compression.get("current_material_sha256_before")
        == compression.get("current_material_sha256_after")
        and compression.get("segment_manifest_sha256_before")
        == compression.get("segment_manifest_sha256_after")
        and compression.get("coverage_ledger_sha256_before")
        == compression.get("coverage_ledger_sha256_after")
        and compression.get("current_evidence_offsets_sha256_before")
        == compression.get("current_evidence_offsets_sha256_after"),
        {
            "mode": compression.get("mode"),
            "ratio": compression.get("compression_ratio"),
            "current_material_compressed": compression.get("current_material_compressed"),
            "invariants_preserved": compression.get("invariants_preserved"),
        },
    )

    current_evidence_count, invalid_evidence = _validate_current_evidence(
        evidence_index, current_material
    )
    memory_evidence_count = sum(
        isinstance(item, dict) and item.get("source_kind") == "long_term_memory"
        for item in evidence_index.values()
    )
    _record_check(
        checks,
        "current_evidence_offsets_grounded",
        current_evidence_count > 0 and not invalid_evidence,
        {"count": current_evidence_count, "invalid_ids": invalid_evidence},
    )
    _record_check(
        checks,
        "history_memory_retrieved",
        memory_evidence_count > 0,
        memory_evidence_count,
    )
    _record_check(
        checks,
        "shadow_review_finished",
        result.get("review_enabled") is True
        and result.get("review_shadow_mode") is True
        and result.get("review_termination") in ACCEPTED_REVIEW_TERMINATIONS,
        {
            "enabled": result.get("review_enabled"),
            "shadow_mode": result.get("review_shadow_mode"),
            "termination": result.get("review_termination"),
        },
    )

    task_types = sorted({item.get("task_type") for item in task_by_id.values()})
    worker_steps = 0
    worker_model_attempts = 0
    worker_tool_results = 0
    worker_duration_ms = 0
    termination_reasons: dict[str, int] = {}
    for task_id in task_by_id:
        worker_result = agent_results.get(task_id) or {}
        execution = worker_result.get("worker_execution") or {}
        worker_steps += len(execution.get("steps", []))
        worker_model_attempts += len(execution.get("model_attempts", []))
        worker_tool_results += len(worker_result.get("tool_results", []))
        worker_duration_ms += int(worker_result.get("duration_ms") or 0)
        reason = str(execution.get("termination_reason") or "unknown")
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1

    review_metrics: dict[str, Any] = {}
    review_path = run_dir / "review_run_result.json"
    if review_path.is_file():
        review_payload = _load_json(review_path)
        review_rounds = review_payload.get("rounds", [])
        review_metrics = {
            "round_count": len(review_rounds),
            "termination": review_payload.get("termination"),
            "recommended_candidate_index": review_payload.get(
                "recommended_candidate_index"
            ),
            "judge_scores": [
                item.get("judge_decision", {}).get("score")
                for item in review_rounds
                if isinstance(item, dict)
            ],
            "improvements": [
                item.get("judge_decision", {}).get("improvement")
                for item in review_rounds
                if isinstance(item, dict)
            ],
        }
    return {
        "run_dir": str(run_dir),
        "passed": all(item["passed"] for item in checks),
        "status": result.get("status"),
        "plan_source": plan.get("source"),
        "plan_signature": _plan_signature(plan),
        "plan_depth": _plan_depth(plan),
        "task_count": len(task_by_id),
        "task_types": task_types,
        "dependency_edge_count": dependency_edges,
        "tool_names": tools,
        "worker_metrics": {
            "step_count": worker_steps,
            "model_attempt_count": worker_model_attempts,
            "tool_result_count": worker_tool_results,
            "duration_ms": worker_duration_ms,
            "termination_reasons": dict(sorted(termination_reasons.items())),
        },
        "current_evidence_count": current_evidence_count,
        "memory_evidence_count": memory_evidence_count,
        "compression_ratio": compression.get("compression_ratio"),
        "replan_count": max(0, len(plan_history) - 1),
        "review_termination": result.get("review_termination"),
        "review_metrics": review_metrics,
        "warnings": result.get("warnings", []),
        "checks": checks,
    }


def evaluate_suite(
    manifest_path: Path,
    ingest_runs: dict[str, Path],
    summary_runs: dict[str, Path],
) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    cases = manifest.get("cases", [])
    case_ids = [str(item["case_id"]) for item in cases]
    expected_ids = set(case_ids)
    suite_checks: list[dict[str, Any]] = []
    suite_observations: list[dict[str, Any]] = []
    _record_check(
        suite_checks,
        "all_manifest_cases_provided",
        set(ingest_runs) == expected_ids and set(summary_runs) == expected_ids,
        {
            "expected": sorted(expected_ids),
            "ingest": sorted(ingest_runs),
            "summary": sorted(summary_runs),
        },
    )

    case_reports: dict[str, Any] = {}
    manifest_dir = manifest_path.parent
    for case in cases:
        case_id = str(case["case_id"])
        if case_id not in ingest_runs or case_id not in summary_runs:
            continue
        ingest_report = _validate_ingest(case, ingest_runs[case_id])
        summary_report = _validate_summary_run(
            case, summary_runs[case_id], manifest_dir
        )
        case_reports[case_id] = {
            "domain": case["domain"],
            "passed": ingest_report["passed"] and summary_report["passed"],
            "ingest": ingest_report,
            "summary": summary_report,
            "manual_focus": case.get("manual_focus", []),
        }

    domains = {str(item["domain"]) for item in cases}
    signatures = {
        report["summary"].get("plan_signature")
        for report in case_reports.values()
        if report["summary"].get("plan_signature")
    }
    worker_sets = {
        tuple(report["summary"].get("task_types", []))
        for report in case_reports.values()
    }
    dependency_edges = sum(
        int(report["summary"].get("dependency_edge_count", 0))
        for report in case_reports.values()
    )
    _record_check(
        suite_checks,
        "three_distinct_domains",
        len(domains) >= 3,
        sorted(domains),
    )
    _record_check(
        suite_checks,
        "model_dag_structures_are_not_identical",
        len(signatures) >= 2 and len(worker_sets) >= 2,
        {
            "plan_signatures": sorted(signatures),
            "worker_type_sets": [list(item) for item in sorted(worker_sets)],
        },
    )
    suite_observations.append(
        {
            "name": "cross_agent_dependency_observed",
            "observed": dependency_edges > 0,
            "detail": dependency_edges,
            "blocking": False,
            "reason": (
                "P6.6 observes naturally planned dependencies but does not require "
                "them; P6.3 separately validates dependency consumption and lineage."
            ),
        }
    )

    cases_passed = len(case_reports) == len(cases) and all(
        item["passed"] for item in case_reports.values()
    )
    passed = cases_passed and all(item["passed"] for item in suite_checks)
    return {
        "schema_version": "summary_multi_agent.p6_6_acceptance.v2",
        "status": "passed" if passed else "failed",
        "scope": {
            "core_multi_agent": True,
            "domain_template_richness": False,
            "active_review_replacement": False,
        },
        "manifest": str(manifest_path),
        "cases": case_reports,
        "suite_checks": suite_checks,
        "suite_observations": suite_observations,
        "manual_review_required": [
            "逐案阅读 summary_multi_agent.md 与推荐候选，按 manual_focus 核对语义质量。",
            "检查 Planner/Worker 的任务和工具选择是否与领域目标相称，而非只满足 JSON 合法性。",
            "记录失败或 fallback 的原始 Prompt、模型输出和审计产物，不为单一样例放宽契约。",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the P6.6 three-domain target-machine runs."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--ingest",
        action="append",
        default=[],
        metavar="CASE_ID=PATH",
        help="Repeat once per case with the concrete Ingest output directory.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="CASE_ID=PATH",
        help="Repeat once per case with the concrete Multi-Agent output directory.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = evaluate_suite(
            args.manifest.expanduser(),
            _parse_path_mapping(args.ingest, "--ingest"),
            _parse_path_mapping(args.run, "--run"),
        )
    except Exception as exc:
        raise SystemExit(f"P6.6 checker failed: {type(exc).__name__}: {exc}") from exc

    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
