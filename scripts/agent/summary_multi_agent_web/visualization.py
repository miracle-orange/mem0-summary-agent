"""把现有运行产物投影为只读、可容错的单次运行可视化模型。"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .artifact_reader import read_artifact_snapshot, sanitize_value
from .schemas import (
    AgentGraphEdge,
    AgentGraphNode,
    VisualizationFlowMetric,
    VisualizationStage,
    VisualizationTimelineItem,
    WebRunVisualization,
)


TRACE_EVENT_LIMIT = 50_000
TRACE_EVENT_TYPES = {
    "pipeline.started",
    "pipeline.finished",
    "pipeline.status_changed",
    "plan.created",
    "plan.replanning",
    "plan.replanned",
    "dag.layer_started",
    "task.started",
    "task.finished",
    "task.ad_hoc_started",
    "task.ad_hoc_finished",
    "context.compressed",
    "review.started",
    "review.round_started",
    "review.finished",
    "evaluation.started",
    "evaluation.finished",
    "run.status_changed",
    "run.global_timeout",
    "run.failed",
}

STAGE_TEMPLATES: dict[str, tuple[tuple[str, str], ...]] = {
    "ingest": (
        ("validation", "Validation"),
        ("preprocessing", "Preprocessing"),
        ("extracting", "Extracting"),
        ("deduplicating", "Deduplicating"),
        ("writing", "Writing"),
        ("verifying", "Verifying"),
    ),
    "summarize": (
        ("store_probe", "Store Probe"),
        ("planning", "Planning"),
        ("agent_execution", "Agent Execution"),
        ("synthesis", "Synthesis"),
        ("review", "Shadow Review"),
        ("evaluation", "Evaluation"),
    ),
    "evaluation_compare": (
        ("validation", "Validation"),
        ("source_snapshot", "Source Snapshot"),
        ("target_annotation", "Target Annotation"),
        ("summary_scoring", "Summary A/B/C Scoring"),
        ("aggregating", "Aggregating"),
        ("comparison_interpretation", "Direct Comparison"),
    ),
}

FINAL_STATUSES = {"completed", "completed_with_warnings", "failed", "cancelled"}
SUCCESS_STATUSES = {"success", "completed"}
WARNING_STATUSES = {"warning", "completed_with_warnings", "partial"}


def _load_json(path: Path, errors: dict[str, str], default: Any) -> Any:
    """读取固定名称 JSON；缺失表示阶段尚未产出，不计为错误。"""
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors[path.name] = f"{type(exc).__name__}: {exc}"
        return default


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _duration_ms(started_at: Any, finished_at: Any) -> int | None:
    start = _parse_time(started_at)
    finish = _parse_time(finished_at)
    if start is None or finish is None or finish < start:
        return None
    return int((finish - start).total_seconds() * 1000)


def _normalize_status(value: Any, *, default: str = "pending") -> str:
    status = str(value or "").strip().lower()
    if status in SUCCESS_STATUSES:
        return "success"
    if status in {"ready", "verified", "accepted"}:
        return "success"
    if status in WARNING_STATUSES:
        return "warning"
    if status in {"failed", "error", "timeout", "timed_out"}:
        return "failed"
    if status in {"skipped", "disabled", "not_applicable"}:
        return "skipped"
    if status in {
        "running",
        "queued",
        "planning",
        "executing",
        "collecting",
        "replanning",
        "synthesizing",
        "reviewing",
        "evaluating",
        "cancelling",
        "cancellation_requested",
    }:
        return "running"
    return default


def _run_kind(run_type: str, run_dir: Path) -> str:
    normalized = run_type.strip().lower()
    if normalized in {"ingest", "e2e_ingest"}:
        return "ingest"
    if normalized == "evaluation_compare":
        return "evaluation_compare"
    if normalized in {"summarize", "multi_agent_summarize", "e2e_run"}:
        return "summarize"
    if (run_dir / "evaluation_compare_run_result.json").is_file():
        return "evaluation_compare"
    if (run_dir / "ingest_run_result.json").is_file() and not (
        run_dir / "plan.json"
    ).is_file():
        return "ingest"
    if (run_dir / "plan.json").is_file():
        return "summarize"
    return "unknown"


def _read_trace(
    run_dir: Path,
    warnings: list[str],
    errors: dict[str, str],
) -> list[dict[str, Any]]:
    """流式读取白名单 trace，忽略损坏行并限制事件数量。"""
    path = run_dir / "execution_trace.jsonl"
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    corrupt_count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if len(events) >= TRACE_EVENT_LIMIT:
                    warnings.append(
                        f"执行 trace 超过 {TRACE_EVENT_LIMIT} 条，展示已截断。"
                    )
                    break
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    corrupt_count += 1
                    continue
                if not isinstance(payload, dict):
                    continue
                event_type = str(payload.get("event_type") or "")
                if event_type not in TRACE_EVENT_TYPES:
                    continue
                event_payload = _as_dict(payload.get("payload"))
                events.append(
                    {
                        "timestamp": str(payload.get("timestamp") or ""),
                        "event_type": event_type,
                        "task_id": str(event_payload.get("task_id") or ""),
                        "status": str(event_payload.get("status") or ""),
                        "round": event_payload.get("round")
                        or event_payload.get("round_index"),
                        "candidate_index": event_payload.get("candidate_index"),
                        "termination": str(event_payload.get("termination") or ""),
                        "error": str(event_payload.get("error") or ""),
                    }
                )
    except (OSError, UnicodeError) as exc:
        errors[path.name] = f"{type(exc).__name__}: {exc}"
        return []
    if corrupt_count:
        warnings.append(f"执行 trace 中有 {corrupt_count} 行损坏，已忽略。")
    return events


def _trace_bounds(
    events: Iterable[dict[str, Any]],
    event_types: set[str],
) -> tuple[str, str, int | None]:
    times = [
        str(item.get("timestamp") or "")
        for item in events
        if item.get("event_type") in event_types and item.get("timestamp")
    ]
    if not times:
        return "", "", None
    return times[0], times[-1], _duration_ms(times[0], times[-1])


def _build_timeline(events: list[dict[str, Any]]) -> list[VisualizationTimelineItem]:
    """把开始/结束事件配对为阶段点和 Worker 区间。"""
    timeline: list[VisualizationTimelineItem] = []
    starts: dict[tuple[str, str], dict[str, Any]] = {}
    task_pairs = {
        "task.started": "task.finished",
        "task.ad_hoc_started": "task.ad_hoc_finished",
    }
    finish_to_start = {finish: start for start, finish in task_pairs.items()}
    labels = {
        "pipeline.started": "Pipeline 启动",
        "pipeline.finished": "Pipeline 完成",
        "plan.created": "Planner 生成计划",
        "plan.replanning": "触发 Replanning",
        "plan.replanned": "Replanner 生成新计划",
        "context.compressed": "Context Compressor 完成",
        "review.started": "Shadow Review 启动",
        "review.round_started": "Review 轮次启动",
        "review.finished": "Shadow Review 完成",
        "evaluation.started": "Evaluation 启动",
        "evaluation.finished": "Evaluation 完成",
        "run.global_timeout": "运行全局超时",
        "run.failed": "运行失败",
    }
    for index, event in enumerate(events):
        event_type = str(event.get("event_type") or "")
        task_id = str(event.get("task_id") or "")
        if event_type in task_pairs:
            starts[(event_type, task_id)] = event
            continue
        if event_type in finish_to_start:
            start_type = finish_to_start[event_type]
            started = starts.pop((start_type, task_id), {})
            status = _normalize_status(event.get("status"), default="success")
            timeline.append(
                VisualizationTimelineItem(
                    item_id=f"task-{index}-{task_id or 'unknown'}",
                    label=("Ad-hoc Worker" if "ad_hoc" in event_type else "Worker")
                    + f" · {task_id or 'unknown'}",
                    category="worker",
                    status=status,
                    started_at=str(started.get("timestamp") or ""),
                    finished_at=str(event.get("timestamp") or ""),
                    duration_ms=_duration_ms(
                        started.get("timestamp"), event.get("timestamp")
                    ),
                    task_id=task_id,
                )
            )
            continue
        if event_type not in labels:
            continue
        suffix = ""
        if event.get("round") is not None:
            suffix = f" · 第 {event['round']} 轮"
        label = labels[event_type] + suffix
        status = (
            "failed"
            if event_type in {"run.failed", "run.global_timeout"}
            else "success"
        )
        if event_type.endswith(".started") or event_type in {"plan.replanning"}:
            status = "running"
        timestamp = str(event.get("timestamp") or "")
        timeline.append(
            VisualizationTimelineItem(
                item_id=f"event-{index}",
                label=label,
                category="event",
                status=status,
                started_at=timestamp,
                finished_at=timestamp,
            )
        )
    for (event_type, task_id), started in starts.items():
        timeline.append(
            VisualizationTimelineItem(
                item_id=f"open-{event_type}-{task_id}",
                label=("Ad-hoc Worker" if "ad_hoc" in event_type else "Worker")
                + f" · {task_id or 'unknown'}",
                category="worker",
                status="running",
                started_at=str(started.get("timestamp") or ""),
                task_id=task_id,
            )
        )
    return sorted(
        timeline,
        key=lambda item: (item.started_at or "~", item.item_id),
    )


def _worker_call_count(worker: dict[str, Any], key: str) -> int | None:
    execution = _as_dict(worker.get("worker_execution"))
    budget = _as_dict(execution.get("final_budget"))
    budget_key = f"{key}_used"
    if budget_key in budget and isinstance(budget[budget_key], int):
        return int(budget[budget_key])
    if key in budget and isinstance(budget[key], int):
        return int(budget[key])
    if key == "model_calls":
        attempts = execution.get("model_attempts")
        return len(attempts) if isinstance(attempts, list) else None
    tools = worker.get("tool_results")
    return len(tools) if isinstance(tools, list) else None


def _build_graph(
    plan: dict[str, Any],
    agent_results: dict[str, Any],
    working_state: dict[str, Any],
) -> tuple[list[AgentGraphNode], list[AgentGraphEdge]]:
    """以最终计划为准构造节点，并用消费记录标记实际协作边。"""
    nodes: list[AgentGraphNode] = []
    edges: list[AgentGraphEdge] = []
    tasks = [item for item in _as_list(plan.get("tasks")) if isinstance(item, dict)]
    task_states = _as_dict(working_state.get("task_states"))
    preserved = {str(item) for item in _as_list(plan.get("preserved_task_ids"))}
    replanned_plan = (
        bool(plan.get("parent_plan_id")) or int(plan.get("replan_round") or 0) > 0
    )
    task_ids = {str(item.get("task_id") or "") for item in tasks}
    for task in tasks:
        task_id = str(task.get("task_id") or "")
        if not task_id:
            continue
        worker = _as_dict(agent_results.get(task_id))
        state = _as_dict(task_states.get(task_id))
        execution = _as_dict(worker.get("worker_execution"))
        consumed = {
            str(item)
            for item in (
                _as_list(worker.get("consumed_result_ids"))
                + _as_list(execution.get("consumed_prior_result_ids"))
            )
        }
        warnings = worker.get("warnings")
        evidence = worker.get("evidence")
        status = worker.get("status") or state.get("status") or "pending"
        nodes.append(
            AgentGraphNode(
                task_id=task_id,
                task_type=str(
                    task.get("task_type") or worker.get("task_type") or "unknown"
                ),
                description=str(task.get("description") or ""),
                status=_normalize_status(status, default=str(status or "pending")),
                critical=bool(task.get("critical")),
                allowed_tools=[
                    str(item) for item in _as_list(task.get("allowed_tools"))
                ],
                duration_ms=(
                    int(worker["duration_ms"])
                    if isinstance(worker.get("duration_ms"), (int, float))
                    else None
                ),
                model_call_count=_worker_call_count(worker, "model_calls"),
                tool_call_count=_worker_call_count(worker, "tool_calls"),
                evidence_count=(len(evidence) if isinstance(evidence, list) else None),
                warning_count=(len(warnings) if isinstance(warnings, list) else None),
                error=str(worker.get("error") or ""),
                fallback_reason=str(execution.get("fallback_reason") or ""),
                consumed_result_ids=sorted(consumed),
                preserved=task_id in preserved,
                replanned=replanned_plan and task_id not in preserved,
            )
        )
        for dependency in _as_list(task.get("dependencies")):
            source_id = str(dependency)
            if source_id and source_id in task_ids:
                edges.append(
                    AgentGraphEdge(
                        source_task_id=source_id,
                        destination_task_id=task_id,
                        consumed=source_id in consumed,
                    )
                )
    return nodes, edges


def _compact_plan_history(value: Any) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for item in _as_list(value):
        if not isinstance(item, dict):
            continue
        tasks = [task for task in _as_list(item.get("tasks")) if isinstance(task, dict)]
        plans.append(
            {
                "plan_id": str(item.get("plan_id") or ""),
                "plan_version": item.get("plan_version"),
                "source": str(item.get("source") or ""),
                "parent_plan_id": str(item.get("parent_plan_id") or ""),
                "replan_round": item.get("replan_round", 0),
                "replan_reason": str(item.get("replan_reason") or ""),
                "preserved_task_ids": [
                    str(task_id) for task_id in _as_list(item.get("preserved_task_ids"))
                ],
                "task_count": len(tasks),
            }
        )
    return plans


def _walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _diagnostic_totals(agent_results: dict[str, Any]) -> tuple[int | None, int | None]:
    raw_total = 0
    accepted_total = 0
    raw_seen = False
    accepted_seen = False
    for worker in agent_results.values():
        if not isinstance(worker, dict):
            continue
        for tool in _as_list(worker.get("tool_results")):
            if not isinstance(tool, dict):
                continue
            for item in _walk_dicts(tool.get("payload")):
                for key in ("backend_raw_count", "raw_count"):
                    if isinstance(item.get(key), int):
                        raw_total += int(item[key])
                        raw_seen = True
                        break
                for key in ("accepted_count", "filtered_count"):
                    if isinstance(item.get(key), int):
                        accepted_total += int(item[key])
                        accepted_seen = True
                        break
    return (raw_total if raw_seen else None, accepted_total if accepted_seen else None)


def _evidence_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        if isinstance(value.get("evidence"), list):
            return [item for item in value["evidence"] if isinstance(item, dict)]
        return [item for item in value.values() if isinstance(item, dict)]
    return []


def _build_memory_flow(
    store_probe: dict[str, Any],
    agent_results: dict[str, Any],
    compression: dict[str, Any],
    evidence: Any,
) -> tuple[list[VisualizationFlowMetric], list[dict[str, Any]]]:
    raw_count, accepted_count = _diagnostic_totals(agent_results)
    matching = store_probe.get("matching_count")
    retained = compression.get("retained_record_count")
    evidence_items = _evidence_items(evidence)
    historical_evidence = [
        item
        for item in evidence_items
        if str(item.get("source_kind") or item.get("source_type") or "").lower()
        in {"long_term_memory", "memory", "historical_memory"}
        or item.get("memory_id")
    ]
    flow = [
        VisualizationFlowMetric(
            metric_id="store_inventory",
            label="Store Probe 库存",
            value=int(matching) if isinstance(matching, int) else None,
            detail=str(store_probe.get("status") or ""),
        ),
        VisualizationFlowMetric(
            metric_id="backend_retrieved",
            label="后端检索数量",
            value=raw_count,
        ),
        VisualizationFlowMetric(
            metric_id="accepted_records",
            label="过滤后接受数量",
            value=accepted_count,
        ),
        VisualizationFlowMetric(
            metric_id="context_retained",
            label="SynthesisContext 保留数量",
            value=int(retained) if isinstance(retained, int) else None,
        ),
        VisualizationFlowMetric(
            metric_id="final_evidence",
            label="最终历史 Evidence",
            value=(len(historical_evidence) if evidence_items else None),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for item in evidence_items:
        span = _as_dict(item.get("source_span"))
        segment_id = item.get("segment_id") or span.get("segment_id")
        start = item.get("start_offset")
        end = item.get("end_offset")
        if start is None:
            start = span.get("start_offset")
        if end is None:
            end = span.get("end_offset")
        rows.append(
            {
                "evidence_id": str(item.get("evidence_id") or item.get("id") or ""),
                "source_kind": str(
                    item.get("source_kind") or item.get("source_type") or ""
                ),
                "source_time": str(item.get("source_time") or ""),
                "segment_id": str(segment_id or ""),
                "offset": (
                    f"{start}:{end}" if start is not None and end is not None else ""
                ),
                "worker_id": str(item.get("worker_id") or item.get("task_id") or ""),
                "memory_id": str(item.get("memory_id") or ""),
            }
        )
    return flow, rows


def _review_projection(review: dict[str, Any]) -> dict[str, Any]:
    result = _as_dict(review.get("run_result"))
    rounds = [item for item in _as_list(result.get("rounds")) if isinstance(item, dict)]
    projected_rounds: list[dict[str, Any]] = []
    for item in rounds:
        red = _as_dict(item.get("red_report"))
        judge = _as_dict(item.get("judge_decision"))
        blue = _as_dict(item.get("blue_result"))
        issues = _as_list(red.get("issues"))
        dispositions = _as_list(red.get("issue_dispositions"))
        repairs = [
            row for row in _as_list(blue.get("repairs")) if isinstance(row, dict)
        ]
        judge_disposition = str(judge.get("disposition") or "")
        projected_rounds.append(
            {
                "round_index": item.get("round_index"),
                "red_candidate_issue_count": red.get(
                    "candidate_issue_count", len(issues)
                ),
                "red_issue_count": len(issues),
                "red_kept_count": sum(
                    1
                    for row in dispositions
                    if isinstance(row, dict) and row.get("disposition") == "kept"
                ),
                "red_dismissed_count": sum(
                    1
                    for row in dispositions
                    if isinstance(row, dict) and row.get("disposition") == "dismissed"
                ),
                "judge_accept_count": 1 if judge_disposition == "accept" else 0,
                "judge_non_accept_count": 1
                if judge_disposition in {"repair", "stop_with_warnings"}
                else 0,
                "blue_applied_count": sum(1 for row in repairs if row.get("applied")),
                "deferred_count": len(_as_list(red.get("deferred_issues"))),
                "remaining_issue_count": len(
                    _as_list(judge.get("remaining_issue_ids"))
                ),
                "judge_disposition": judge_disposition,
                "judge_score": judge.get("score"),
            }
        )
    return {
        "enabled": bool(result.get("enabled")) if result else False,
        "shadow_mode": result.get("shadow_mode"),
        "termination": str(result.get("termination") or ""),
        "recommended_candidate_index": result.get("recommended_candidate_index"),
        "round_count": len(rounds),
        "rounds": projected_rounds,
    }


def _evaluation_projection(evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics = _as_dict(evaluation.get("metrics"))
    if evaluation.get("mode") == "lite":
        candidates = _as_list(metrics.get("candidates"))
        rows = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "summary_id": item.get("summary_id"),
                    "display_name": item.get("display_name"),
                    "status": item.get("status"),
                    "support_scope": item.get("support_scope"),
                    "metrics": sanitize_value(_as_dict(item.get("metrics"))),
                    "claim_count": len(_as_list(item.get("claims"))),
                    "missing_reference_fact_ids": sanitize_value(
                        _as_list(item.get("missing_reference_fact_ids"))
                    ),
                    "unsupported_claim_ids": sanitize_value(
                        _as_list(item.get("unsupported_claim_ids"))
                    ),
                    "warnings": sanitize_value(_as_list(item.get("warnings"))),
                }
            )
        reference = _as_dict(metrics.get("reference"))
        return {
            "mode": "lite",
            "comparison": bool(evaluation.get("comparison")),
            "status": str(evaluation.get("status") or "disabled"),
            "rows": rows,
            "baseline_summary_id": metrics.get("baseline_summary_id"),
            "deltas_from_baseline": sanitize_value(
                _as_dict(metrics.get("deltas_from_baseline"))
            ),
            "reference_fact_count": len(_as_list(reference.get("facts"))),
            "execution_stats": sanitize_value(_as_dict(metrics.get("execution_stats"))),
            "warnings": sanitize_value(_as_list(metrics.get("warnings"))),
            "personalization": {"status": "disabled", "macro": {}, "per_target": []},
        }
    if evaluation.get("comparison"):
        return {
            "comparison": True,
            "status": str(evaluation.get("status") or ""),
            "rows": sanitize_value(_as_list(metrics.get("rows"))),
            "deltas_from_first_summary": sanitize_value(
                _as_dict(metrics.get("deltas_from_first_summary"))
            ),
            "scope_deltas_from_first_summary": sanitize_value(
                _as_dict(metrics.get("scope_deltas_from_first_summary"))
            ),
            "scope_problem_claims": sanitize_value(
                _as_dict(metrics.get("scope_problem_claims"))
            ),
            "scope_content_differences": sanitize_value(
                _as_dict(metrics.get("scope_content_differences"))
            ),
            "qualitative_comparison": sanitize_value(
                _as_dict(metrics.get("qualitative_comparison"))
            ),
            "verified_content_differences": sanitize_value(
                _as_dict(metrics.get("verified_content_differences"))
            ),
            "comparison_interpretation_status": str(
                metrics.get("comparison_interpretation_status") or "disabled"
            ),
            "baseline_summary_id": metrics.get("baseline_summary_id"),
            "successful_summary_count": metrics.get("successful_summary_count"),
            "failed_summary_count": metrics.get("failed_summary_count"),
            "total_model_call_count": metrics.get("total_model_call_count"),
        }
    personalization = _as_dict(evaluation.get("personalization"))
    personalization_metrics = _as_dict(personalization.get("metrics"))
    return {
        "comparison": False,
        "status": str(evaluation.get("status") or "disabled"),
        "overall": {
            "core": sanitize_value(_as_dict(metrics.get("core_metrics"))),
            "history": sanitize_value(_as_dict(metrics.get("history_metrics"))),
            "transition": sanitize_value(_as_dict(metrics.get("transition_metrics"))),
            "safety": sanitize_value(_as_dict(metrics.get("safety_metrics"))),
            "conciseness": sanitize_value(_as_dict(metrics.get("conciseness_metrics"))),
        },
        "personalization": {
            "status": str(personalization.get("status") or "disabled"),
            "macro": sanitize_value(
                _as_dict(personalization_metrics.get("macro_metrics"))
            ),
            "per_target": sanitize_value(
                _as_list(personalization_metrics.get("per_target"))
            ),
        },
    }


def _stage_current_id(kind: str, raw_stage: str) -> str:
    stage = raw_stage.lower()
    if kind == "ingest":
        return (
            stage
            if stage in {item[0] for item in STAGE_TEMPLATES[kind]}
            else "validation"
        )
    if kind == "evaluation_compare":
        if stage.startswith("scoring_summary_"):
            return "summary_scoring"
        aliases = {
            "validating": "validation",
            "building_source_snapshot": "source_snapshot",
            "annotating_target_sources": "target_annotation",
            "explaining_comparison": "comparison_interpretation",
            "aggregating": "aggregating",
            "lite_reference_facts": "source_snapshot",
            "lite_scoring": "summary_scoring",
            "lite_aggregating": "aggregating",
        }
        return aliases.get(stage, "validation")
    aliases = {
        "pending": "store_probe",
        "queued": "store_probe",
        "running": "store_probe",
        "planning": "planning",
        "replanning": "planning",
        "executing": "agent_execution",
        "collecting": "agent_execution",
        "synthesizing": "synthesis",
        "reviewing": "review",
        "evaluating": "evaluation",
    }
    return aliases.get(stage, "store_probe")


def _stage_artifact_statuses(
    kind: str,
    run_dir: Path,
    snapshot: Any,
    run_info: dict[str, Any],
) -> dict[str, str]:
    values: dict[str, str] = {}
    if kind == "ingest":
        stage_report = _as_dict(snapshot.ingest.get("stage_report"))
        current = str(stage_report.get("stage") or run_info.get("stage") or "")
        status = _normalize_status(stage_report.get("status"), default="pending")
        for stage_id, _ in STAGE_TEMPLATES[kind]:
            if (
                run_dir
                / {
                    "preprocessing": "e2e_preprocess_info.json",
                    "extracting": "typed_memories.json",
                    "deduplicating": "typed_memory_dedup_report.json",
                    "writing": "ingest_write_journal.json",
                    "verifying": "ingest_readback_report.json",
                }.get(stage_id, "web_run_info.json")
            ).is_file():
                values[stage_id] = "success"
        if current in {item[0] for item in STAGE_TEMPLATES[kind]} and status in {
            "running",
            "warning",
            "failed",
        }:
            values[current] = status
        return values
    if kind == "evaluation_compare":
        result = _load_json(run_dir / "evaluation_compare_run_result.json", {}, {})
        result = _as_dict(result)
        if (run_dir / "evaluation_lite_reference.json").is_file():
            values["validation"] = "success"
            values["source_snapshot"] = "success"
            values["target_annotation"] = "skipped"
        if (run_dir / "evaluation_lite_result.json").is_file():
            lite_result = _as_dict(
                _load_json(run_dir / "evaluation_lite_result.json", {}, {})
            )
            statuses = {
                _normalize_status(item.get("status"))
                for item in _as_list(lite_result.get("candidates"))
                if isinstance(item, dict)
            }
            values["summary_scoring"] = (
                "failed"
                if statuses == {"failed"}
                else "warning"
                if "failed" in statuses
                else "success"
            )
            values["aggregating"] = _normalize_status(
                lite_result.get("status"), default="warning"
            )
            values["comparison_interpretation"] = "skipped"
            return values
        if (run_dir / "evaluation_source_snapshot.json").is_file():
            values["source_snapshot"] = "success"
            values["validation"] = "success"
        if (run_dir / "evaluation_personalization_source_annotations.json").is_file():
            values["target_annotation"] = "success"
        elif not run_info.get("target_entities"):
            values["target_annotation"] = "skipped"
        summaries = _as_list(result.get("summaries"))
        if summaries:
            statuses = {
                _normalize_status(item.get("status"))
                for item in summaries
                if isinstance(item, dict)
            }
            values["summary_scoring"] = (
                "failed"
                if statuses == {"failed"}
                else "warning"
                if "failed" in statuses
                else "success"
            )
        if (run_dir / "evaluation_comparison_metrics.json").is_file():
            values["aggregating"] = "success"
            interpretation_status = str(
                _as_dict(
                    _load_json(run_dir / "evaluation_comparison_metrics.json", {}, {})
                ).get("comparison_interpretation_status")
                or "disabled"
            )
            values["comparison_interpretation"] = (
                "skipped"
                if interpretation_status == "disabled"
                else _normalize_status(interpretation_status, default="warning")
            )
        if str(result.get("status") or "") == "failed":
            failed_stage = _stage_current_id(
                "evaluation_compare", str(result.get("stage") or "")
            )
            values[failed_stage] = "failed"
        return values
    probe_status = str(_as_dict(snapshot.store_probe).get("status") or "")
    if probe_status:
        values["store_probe"] = _normalize_status(probe_status, default="warning")
    if _as_dict(snapshot.planner.get("plan")):
        values["planning"] = "success"
    if snapshot.workers:
        worker_statuses = {
            _normalize_status(item.get("status")) for item in snapshot.workers
        }
        values["agent_execution"] = (
            "failed"
            if worker_statuses == {"failed"}
            else "warning"
            if "failed" in worker_statuses
            else "success"
        )
    if snapshot.formal_summary:
        values["synthesis"] = "success"
    review = _as_dict(snapshot.review.get("run_result"))
    if review:
        values["review"] = (
            "success" if review.get("termination") == "accepted" else "warning"
        )
    else:
        values["review"] = "skipped"
    evaluation_status = str(_as_dict(snapshot.evaluation).get("status") or "disabled")
    values["evaluation"] = _normalize_status(evaluation_status, default="skipped")
    return values


def _build_stages(
    kind: str,
    run_dir: Path,
    snapshot: Any,
    run_info: dict[str, Any],
    events: list[dict[str, Any]],
) -> list[VisualizationStage]:
    template = STAGE_TEMPLATES.get(kind, ())
    if not template:
        return []
    current_id = _stage_current_id(kind, str(run_info.get("stage") or ""))
    artifact_statuses = _stage_artifact_statuses(kind, run_dir, snapshot, run_info)
    failed_from_artifact = next(
        (
            stage_id
            for stage_id, status in artifact_statuses.items()
            if status == "failed"
        ),
        "",
    )
    if failed_from_artifact:
        current_id = failed_from_artifact
    current_index = next(
        (i for i, item in enumerate(template) if item[0] == current_id), 0
    )
    final_status = str(
        run_info.get("status")
        or getattr(snapshot.status, "value", snapshot.status)
        or ""
    )
    trace_groups = {
        "planning": {"plan.created", "plan.replanning", "plan.replanned"},
        "agent_execution": {
            "task.started",
            "task.finished",
            "task.ad_hoc_started",
            "task.ad_hoc_finished",
        },
        "synthesis": {"context.compressed"},
        "review": {"review.started", "review.round_started", "review.finished"},
        "evaluation": {"evaluation.started", "evaluation.finished"},
    }
    ingest_preprocess: dict[str, Any] = {}
    ingest_typed: list[Any] = []
    ingest_dedup: dict[str, Any] = {}
    ingest_stage_report: dict[str, Any] = {}
    ingest_readback: dict[str, Any] = {}
    comparison_metrics: dict[str, Any] = {}
    if kind == "ingest":
        ingest_preprocess = _as_dict(
            _load_json(run_dir / "e2e_preprocess_info.json", {}, {})
        )
        ingest_typed = _as_list(_load_json(run_dir / "typed_memories.json", {}, []))
        ingest_dedup = _as_dict(
            _load_json(run_dir / "typed_memory_dedup_report.json", {}, {})
        )
        ingest_stage_report = _as_dict(snapshot.ingest.get("stage_report"))
        ingest_readback = _as_dict(snapshot.ingest.get("readback"))
    elif kind == "evaluation_compare":
        comparison_metrics = _as_dict(snapshot.evaluation.get("metrics"))
    stages: list[VisualizationStage] = []
    for index, (stage_id, label) in enumerate(template):
        status = artifact_statuses.get(stage_id, "pending")
        if status == "pending":
            if final_status in {"completed", "completed_with_warnings"}:
                status = (
                    "skipped"
                    if stage_id in {"review", "evaluation", "target_annotation"}
                    else "success"
                )
            elif final_status == "failed":
                status = (
                    "success"
                    if index < current_index
                    else "failed"
                    if index == current_index
                    else "pending"
                )
            elif index < current_index:
                status = "success"
            elif index == current_index:
                status = "running"
        started_at, finished_at, duration = _trace_bounds(
            events, trace_groups.get(stage_id, set())
        )
        counters: dict[str, Any] = {}
        if kind == "ingest":
            if stage_id == "preprocessing":
                counters = {
                    "sources": ingest_preprocess.get("history_count"),
                    "chunks": ingest_preprocess.get("chunk_count"),
                }
            elif stage_id == "extracting":
                counters = {
                    "records": sum(
                        len(_as_list(item.get("typed_memories")))
                        for item in ingest_typed
                        if isinstance(item, dict)
                    )
                }
            elif stage_id == "deduplicating":
                outcomes = ingest_dedup.get("candidate_outcomes")
                counters = {
                    "removed": len(_as_list(ingest_dedup.get("removed_records"))),
                    "candidates": (
                        len(outcomes) if isinstance(outcomes, (list, dict)) else None
                    ),
                }
            elif stage_id == "writing":
                counters = {
                    "add calls": ingest_stage_report.get("add_call_count"),
                    "operations": ingest_stage_report.get("operation_count"),
                }
            elif stage_id == "verifying":
                counters = {
                    "expected": ingest_readback.get("expected_operation_count"),
                    "recovered": ingest_readback.get("recovered_operation_count"),
                    "missing": len(
                        _as_list(ingest_readback.get("missing_operation_ids"))
                    ),
                }
        elif kind == "evaluation_compare":
            if snapshot.evaluation.get("mode") == "lite":
                lite_candidates = _as_list(comparison_metrics.get("candidates"))
                if stage_id == "source_snapshot":
                    counters = {
                        "facts": len(
                            _as_list(
                                _as_dict(comparison_metrics.get("reference")).get(
                                    "facts"
                                )
                            )
                        )
                    }
                elif stage_id == "summary_scoring":
                    counters = {
                        "success": sum(
                            1
                            for item in lite_candidates
                            if isinstance(item, dict) and item.get("status") != "failed"
                        ),
                        "failed": sum(
                            1
                            for item in lite_candidates
                            if isinstance(item, dict) and item.get("status") == "failed"
                        ),
                    }
                elif stage_id == "aggregating":
                    counters = {
                        "model calls": _as_dict(
                            comparison_metrics.get("execution_stats")
                        ).get("model_call_count")
                    }
            else:
                if stage_id == "source_snapshot":
                    counters = {
                        "facts": comparison_metrics.get("source_fact_count"),
                        "relations": comparison_metrics.get("fact_relation_count"),
                    }
                elif stage_id == "summary_scoring":
                    counters = {
                        "success": comparison_metrics.get("successful_summary_count"),
                        "failed": comparison_metrics.get("failed_summary_count"),
                    }
                elif stage_id == "aggregating":
                    counters = {
                        "model calls": comparison_metrics.get("total_model_call_count")
                    }
        elif stage_id == "agent_execution":
            counters = {
                "planned": len(snapshot.workers)
                or len(snapshot.planner.get("plan", {}).get("tasks", [])),
                "success": sum(
                    1
                    for item in snapshot.workers
                    if _normalize_status(item.get("status")) == "success"
                ),
            }
        elif stage_id == "store_probe":
            counters = {"matching": snapshot.store_probe.get("matching_count")}
        elif stage_id == "planning":
            counters = {
                "tasks": len(
                    _as_list(_as_dict(snapshot.planner.get("plan")).get("tasks"))
                )
            }
        elif stage_id == "review":
            counters = {
                "rounds": len(
                    _as_list(_as_dict(snapshot.review.get("run_result")).get("rounds"))
                )
            }
        elif stage_id == "evaluation" and snapshot.evaluation:
            counters = {"status": snapshot.evaluation.get("status")}
        stages.append(
            VisualizationStage(
                stage_id=stage_id,
                label=label,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=duration,
                counters=sanitize_value(counters),
            )
        )
    return stages


def _overview(
    run_info: dict[str, Any],
    snapshot: Any,
    graph_nodes: list[AgentGraphNode],
    memory_flow: list[VisualizationFlowMetric],
    plan_history: list[dict[str, Any]],
    review: dict[str, Any],
    evaluation: dict[str, Any],
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    finished_at = run_info.get("finished_at")
    if not finished_at and events:
        finished_at = events[-1].get("timestamp")
    model_calls = sum(
        item.model_call_count
        for item in graph_nodes
        if item.model_call_count is not None
    )
    tool_calls = sum(
        item.tool_call_count for item in graph_nodes if item.tool_call_count is not None
    )
    model_known = any(item.model_call_count is not None for item in graph_nodes)
    tool_known = any(item.tool_call_count is not None for item in graph_nodes)
    coverage = _as_dict(snapshot.coverage)
    complete = coverage.get("complete")
    if complete is None:
        complete = coverage.get("coverage_complete")
    if not model_known and isinstance(evaluation.get("total_model_call_count"), int):
        model_calls = int(evaluation["total_model_call_count"])
        model_known = True
    return sanitize_value(
        {
            "status": str(
                run_info.get("status")
                or getattr(snapshot.status, "value", snapshot.status)
                or "unknown"
            ),
            "stage": str(run_info.get("stage") or ""),
            "started_at": str(run_info.get("started_at") or ""),
            "finished_at": str(run_info.get("finished_at") or ""),
            "duration_ms": _duration_ms(run_info.get("started_at"), finished_at),
            "planned_task_count": len(graph_nodes) if graph_nodes else None,
            "successful_task_count": sum(
                item.status == "success" for item in graph_nodes
            )
            if graph_nodes
            else None,
            "model_call_count": model_calls if model_known else None,
            "tool_call_count": tool_calls if tool_known else None,
            "replan_count": max(0, len(plan_history) - 1) if plan_history else 0,
            "coverage_complete": complete,
            "memory_flow": {item.metric_id: item.value for item in memory_flow},
            "review_status": review.get("termination")
            or ("disabled" if not review.get("enabled") else "running"),
            "evaluation_status": evaluation.get("status") or "disabled",
            "store": sanitize_value(_as_dict(run_info.get("store"))),
            "filters": sanitize_value(_as_dict(run_info.get("filters"))),
            "memory_access": str(run_info.get("memory_access") or ""),
        }
    )


def build_run_visualization(run_dir: Path) -> WebRunVisualization:
    """只读构建单次运行可视化；不会初始化模型、Memory 或写回产物。"""
    root = run_dir.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"run output directory does not exist: {root}")
    errors: dict[str, str] = {}
    warnings: list[str] = []
    run_info = _as_dict(_load_json(root / "web_run_info.json", errors, {}))
    snapshot = read_artifact_snapshot(root)
    errors.update(
        {
            name: error
            for name, error in snapshot.errors.items()
            if error != "产物不存在"
        }
    )
    run_type = str(run_info.get("run_type") or "")
    kind = _run_kind(run_type, root)
    if not run_type:
        run_type = kind
    events = _read_trace(root, warnings, errors)
    plan = _as_dict(snapshot.planner.get("plan"))
    raw_agent_results = _as_dict(_load_json(root / "agent_results.json", errors, {}))
    raw_working_state = _as_dict(_load_json(root / "working_state.json", errors, {}))
    graph_nodes, graph_edges = (
        _build_graph(plan, raw_agent_results, raw_working_state)
        if kind == "summarize"
        else ([], [])
    )
    raw_history = _load_json(root / "plan_history.json", errors, [])
    plan_history = _compact_plan_history(raw_history)
    memory_flow, evidence_rows = _build_memory_flow(
        _as_dict(snapshot.store_probe),
        raw_agent_results,
        _as_dict(snapshot.compression),
        snapshot.evidence,
    )
    review = _review_projection(_as_dict(snapshot.review))
    evaluation = _evaluation_projection(_as_dict(snapshot.evaluation))
    stages = _build_stages(kind, root, snapshot, run_info, events)
    warnings.extend(snapshot.warnings)
    if kind in {"ingest", "evaluation_compare"}:
        warnings.append("该运行类型不产生 Agent DAG；Agent 图页仅展示对应阶段流程。")
    return WebRunVisualization(
        run_id=str(run_info.get("run_id") or snapshot.run_id or ""),
        run_type=run_type or "unknown",
        output_dir=root,
        overview=_overview(
            run_info,
            snapshot,
            graph_nodes,
            memory_flow,
            plan_history,
            review,
            evaluation,
            events,
        ),
        stages=stages,
        graph_nodes=graph_nodes,
        graph_edges=graph_edges,
        plan_history=plan_history,
        timeline=_build_timeline(events),
        memory_flow=memory_flow,
        evidence_rows=evidence_rows,
        review=review,
        evaluation=evaluation,
        warnings=list(dict.fromkeys(str(item) for item in warnings if item)),
        errors=errors,
    )
