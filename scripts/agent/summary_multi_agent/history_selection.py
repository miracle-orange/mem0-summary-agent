"""Build a read-only audit of the history records visible to synthesis."""

from __future__ import annotations

from typing import Any

from .schemas import AgentStatus, SummaryWorkingState, TaskType


HISTORY_SELECTION_REPORT_VERSION = (
    "summary_multi_agent.history_selection_report.v2"
)


def _record_ids(records: object) -> list[str]:
    """Return stable, unique memory IDs from a model or tool payload."""

    if not isinstance(records, (list, tuple)):
        return []
    values: list[str] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        memory_id = str(item.get("memory_id") or "").strip()
        if memory_id and memory_id not in values:
            values.append(memory_id)
    return values


def build_history_selection_report(
    state: SummaryWorkingState,
) -> dict[str, Any]:
    """Distinguish unplanned, failed, empty and record-bearing history paths."""

    planned = [
        task
        for task in (state.plan.tasks if state.plan is not None else ())
        if task.task_type == TaskType.SEMANTIC_MEMORY
    ]
    task_reports: list[dict[str, Any]] = []
    selected_union: list[str] = []
    available_union: list[str] = []
    merge_rows: list[dict[str, Any]] = []
    total_occurrences = 0
    total_unique_ids = 0
    total_retained = 0

    for task in planned:
        result = state.agent_results.get(task.task_id)
        if result is None:
            task_reports.append(
                {
                    "task_id": task.task_id,
                    "status": "not_executed",
                    "agent_status": None,
                    "selection_mode": None,
                    "available_memory_ids": [],
                    "requested_memory_ids": None,
                    "selected_memory_ids": [],
                    "equivalent_record_merges": [],
                    "error": "semantic task has no terminal AgentResult",
                    "execution_origin": None,
                    "retrieval_complete": False,
                    "retrieval_stats": {},
                    "record_selection_completed": False,
                    "record_selection_origin": None,
                    "reused_search_count": 0,
                    "new_search_count": 0,
                }
            )
            continue

        payload = result.payload if isinstance(result.payload, dict) else {}
        selected = payload.get("selected_memory_ids")
        if not isinstance(selected, list):
            selected = _record_ids(payload.get("records"))
        available = payload.get("available_record_ids")
        if not isinstance(available, list):
            available = list(selected)
        requested = payload.get("requested_memory_ids")
        if not isinstance(requested, list):
            requested = None
        merges = payload.get("equivalent_record_merges")
        if not isinstance(merges, list):
            merges = []
        retrieval_stats = payload.get("retrieval_stats")
        if not isinstance(retrieval_stats, dict):
            retrieval_stats = {}
        total_occurrences += int(
            retrieval_stats.get("record_occurrence_count") or len(available)
        )
        total_unique_ids += int(
            retrieval_stats.get("unique_memory_id_count") or len(available)
        )
        total_retained += int(
            retrieval_stats.get("retained_record_count") or len(selected)
        )
        execution = result.worker_execution
        execution_origin = payload.get("execution_origin")
        if not isinstance(execution_origin, str):
            if selection_mode := payload.get("selection_mode"):
                execution_origin = str(selection_mode)
            elif execution is not None:
                execution_origin = execution.mode.value
            else:
                execution_origin = "legacy_rule"

        if result.status != AgentStatus.SUCCESS:
            status = "failed"
        elif not selected:
            status = "empty"
        else:
            status = "records"
        selection_mode = payload.get("selection_mode")
        if not isinstance(selection_mode, str):
            selection_mode = "legacy_observation_all"
        task_reports.append(
            {
                "task_id": task.task_id,
                "status": status,
                "agent_status": result.status.value,
                "selection_mode": selection_mode,
                "available_memory_ids": list(dict.fromkeys(available)),
                "requested_memory_ids": requested,
                "selected_memory_ids": list(dict.fromkeys(selected)),
                "equivalent_record_merges": merges,
                "record_version_conflicts": payload.get(
                    "record_version_conflicts", []
                ),
                "execution_origin": execution_origin,
                "retrieval_complete": payload.get("retrieval_complete"),
                "retrieval_stats": retrieval_stats,
                "record_selection_completed": bool(
                    payload.get("record_selection_completed", False)
                ),
                "record_selection_origin": payload.get(
                    "record_selection_origin"
                ),
                "reused_search_count": int(
                    payload.get("reused_search_count") or 0
                ),
                "new_search_count": int(
                    payload.get("new_search_count") or 0
                ),
                "error": result.error,
            }
        )
        for memory_id in available:
            if memory_id not in available_union:
                available_union.append(memory_id)
        for memory_id in selected:
            if memory_id not in selected_union:
                selected_union.append(memory_id)
        merge_rows.extend(item for item in merges if isinstance(item, dict))

    if not planned:
        overall_status = "not_planned"
    elif any(item["status"] == "records" for item in task_reports):
        overall_status = "records"
    elif any(item["status"] in {"failed", "not_executed"} for item in task_reports):
        overall_status = "unavailable"
    else:
        overall_status = "empty"

    visible_ids: list[str] = []
    if state.synthesis_context is not None:
        visible_ids.extend(state.synthesis_context.historical_records)
        for task_id, result in state.synthesis_context.worker_results.items():
            if not isinstance(result, dict):
                continue
            payload = result.get("payload")
            if not isinstance(payload, dict):
                continue
            for index, record in enumerate(payload.get("records", [])):
                if not isinstance(record, dict):
                    continue
                record_id = str(
                    record.get("memory_id")
                    or record.get("record_ref_id")
                    or f"{task_id}:record:{index}"
                )
                if record_id not in visible_ids:
                    visible_ids.append(record_id)
    return {
        "schema_version": HISTORY_SELECTION_REPORT_VERSION,
        "status": overall_status,
        "semantic_task_count": len(planned),
        "tasks": task_reports,
        "available_memory_ids": available_union,
        "selected_memory_ids": selected_union,
        "equivalent_record_merges": merge_rows,
        "record_occurrence_count": total_occurrences,
        "unique_memory_id_count": total_unique_ids,
        "retained_record_count": total_retained,
        "synthesis_visible_history_ids": visible_ids,
        "synthesis_visible_history_count": len(visible_ids),
    }
