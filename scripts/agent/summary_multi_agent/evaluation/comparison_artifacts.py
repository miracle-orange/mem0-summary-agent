"""持久化独立多纪要横评产物，供 CLI、Web 和审计共用。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .comparison import EvaluationComparisonResult
from .comparison_scope import build_comparison_basis
from .schemas import EvaluationCompareRequest


def _write_json(path: Path, value: Any) -> None:
    """以 UTF-8 和稳定缩进写入 JSON。"""
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_comparison_artifacts(
    output_dir: Path,
    request: EvaluationCompareRequest,
    result: EvaluationComparisonResult,
) -> dict[str, str]:
    """保存共享 Snapshot、逐份评分和横向报告。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = result.snapshot_bundle.snapshot
    comparison_basis = result.comparison_basis or build_comparison_basis(
        snapshot_id=snapshot.snapshot_id,
        source_facts=snapshot.source_facts,
        fact_relations=snapshot.fact_relations,
    )
    _write_json(
        output_dir / "evaluation_compare_request.json",
        request.model_dump(mode="json"),
    )
    _write_json(
        output_dir / "evaluation_source_snapshot.json",
        snapshot.model_dump(mode="json"),
    )
    _write_json(
        output_dir / "evaluation_source_relations.json",
        [item.model_dump(mode="json") for item in snapshot.fact_relations],
    )
    _write_json(
        output_dir / "evaluation_comparison_basis.json",
        comparison_basis.model_dump(mode="json"),
    )
    _write_json(
        output_dir / "evaluation_comparison_metrics.json",
        result.comparison_metrics,
    )
    (output_dir / "evaluation_comparison_report.md").write_text(
        result.report_markdown,
        encoding="utf-8",
    )
    _write_json(
        output_dir / "evaluation_source_model_trace.json",
        list(result.source_model_trace),
    )
    _write_json(
        output_dir / "evaluation_source_personalization_model_trace.json",
        list(result.source_personalization_model_trace),
    )
    _write_json(
        output_dir / "evaluation_comparison_interpretation_model_trace.json",
        list(result.comparison_interpretation_model_trace),
    )
    summary_paths: dict[str, dict[str, str]] = {}
    for item in result.summaries:
        summary_dir = output_dir / "summaries" / item.summary_id
        summary_dir.mkdir(parents=True, exist_ok=True)
        _write_json(
            summary_dir / "evaluation_model_trace.json",
            list(item.model_trace),
        )
        _write_json(
            summary_dir / "personalization_model_trace.json",
            list(item.personalization_model_trace),
        )
        _write_json(
            summary_dir / "evaluation_scope_model_trace.json",
            list(item.scope_model_trace),
        )
        if item.artifacts is not None:
            artifacts = item.artifacts
            _write_json(
                summary_dir / "summary_facts.json",
                [value.model_dump(mode="json") for value in artifacts.summary_facts],
            )
            _write_json(
                summary_dir / "fact_matches.json",
                [value.model_dump(mode="json") for value in artifacts.matches],
            )
            _write_json(
                summary_dir / "evaluation_metrics.json",
                artifacts.metrics.model_dump(mode="json"),
            )
            (summary_dir / "evaluation_report.md").write_text(
                artifacts.report_markdown,
                encoding="utf-8",
            )
            personalization = artifacts.personalization
            if personalization is not None:
                _write_json(
                    summary_dir / "personalization_source_annotations.json",
                    [
                        value.model_dump(mode="json")
                        for value in personalization.source_annotations
                    ],
                )
                _write_json(
                    summary_dir / "personalization_claim_annotations.json",
                    [
                        value.model_dump(mode="json")
                        for value in personalization.claim_annotations
                    ],
                )
                _write_json(
                    summary_dir / "personalization_metrics.json",
                    personalization.metrics.model_dump(mode="json"),
                )
                (summary_dir / "personalization_report.md").write_text(
                    personalization.report_markdown,
                    encoding="utf-8",
                )
            if item.scope_result is not None:
                scope_payload = item.scope_result.model_dump(
                    mode="json",
                    exclude={"all_sources_matches", "current_only_matches"},
                )
                _write_json(
                    summary_dir / "evaluation_scope_metrics.json",
                    scope_payload,
                )
                _write_json(
                    summary_dir / "fact_matches_all_sources.json",
                    [
                        value.model_dump(mode="json")
                        for value in item.scope_result.all_sources_matches
                    ],
                )
                _write_json(
                    summary_dir / "fact_matches_current_only.json",
                    [
                        value.model_dump(mode="json")
                        for value in item.scope_result.current_only_matches
                    ],
                )
        else:
            _write_json(
                summary_dir / "evaluation_failure.json",
                {"status": "failed", "error": item.error},
            )
        summary_paths[item.summary_id] = {
            "directory": str(summary_dir),
            "status": item.status,
        }
    run_result = {
        "schema_version": "summary_multi_agent.evaluation_compare_run_result.v1",
        "status": result.status,
        "snapshot_id": snapshot.snapshot_id,
        "warnings": list(result.warnings),
        "evaluation_status": (
            "failed" if result.status == "failed" else "completed"
        ),
        "personalization_evaluation_status": (
            "disabled"
            if not request.target_entities
            else (
                "failed"
                if all(
                    item.artifacts is None
                    or item.artifacts.personalization is None
                    or item.artifacts.personalization.metrics.status.value
                    == "failed"
                    for item in result.summaries
                )
                else (
                    "completed_with_warnings"
                    if any(
                        item.artifacts is None
                        or item.artifacts.personalization is None
                        or item.artifacts.personalization.metrics.status.value
                        == "failed"
                        for item in result.summaries
                    )
                    else "completed"
                )
            )
        ),
        "comparison_interpretation_status": result.comparison_metrics.get(
            "comparison_interpretation_status", "disabled"
        ),
        "summaries": [
            {
                "summary_id": item.summary_id,
                "display_name": item.display_name,
                "status": item.status,
                "error": item.error,
                "artifact_dir": str(output_dir / "summaries" / item.summary_id),
            }
            for item in result.summaries
        ],
        "artifact_paths": {
            "evaluation_compare_request.json": str(
                output_dir / "evaluation_compare_request.json"
            ),
            "evaluation_source_snapshot.json": str(
                output_dir / "evaluation_source_snapshot.json"
            ),
            "evaluation_source_relations.json": str(
                output_dir / "evaluation_source_relations.json"
            ),
            "evaluation_comparison_basis.json": str(
                output_dir / "evaluation_comparison_basis.json"
            ),
            "evaluation_comparison_metrics.json": str(
                output_dir / "evaluation_comparison_metrics.json"
            ),
            "evaluation_comparison_report.md": str(
                output_dir / "evaluation_comparison_report.md"
            ),
        },
        "summary_artifacts": summary_paths,
    }
    _write_json(
        output_dir / "evaluation_compare_run_result.json",
        run_result,
    )
    return run_result["artifact_paths"]


def write_comparison_failure_artifacts(
    output_dir: Path,
    request: EvaluationCompareRequest,
    *,
    stage: str,
    error: str,
    source_model_trace: tuple[dict[str, Any], ...] = (),
    source_personalization_model_trace: tuple[dict[str, Any], ...] = (),
    source_execution_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    """在 Snapshot 失败时保留请求、阶段、统计和本地模型审计 trace。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "evaluation_compare_request.json",
        request.model_dump(mode="json"),
    )
    _write_json(
        output_dir / "evaluation_source_model_trace.json",
        list(source_model_trace),
    )
    _write_json(
        output_dir / "evaluation_source_personalization_model_trace.json",
        list(source_personalization_model_trace),
    )
    run_result = {
        "schema_version": "summary_multi_agent.evaluation_compare_run_result.v1",
        "status": "failed",
        "stage": stage,
        "error": error,
        "source_execution_stats": dict(source_execution_stats or {}),
        "artifact_paths": {
            "evaluation_compare_request.json": str(
                output_dir / "evaluation_compare_request.json"
            ),
            "evaluation_source_model_trace.json": str(
                output_dir / "evaluation_source_model_trace.json"
            ),
        },
    }
    _write_json(output_dir / "evaluation_compare_run_result.json", run_result)
    return run_result
