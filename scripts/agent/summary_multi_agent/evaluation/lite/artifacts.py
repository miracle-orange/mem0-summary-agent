"""精简评测的业务报告与私有模型审计产物。"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from ..schemas import EvaluationCompareRequest
from .schemas import LiteEvaluationResult, LiteReferenceBundle

LITE_ARTIFACT_NAMES = (
    "evaluation_lite_reference.json",
    "evaluation_lite_result.json",
    "evaluation_lite_report.md",
    "evaluation_lite_report.html",
    "evaluation_lite_model_trace.json",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _metric_text(metric: Any) -> str:
    if metric is None or metric.value is None:
        return "N/A"
    return f"{metric.value:.3f} ({metric.numerator}/{metric.denominator})"


def build_lite_report(result: LiteEvaluationResult) -> str:
    lines = [
        "# 会议纪要精简评测",
        "",
        "参考事实由评审模型从当前材料提取，并非人工金标准。材料支持率基于本次明确允许的证据范围。",
        "",
        "| 纪要 | 状态 | 当前关键事实覆盖率 | 材料支持率 | 关键事实关联率 |",
        "|---|---|---:|---:|---:|",
    ]
    for candidate in result.candidates:
        metrics = candidate.metrics
        lines.append(
            "| "
            + " | ".join(
                (
                    candidate.display_name.replace("|", "\\|"),
                    candidate.status,
                    _metric_text(
                        metrics.current_key_fact_coverage if metrics else None
                    ),
                    _metric_text(metrics.material_support_rate if metrics else None),
                    _metric_text(metrics.key_fact_relevance_rate if metrics else None),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"基线纪要：`{result.baseline_summary_id}`。这里只展示差值，不自动推荐最佳纪要。",
            "",
        ]
    )
    fact_by_id = {item.fact_id: item.text for item in result.reference.facts}
    for candidate in result.candidates:
        claim_by_id = {item.claim_id: item.text for item in candidate.claims}
        lines.extend(
            [
                f"## {candidate.display_name}",
                "",
                f"- 评测状态：`{candidate.status}`",
                f"- 支持性证据范围：`{candidate.support_scope}`",
                f"- 原子事实数：{len(candidate.claims)}",
                "",
                "未覆盖的当前关键事实：",
                "",
            ]
        )
        coverage_available = bool(
            metrics and metrics.current_key_fact_coverage.value is not None
        )
        support_available = bool(
            metrics and metrics.material_support_rate.value is not None
        )
        if not coverage_available:
            lines.append("- N/A（覆盖匹配未完整完成）")
        elif candidate.missing_reference_fact_ids:
            lines.extend(
                f"- `{fact_id}` {fact_by_id.get(fact_id, '')}"
                for fact_id in candidate.missing_reference_fact_ids
            )
        else:
            lines.append("- 无")
        lines.extend(["", "未获材料支持的纪要事实：", ""])
        if not support_available:
            lines.append("- N/A（材料支持判断未完整完成）")
        elif candidate.unsupported_claim_ids:
            lines.extend(
                f"- `{claim_id}` {claim_by_id.get(claim_id, '')}"
                for claim_id in candidate.unsupported_claim_ids
            )
        else:
            lines.append("- 无")
        if candidate.warnings:
            lines.extend(["", "Warnings：", ""])
            lines.extend(f"- {item}" for item in candidate.warnings)
        lines.append("")
    lines.extend(
        [
            "## 调用统计",
            "",
            f"- 模型调用：{result.execution_stats.get('model_call_count', 0)}",
            f"- JSON 修复调用：{result.execution_stats.get('repair_call_count', 0)}",
            f"- 运行内缓存命中：{result.execution_stats.get('cache_hit_count', 0)}",
            f"- 总耗时：{result.execution_stats.get('elapsed_ms', 0)} ms",
            "",
        ]
    )
    return "\n".join(lines)


def _build_html(markdown_report: str, result: LiteEvaluationResult) -> str:
    rows = []
    for candidate in result.candidates:
        metrics = candidate.metrics
        rows.append(
            "<tr>"
            f"<td>{html.escape(candidate.display_name)}</td>"
            f"<td>{html.escape(candidate.status)}</td>"
            f"<td>{html.escape(_metric_text(metrics.current_key_fact_coverage if metrics else None))}</td>"
            f"<td>{html.escape(_metric_text(metrics.material_support_rate if metrics else None))}</td>"
            f"<td>{html.escape(_metric_text(metrics.key_fact_relevance_rate if metrics else None))}</td>"
            "</tr>"
        )
    return (
        """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>会议纪要精简评测</title><style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;color:#172033;background:#f5f7fa}
main{max-width:1080px;margin:0 auto;padding:28px}section{background:#fff;border:1px solid #d7dee8;border-radius:6px;padding:20px;margin:14px 0}
table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #e4e9f0;text-align:left}th{background:#f1f4f8}
pre{white-space:pre-wrap;line-height:1.65;font-family:inherit}@media(max-width:720px){main{padding:12px}.scroll{overflow:auto}table{min-width:760px}}
</style></head><body><main><section><h1>会议纪要精简评测</h1>
<p>参考事实由模型从当前材料提取，并非人工金标准；不自动推荐最佳纪要。</p>
<div class="scroll"><table><thead><tr><th>纪要</th><th>状态</th><th>当前关键事实覆盖率</th><th>材料支持率</th><th>关键事实关联率</th></tr></thead><tbody>"""
        + "".join(rows)
        + """</tbody></table></div></section>
<section><pre>"""
        + html.escape(markdown_report)
        + """</pre></section></main></body></html>"""
    )


def write_lite_evaluation_artifacts(
    output_dir: Path,
    result: LiteEvaluationResult,
    *,
    model_trace: tuple[dict[str, Any], ...] = (),
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_lite_report(result)
    _write_json(
        output_dir / "evaluation_lite_reference.json",
        result.reference.model_dump(mode="json"),
    )
    _write_json(
        output_dir / "evaluation_lite_result.json",
        result.model_dump(mode="json"),
    )
    (output_dir / "evaluation_lite_report.md").write_text(report, encoding="utf-8")
    (output_dir / "evaluation_lite_report.html").write_text(
        _build_html(report, result), encoding="utf-8"
    )
    _write_json(output_dir / "evaluation_lite_model_trace.json", list(model_trace))
    return {name: str(output_dir / name) for name in LITE_ARTIFACT_NAMES}


def write_lite_failure_artifacts(
    output_dir: Path,
    *,
    error: str,
    stage: str,
    reference: LiteReferenceBundle | None = None,
    model_trace: tuple[dict[str, Any], ...] = (),
    execution_stats: dict[str, int] | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "evaluation_lite_reference.json",
        reference.model_dump(mode="json") if reference is not None else None,
    )
    payload = {
        "schema_version": "summary_multi_agent.evaluation_lite_failure.v1",
        "status": "failed",
        "evaluation_mode": "lite",
        "stage": stage,
        "error": error,
        "execution_stats": dict(execution_stats or {}),
    }
    _write_json(output_dir / "evaluation_lite_result.json", payload)
    report = (
        "# 会议纪要精简评测\n\n评测失败，已生成的正式纪要保持不变。\n\n"
        f"阶段：`{stage}`\n\n错误：{error}\n"
    )
    (output_dir / "evaluation_lite_report.md").write_text(report, encoding="utf-8")
    (output_dir / "evaluation_lite_report.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><body><pre>'
        + html.escape(report)
        + "</pre></body></html>",
        encoding="utf-8",
    )
    _write_json(output_dir / "evaluation_lite_model_trace.json", list(model_trace))
    return {name: str(output_dir / name) for name in LITE_ARTIFACT_NAMES}


def write_lite_comparison_artifacts(
    output_dir: Path,
    request: EvaluationCompareRequest,
    result: LiteEvaluationResult,
    *,
    model_trace: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    paths = write_lite_evaluation_artifacts(output_dir, result, model_trace=model_trace)
    _write_json(
        output_dir / "evaluation_compare_request.json",
        request.model_dump(mode="json"),
    )
    payload = {
        "schema_version": "summary_multi_agent.evaluation_compare_run_result.v2",
        "status": result.status,
        "evaluation_mode": "lite",
        "evaluation_status": ("failed" if result.status == "failed" else "completed"),
        "personalization_evaluation_status": "disabled",
        "comparison_interpretation_status": "disabled",
        "baseline_summary_id": result.baseline_summary_id,
        "warnings": list(result.warnings),
        "summaries": [
            {
                "summary_id": item.summary_id,
                "display_name": item.display_name,
                "status": item.status,
                "error": item.error,
            }
            for item in result.candidates
        ],
        "artifact_paths": {
            **paths,
            "evaluation_compare_request.json": str(
                output_dir / "evaluation_compare_request.json"
            ),
        },
    }
    _write_json(output_dir / "evaluation_compare_run_result.json", payload)
    return payload


def write_lite_comparison_failure_artifacts(
    output_dir: Path,
    request: EvaluationCompareRequest,
    *,
    error: str,
    stage: str,
    reference: LiteReferenceBundle | None = None,
    model_trace: tuple[dict[str, Any], ...] = (),
    execution_stats: dict[str, int] | None = None,
) -> dict[str, Any]:
    paths = write_lite_failure_artifacts(
        output_dir,
        error=error,
        stage=stage,
        reference=reference,
        model_trace=model_trace,
        execution_stats=execution_stats,
    )
    _write_json(
        output_dir / "evaluation_compare_request.json",
        request.model_dump(mode="json"),
    )
    payload = {
        "schema_version": "summary_multi_agent.evaluation_compare_run_result.v2",
        "status": "failed",
        "evaluation_mode": "lite",
        "evaluation_status": "failed",
        "personalization_evaluation_status": "disabled",
        "comparison_interpretation_status": "disabled",
        "stage": stage,
        "error": error,
        "warnings": [error],
        "summaries": [],
        "artifact_paths": {
            **paths,
            "evaluation_compare_request.json": str(
                output_dir / "evaluation_compare_request.json"
            ),
        },
    }
    _write_json(output_dir / "evaluation_compare_run_result.json", payload)
    return payload
