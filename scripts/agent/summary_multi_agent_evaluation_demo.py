"""独立多纪要事实级横评 CLI：一次 Snapshot，串行评测 2 至 5 份纪要。"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from summary_memory_llm import call_summary_llm
from summary_multi_agent.adapters.model import SyncCallableModelClient
from summary_multi_agent.evaluation.comparison import EvaluationComparisonRunner
from summary_multi_agent.evaluation.comparison_artifacts import (
    write_comparison_artifacts,
    write_comparison_failure_artifacts,
)
from summary_multi_agent.evaluation.lite import (
    LiteEvaluationComparisonRunner,
    write_lite_comparison_artifacts,
    write_lite_comparison_failure_artifacts,
)
from summary_multi_agent.evaluation.schemas import (
    EvaluationCompareConfig,
    EvaluationCompareRequest,
    EvaluationSourceDocument,
    EvaluationSummaryDocument,
)
from summary_multi_agent.schemas import EvaluationMode, LiteEvaluationConfig


def _parse_spec(raw: str, *, expected: int, option: str) -> tuple[str, ...]:
    """解析以竖线分隔且保留显示文本空格的 CLI 文档规格。"""
    values = tuple(item.strip() for item in raw.split("|", expected - 1))
    if len(values) != expected or any(not item for item in values):
        raise ValueError(
            f"{option} must contain {expected} non-empty fields separated by |"
        )
    return values


def parse_args() -> argparse.Namespace:
    """解析 JSON 请求或显式文件参数。"""
    parser = argparse.ArgumentParser(
        description="Compare 2-5 minutes against one frozen source snapshot."
    )
    parser.add_argument("--request-json", type=Path)
    parser.add_argument("--project", default="")
    parser.add_argument("--summary-request", default="")
    parser.add_argument("--current-file", type=Path)
    parser.add_argument("--current-source-id", default="current")
    parser.add_argument("--current-label", default="当前材料")
    parser.add_argument("--current-source-time", default="")
    parser.add_argument(
        "--history",
        action="append",
        default=[],
        metavar="ID|LABEL|SOURCE_TIME|PATH",
    )
    parser.add_argument(
        "--summary",
        action="append",
        default=[],
        metavar="ID|DISPLAY_NAME|PATH",
    )
    parser.add_argument("--target-entity", action="append", default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/agent_multi_evaluation_compare"),
    )
    parser.add_argument("--comparison-max-summaries", type=int, default=5)
    parser.add_argument(
        "--evaluation-mode", choices=("strict", "lite"), default="strict"
    )
    parser.add_argument("--lite-segment-max-chars", type=int, default=3000)
    parser.add_argument("--lite-segment-overlap-chars", type=int, default=100)
    parser.add_argument("--lite-reference-max-tokens", type=int, default=4096)
    parser.add_argument("--lite-claim-max-tokens", type=int, default=4096)
    parser.add_argument("--lite-match-max-tokens", type=int, default=4096)
    parser.add_argument("--lite-verification-max-tokens", type=int, default=4096)
    parser.add_argument("--lite-repair-attempts", type=int, default=1)
    parser.add_argument("--lite-model-timeout-seconds", type=int, default=180)
    parser.add_argument("--lite-timeout-seconds", type=int, default=1200)
    parser.add_argument("--lite-max-model-calls", type=int, default=64)
    parser.add_argument("--lite-max-concurrent-model-calls", type=int, default=1)
    parser.add_argument("--lite-model-context-window-tokens", type=int)
    parser.add_argument("--lite-context-safety-margin-tokens", type=int, default=1024)
    parser.add_argument("--comparison-source-max-model-calls", type=int, default=48)
    parser.add_argument(
        "--comparison-per-summary-max-model-calls", type=int, default=96
    )
    parser.add_argument(
        "--comparison-personalization-source-max-model-calls",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--comparison-personalization-per-summary-max-model-calls",
        type=int,
        default=24,
    )
    parser.add_argument("--comparison-max-total-model-calls", type=int, default=320)
    parser.add_argument(
        "--comparison-interpretation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用横评直接定性说明与机械覆盖差异二次核验",
    )
    parser.add_argument(
        "--comparison-interpretation-max-model-calls", type=int, default=48
    )
    parser.add_argument(
        "--comparison-interpretation-max-tokens", type=int, default=2400
    )
    parser.add_argument(
        "--comparison-difference-verification-max-units-per-pair-scope",
        type=int,
        default=48,
    )
    parser.add_argument("--comparison-timeout-seconds", type=int, default=7200)
    parser.add_argument(
        "--comparison-per-summary-timeout-seconds", type=int, default=1200
    )
    parser.add_argument("--comparison-model-timeout-seconds", type=int, default=180)
    parser.add_argument("--evaluation-segment-max-chars", type=int, default=500)
    parser.add_argument("--evaluation-source-fact-max-tokens", type=int, default=3000)
    parser.add_argument("--evaluation-summary-fact-max-tokens", type=int, default=3000)
    parser.add_argument("--evaluation-match-max-tokens", type=int, default=3000)
    parser.add_argument("--evaluation-repair-attempts", type=int, default=2)
    parser.add_argument("--evaluation-match-batch-size", type=int, default=6)
    parser.add_argument("--evaluation-history-batch-size", type=int, default=12)
    parser.add_argument("--evaluation-match-full-scan-threshold", type=int, default=48)
    parser.add_argument(
        "--evaluation-match-candidate-limit-per-fact", type=int, default=12
    )
    parser.add_argument(
        "--evaluation-match-candidate-union-limit", type=int, default=48
    )
    parser.add_argument("--evaluation-personalization-batch-size", type=int, default=24)
    parser.add_argument(
        "--evaluation-personalization-max-tokens", type=int, default=2400
    )
    parser.add_argument(
        "--evaluation-adaptive-split",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--evaluation-adaptive-split-min-chars", type=int, default=500)
    parser.add_argument("--evaluation-adaptive-split-max-depth", type=int, default=1)
    return parser.parse_args()


def build_request(args: argparse.Namespace) -> EvaluationCompareRequest:
    """从 JSON 或显式文件参数构建严格横评请求。"""
    if args.request_json is not None:
        payload = json.loads(args.request_json.expanduser().read_text(encoding="utf-8"))
        return EvaluationCompareRequest.model_validate(payload)
    if not args.project or not args.summary_request:
        raise ValueError(
            "--project and --summary-request are required without --request-json"
        )
    if args.current_file is None or not args.current_source_time:
        raise ValueError("--current-file and --current-source-time are required")
    histories = []
    for raw in args.history:
        source_id, label, source_time, path = _parse_spec(
            raw, expected=4, option="--history"
        )
        histories.append(
            EvaluationSourceDocument(
                source_id=source_id,
                label=label,
                source_time=source_time,
                text=Path(path).expanduser().read_text(encoding="utf-8"),
            )
        )
    summaries = []
    for raw in args.summary:
        summary_id, display_name, path = _parse_spec(
            raw, expected=3, option="--summary"
        )
        summaries.append(
            EvaluationSummaryDocument(
                summary_id=summary_id,
                display_name=display_name,
                text=Path(path).expanduser().read_text(encoding="utf-8"),
            )
        )
    return EvaluationCompareRequest(
        project=args.project,
        summary_request=args.summary_request,
        current_source=EvaluationSourceDocument(
            source_id=args.current_source_id,
            label=args.current_label,
            source_time=args.current_source_time,
            text=args.current_file.expanduser().read_text(encoding="utf-8"),
        ),
        historical_sources=tuple(histories),
        summaries=tuple(summaries),
        target_entities=tuple(
            dict.fromkeys(item.strip() for item in args.target_entity if item.strip())
        ),
        evaluation_config=EvaluationCompareConfig(
            evaluation_mode=args.evaluation_mode,
            lite_evaluation=LiteEvaluationConfig(
                segment_max_chars=args.lite_segment_max_chars,
                segment_overlap_chars=args.lite_segment_overlap_chars,
                reference_max_tokens=args.lite_reference_max_tokens,
                claim_max_tokens=args.lite_claim_max_tokens,
                match_max_tokens=args.lite_match_max_tokens,
                verification_max_tokens=args.lite_verification_max_tokens,
                repair_attempts=args.lite_repair_attempts,
                model_timeout_seconds=args.lite_model_timeout_seconds,
                timeout_seconds=args.lite_timeout_seconds,
                max_model_calls=args.lite_max_model_calls,
                max_concurrent_model_calls=(args.lite_max_concurrent_model_calls),
                model_context_window_tokens=(args.lite_model_context_window_tokens),
                context_safety_margin_tokens=(args.lite_context_safety_margin_tokens),
            ),
            comparison_max_summaries=args.comparison_max_summaries,
            comparison_source_max_model_calls=(args.comparison_source_max_model_calls),
            comparison_per_summary_max_model_calls=(
                args.comparison_per_summary_max_model_calls
            ),
            comparison_personalization_source_max_model_calls=(
                args.comparison_personalization_source_max_model_calls
            ),
            comparison_personalization_per_summary_max_model_calls=(
                args.comparison_personalization_per_summary_max_model_calls
            ),
            comparison_max_total_model_calls=args.comparison_max_total_model_calls,
            comparison_interpretation_enabled=args.comparison_interpretation,
            comparison_interpretation_max_model_calls=(
                args.comparison_interpretation_max_model_calls
            ),
            comparison_interpretation_max_tokens=(
                args.comparison_interpretation_max_tokens
            ),
            comparison_difference_verification_max_units_per_pair_scope=(
                args.comparison_difference_verification_max_units_per_pair_scope
            ),
            comparison_timeout_seconds=args.comparison_timeout_seconds,
            comparison_per_summary_timeout_seconds=(
                args.comparison_per_summary_timeout_seconds
            ),
            comparison_model_timeout_seconds=args.comparison_model_timeout_seconds,
            segment_max_chars=args.evaluation_segment_max_chars,
            source_fact_max_tokens=args.evaluation_source_fact_max_tokens,
            summary_fact_max_tokens=args.evaluation_summary_fact_max_tokens,
            match_max_tokens=args.evaluation_match_max_tokens,
            repair_attempts=args.evaluation_repair_attempts,
            match_batch_size=args.evaluation_match_batch_size,
            history_batch_size=args.evaluation_history_batch_size,
            match_full_scan_threshold=(args.evaluation_match_full_scan_threshold),
            match_candidate_limit_per_fact=(
                args.evaluation_match_candidate_limit_per_fact
            ),
            match_candidate_union_limit=(args.evaluation_match_candidate_union_limit),
            personalization_batch_size=(args.evaluation_personalization_batch_size),
            personalization_max_tokens=(args.evaluation_personalization_max_tokens),
            adaptive_split_enabled=args.evaluation_adaptive_split,
            adaptive_split_min_chars=(args.evaluation_adaptive_split_min_chars),
            adaptive_split_max_depth=(args.evaluation_adaptive_split_max_depth),
        ),
    )


def create_run_dir(root: Path, project: str) -> Path:
    """创建独立且不覆盖既有横评产物的运行目录。"""
    safe_project = (
        "".join(
            char if char.isalnum() or char in "_.-" else "_" for char in project
        ).strip("._-")
        or "project"
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root.expanduser() / f"{timestamp}_evaluate_{safe_project}"
    suffix = 2
    while path.exists():
        path = root.expanduser() / f"{timestamp}_evaluate_{safe_project}-{suffix}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


async def run(args: argparse.Namespace) -> int:
    """执行横评并打印运行目录和逐份状态。"""
    request = build_request(args)
    output_dir = create_run_dir(args.output_dir, request.project)
    model_client = SyncCallableModelClient(call_summary_llm)
    if request.evaluation_config.evaluation_mode == EvaluationMode.LITE:
        runner = LiteEvaluationComparisonRunner(
            model_client=model_client, output_dir=output_dir
        )
        try:
            result = await runner.run(request)
        except Exception as exc:
            payload = write_lite_comparison_failure_artifacts(
                output_dir,
                request,
                stage=runner.last_stage,
                error=f"{type(exc).__name__}: {exc}",
                reference=runner.reference,
                model_trace=runner.model_trace,
                execution_stats=runner.execution_stats,
            )
            print("STATUS: failed")
            print(f"OUTPUT_DIR: {output_dir}")
            print(f"ERROR: {payload['error']}")
            return 1
        write_lite_comparison_artifacts(
            output_dir,
            request,
            result,
            model_trace=runner.model_trace,
        )
        print(f"STATUS: {result.status}")
        print(f"OUTPUT_DIR: {output_dir}")
        print(f"REFERENCE_FACTS: {len(result.reference.facts)}")
        print(f"MODEL_CALLS: {result.execution_stats.get('model_call_count', 0)}")
        for item in result.candidates:
            print(f"SUMMARY {item.summary_id}: {item.status}")
        print()
        print((output_dir / "evaluation_lite_report.md").read_text(encoding="utf-8"))
        return 1 if result.status == "failed" else 0

    runner = EvaluationComparisonRunner(model_client=model_client)
    try:
        result = await runner.run(request)
    except Exception as exc:
        payload = write_comparison_failure_artifacts(
            output_dir,
            request,
            stage=runner.last_stage,
            error=f"{type(exc).__name__}: {exc}",
            source_model_trace=runner.last_source_model_trace,
            source_personalization_model_trace=(
                runner.last_source_personalization_model_trace
            ),
            source_execution_stats=runner.last_source_execution_stats,
        )
        print("STATUS: failed")
        print(f"OUTPUT_DIR: {output_dir}")
        print(f"ERROR: {payload['error']}")
        return 1
    write_comparison_artifacts(output_dir, request, result)
    print(f"STATUS: {result.status}")
    print(f"OUTPUT_DIR: {output_dir}")
    print(f"SNAPSHOT_ID: {result.snapshot_bundle.snapshot.snapshot_id}")
    print(f"MODEL_CALLS: {result.comparison_metrics.get('total_model_call_count')}")
    for item in result.summaries:
        print(f"SUMMARY {item.summary_id}: {item.status}")
    print()
    print(result.report_markdown)
    return 1 if result.status == "failed" else 0


def main() -> None:
    """CLI 入口。"""
    raise SystemExit(asyncio.run(run(parse_args())))


if __name__ == "__main__":
    main()
