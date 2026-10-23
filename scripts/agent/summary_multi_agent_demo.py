"""Summary Multi-Agent V2 命令行入口：构造请求、配置运行时并执行 Pipeline。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from summary_memory_llm import load_runtime_env, validate_llm_configuration

load_runtime_env()

from summary_memory_case_demo import (
    QDRANT_SUMMARY_PATH,
    SUMMARY_COLLECTION_NAME,
    build_summary_memory,
    close_summary_memory,
)
from summary_memory_llm import call_summary_llm, get_llm_runtime_info
from summary_multi_agent.runtime import build_multi_agent_v2_pipeline
from summary_multi_agent.profiles import CLOUD_FAST_DEFAULTS
from summary_multi_agent.schemas import (
    LiteEvaluationConfig,
    RunConfig,
    RunStatus,
    SummaryRequest,
)


def env_bool(name: str, default: bool = False) -> bool:
    """读取布尔环境变量，拒绝无法明确解释的值。"""
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {raw!r}")


def parse_args() -> argparse.Namespace:
    """解析 CLI 输入；具体运行参数仍由 RunConfig 统一校验。"""
    parser = argparse.ArgumentParser(
        description="Run the standalone Summary Multi-Agent V2 summarize pipeline."
    )
    parser.add_argument(
        "--execution-profile", choices=("cloud_fast", "legacy"),
        default=os.getenv("SMW_EXECUTION_PROFILE", "cloud_fast"),
    )
    parser.add_argument("--current-file", type=Path, required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--source-time", default="")
    parser.add_argument(
        "--summary-request",
        default="生成忠实、可追溯的个性化摘要。",
    )
    parser.add_argument("--target-entity", action="append", default=[])
    parser.add_argument(
        "--summary-style",
        choices=("structured", "natural"),
        default="structured",
    )
    parser.add_argument(
        "--preferences-json",
        default="{}",
        help="JSON object merged with summary_style for this request.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/agent_multi_v2"),
    )
    parser.add_argument(
        "--top-k-per-plan-item",
        type=int,
        default=int(os.getenv("SMW_V2_RETRIEVAL_TOP_K", "10")),
    )
    parser.add_argument(
        "--context-compression-mode",
        choices=("disabled", "deterministic"),
        default=os.getenv("SMW_V2_CONTEXT_COMPRESSION_MODE", "deterministic"),
        help="Compress only auxiliary synthesis/review context; never current material.",
    )
    parser.add_argument(
        "--context-compression-top-k-per-type",
        type=int,
        default=int(os.getenv("SMW_V2_CONTEXT_COMPRESSION_TOP_K_PER_TYPE", "10")),
        help=(
            "Bound unprotected auxiliary memory types after global identity "
            "deduplication; time/decision/constraint/task/risk remain unbounded."
        ),
    )
    parser.add_argument(
        "--raw-segment-max-chars",
        type=int,
        default=int(os.getenv("SMW_V2_RAW_SEGMENT_MAX_CHARS", "3000")),
    )
    parser.add_argument(
        "--raw-segment-overlap-ratio",
        type=float,
        default=float(os.getenv("SMW_V2_RAW_SEGMENT_OVERLAP_RATIO", "0.05")),
    )
    parser.add_argument(
        "--max-concurrent-workers",
        type=int,
        default=int(os.getenv("SMW_V2_MAX_CONCURRENT_WORKERS", "4")),
    )
    parser.add_argument(
        "--max-concurrent-segment-maps",
        type=int,
        default=int(os.getenv("SMW_V2_MAX_CONCURRENT_SEGMENT_MAPS", "2")),
    )
    parser.add_argument(
        "--current-map-max-tokens",
        type=int,
        default=int(os.getenv("SMW_V2_CURRENT_MAP_MAX_TOKENS", "2400")),
    )
    parser.add_argument(
        "--summary-max-tokens",
        type=int,
        default=int(os.getenv("SMW_V2_SUMMARY_MAX_TOKENS", "2400")),
    )
    parser.add_argument(
        "--minutes-output-mode",
        choices=("classic", "rich"),
        default=os.getenv("SMW_V2_MINUTES_OUTPUT_MODE", "classic"),
        help="Generate the classic Markdown summary or a structured rich report.",
    )
    parser.add_argument(
        "--rich-summary-max-tokens",
        type=int,
        default=int(os.getenv("SMW_V2_RICH_SUMMARY_MAX_TOKENS", "6000")),
    )
    parser.add_argument(
        "--enable-evaluation",
        action="store_true",
        default=env_bool("SMW_V2_EVALUATION_ENABLED", False),
        help="Run the read-only fact-level evaluation sidecar after the formal summary.",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("strict", "lite"),
        default=os.getenv("SMW_V2_EVALUATION_MODE", "strict"),
        help="Use the existing strict evaluator or the fast lite evaluator.",
    )
    parser.add_argument("--lite-evaluation-segment-max-chars", type=int, default=3000)
    parser.add_argument(
        "--lite-evaluation-segment-overlap-chars", type=int, default=100
    )
    parser.add_argument(
        "--lite-evaluation-reference-max-tokens", type=int, default=4096
    )
    parser.add_argument("--lite-evaluation-claim-max-tokens", type=int, default=4096)
    parser.add_argument("--lite-evaluation-match-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--lite-evaluation-verification-max-tokens", type=int, default=4096
    )
    parser.add_argument("--lite-evaluation-repair-attempts", type=int, default=1)
    parser.add_argument(
        "--lite-evaluation-model-timeout-seconds", type=int, default=180
    )
    parser.add_argument("--lite-evaluation-timeout-seconds", type=int, default=1200)
    parser.add_argument("--lite-evaluation-max-model-calls", type=int, default=64)
    parser.add_argument(
        "--lite-evaluation-max-concurrent-model-calls", type=int, default=1
    )
    parser.add_argument(
        "--evaluation-segment-max-chars",
        type=int,
        default=int(os.getenv("SMW_V2_EVALUATION_SEGMENT_MAX_CHARS", "1000")),
        help="Independent segment size for source/summary fact extraction.",
    )
    parser.add_argument(
        "--evaluation-source-fact-max-tokens",
        type=int,
        default=int(os.getenv("SMW_V2_EVALUATION_SOURCE_FACT_MAX_TOKENS", "3000")),
    )
    parser.add_argument(
        "--evaluation-summary-fact-max-tokens",
        type=int,
        default=int(os.getenv("SMW_V2_EVALUATION_SUMMARY_FACT_MAX_TOKENS", "3000")),
    )
    parser.add_argument(
        "--evaluation-match-max-tokens",
        type=int,
        default=int(os.getenv("SMW_V2_EVALUATION_MATCH_MAX_TOKENS", "3000")),
    )
    parser.add_argument(
        "--evaluation-repair-attempts",
        type=int,
        default=int(os.getenv("SMW_V2_EVALUATION_REPAIR_ATTEMPTS", "2")),
    )
    parser.add_argument(
        "--evaluation-timeout-seconds",
        type=int,
        default=int(os.getenv("SMW_V2_EVALUATION_TIMEOUT_SECONDS", "600")),
    )
    parser.add_argument(
        "--evaluation-model-timeout-seconds",
        type=int,
        default=int(os.getenv("SMW_V2_EVALUATION_MODEL_TIMEOUT_SECONDS", "180")),
    )
    parser.add_argument(
        "--evaluation-max-model-calls",
        type=int,
        default=int(os.getenv("SMW_V2_EVALUATION_MAX_MODEL_CALLS", "48")),
    )
    parser.add_argument(
        "--evaluation-max-concurrent-segment-maps",
        type=int,
        default=int(os.getenv("SMW_V2_EVALUATION_MAX_CONCURRENT_SEGMENT_MAPS", "1")),
    )
    parser.add_argument(
        "--evaluation-match-batch-size",
        type=int,
        default=int(os.getenv("SMW_V2_EVALUATION_MATCH_BATCH_SIZE", "6")),
    )
    parser.add_argument(
        "--evaluation-history-batch-size",
        type=int,
        default=int(os.getenv("SMW_V2_EVALUATION_HISTORY_BATCH_SIZE", "12")),
    )
    parser.add_argument(
        "--evaluation-match-full-scan-threshold",
        type=int,
        default=int(os.getenv("SMW_V2_EVALUATION_MATCH_FULL_SCAN_THRESHOLD", "48")),
    )
    parser.add_argument(
        "--evaluation-match-candidate-limit-per-fact",
        type=int,
        default=int(
            os.getenv("SMW_V2_EVALUATION_MATCH_CANDIDATE_LIMIT_PER_FACT", "12")
        ),
    )
    parser.add_argument(
        "--evaluation-match-candidate-union-limit",
        type=int,
        default=int(os.getenv("SMW_V2_EVALUATION_MATCH_CANDIDATE_UNION_LIMIT", "48")),
    )
    parser.add_argument(
        "--evaluation-personalization-batch-size",
        type=int,
        default=int(os.getenv("SMW_V2_EVALUATION_PERSONALIZATION_BATCH_SIZE", "24")),
        help="Facts classified per target-entity personalization model call.",
    )
    parser.add_argument(
        "--evaluation-personalization-max-tokens",
        type=int,
        default=int(os.getenv("SMW_V2_EVALUATION_PERSONALIZATION_MAX_TOKENS", "2400")),
    )
    parser.add_argument(
        "--evaluation-personalization-max-model-calls",
        type=int,
        default=int(
            os.getenv("SMW_V2_EVALUATION_PERSONALIZATION_MAX_MODEL_CALLS", "24")
        ),
    )
    parser.add_argument(
        "--evaluation-personalization-timeout-seconds",
        type=int,
        default=int(
            os.getenv("SMW_V2_EVALUATION_PERSONALIZATION_TIMEOUT_SECONDS", "300")
        ),
    )
    parser.add_argument(
        "--temporal-max-tool-calls",
        type=int,
        default=int(os.getenv("SMW_V2_TEMPORAL_MAX_TOOL_CALLS", "128")),
    )
    parser.add_argument(
        "--enable-review",
        action="store_true",
        default=env_bool("SMW_V2_REVIEW_ENABLED", False),
        help="Enable the P5 Red-Blue-Judge review loop in shadow mode.",
    )
    parser.add_argument(
        "--max-review-rounds",
        type=int,
        default=int(os.getenv("SMW_V2_MAX_REVIEW_ROUNDS", "3")),
    )
    parser.add_argument(
        "--review-red-max-tokens",
        type=int,
        default=int(os.getenv("SMW_V2_REVIEW_RED_MAX_TOKENS", "2400")),
    )
    parser.add_argument(
        "--review-red-max-candidates-per-audit",
        type=int,
        default=int(os.getenv("SMW_V2_REVIEW_RED_MAX_CANDIDATES_PER_AUDIT", "6")),
    )
    parser.add_argument(
        "--review-red-adjudication-max-tokens",
        type=int,
        default=int(os.getenv("SMW_V2_REVIEW_RED_ADJUDICATION_MAX_TOKENS", "1600")),
    )
    parser.add_argument(
        "--review-red-adjudication-batch-size",
        type=int,
        default=int(os.getenv("SMW_V2_REVIEW_RED_ADJUDICATION_BATCH_SIZE", "3")),
    )
    parser.add_argument(
        "--review-blue-max-tokens",
        type=int,
        default=int(os.getenv("SMW_V2_REVIEW_BLUE_MAX_TOKENS", "3000")),
    )
    parser.add_argument(
        "--review-judge-max-tokens",
        type=int,
        default=int(os.getenv("SMW_V2_REVIEW_JUDGE_MAX_TOKENS", "1600")),
    )
    parser.add_argument(
        "--review-max-concurrent-segment-audits",
        type=int,
        default=int(os.getenv("SMW_V2_REVIEW_MAX_CONCURRENT_SEGMENT_AUDITS", "1")),
    )
    parser.add_argument(
        "--review-max-issues-per-round",
        type=int,
        default=int(os.getenv("SMW_V2_REVIEW_MAX_ISSUES_PER_ROUND", "6")),
    )
    parser.add_argument(
        "--review-accept-score",
        type=float,
        default=float(os.getenv("SMW_V2_REVIEW_ACCEPT_SCORE", "0.85")),
    )
    parser.add_argument(
        "--review-min-improvement",
        type=float,
        default=float(os.getenv("SMW_V2_REVIEW_MIN_IMPROVEMENT", "0.02")),
    )
    parser.add_argument(
        "--review-text-change-convergence-threshold",
        type=float,
        default=float(
            os.getenv(
                "SMW_V2_REVIEW_TEXT_CHANGE_CONVERGENCE_THRESHOLD",
                "0.02",
            )
        ),
    )
    parser.add_argument(
        "--worker-timeout-seconds",
        type=int,
        default=int(os.getenv("SMW_V2_WORKER_TIMEOUT_SECONDS", "180")),
    )
    parser.add_argument(
        "--worker-mode",
        choices=("rule", "model", "model_with_rule_fallback"),
        default=os.getenv("SMW_V2_WORKER_MODE", "model_with_rule_fallback"),
        help=(
            "Use fixed rule workers, strict model-driven workers, or model workers "
            "with an audited rule fallback."
        ),
    )
    parser.add_argument(
        "--worker-max-steps",
        type=int,
        default=int(os.getenv("SMW_V2_WORKER_MAX_STEPS", "6")),
    )
    parser.add_argument(
        "--worker-max-model-calls",
        type=int,
        default=int(os.getenv("SMW_V2_WORKER_MAX_MODEL_CALLS", "4")),
    )
    parser.add_argument(
        "--worker-action-max-tokens",
        type=int,
        default=int(os.getenv("SMW_V2_WORKER_ACTION_MAX_TOKENS", "3000")),
    )
    parser.add_argument(
        "--worker-action-repair-attempts",
        type=int,
        default=int(os.getenv("SMW_V2_WORKER_ACTION_REPAIR_ATTEMPTS", "1")),
    )
    parser.add_argument(
        "--worker-model-timeout-seconds",
        type=int,
        default=int(os.getenv("SMW_V2_WORKER_MODEL_TIMEOUT_SECONDS", "180")),
    )
    parser.add_argument(
        "--max-concurrent-model-calls",
        type=int,
        default=int(os.getenv("SMW_V2_MAX_CONCURRENT_MODEL_CALLS", "1")),
    )
    parser.add_argument(
        "--global-timeout-seconds",
        type=int,
        default=int(os.getenv("SMW_V2_GLOBAL_TIMEOUT_SECONDS", "600")),
    )
    parser.add_argument(
        "--max-replan-rounds",
        type=int,
        default=int(os.getenv("SMW_V2_MAX_REPLAN_ROUNDS", "2")),
    )
    parser.add_argument(
        "--planner-mode",
        choices=("model", "rule"),
        default=os.getenv("SMW_V2_PLANNER_MODE", "model"),
        help="Use the model-first planner/replanner or the explicit rule planner.",
    )
    parser.add_argument(
        "--planner-max-tokens",
        type=int,
        default=int(os.getenv("SMW_V2_PLANNER_MAX_TOKENS", "1600")),
    )
    parser.add_argument(
        "--planner-repair-max-tokens",
        type=int,
        default=int(os.getenv("SMW_V2_PLANNER_REPAIR_MAX_TOKENS", "1600")),
    )
    parser.add_argument(
        "--planner-repair-attempts",
        type=int,
        default=int(os.getenv("SMW_V2_PLANNER_REPAIR_ATTEMPTS", "2")),
    )
    parser.add_argument(
        "--planner-timeout-seconds",
        type=int,
        default=int(os.getenv("SMW_V2_PLANNER_TIMEOUT_SECONDS", "180")),
    )
    parser.add_argument(
        "--replanner-max-tokens",
        type=int,
        default=int(os.getenv("SMW_V2_REPLANNER_MAX_TOKENS", "1800")),
    )
    parser.add_argument(
        "--replanner-repair-max-tokens",
        type=int,
        default=int(os.getenv("SMW_V2_REPLANNER_REPAIR_MAX_TOKENS", "1800")),
    )
    parser.add_argument(
        "--replanner-repair-attempts",
        type=int,
        default=int(os.getenv("SMW_V2_REPLANNER_REPAIR_ATTEMPTS", "2")),
    )
    parser.add_argument(
        "--replanner-timeout-seconds",
        type=int,
        default=int(os.getenv("SMW_V2_REPLANNER_TIMEOUT_SECONDS", "180")),
    )
    parser.add_argument(
        "--planner-max-tasks",
        type=int,
        default=int(os.getenv("SMW_V2_PLANNER_MAX_TASKS", "16")),
    )
    parser.add_argument(
        "--planner-max-depth",
        type=int,
        default=int(os.getenv("SMW_V2_PLANNER_MAX_DEPTH", "8")),
    )
    parser.add_argument(
        "--planner-max-tasks-per-capability",
        type=int,
        default=int(os.getenv("SMW_V2_PLANNER_MAX_TASKS_PER_CAPABILITY", "4")),
    )
    parser.add_argument(
        "--planner-max-total-model-calls",
        type=int,
        default=int(os.getenv("SMW_V2_PLANNER_MAX_TOTAL_MODEL_CALLS", "64")),
    )
    parser.add_argument(
        "--planner-max-total-tool-calls",
        type=int,
        default=int(os.getenv("SMW_V2_PLANNER_MAX_TOTAL_TOOL_CALLS", "2048")),
    )
    parser.add_argument(
        "--planner-max-task-timeout-seconds",
        type=int,
        default=int(os.getenv("SMW_V2_PLANNER_MAX_TASK_TIMEOUT_SECONDS", "600")),
    )
    parser.add_argument(
        "--schema-repair-attempts",
        type=int,
        default=int(os.getenv("SMW_V2_SCHEMA_REPAIR_ATTEMPTS", "2")),
    )
    configured_context_window = os.getenv("SMW_MODEL_CONTEXT_WINDOW_TOKENS", "").strip()
    parser.add_argument(
        "--model-context-window-tokens",
        type=int,
        default=(int(configured_context_window) if configured_context_window else None),
        help="Optional known model context window; unset disables hard preflight.",
    )
    parser.add_argument(
        "--model-context-safety-margin-tokens",
        type=int,
        default=int(os.getenv("SMW_MODEL_CONTEXT_SAFETY_MARGIN_TOKENS", "1024")),
    )
    profile_parser = argparse.ArgumentParser(add_help=False)
    profile_parser.add_argument(
        "--execution-profile", choices=("cloud_fast", "legacy"),
        default=os.getenv("SMW_EXECUTION_PROFILE", "cloud_fast"),
    )
    profile, _ = profile_parser.parse_known_args()
    if profile.execution_profile == "cloud_fast":
        parser.set_defaults(**{
            name: value for name, value in CLOUD_FAST_DEFAULTS.items()
            if any(action.dest == name for action in parser._actions)
            and f"SMW_V2_{name.upper()}" not in os.environ
        })
        parser.set_defaults(
            lite_evaluation_segment_max_chars=12000,
            lite_evaluation_reference_max_tokens=8192,
            lite_evaluation_claim_max_tokens=8192,
            lite_evaluation_max_concurrent_model_calls=4,
        )
    return parser.parse_args()


def parse_preferences(raw: str, summary_style: str) -> dict[str, Any]:
    """解析偏好 JSON，并用显式命令行风格覆盖同名偏好。"""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--preferences-json is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("--preferences-json must be a JSON object")
    preferences = dict(value)
    preferences["summary_style"] = summary_style
    return preferences


def create_run_dir(root: Path, project: str) -> Path:
    """创建不会覆盖既有审计文件的唯一 V2 运行目录。"""
    # 每次运行创建独立审计目录，重试时不覆盖已有 prompt、trace 或 coverage 产物。
    safe_project = (
        "".join(
            character if character.isalnum() or character in "_.-" else "_"
            for character in project
        ).strip("._-")
        or "project"
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = root.expanduser() / f"{timestamp}_multi_agent_v2_{safe_project}"
    suffix = 2
    while run_dir.exists():
        run_dir = root.expanduser() / (
            f"{timestamp}_multi_agent_v2_{safe_project}-{suffix}"
        )
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


async def run(args: argparse.Namespace) -> int:
    """构造 SummaryRequest/RunConfig，运行 Pipeline 并输出终态摘要与指标。"""
    current_material = args.current_file.expanduser().read_text(encoding="utf-8")
    preferences = parse_preferences(args.preferences_json, args.summary_style)
    request = SummaryRequest(
        current_material=current_material,
        source_time=args.source_time.strip() or None,
        summary_request=args.summary_request,
        target_entities=args.target_entity,
        personalization_preferences=preferences,
    )
    config = RunConfig(
        workflow_mode="multi_agent_v2",
        execution_profile=args.execution_profile,
        planner_mode=args.planner_mode,
        planner_max_tokens=args.planner_max_tokens,
        planner_repair_max_tokens=args.planner_repair_max_tokens,
        planner_repair_attempts=args.planner_repair_attempts,
        planner_timeout_seconds=args.planner_timeout_seconds,
        replanner_max_tokens=args.replanner_max_tokens,
        replanner_repair_max_tokens=args.replanner_repair_max_tokens,
        replanner_repair_attempts=args.replanner_repair_attempts,
        replanner_timeout_seconds=args.replanner_timeout_seconds,
        worker_mode=args.worker_mode,
        worker_max_steps=args.worker_max_steps,
        worker_max_model_calls=args.worker_max_model_calls,
        worker_action_max_tokens=args.worker_action_max_tokens,
        worker_action_repair_attempts=args.worker_action_repair_attempts,
        worker_model_timeout_seconds=args.worker_model_timeout_seconds,
        model_context_window_tokens=args.model_context_window_tokens,
        model_context_safety_margin_tokens=(args.model_context_safety_margin_tokens),
        max_concurrent_model_calls=args.max_concurrent_model_calls,
        retrieval_top_k_per_plan_item=args.top_k_per_plan_item,
        context_compression_mode=args.context_compression_mode,
        context_compression_top_k_per_type=(args.context_compression_top_k_per_type),
        raw_segment_max_chars=args.raw_segment_max_chars,
        raw_segment_overlap_ratio=args.raw_segment_overlap_ratio,
        max_concurrent_workers=args.max_concurrent_workers,
        max_concurrent_segment_maps=args.max_concurrent_segment_maps,
        current_map_max_tokens=args.current_map_max_tokens,
        summary_max_tokens=args.summary_max_tokens,
        minutes_output_mode=args.minutes_output_mode,
        rich_summary_max_tokens=args.rich_summary_max_tokens,
        temporal_max_tool_calls=args.temporal_max_tool_calls,
        review_enabled=args.enable_review,
        review_shadow_mode=True,
        max_review_rounds=args.max_review_rounds,
        review_red_max_tokens=args.review_red_max_tokens,
        review_red_max_candidates_per_audit=(args.review_red_max_candidates_per_audit),
        review_red_adjudication_max_tokens=(args.review_red_adjudication_max_tokens),
        review_red_adjudication_batch_size=(args.review_red_adjudication_batch_size),
        review_blue_max_tokens=args.review_blue_max_tokens,
        review_judge_max_tokens=args.review_judge_max_tokens,
        review_max_concurrent_segment_audits=(
            args.review_max_concurrent_segment_audits
        ),
        review_max_issues_per_round=args.review_max_issues_per_round,
        review_accept_score=args.review_accept_score,
        review_min_improvement=args.review_min_improvement,
        review_text_change_convergence_threshold=(
            args.review_text_change_convergence_threshold
        ),
        evaluation_enabled=args.enable_evaluation,
        evaluation_mode=args.evaluation_mode,
        lite_evaluation=LiteEvaluationConfig(
            reference_chunking="auto" if args.execution_profile == "cloud_fast" else "fixed",
            segment_max_chars=args.lite_evaluation_segment_max_chars,
            segment_overlap_chars=args.lite_evaluation_segment_overlap_chars,
            reference_max_tokens=args.lite_evaluation_reference_max_tokens,
            claim_max_tokens=args.lite_evaluation_claim_max_tokens,
            match_max_tokens=args.lite_evaluation_match_max_tokens,
            verification_max_tokens=(args.lite_evaluation_verification_max_tokens),
            repair_attempts=args.lite_evaluation_repair_attempts,
            model_timeout_seconds=(args.lite_evaluation_model_timeout_seconds),
            timeout_seconds=args.lite_evaluation_timeout_seconds,
            max_model_calls=args.lite_evaluation_max_model_calls,
            max_concurrent_model_calls=(
                args.lite_evaluation_max_concurrent_model_calls
            ),
        ),
        evaluation_segment_max_chars=args.evaluation_segment_max_chars,
        evaluation_source_fact_max_tokens=(args.evaluation_source_fact_max_tokens),
        evaluation_summary_fact_max_tokens=(args.evaluation_summary_fact_max_tokens),
        evaluation_match_max_tokens=args.evaluation_match_max_tokens,
        evaluation_repair_attempts=args.evaluation_repair_attempts,
        evaluation_timeout_seconds=args.evaluation_timeout_seconds,
        evaluation_model_timeout_seconds=args.evaluation_model_timeout_seconds,
        evaluation_max_model_calls=args.evaluation_max_model_calls,
        evaluation_max_concurrent_segment_maps=(
            args.evaluation_max_concurrent_segment_maps
        ),
        evaluation_match_batch_size=args.evaluation_match_batch_size,
        evaluation_history_batch_size=args.evaluation_history_batch_size,
        evaluation_match_full_scan_threshold=(
            args.evaluation_match_full_scan_threshold
        ),
        evaluation_match_candidate_limit_per_fact=(
            args.evaluation_match_candidate_limit_per_fact
        ),
        evaluation_match_candidate_union_limit=(
            args.evaluation_match_candidate_union_limit
        ),
        evaluation_personalization_batch_size=(
            args.evaluation_personalization_batch_size
        ),
        evaluation_personalization_max_tokens=(
            args.evaluation_personalization_max_tokens
        ),
        evaluation_personalization_max_model_calls=(
            args.evaluation_personalization_max_model_calls
        ),
        evaluation_personalization_timeout_seconds=(
            args.evaluation_personalization_timeout_seconds
        ),
        worker_timeout_seconds=args.worker_timeout_seconds,
        global_timeout_seconds=args.global_timeout_seconds,
        max_replan_rounds=args.max_replan_rounds,
        planner_max_tasks=args.planner_max_tasks,
        planner_max_depth=args.planner_max_depth,
        planner_max_tasks_per_capability=(args.planner_max_tasks_per_capability),
        planner_max_total_model_calls=args.planner_max_total_model_calls,
        planner_max_total_tool_calls=args.planner_max_total_tool_calls,
        planner_max_task_timeout_seconds=(args.planner_max_task_timeout_seconds),
        schema_repair_attempts=args.schema_repair_attempts,
    )
    run_dir = create_run_dir(args.output_dir, args.project)
    (run_dir / "run_info.json").write_text(
        json.dumps(
            {
                "workflow_mode": "multi_agent_v2",
                "created_at": datetime.now().isoformat(),
                "user_id": args.user_id,
                "project": args.project,
                "source_time": request.source_time,
                "current_file": str(args.current_file),
                "store": {
                    "qdrant_path": QDRANT_SUMMARY_PATH,
                    "collection": SUMMARY_COLLECTION_NAME,
                },
                "filters": {
                    "user_id": args.user_id,
                    "project": args.project,
                    "memory_schema_version": "summary_typed_memory.v1",
                },
                "llm_runtime": get_llm_runtime_info(),
                "config": config.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    memory = build_summary_memory()
    try:
        pipeline = build_multi_agent_v2_pipeline(
            memory=memory,
            llm_call=call_summary_llm,
            user_id=args.user_id,
            project=args.project,
            output_dir=run_dir,
            config=config,
        )
        result = await pipeline.run(request)
    finally:
        # CLI 负责 Memory 生命周期；Pipeline 内的 Summarize 工具保持只读。
        close_summary_memory(memory)

    print(f"STATUS: {result.status.value}")
    print(f"OUTPUT_DIR: {run_dir}")
    print(f"COVERAGE_COMPLETE: {result.coverage_complete}")
    if result.review_enabled:
        print("REVIEW_MODE: shadow")
        print(
            "REVIEW_TERMINATION: "
            f"{result.review_termination.value if result.review_termination else 'unknown'}"
        )
        print(f"RECOMMENDED_CANDIDATE_INDEX: {result.recommended_candidate_index}")
    if result.evaluation_enabled:
        print(f"EVALUATION_STATUS: {result.evaluation_status.value}")
        print(
            "PERSONALIZATION_EVALUATION_STATUS: "
            f"{result.personalization_evaluation_status.value}"
        )
        metrics_path = Path(result.evaluation_metrics_path or "")
        if metrics_path.is_file():
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            core_metrics = metrics_payload.get("core_metrics", {})
            for label, key in (
                ("MUST_FACT_RECALL", "must_fact_recall"),
                ("USEFUL_FACT_PRECISION", "useful_fact_precision"),
                ("FACT_SUPPORT_PRECISION", "fact_support_precision"),
                ("MINUTES_FACT_F1", "minutes_fact_f1"),
            ):
                value = core_metrics.get(key, {}).get("value")
                print(f"{label}: {'N/A' if value is None else value}")
            for section_name, label, key in (
                (
                    "history_metrics",
                    "HISTORICAL_FACT_F1",
                    "historical_fact_f1",
                ),
                (
                    "transition_metrics",
                    "STATE_TRANSITION_F1",
                    "state_transition_f1",
                ),
            ):
                value = metrics_payload.get(section_name, {}).get(key, {}).get("value")
                print(f"{label}: {'N/A' if value is None else value}")
        personalization_path = Path(
            result.personalization_evaluation_metrics_path or ""
        )
        if personalization_path.is_file():
            personalization_payload = json.loads(
                personalization_path.read_text(encoding="utf-8")
            )
            for target in personalization_payload.get("per_target", []):
                entity = target.get("target_entity", "unknown")
                core = target.get("core_metrics", {})
                print(f"TARGET_ENTITY: {entity}")
                for label, key in (
                    (
                        "TARGET_CONTENT_MUST_RECALL",
                        "target_content_must_recall",
                    ),
                    (
                        "TARGET_CONTENT_USEFUL_PRECISION",
                        "target_content_useful_precision",
                    ),
                    ("TARGET_CONTENT_F1", "target_content_f1"),
                    (
                        "TARGET_DIRECTED_MUST_RECALL",
                        "target_directed_must_recall",
                    ),
                    (
                        "TARGET_DIRECTED_USEFUL_PRECISION",
                        "target_directed_useful_precision",
                    ),
                    ("TARGET_DIRECTED_F1", "target_directed_f1"),
                ):
                    value = core.get(key, {}).get("value")
                    print(f"{label}: {'N/A' if value is None else value}")
    if result.final_summary:
        print("\n" + result.final_summary)
    if result.warnings:
        print("\n[WARNINGS]")
        for warning in result.warnings:
            print(f"- {warning}")
    return (
        0
        if result.status
        in {
            RunStatus.COMPLETED,
            RunStatus.COMPLETED_WITH_WARNINGS,
        }
        else 1
    )


def main() -> None:
    """通过 asyncio 运行 CLI 主流程，并将结果状态转换为进程退出码。"""
    os.environ.setdefault("SMW_LLM_BACKEND", "openai_compatible")
    args = parse_args()
    validate_llm_configuration()
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
