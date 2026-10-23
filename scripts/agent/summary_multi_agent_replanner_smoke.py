"""对 Model Replanner 进行真实模型只读冒烟验证，不写入长期 Memory。"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from summary_memory_llm import call_summary_llm, get_llm_runtime_info
from summary_multi_agent.adapters import SyncCallableModelClient
from summary_multi_agent.planner import (
    ModelDrivenPlanner,
    RuleBasedPlanner,
    build_planner_input,
    validate_plan_against_input,
)
from summary_multi_agent.raw_material import build_raw_material_bundle
from summary_multi_agent.schemas import (
    AgentResult,
    AgentStatus,
    EvidenceRef,
    ExecutionPlan,
    PlanSource,
    RunConfig,
    SourceKind,
    SubTask,
    SummaryRequest,
    TaskType,
    ToolCapabilitySpec,
    WorkerCapabilitySpec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a read-only real-model P6.5 replanner smoke scenario. "
            "No Memory or Qdrant operation is performed."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/agent_multi_v2_p65_replanner_smoke"),
    )
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--repair-max-tokens", type=int, default=1800)
    parser.add_argument("--repair-attempts", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args()


def build_scenario() -> tuple[
    SummaryRequest,
    ExecutionPlan,
    dict[str, AgentResult],
    tuple[WorkerCapabilitySpec, ...],
    tuple[ToolCapabilitySpec, ...],
    RunConfig,
]:
    request = SummaryRequest(
        current_material=(
            "项目组确认需求证据已经检索完成。原计划要求明天中午前完成时间安排核验，"
            "但日历服务发生一次临时连接失败，请在保留已成功检索结果的前提下重新规划。"
        ),
        source_time="2026-08-10",
        summary_request="生成突出任务、证据和时间安排的结构化摘要。",
    )
    workers = (
        WorkerCapabilitySpec(
            capability_id=TaskType.SEMANTIC_MEMORY,
            description="Retrieve typed historical evidence.",
            allowed_tools=("typed_memory_search",),
        ),
        WorkerCapabilitySpec(
            capability_id=TaskType.TEMPORAL_REASONING,
            description="Resolve current time expressions.",
            allowed_tools=("calendar_resolve",),
        ),
    )
    tools = (
        ToolCapabilitySpec(
            tool_name="typed_memory_search",
            description="Read-only typed memory search.",
            allowed_task_types=(TaskType.SEMANTIC_MEMORY,),
        ),
        ToolCapabilitySpec(
            tool_name="calendar_resolve",
            description="Resolve a supported temporal expression.",
            allowed_task_types=(TaskType.TEMPORAL_REASONING,),
        ),
    )
    previous_plan = ExecutionPlan(
        source=PlanSource.MODEL,
        tasks=(
            SubTask(
                task_id="retrieve_project_evidence",
                task_type=TaskType.SEMANTIC_MEMORY,
                description="Retrieve historical project task evidence.",
                allowed_tools=("typed_memory_search",),
                inputs={
                    "queries": ["项目任务 时间安排"],
                    "memory_types": ["task", "time"],
                    "top_k_per_type": 5,
                },
                max_model_calls=0,
                max_tool_calls=1,
            ),
            SubTask(
                task_id="resolve_project_deadline",
                task_type=TaskType.TEMPORAL_REASONING,
                description="Resolve the required project deadline expression.",
                critical=True,
                allowed_tools=("calendar_resolve",),
                max_model_calls=0,
                max_tool_calls=1,
            ),
        ),
        rationale="Retrieve evidence and normalize the required deadline.",
    )
    results = {
        "retrieve_project_evidence": AgentResult(
            task_id="retrieve_project_evidence",
            task_type=TaskType.SEMANTIC_MEMORY,
            status=AgentStatus.SUCCESS,
            payload={
                "records": [
                    {
                        "memory_id": "smoke_memory_001",
                        "memory_type": "task",
                        "content": "项目需求证据已经确认并可供后续任务复用。",
                    }
                ]
            },
            evidence=(
                EvidenceRef(
                    source_kind=SourceKind.LONG_TERM_MEMORY,
                    memory_id="smoke_memory_001",
                    evidence_text="项目需求证据已经确认并可供后续任务复用。",
                ),
            ),
        ),
        "resolve_project_deadline": AgentResult(
            task_id="resolve_project_deadline",
            task_type=TaskType.TEMPORAL_REASONING,
            status=AgentStatus.FAILED,
            error="transient calendar connection failure",
        ),
    }
    config = RunConfig(
        workflow_mode="multi_agent_v2",
        planner_mode="model",
        worker_mode="rule",
        max_replan_rounds=2,
        planner_max_total_model_calls=8,
        planner_max_total_tool_calls=16,
    )
    return request, previous_plan, results, workers, tools, config


async def run(args: argparse.Namespace) -> Path:
    run_dir = args.output_dir / (
        datetime.now().strftime("%Y%m%d-%H%M%S") + "_p65_replanner_smoke"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    request, previous_plan, results, workers, tools, config = build_scenario()
    raw_material = build_raw_material_bundle(
        request.current_material,
        max_segment_chars=1000,
    )
    planner_input = build_planner_input(
        request=request,
        raw_material=raw_material,
        worker_capabilities=workers,
        tool_capabilities=tools,
        config=config,
        previous_plan=previous_plan,
        results=results,
        plan_history=(previous_plan,),
        task_attempts={
            "retrieve_project_evidence": 1,
            "resolve_project_deadline": 1,
        },
        worker_tool_calls_used=2,
        replan_round=1,
        failure_reasons=(
            "resolve_project_deadline: transient calendar connection failure",
        ),
    )
    planner = ModelDrivenPlanner(
        model_client=SyncCallableModelClient(call_summary_llm),
        fallback_planner=RuleBasedPlanner(),
        output_dir=run_dir,
        replanner_max_tokens=args.max_tokens,
        replanner_repair_max_tokens=args.repair_max_tokens,
        replanner_repair_attempts=args.repair_attempts,
        replanner_timeout_seconds=args.timeout_seconds,
    )
    plan = validate_plan_against_input(
        await planner.replan(planner_input),
        planner_input,
    )
    (run_dir / "replan_plan.json").write_text(
        json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "smoke_run_info.json").write_text(
        json.dumps(
            {
                "schema_version": "summary_multi_agent.replanner_smoke.v2",
                "llm_runtime": get_llm_runtime_info(),
                "memory_operations": [],
                "previous_plan_id": previous_plan.plan_id,
                "selected_plan_id": plan.plan_id,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    report = json.loads(
        (run_dir / "replanner_validation_report.json").read_text("utf-8")
    )
    if report.get("status") != "model_replan_accepted":
        raise RuntimeError(
            "real-model replanner smoke used fallback; inspect replanner artifacts"
        )
    print("STATUS: model_replan_accepted")
    print(f"STRATEGY: {report.get('selected_strategy')}")
    print(f"OUTPUT_DIR: {run_dir}")
    print(f"PRESERVED_TASK_IDS: {list(plan.preserved_task_ids)}")
    print(f"TASK_IDS: {[task.task_id for task in plan.tasks]}")
    print("MEMORY_OPERATIONS: []")
    return run_dir


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
