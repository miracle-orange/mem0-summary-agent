"""读取、脱敏并安全归档 Web 运行产物，供 Inspector 展示。"""

from __future__ import annotations

import json
import os
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schemas import WebArtifactSnapshot, WebRunStatus, status_from_runtime


SENSITIVE_KEY_NAMES = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "sender",
}
_PROMPT_KEY_RE = re.compile(r"(?:^|_)(?:prompt|raw_output|thinking)(?:$|_)", re.I)
TEXT_PREVIEW_LIMIT = 200_000
SAFE_ARCHIVE_NAMES = {
    "add_results.json",
    "agent_results.json",
    "artifact_board.json",
    "context_compression_report.json",
    "coverage_ledger.json",
    "e2e_preprocess_info.json",
    "e2e_run_result.json",
    "ingest_readback_report.json",
    "ingest_run_result.json",
    "ingest_stage_report.json",
    "ingest_write_journal.json",
    "evidence_index.json",
    "evaluation_fact_matches.json",
    "evaluation_comparison_basis.json",
    "evaluation_comparison_metrics.json",
    "evaluation_comparison_report.md",
    "evaluation_compare_run_result.json",
    "evaluation_source_relations.json",
    "evaluation_metrics.json",
    "evaluation_lite_reference.json",
    "evaluation_lite_result.json",
    "evaluation_lite_report.md",
    "evaluation_lite_report.html",
    "evaluation_personalization_claim_annotations.json",
    "evaluation_personalization_metrics.json",
    "evaluation_personalization_report.md",
    "evaluation_personalization_source_annotations.json",
    "evaluation_report.md",
    "evaluation_source_facts.json",
    "evaluation_summary_facts.json",
    "fact_matches.json",
    "fact_matches_all_sources.json",
    "fact_matches_current_only.json",
    "evaluation_scope_metrics.json",
    "personalization_claim_annotations.json",
    "personalization_metrics.json",
    "personalization_report.md",
    "personalization_source_annotations.json",
    "generated_history_summaries.md",
    "memory_store_probe.json",
    "model_input_report.json",
    "model_call_input_report.json",
    "history_selection_report.json",
    "minutes_document.json",
    "minutes_sources.json",
    "minutes_report.html",
    "minutes_report_manifest.json",
    "plan.json",
    "plan_history.json",
    "planner_validation_report.json",
    "replanner_validation_report.json",
    "review_blue_results.json",
    "review_candidate_summaries.json",
    "review_judge_decisions.json",
    "review_run_result.json",
    "review_verification_results.json",
    "summary_multi_agent.md",
    "summary_run_result.json",
    "typed_memory_dedup_report.json",
    "typed_memories.json",
    "web_ingest_info.json",
    "web_run_info.json",
    "working_state.json",
}


@dataclass(frozen=True)
class ArtifactRead:
    """表示单个产物读取到的值，或可展示给 Inspector 的错误。"""

    value: Any = None
    error: str = ""


def _is_sensitive_key(key: str) -> bool:
    """判断键名是否可能包含凭据、Prompt 或模型原始输出。"""
    normalized = key.strip().lower().replace("-", "_")
    return normalized in SENSITIVE_KEY_NAMES or bool(_PROMPT_KEY_RE.search(normalized))


def sanitize_value(value: Any, *, remove_prompts: bool = True) -> Any:
    """将 JSON 投影为可安全展示在 Inspector 中的结构化数据。"""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if key.lower() in {"raw_output", "prompt", "thinking"}:
                continue
            if remove_prompts and _is_sensitive_key(key):
                continue
            result[key] = sanitize_value(raw_value, remove_prompts=remove_prompts)
        return result
    if isinstance(value, list):
        return [sanitize_value(item, remove_prompts=remove_prompts) for item in value]
    if isinstance(value, tuple):
        return [sanitize_value(item, remove_prompts=remove_prompts) for item in value]
    return value


def _redact_sensitive_text(text: str) -> str:
    """在文本归档前替换环境变量凭据和常见敏感键值。"""
    redacted = text
    for env_name in (
        "SMW_DEEPSEEK_API_KEY",
        "VLLM_API_KEY",
        "OPENAI_API_KEY",
    ):
        secret = os.getenv(env_name, "")
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return re.sub(
        r"(?i)(api[_-]?key|authorization|password|secret)\s*[:=]\s*[^\s,}]+",
        r"\1=[REDACTED]",
        redacted,
    )


def _safe_relative_path(root: Path, relative_name: str) -> Path:
    """解析 run 目录内相对路径，并阻止绝对路径和目录逃逸。"""
    # 先拒绝绝对路径和父目录片段，再做 resolve/relative_to 双重校验，
    # 防止 Inspector 或下载接口读取当前 run 目录之外的文件。
    candidate_name = str(relative_name)
    candidate = Path(candidate_name)
    if candidate.is_absolute() or candidate_name in {"", "."}:
        raise ValueError("artifact path must be a non-empty relative path")
    if ".." in candidate.parts:
        raise ValueError("artifact path may not contain '..'")
    root_resolved = root.expanduser().resolve()
    resolved = (root_resolved / candidate).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError("artifact path escapes the current run directory") from exc
    return resolved


class ArtifactReader:
    """读取 Web 运行产物并构造脱敏、可容错的 Inspector 视图。"""

    def __init__(self, run_dir: Path) -> None:
        """规范化并校验运行目录，初始化按文件累计的读取错误表。"""
        self.run_dir = run_dir.expanduser().resolve()
        if not self.run_dir.exists() or not self.run_dir.is_dir():
            raise ValueError(f"run output directory does not exist: {self.run_dir}")
        self.errors: dict[str, str] = {}

    def read_json(self, relative_name: str) -> ArtifactRead:
        """安全读取并解析 JSON；缺失、损坏或越界路径转为 ``ArtifactRead.error``。"""
        try:
            path = _safe_relative_path(self.run_dir, relative_name)
        except ValueError as exc:
            error = str(exc)
            self.errors[relative_name] = error
            return ArtifactRead(error=error)
        if not path.exists():
            error = "产物不存在"
            self.errors[relative_name] = error
            return ArtifactRead(error=error)
        if not path.is_file():
            error = "产物不是文件"
            self.errors[relative_name] = error
            return ArtifactRead(error=error)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            error = f"JSON 读取失败：{type(exc).__name__}: {exc}"
            self.errors[relative_name] = error
            return ArtifactRead(error=error)
        return ArtifactRead(value=value)

    def read_text(self, relative_name: str) -> ArtifactRead:
        """安全读取 UTF-8 文本，并限制 Inspector 预览长度。"""
        try:
            path = _safe_relative_path(self.run_dir, relative_name)
        except ValueError as exc:
            error = str(exc)
            self.errors[relative_name] = error
            return ArtifactRead(error=error)
        if not path.exists():
            error = "产物不存在"
            self.errors[relative_name] = error
            return ArtifactRead(error=error)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            error = f"文本读取失败：{type(exc).__name__}: {exc}"
            self.errors[relative_name] = error
            return ArtifactRead(error=error)
        return ArtifactRead(value=text[:TEXT_PREVIEW_LIMIT])

    def _json(self, name: str, *, default: Any) -> Any:
        """读取 JSON，出错时返回指定默认值并保留错误记录。"""
        result = self.read_json(name)
        return default if result.error else result.value

    def build_snapshot(self) -> WebArtifactSnapshot:
        """按运行类型汇总 Planner、Worker、Evidence、Review、Evaluation 和 Ingest。

        运行产物会分阶段落盘，因此缺失文件按“暂不可用”记录，不让刷新 Inspector
        的 UI 回调因部分运行尚未完成而失败；展示数据会先脱敏。
        """
        # 产物是按阶段陆续写入的；缺失或损坏文件记录为可读错误，不能让
        # 用户在运行中刷新 Inspector 时触发整个页面回调异常。
        run_info = (
            self._json("web_run_info.json", default={})
            if (self.run_dir / "web_run_info.json").is_file()
            else {}
        )
        if (
            isinstance(run_info, dict)
            and run_info.get("run_type") == "evaluation_compare"
        ):
            return self._build_comparison_snapshot(run_info)
        declared_ingest = (
            isinstance(run_info, dict) and run_info.get("run_type") == "ingest"
        )
        summary_result_path = self.run_dir / "summary_run_result.json"
        result = (
            self._json("summary_run_result.json", default={})
            if summary_result_path.is_file()
            else (
                {}
                if declared_ingest
                else self._json("summary_run_result.json", default={})
            )
        )
        ingest_result = (
            self._json("ingest_run_result.json", default={})
            if (self.run_dir / "ingest_run_result.json").is_file()
            else {}
        )
        ingest_only = (declared_ingest or bool(ingest_result)) and not (
            summary_result_path.is_file()
        )

        def artifact_json(name: str, *, default: Any) -> Any:
            """对独立 Ingest 屏蔽不适用的 Summarize 产物。"""
            # 独立 Ingest 不产生 Planner/Worker/Review/Evaluation 产物；这些
            # 文件对该运行类型是“不适用”，不是失败。
            if ingest_only:
                return default
            return self._json(name, default=default)

        if not isinstance(result, dict) or not result:
            result = ingest_result
        if not isinstance(result, dict):
            self.errors["summary_run_result.json"] = "运行结果不是 JSON 对象"
            result = {}

        formal = (
            self.read_text("summary_multi_agent.md")
            if (self.run_dir / "summary_multi_agent.md").is_file()
            else (
                ArtifactRead(value="")
                if ingest_only
                else self.read_text("summary_multi_agent.md")
            )
        )
        formal_summary = (
            formal.value if not formal.error and isinstance(formal.value, str) else ""
        )

        candidate_payload = artifact_json("review_candidate_summaries.json", default=[])
        shadow_candidates: list[dict[str, Any]] = []
        if isinstance(candidate_payload, list):
            for item in candidate_payload:
                if not isinstance(item, dict) or item.get("candidate_index") == 0:
                    continue
                shadow_candidates.append(
                    {
                        "candidate_index": item.get("candidate_index"),
                        "summary": str(item.get("summary", "")),
                        "formal": False,
                    }
                )
        elif candidate_payload:
            self.errors["review_candidate_summaries.json"] = "候选摘要不是 JSON 数组"

        plan = artifact_json("plan.json", default=None)
        plan_history = artifact_json("plan_history.json", default=[])
        planner_report = artifact_json("planner_validation_report.json", default={})
        replanner_report = artifact_json("replanner_validation_report.json", default={})
        planner = {
            "plan": sanitize_value(plan) if plan is not None else {},
            "plan_history_count": len(plan_history)
            if isinstance(plan_history, list)
            else 0,
            "validation_report": sanitize_value(planner_report),
        }

        raw_workers = artifact_json("agent_results.json", default={})
        workers: list[dict[str, Any]] = []
        tool_counter: Counter[tuple[str, str]] = Counter()
        if isinstance(raw_workers, dict):
            for task_id, raw_worker in raw_workers.items():
                if not isinstance(raw_worker, dict):
                    continue
                raw_tools = raw_worker.get("tool_results", [])
                tool_summaries: list[dict[str, Any]] = []
                if isinstance(raw_tools, list):
                    for raw_tool in raw_tools:
                        if not isinstance(raw_tool, dict):
                            continue
                        tool_name = str(raw_tool.get("tool_name", "unknown"))
                        tool_status = str(raw_tool.get("status", "unknown"))
                        tool_counter[(tool_name, tool_status)] += 1
                        tool_summaries.append(
                            {
                                "call_id": raw_tool.get("call_id"),
                                "tool_name": tool_name,
                                "status": tool_status,
                                "attempts": raw_tool.get("attempts"),
                                "duration_ms": raw_tool.get("duration_ms"),
                                "diagnostics": sanitize_value(
                                    raw_tool.get("payload", {}).get("diagnostics", {})
                                    if isinstance(raw_tool.get("payload"), dict)
                                    else {}
                                ),
                            }
                        )
                execution = raw_worker.get("worker_execution")
                execution_summary = {}
                if isinstance(execution, dict):
                    execution_summary = {
                        "mode": execution.get("mode"),
                        "termination_reason": execution.get("termination_reason"),
                        "step_count": len(execution.get("steps", [])),
                        "observation_count": len(execution.get("observations", [])),
                        "model_attempt_count": len(execution.get("model_attempts", [])),
                        "fallback_reason": execution.get("fallback_reason"),
                        "consumed_prior_result_ids": execution.get(
                            "consumed_prior_result_ids", []
                        ),
                        "consumed_prior_evidence_ids": execution.get(
                            "consumed_prior_evidence_ids", []
                        ),
                    }
                workers.append(
                    {
                        "task_id": task_id,
                        "task_type": raw_worker.get("task_type"),
                        "status": raw_worker.get("status"),
                        "is_empty": raw_worker.get("is_empty", False),
                        "confidence": raw_worker.get("confidence"),
                        "warning_count": len(raw_worker.get("warnings", [])),
                        "warnings": raw_worker.get("warnings", []),
                        "error": raw_worker.get("error"),
                        "duration_ms": raw_worker.get("duration_ms"),
                        "evidence_ids": [
                            item.get("evidence_id")
                            for item in raw_worker.get("evidence", [])
                            if isinstance(item, dict)
                        ],
                        "tool_calls": tool_summaries,
                        "execution": execution_summary,
                    }
                )
        elif raw_workers:
            self.errors["agent_results.json"] = "Worker 结果不是 JSON 对象"

        tools = [
            {"tool_name": name, "status": status, "count": count}
            for (name, status), count in sorted(tool_counter.items())
        ]

        evidence = artifact_json("evidence_index.json", default={})
        coverage = artifact_json("coverage_ledger.json", default={})
        compression = artifact_json("context_compression_report.json", default={})
        store_probe = (
            self._json("memory_store_probe.json", default={})
            if (self.run_dir / "memory_store_probe.json").is_file()
            else {}
        )
        ingest = {
            "run_result": sanitize_value(ingest_result),
            "stage_report": sanitize_value(
                self._json("ingest_stage_report.json", default={})
                if (self.run_dir / "ingest_stage_report.json").is_file()
                else {}
            ),
            "readback": sanitize_value(
                self._json("ingest_readback_report.json", default={})
                if (self.run_dir / "ingest_readback_report.json").is_file()
                else {}
            ),
            "write_journal": sanitize_value(
                self._json("ingest_write_journal.json", default=[])
                if (self.run_dir / "ingest_write_journal.json").is_file()
                else []
            ),
        }
        working_state = artifact_json("working_state.json", default={})
        if isinstance(working_state, dict):
            working_memory = {
                "run_id": working_state.get("run_id"),
                "status": working_state.get("status"),
                "replan_round": working_state.get("replan_round", 0),
                "task_state_count": len(working_state.get("task_states", {})),
                "agent_result_count": len(working_state.get("agent_results", {})),
                "evidence_count": len(working_state.get("evidence_index", {})),
                "candidate_count": len(working_state.get("candidate_summaries", [])),
                "warnings": working_state.get("warnings", []),
                "errors": working_state.get("errors", []),
            }
        else:
            working_memory = {}

        review = {
            "run_result": sanitize_value(
                artifact_json("review_run_result.json", default={})
            ),
            "judge_decisions": sanitize_value(
                artifact_json("review_judge_decisions.json", default=[])
            ),
            "blue_results": sanitize_value(
                artifact_json("review_blue_results.json", default=[])
            ),
            "verification_results": sanitize_value(
                artifact_json("review_verification_results.json", default=[])
            ),
        }
        evaluation: dict[str, Any] = {}
        evaluation_enabled = bool(result.get("evaluation_enabled"))
        if (self.run_dir / "evaluation_lite_result.json").is_file():
            lite_result = artifact_json("evaluation_lite_result.json", default={})
            lite_reference = artifact_json("evaluation_lite_reference.json", default={})
            report_read = self.read_text("evaluation_lite_report.md")
            evaluation = {
                "mode": "lite",
                "status": result.get("evaluation_status", "unknown"),
                "comparison": False,
                "metrics": sanitize_value(lite_result),
                "reference": sanitize_value(lite_reference),
                "report_markdown": (
                    report_read.value
                    if not report_read.error and isinstance(report_read.value, str)
                    else ""
                ),
                "personalization": {
                    "status": "disabled",
                    "metrics": {},
                    "report_markdown": "",
                },
            }
        elif evaluation_enabled or (self.run_dir / "evaluation_metrics.json").is_file():
            metrics = artifact_json("evaluation_metrics.json", default={})
            source_snapshot = (
                artifact_json("evaluation_source_snapshot.json", default={})
                if (self.run_dir / "evaluation_source_snapshot.json").is_file()
                else {}
            )
            source_relations = (
                artifact_json("evaluation_source_relations.json", default=[])
                if (self.run_dir / "evaluation_source_relations.json").is_file()
                else []
            )
            source_facts = artifact_json("evaluation_source_facts.json", default=[])
            summary_facts = artifact_json("evaluation_summary_facts.json", default=[])
            matches = artifact_json("evaluation_fact_matches.json", default=[])
            report_read = self.read_text("evaluation_report.md")
            personalization_metrics = (
                artifact_json("evaluation_personalization_metrics.json", default={})
                if (self.run_dir / "evaluation_personalization_metrics.json").is_file()
                else {}
            )
            personalization_sources = (
                artifact_json(
                    "evaluation_personalization_source_annotations.json", default=[]
                )
                if (
                    self.run_dir / "evaluation_personalization_source_annotations.json"
                ).is_file()
                else []
            )
            personalization_claims = (
                artifact_json(
                    "evaluation_personalization_claim_annotations.json", default=[]
                )
                if (
                    self.run_dir / "evaluation_personalization_claim_annotations.json"
                ).is_file()
                else []
            )
            personalization_report_read = (
                self.read_text("evaluation_personalization_report.md")
                if (self.run_dir / "evaluation_personalization_report.md").is_file()
                else ArtifactRead(value="")
            )
            evaluation = {
                "mode": "strict",
                "status": result.get("evaluation_status", "unknown"),
                "metrics": sanitize_value(metrics),
                "snapshot_id": (
                    source_snapshot.get("snapshot_id")
                    if isinstance(source_snapshot, dict)
                    else None
                ),
                "source_relation_count": (
                    len(source_relations) if isinstance(source_relations, list) else 0
                ),
                "source_relations": sanitize_value(source_relations),
                "source_fact_count": (
                    len(source_facts) if isinstance(source_facts, list) else 0
                ),
                "summary_fact_count": (
                    len(summary_facts) if isinstance(summary_facts, list) else 0
                ),
                "problem_matches": sanitize_value(
                    [
                        item
                        for item in matches
                        if isinstance(item, dict)
                        and (
                            item.get("match_status") != "supported_full"
                            or item.get("stale_history")
                            or item.get("duplicate_of_summary_fact_id")
                        )
                    ]
                    if isinstance(matches, list)
                    else []
                ),
                "report_markdown": (
                    report_read.value
                    if not report_read.error and isinstance(report_read.value, str)
                    else ""
                ),
                "personalization": {
                    "status": result.get(
                        "personalization_evaluation_status", "disabled"
                    ),
                    "metrics": sanitize_value(personalization_metrics),
                    "source_annotation_count": (
                        len(personalization_sources)
                        if isinstance(personalization_sources, list)
                        else 0
                    ),
                    "claim_annotation_count": (
                        len(personalization_claims)
                        if isinstance(personalization_claims, list)
                        else 0
                    ),
                    "report_markdown": (
                        personalization_report_read.value
                        if not personalization_report_read.error
                        and isinstance(personalization_report_read.value, str)
                        else ""
                    ),
                },
            }
        replanning = {
            "plan_history": sanitize_value(plan_history),
            "validation_report": sanitize_value(replanner_report),
        }
        minutes_document = (
            artifact_json("minutes_document.json", default={})
            if (self.run_dir / "minutes_document.json").is_file()
            else {}
        )
        minutes_sources = (
            artifact_json("minutes_sources.json", default={})
            if (self.run_dir / "minutes_sources.json").is_file()
            else {}
        )
        minutes_manifest = (
            artifact_json("minutes_report_manifest.json", default={})
            if (self.run_dir / "minutes_report_manifest.json").is_file()
            else {}
        )

        status_value = result.get("status")
        status = status_from_runtime(str(status_value)) if status_value else None
        warnings = (
            list(result.get("warnings", []))
            if isinstance(result.get("warnings"), list)
            else []
        )
        warnings.extend(working_memory.get("warnings", []))
        errors = dict(self.errors)
        raw_errors = working_memory.get("errors", [])
        if raw_errors:
            errors["working_state"] = "; ".join(str(item) for item in raw_errors)

        return WebArtifactSnapshot(
            run_id=str(result.get("run_id", working_memory.get("run_id", ""))),
            output_dir=self.run_dir,
            status=status,
            formal_summary=formal_summary,
            shadow_candidates=shadow_candidates,
            planner=sanitize_value(planner),
            workers=sanitize_value(workers),
            tools=tools,
            evidence=sanitize_value(evidence),
            coverage=sanitize_value(coverage),
            working_memory=sanitize_value(working_memory),
            compression=sanitize_value(compression),
            store_probe=sanitize_value(store_probe),
            ingest=ingest,
            replanning=sanitize_value(replanning),
            review=review,
            evaluation=evaluation,
            minutes_document=sanitize_value(minutes_document),
            minutes_sources=sanitize_value(minutes_sources),
            minutes_manifest=sanitize_value(minutes_manifest),
            warnings=list(dict.fromkeys(str(item) for item in warnings if item)),
            errors=errors,
        )

    def _build_comparison_snapshot(
        self,
        run_info: dict[str, Any],
    ) -> WebArtifactSnapshot:
        """构造独立横评的脱敏 Inspector，不读取原始请求或模型 trace。"""
        result = self._json("evaluation_compare_run_result.json", default={})
        if (self.run_dir / "evaluation_lite_result.json").is_file():
            lite_result = self._json("evaluation_lite_result.json", default={})
            report_read = self.read_text("evaluation_lite_report.md")
            status_raw = str(
                lite_result.get("status")
                or result.get("status")
                or run_info.get("status")
                or "failed"
            )
            try:
                status = WebRunStatus(status_raw)
            except ValueError:
                status = WebRunStatus.FAILED
            warnings = lite_result.get("warnings", [])
            if not isinstance(warnings, list):
                warnings = []
            return WebArtifactSnapshot(
                run_id=str(run_info.get("run_id") or ""),
                output_dir=self.run_dir,
                status=status,
                evaluation={
                    "mode": "lite",
                    "status": status_raw,
                    "comparison": True,
                    "metrics": sanitize_value(lite_result),
                    "reference": sanitize_value(
                        self._json("evaluation_lite_reference.json", default={})
                    ),
                    "report_markdown": (
                        report_read.value
                        if not report_read.error and isinstance(report_read.value, str)
                        else ""
                    ),
                    "personalization": {
                        "status": "disabled",
                        "metrics": {},
                        "report_markdown": "",
                    },
                },
                warnings=[str(item) for item in warnings],
                errors=dict(self.errors),
            )
        metrics = self._json("evaluation_comparison_metrics.json", default={})
        relations = self._json("evaluation_source_relations.json", default=[])
        report_read = self.read_text("evaluation_comparison_report.md")
        status_raw = str(result.get("status") or run_info.get("status") or "failed")
        try:
            status = WebRunStatus(status_raw)
        except ValueError:
            status = WebRunStatus.FAILED
        warnings = result.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = []
        return WebArtifactSnapshot(
            run_id=str(run_info.get("run_id") or ""),
            output_dir=self.run_dir,
            status=status,
            evaluation={
                "mode": "strict",
                "status": status_raw,
                "comparison": True,
                "metrics": sanitize_value(metrics),
                "relations": sanitize_value(relations),
                "report_markdown": (
                    report_read.value
                    if not report_read.error and isinstance(report_read.value, str)
                    else ""
                ),
                "summaries": sanitize_value(result.get("summaries", [])),
            },
            warnings=[str(item) for item in warnings],
            errors=dict(self.errors),
        )


def read_artifact_snapshot(run_dir: Path) -> WebArtifactSnapshot:
    """读取指定运行目录并返回脱敏 Inspector 快照。"""
    return ArtifactReader(run_dir).build_snapshot()


def create_safe_artifact_archive(run_dir: Path) -> Path:
    """创建只包含白名单结构化产物的 ZIP，并排除 Prompt/raw output/隐藏思维链。"""
    root = run_dir.expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ValueError(f"run output directory does not exist: {root}")
    # 归档只允许结构化审计文件。Prompt、模型 raw output 和当前原文可能
    # 含敏感信息或隐藏推理，不进入 Web 下载包。
    archive_path = root / "web_artifacts_safe.zip"
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name not in SAFE_ARCHIVE_NAMES:
                continue
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError("artifact symlink escapes the run directory") from exc
            if path.suffix.lower() == ".json":
                try:
                    value = json.loads(path.read_text(encoding="utf-8"))
                    content = json.dumps(
                        sanitize_value(value),
                        ensure_ascii=False,
                        indent=2,
                    )
                except (OSError, UnicodeError, json.JSONDecodeError):
                    continue
            else:
                try:
                    content = path.read_text(encoding="utf-8")
                except (OSError, UnicodeError):
                    continue
            archive.writestr(str(relative), _redact_sensitive_text(content))
    return archive_path
