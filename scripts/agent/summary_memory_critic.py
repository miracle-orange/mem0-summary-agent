"""构造 Summary Critic、规范化问题报告并生成不覆盖原摘要的修订草稿。"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from summary_memory_typed_memory import extract_json_value


SUMMARY_CRITIC_SCHEMA_VERSION = "summary_critic_report.v2"
SUMMARY_REWRITE_SCHEMA_VERSION = "summary_rewrite_report.v1"

CRITIC_ISSUE_FIELDS = (
    "issue_id",
    "severity",
    "issue_type",
    "summary_style",
    "location",
    "problem",
    "source_evidence",
    "summary_quote",
    "evidence_refs",
    "conflicting_artifact",
    "suggested_fix",
    "required_action",
    "rewrite_instruction",
    "needs_retrieval",
    "route_to",
)

ALLOWED_SEVERITIES = {"high", "medium", "low"}
ALLOWED_ROUTES = {
    "summary_writer",
    "time_agent",
    "decision_constraint_agent",
    "task_agent",
    "risk_agent",
    "preference_agent",
    "artifact_board_builder",
    "unknown",
}
ALLOWED_REQUIRED_ACTIONS = {
    "keep",
    "soften",
    "remove",
    "correct_date",
    "correct_owner",
    "mark_unclear",
    "add_missing_constraint",
}
REWRITE_TRIGGER_SEVERITIES = {"high", "medium"}


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _normalize_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _normalize_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "pass", "passed", "1"}:
            return True
        if normalized in {"false", "no", "fail", "failed", "0"}:
            return False
    return default


def _summaries_block(summaries: dict[str, str]) -> str:
    if not summaries:
        return "无摘要文本。"

    ordered_styles = [style for style in ("structured", "natural") if style in summaries]
    ordered_styles.extend(style for style in summaries if style not in ordered_styles)
    parts = []
    for style in ordered_styles:
        parts.append(f"# {style}\n\n{summaries.get(style, '').strip()}")
    return "\n\n".join(parts)


def _stable_issue_id(raw_issue: dict[str, Any], index: int) -> str:
    explicit = _normalize_text(raw_issue.get("issue_id"))
    if explicit:
        return explicit
    seed = "|".join(
        [
            _normalize_text(raw_issue.get("issue_type")),
            _normalize_text(raw_issue.get("summary_style")),
            _normalize_text(raw_issue.get("location")),
            _normalize_text(raw_issue.get("problem")),
            _normalize_text(raw_issue.get("summary_quote")),
        ]
    )
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:10]
    return f"issue_{index}_{digest}"


def _infer_required_action(raw_issue: dict[str, Any]) -> str:
    explicit = _normalize_text(raw_issue.get("required_action")).lower()
    if explicit in ALLOWED_REQUIRED_ACTIONS:
        return explicit

    issue_type = _normalize_text(raw_issue.get("issue_type")).lower()
    suggested = _normalize_text(raw_issue.get("suggested_fix")).lower()
    text = f"{issue_type} {suggested}"
    if "owner" in text or "负责人" in text:
        return "correct_owner"
    if "date" in text or "time" in text or "日期" in text or "时间" in text:
        return "correct_date"
    if "constraint" in text or "约束" in text:
        return "add_missing_constraint"
    if "remove" in text or "删除" in text or "去掉" in text:
        return "remove"
    if "unclear" in text or "待确认" in text or "不明" in text:
        return "mark_unclear"
    if "keep" in text or "保留" in text:
        return "keep"
    return "soften"


def build_summary_critic_prompt(
    current_content: str,
    source_time: str,
    artifact_board: dict[str, Any],
    retrieved_typed_memories: list[dict[str, Any]],
    summaries: dict[str, str],
) -> str:
    artifact_block = _json_block(artifact_board or {})
    typed_memory_block = _json_block(retrieved_typed_memories or [])
    summary_block = _summaries_block(summaries)
    return f"""你是 Summary Critic Agent，负责检查候选增强摘要是否忠实于当前原始材料和有证据的历史记忆。当前事实以当前原始材料为最高优先级；artifact board 只整理历史上下文。

你的任务不是重写摘要，也不是评分。你只输出结构化检查报告，指出具体问题、证据依据和建议修正方向。
报告必须足够可执行，便于后续 summary rewrite 或专门 agent 路由。

当前资料时间锚点 source_time：
{source_time or "未提供"}

当前原始材料（当前事实的最高优先级）：
{current_content}

artifact board：
{artifact_block}

retrieved typed memories：
{typed_memory_block}

待检查摘要：
{summary_block}

必须重点检查：
1. 时间日期和星期是否一致。
2. 当前材料之后的后续任务是否被写到了早于 source_time 的日期；若早于 source_time，摘要是否明确说明这是旧计划、逾期或待确认。
3. 历史任务、历史截止窗口或 completion_status/status=unknown 的事项是否被摘要误写成已完成、已按期完成或已兑现。
4. 当前材料中的“重新讨论、建议、考虑、评估、待确认”是否被误写成新决策或已确认结论。
5. 提出者、建议者、参与者是否被误写成负责人；只有当前材料或证据明确负责，才能写成负责人。
6. 历史约束、范围边界、active constraint 是否被摘要忽略、推翻或写成已改变；除非当前材料有明确证据。
7. artifact board warnings 是否被最终摘要违反。
8. preference memory 是否被合理使用；不要把弱推断偏好写成硬事实，也不要过度拟合场景。
9. 摘要是否引入当前材料、artifact board、typed memories 都没有支持的新事实。

判断原则：
- 如果摘要没有写某个历史事实，不一定是问题；只有影响当前结论、状态、约束或行动项时才记录 issue。
- 如果证据不足，问题类型应写为 evidence_gap 或 state_unclear，不要替摘要补编事实。
- high：会导致错误决策、错误负责人、错误完成状态、推翻约束或重大时间错误。
- medium：有明显过度推断、遗漏关键待确认项或表达可能误导。
- low：轻微表述风险、证据呈现不足或结构建议。
- required_action 只能是：keep / soften / remove / correct_date / correct_owner / mark_unclear / add_missing_constraint。
- needs_retrieval=true 仅当现有 artifact board / typed memories / 当前材料不足以支持直接改摘要，必须补检索时才设为 true。
- rewrite_instruction 要写成可直接执行的修改指令，而不是空泛建议。

输出必须是严格 JSON 对象，不要输出 markdown，不要输出解释文字。
输出格式：
{{
  "schema_version": "{SUMMARY_CRITIC_SCHEMA_VERSION}",
  "passed": true,
  "summary_results": {{
    "structured": {{
      "passed": true,
      "issues": []
    }},
    "natural": {{
      "passed": true,
      "issues": []
    }}
  }},
  "issues": [
    {{
      "issue_id": "issue_1",
      "severity": "high|medium|low",
      "issue_type": "time_error|task_status_overclaim|decision_state_error|owner_misattribution|constraint_violation|warning_violation|preference_overfit|unsupported_fact|evidence_gap|other",
      "summary_style": "structured|natural|all",
      "location": "摘要中的章节、段落或位置；找不到则写 unknown",
      "problem": "具体说明摘要哪里错了或风险是什么",
      "source_evidence": "支撑该判断的当前材料 / artifact board / typed memory 证据摘要",
      "summary_quote": "摘要中的原句或关键片段；没有则写空字符串",
      "evidence_refs": ["typed_memory_id、artifact section 或当前材料片段"],
      "conflicting_artifact": "相关 artifact section；没有则写空字符串",
      "suggested_fix": "建议如何改；如果需要补充检索，说明需要确认什么",
      "required_action": "keep|soften|remove|correct_date|correct_owner|mark_unclear|add_missing_constraint",
      "rewrite_instruction": "给 summary rewrite 的可执行修改指令",
      "needs_retrieval": false,
      "route_to": "summary_writer|time_agent|decision_constraint_agent|task_agent|risk_agent|preference_agent|artifact_board_builder|unknown"
    }}
  ],
  "checked_items": [
    "time_consistency",
    "future_task_dates",
    "task_completion_status",
    "decision_state",
    "owner_attribution",
    "historical_constraints",
    "artifact_warnings",
    "preference_usage",
    "unsupported_facts"
  ],
  "warnings": []
}}
"""


def normalize_critic_issue(raw_issue: Any, index: int = 1) -> Optional[dict[str, Any]]:
    if not isinstance(raw_issue, dict):
        return None

    severity = _normalize_text(raw_issue.get("severity")).lower() or "medium"
    if severity not in ALLOWED_SEVERITIES:
        severity = "medium"

    route_to = _normalize_text(raw_issue.get("route_to")) or "unknown"
    if route_to not in ALLOWED_ROUTES:
        route_to = "unknown"

    evidence_refs = [_normalize_text(item) for item in _normalize_list(raw_issue.get("evidence_refs"))]
    evidence_refs = [item for item in evidence_refs if item]

    source_evidence = _normalize_text(raw_issue.get("source_evidence"))
    if not source_evidence and evidence_refs:
        source_evidence = "; ".join(evidence_refs)

    suggested_fix = _normalize_text(raw_issue.get("suggested_fix"))
    rewrite_instruction = _normalize_text(raw_issue.get("rewrite_instruction")) or suggested_fix
    required_action = _infer_required_action({**raw_issue, "suggested_fix": suggested_fix})

    issue = {
        "issue_id": _stable_issue_id(raw_issue, index),
        "severity": severity,
        "issue_type": _normalize_text(raw_issue.get("issue_type")) or "other",
        "summary_style": _normalize_text(raw_issue.get("summary_style")) or "all",
        "location": _normalize_text(raw_issue.get("location")) or "unknown",
        "problem": _normalize_text(raw_issue.get("problem")),
        "source_evidence": source_evidence,
        "summary_quote": _normalize_text(raw_issue.get("summary_quote")),
        "evidence_refs": evidence_refs,
        "conflicting_artifact": _normalize_text(raw_issue.get("conflicting_artifact")),
        "suggested_fix": suggested_fix,
        "required_action": required_action,
        "rewrite_instruction": rewrite_instruction,
        "needs_retrieval": _normalize_bool(raw_issue.get("needs_retrieval"), False),
        "route_to": route_to,
    }
    if not issue["problem"]:
        return None
    return issue


def _collect_nested_issues(raw_report: dict[str, Any]) -> list[dict[str, Any]]:
    nested_issues: list[dict[str, Any]] = []
    raw_summary_results = raw_report.get("summary_results")
    if not isinstance(raw_summary_results, dict):
        return nested_issues

    index = 1
    for style, style_result in raw_summary_results.items():
        if not isinstance(style_result, dict):
            continue
        for raw_issue in _normalize_list(style_result.get("issues")):
            if isinstance(raw_issue, dict) and not raw_issue.get("summary_style"):
                raw_issue = {**raw_issue, "summary_style": style}
            issue = normalize_critic_issue(raw_issue, index=index)
            if issue is not None:
                nested_issues.append(issue)
                index += 1
    return nested_issues


def normalize_summary_critic_report(raw_report: dict[str, Any]) -> dict[str, Any]:
    raw_issues = _normalize_list(raw_report.get("issues"))
    issues: list[dict[str, Any]] = []
    for index, item in enumerate(raw_issues, start=1):
        issue = normalize_critic_issue(item, index=index)
        if issue is not None:
            issues.append(issue)
    if not issues:
        issues = _collect_nested_issues(raw_report)

    # Ensure unique issue_id after normalization.
    seen: dict[str, int] = {}
    for issue in issues:
        issue_id = str(issue.get("issue_id") or "issue")
        count = seen.get(issue_id, 0) + 1
        seen[issue_id] = count
        if count > 1:
            issue["issue_id"] = f"{issue_id}_{count}"

    raw_summary_results = raw_report.get("summary_results")
    summary_results: dict[str, Any] = {}
    if isinstance(raw_summary_results, dict):
        for style, style_result in raw_summary_results.items():
            if not isinstance(style_result, dict):
                continue
            style_issues = [
                issue for issue in issues if issue.get("summary_style") in {style, "all"}
            ]
            summary_results[str(style)] = {
                "passed": _normalize_bool(style_result.get("passed"), not style_issues),
                "issues": style_issues,
            }

    for style in ("structured", "natural"):
        if style in summary_results:
            continue
        style_issues = [issue for issue in issues if issue.get("summary_style") in {style, "all"}]
        if style_issues:
            summary_results[style] = {"passed": False, "issues": style_issues}

    passed = _normalize_bool(raw_report.get("passed"), not issues)
    if issues:
        passed = False

    checked_items = [_normalize_text(item) for item in _normalize_list(raw_report.get("checked_items"))]
    warnings = [_normalize_text(item) for item in _normalize_list(raw_report.get("warnings"))]

    return {
        "schema_version": SUMMARY_CRITIC_SCHEMA_VERSION,
        "passed": passed,
        "issue_count": len(issues),
        "summary_results": summary_results,
        "issues": issues,
        "checked_items": [item for item in checked_items if item],
        "warnings": [item for item in warnings if item],
    }


def _fallback_critic_report(problem: str, suggested_fix: str, raw_output: str = "", parse_error: bool = False) -> dict[str, Any]:
    issue = normalize_critic_issue(
        {
            "issue_id": "issue_critic_error",
            "severity": "high",
            "issue_type": "critic_parse_error" if parse_error else "critic_runtime_error",
            "summary_style": "all",
            "location": "critic_output",
            "problem": problem,
            "source_evidence": "",
            "summary_quote": "",
            "evidence_refs": [],
            "conflicting_artifact": "",
            "suggested_fix": suggested_fix,
            "required_action": "mark_unclear",
            "rewrite_instruction": suggested_fix,
            "needs_retrieval": False,
            "route_to": "unknown",
        },
        index=1,
    )
    report = {
        "schema_version": SUMMARY_CRITIC_SCHEMA_VERSION,
        "passed": False,
        "issue_count": 1,
        "summary_results": {},
        "issues": [issue] if issue else [],
        "checked_items": [],
        "warnings": ["critic output parse failed" if parse_error else "summary critic runtime failed"],
    }
    if raw_output:
        report["raw_output"] = raw_output
    if parse_error:
        report["parse_error"] = True
    else:
        report["runtime_error"] = True
    return report


def parse_summary_critic_json(raw_output: str) -> dict[str, Any]:
    try:
        parsed = extract_json_value(raw_output)
    except Exception as exc:  # noqa: BLE001 - keep report generation robust.
        return _fallback_critic_report(
            problem=f"Summary Critic JSON 解析失败：{exc}",
            suggested_fix="查看 raw_output，必要时提高 SMW_CRITIC_MAX_TOKENS 或重试。",
            raw_output=raw_output,
            parse_error=True,
        )

    if not isinstance(parsed, dict):
        return _fallback_critic_report(
            problem="Summary Critic 输出不是 JSON 对象。",
            suggested_fix="查看 raw_output，必要时调整 critic prompt。",
            raw_output=raw_output,
            parse_error=True,
        )

    return normalize_summary_critic_report(parsed)


def format_summary_critic_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Summary Critic Report",
        "",
        f"- Schema Version: {report.get('schema_version', SUMMARY_CRITIC_SCHEMA_VERSION)}",
        f"- Passed: {report.get('passed')}",
        f"- Issue Count: {report.get('issue_count', 0)}",
    ]

    warnings = report.get("warnings")
    if isinstance(warnings, list) and warnings:
        lines.extend(["", "## Warnings"])
        for warning in warnings:
            lines.append(f"- {_normalize_text(warning)}")

    issues = report.get("issues")
    if not isinstance(issues, list) or not issues:
        lines.extend(["", "## Issues", "", "No issues found."])
    else:
        lines.extend(["", "## Issues"])
        for index, issue in enumerate(issues, start=1):
            lines.extend(
                [
                    "",
                    f"### {index}. [{issue.get('severity', 'medium')}] {issue.get('issue_type', 'other')}",
                    f"- Issue ID: {issue.get('issue_id', '')}",
                    f"- Summary Style: {issue.get('summary_style', 'all')}",
                    f"- Location: {issue.get('location', 'unknown')}",
                    f"- Problem: {issue.get('problem', '')}",
                    f"- Summary Quote: {issue.get('summary_quote', '')}",
                    f"- Source Evidence: {issue.get('source_evidence', '')}",
                    f"- Conflicting Artifact: {issue.get('conflicting_artifact', '')}",
                    f"- Required Action: {issue.get('required_action', 'soften')}",
                    f"- Rewrite Instruction: {issue.get('rewrite_instruction', '')}",
                    f"- Needs Retrieval: {issue.get('needs_retrieval', False)}",
                    f"- Suggested Fix: {issue.get('suggested_fix', '')}",
                    f"- Route To: {issue.get('route_to', 'unknown')}",
                ]
            )
            evidence_refs = issue.get("evidence_refs")
            if isinstance(evidence_refs, list) and evidence_refs:
                lines.append(f"- Evidence Refs: {', '.join(_normalize_text(item) for item in evidence_refs)}")

    checked_items = report.get("checked_items")
    if isinstance(checked_items, list) and checked_items:
        lines.extend(["", "## Checked Items"])
        for item in checked_items:
            lines.append(f"- {_normalize_text(item)}")

    return "\n".join(lines).strip() + "\n"


def select_rewrite_issues(report: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(report, dict):
        return []
    issues = report.get("issues")
    if not isinstance(issues, list):
        return []
    selected = []
    for issue in issues:
        if not isinstance(issue, dict):
            continue
        severity = _normalize_text(issue.get("severity")).lower()
        if severity in REWRITE_TRIGGER_SEVERITIES:
            selected.append(issue)
    return selected


def should_rewrite_from_critic(report: Optional[dict[str, Any]]) -> bool:
    return bool(select_rewrite_issues(report))


def build_summary_rewrite_prompt(
    current_content: str,
    source_time: str,
    artifact_board: dict[str, Any],
    retrieved_typed_memories: list[dict[str, Any]],
    summaries: dict[str, str],
    critic_report: dict[str, Any],
) -> str:
    rewrite_issues = select_rewrite_issues(critic_report)
    issue_block = _json_block(rewrite_issues)
    artifact_block = _json_block(artifact_board or {})
    typed_memory_block = _json_block(retrieved_typed_memories or [])
    summary_block = _summaries_block(summaries)
    styles = [style for style in ("structured", "natural") if style in summaries]
    if not styles:
        styles = list(summaries.keys()) or ["structured"]
    style_list = ", ".join(styles)

    return f"""你是 Summary Rewrite Agent。你的任务是根据 Summary Critic 的 high/medium issues 生成修订草稿。

重要约束：
1. 这是 draft，不要声称覆盖原摘要。
2. 只修正 critic 指出的问题，不要借机大幅改写无关内容。
3. 不要编造当前原始材料、artifact board、typed memories 中没有支持的新事实；当前事实以当前原始材料为最高优先级。
4. 若 issue.needs_retrieval=true，对应位置应改为更谨慎表述（如待确认/证据不足），不要假装已经补检索成功。
5. 保持原有摘要风格；需要输出的风格：{style_list}
6. required_action 与 rewrite_instruction 优先执行。

当前资料时间锚点 source_time：
{source_time or "未提供"}

当前原始材料（当前事实的最高优先级）：
{current_content}

artifact board：
{artifact_block}

retrieved typed memories：
{typed_memory_block}

原摘要：
{summary_block}

需要处理的 critic issues（仅 high/medium）：
{issue_block}

请直接输出修订后的摘要草稿，使用以下格式，不要输出额外解释：

# structured

（若本次需要 structured，在此给出修订后的 structured 摘要；否则可省略该节）

# natural

（若本次需要 natural，在此给出修订后的 natural 摘要；否则可省略该节）
"""


def parse_rewritten_summaries(raw_output: str, original_summaries: dict[str, str]) -> dict[str, str]:
    text = (raw_output or "").replace("\r\n", "\n").strip()
    if not text:
        return {}

    sections: dict[str, list[str]] = {}
    current = None
    for line in text.split("\n"):
        stripped = line.strip()
        lower = stripped.lower()
        if lower in {"# structured", "## structured"} or lower == "structured":
            current = "structured"
            sections.setdefault(current, [])
            continue
        if lower in {"# natural", "## natural"} or lower == "natural":
            current = "natural"
            sections.setdefault(current, [])
            continue
        if current is None:
            # If model omitted headers and only one style exists, capture as that style.
            if len(original_summaries) == 1:
                current = next(iter(original_summaries.keys()))
                sections.setdefault(current, [])
            else:
                continue
        sections.setdefault(current, []).append(line)

    rewritten: dict[str, str] = {}
    for style, lines in sections.items():
        body = "\n".join(lines).strip()
        if body:
            rewritten[style] = body

    # Fallback: whole output for single-style runs.
    if not rewritten and len(original_summaries) == 1:
        style = next(iter(original_summaries.keys()))
        rewritten[style] = text
    return rewritten


def build_summary_rewrite_report(
    critic_report: dict[str, Any],
    rewritten_summaries: dict[str, str],
    enabled: bool,
    triggered: bool,
    notes: str = "",
) -> dict[str, Any]:
    rewrite_issues = select_rewrite_issues(critic_report)
    return {
        "schema_version": SUMMARY_REWRITE_SCHEMA_VERSION,
        "enabled": enabled,
        "triggered": triggered,
        "issue_count": len(rewrite_issues),
        "issue_ids": [issue.get("issue_id", "") for issue in rewrite_issues],
        "needs_retrieval_issue_ids": [
            issue.get("issue_id", "")
            for issue in rewrite_issues
            if issue.get("needs_retrieval")
        ],
        "styles": list(rewritten_summaries.keys()),
        "notes": notes
        or "Rewrite draft only. Original summary_structured.md / summary_natural.md are not overwritten.",
    }
