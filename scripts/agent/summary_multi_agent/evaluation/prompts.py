"""事实级评测各阶段 Prompt。"""

from __future__ import annotations

import json
from typing import Any

from ..schemas import RawSegment, SummaryRequest
from .schemas import (
    EvaluationFactMatch,
    EvaluationFactRelation,
    EvaluationSourceFact,
    EvaluationSummaryFact,
    PersonalizationSourceAnnotation,
)


def _json(value: Any) -> str:
    """将评测输入稳定序列化为 Prompt 片段。"""
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def build_current_fact_prompt(request: SummaryRequest, segment: RawSegment) -> str:
    """构造独立抽取当前源事实的评测 Prompt。"""
    return f"""你是 Summary Multi-Agent 的 Evaluation Source Fact Builder。

请独立分析当前会议原始材料分段，抽取可用于评价最终纪要的原子事实。
这是评测事实构建，不得参考已经生成的摘要。

规则：
1. 每条事实只能表达一个原子命题。
2. evidence_quote 必须是分段中的连续逐字原文。
3. 不得猜测负责人、状态、日期、数值或规格。
4. 建议、提议、否决、未决、确认、完成和取消必须区分。
5. salience 只能是 must_include、useful_optional、exclude。
6. must_include 表示根据 summary_request，缺失会明显损害纪要用途。
7. useful_optional 表示写入有价值但不强制；exclude 表示闲聊、重复或已否决且无需说明。
8. state 只能是 active、confirmed、proposed、rejected、pending、completed、cancelled、superseded、unknown。
9. classification_reason 最多 30 个中文字符；不要重复 evidence 或事实正文。
10. 只返回 JSON，不要输出评分或解释性正文。

输出格式：
{{
  "facts": [
    {{
      "fact": "原子事实",
      "fact_type": "fact/decision/constraint/task/risk/preference/question/result/progress/other",
      "state": "confirmed",
      "salience": "must_include",
      "entities": ["实体"],
      "evidence_quote": "逐字原文",
      "speaker": null,
      "turn_id": null,
      "classification_reason": "简短原因"
    }}
  ]
}}

request_id: {request.request_id}
source_time: {request.source_time or "未提供"}
summary_request: {request.summary_request}
segment_id: {segment.segment_id}
segment_offset: [{segment.start_offset}, {segment.end_offset}]

<evaluation_current_segment>
{segment.text}
</evaluation_current_segment>
"""


def build_history_fact_prompt(
    request: SummaryRequest,
    current_facts: tuple[EvaluationSourceFact, ...],
    history_batch: dict[str, dict[str, Any]],
) -> str:
    """构造结合当前事实分类可见历史记录的评测 Prompt。"""
    compact_current = [
        {
            "fact_id": item.fact_id,
            "fact": item.fact,
            "fact_type": item.fact_type,
            "state": item.state.value,
            "salience": item.salience.value,
            "entities": list(item.entities),
        }
        for item in current_facts
    ]
    return f"""你是 Summary Multi-Agent 的 Evaluation Historical Fact Classifier。

请把本轮 Summarizer 实际可见的 historical_records 转换为原子历史事实，并结合当前事实判断其当前有效性和写入必要性。

规则：
1. 只能使用输入中的历史记录和当前事实，不得补充外部知识。
2. 每条输出必须引用一个已给出的 record_id。
3. disposition 只能是 active、completed、updated、superseded、contradicted、background、irrelevant、uncertain。
4. salience 只能是 must_include、useful_optional、exclude。
5. 当前事实优先。历史 pending 被当前 completed 更新时，不得继续标成 active pending。
6. 与当前请求无关的历史记录标为 background 或 irrelevant，不得为了提高覆盖率全部标 must_include。
7. 一条历史记录包含多个原子事实时可以拆分，但不得合并不同状态。
8. relation_type 只能是 history_only、confirmed、completed、updated、replaced、cancelled、unresolved_conflict。
9. history_only 不得引用当前事实；其他关系必须引用输入中的 related_current_fact_ids。无明确更新证据的冲突必须使用 unresolved_conflict。
10. classification_reason 和 relation_reason 各不超过 30 个中文字符；不要重复输入记录正文。
11. 只返回 JSON，不要直接评分。

输出格式：
{{
  "facts": [
    {{
      "record_id": "输入中的 record_id",
      "fact": "原子历史事实",
      "fact_type": "task",
      "state": "completed",
      "salience": "must_include",
      "disposition": "updated",
      "entities": ["实体"],
      "related_current_fact_ids": ["eval_current_..."],
      "relation_type": "completed",
      "relation_reason": "当前事实确认任务完成",
      "classification_reason": "简短原因"
    }}
  ]
}}

source_time: {request.source_time or "未提供"}
summary_request: {request.summary_request}

<evaluation_current_facts>
{_json(compact_current)}
</evaluation_current_facts>

<evaluation_visible_historical_records>
{_json(history_batch)}
</evaluation_visible_historical_records>
"""


def build_raw_history_fact_prompt(
    *,
    summary_request: str,
    source_id: str,
    source_time: str,
    segment: RawSegment,
    current_facts: tuple[EvaluationSourceFact, ...],
) -> str:
    """构造独立横评中带逐字 Evidence 的历史原文抽取 Prompt。"""
    compact_current = [
        {
            "fact_id": item.fact_id,
            "fact": item.fact,
            "fact_type": item.fact_type,
            "state": item.state.value,
            "salience": item.salience.value,
            "entities": list(item.entities),
        }
        for item in current_facts
    ]
    return f"""你是 Summary Multi-Agent 的 Evaluation Historical Source Builder。

请从一份历史原始材料分段抽取原子事实，并结合当前事实标注其当前有效性和状态关系。不得参考待评测纪要。

规则：
1. evidence_quote 必须是历史分段中的连续逐字原文；每条只表达一个原子命题。
2. disposition 只能是 active、completed、updated、superseded、contradicted、background、irrelevant、uncertain。
3. salience 只能是 must_include、useful_optional、exclude，并服从 summary_request。
4. relation_type 只能是 history_only、confirmed、completed、updated、replaced、cancelled、unresolved_conflict。
5. history_only 不得引用当前事实；其他关系必须引用输入中的 related_current_fact_ids。
6. completed/updated/replaced/cancelled 必须有当前材料的明确状态变化证据；不能把普通矛盾猜成变化。
7. 当前事实优先；旧状态已变化时不得把旧状态继续标成当前有效状态。
8. state 只能是 active、confirmed、proposed、rejected、pending、completed、cancelled、superseded、unknown。
9. classification_reason 和 relation_reason 各不超过 30 个中文字符。
10. 只返回完整 JSON，不输出评分或解释正文。

输出格式：
{{
  "facts": [
    {{
      "fact": "原子历史事实",
      "fact_type": "task",
      "state": "pending",
      "salience": "must_include",
      "disposition": "updated",
      "entities": ["实体"],
      "evidence_quote": "历史原文逐字片段",
      "speaker": null,
      "turn_id": null,
      "related_current_fact_ids": ["eval_current_..."],
      "relation_type": "completed",
      "relation_reason": "当前确认任务完成",
      "classification_reason": "简短原因"
    }}
  ]
}}

source_id: {source_id}
source_time: {source_time}
summary_request: {summary_request}
segment_id: {segment.segment_id}
segment_offset: [{segment.start_offset}, {segment.end_offset}]

<evaluation_current_facts>
{_json(compact_current)}
</evaluation_current_facts>

<evaluation_historical_segment>
{segment.text}
</evaluation_historical_segment>
"""


def build_summary_fact_prompt(summary_segment: RawSegment) -> str:
    """构造从正式摘要分段抽取原子 claim 的评测 Prompt。"""
    return f"""你是 Summary Multi-Agent 的 Evaluation Summary Fact Extractor。

请从正式输出纪要分段中抽取实际表达出的全部原子事实。

规则：
1. 每条只表达一个可核验命题。
2. evidence_quote 必须是纪要分段中的连续逐字原文。
3. 不判断事实真假，不与源材料匹配，不输出分数。
4. claimed_scope 只能是 current、historical、mixed、unknown；不确定时使用 unknown。
5. 标题和纯格式文字不是事实。
6. 重复表达仍需分别抽取，后续由确定性逻辑识别冗余。
7. 只返回 JSON。

输出格式：
{{
  "facts": [
    {{
      "fact": "摘要中的原子事实",
      "evidence_quote": "摘要逐字片段",
      "claimed_scope": "current",
      "entities": ["实体"]
    }}
  ]
}}

summary_segment_id: {summary_segment.segment_id}
summary_segment_offset: [{summary_segment.start_offset}, {summary_segment.end_offset}]

<evaluation_summary_segment>
{summary_segment.text}
</evaluation_summary_segment>
"""


def build_match_prompt(
    summary_facts: tuple[EvaluationSummaryFact, ...],
    source_facts: tuple[EvaluationSourceFact, ...],
    *,
    source_scope: str,
    temporal_support: dict[str, Any],
    fact_relations: tuple[EvaluationFactRelation, ...] = (),
) -> str:
    """构造当前优先的摘要事实与源事实匹配 Prompt。"""
    claims = [
        {
            "summary_fact_id": item.summary_fact_id,
            "fact": item.fact,
            "claimed_scope": item.claimed_scope.value,
        }
        for item in summary_facts
    ]
    sources = [
        {
            "fact_id": item.fact_id,
            "fact": item.fact,
            "fact_type": item.fact_type,
            "state": item.state.value,
            "salience": item.salience.value,
            "history_disposition": (
                item.history_disposition.value
                if item.history_disposition is not None
                else None
            ),
            "entities": list(item.entities),
            "evidence": [evidence.evidence_text for evidence in item.evidence],
        }
        for item in source_facts
    ]
    relations = [
        {
            "relation_id": item.relation_id,
            "fact_family_id": item.fact_family_id,
            "historical_fact_id": item.historical_fact_id,
            "current_fact_ids": list(item.current_fact_ids),
            "relation_type": item.relation_type.value,
            "historical_state": item.historical_state.value,
            "current_state": item.current_state.value,
        }
        for item in fact_relations
    ]
    return f"""你是 Summary Multi-Agent 的 Evaluation Fact Matcher。

请判断每条正式摘要事实是否被本批源事实支持。你只做分类，不给分数。

规则：
1. status 只能是 supported_full、supported_partial、contradicted、unsupported。
2. 内容、状态、负责人、时间、数值和规格都正确，才能是 supported_full。
3. 只有部分正确或缺少关键限定时是 supported_partial。
4. 与源事实的最终状态冲突时是 contradicted。
5. 输入的 source facts 已是系统筛选出的完整候选集合；没有对应事实时是 unsupported，禁止猜测。
6. 只能引用输入中的 fact_id 和 summary_fact_id。
7. Temporal Worker 结果只用于核对已完成的日期解析，不得自行重新计算相对日期。
8. 历史 superseded/contradicted 内容被当作当前状态陈述时，stale_history=true。
9. source_relations 只用于识别历史语义和状态变化；当前事实仍是当前状态的第一真值来源。
10. transition_assessment 只能是 not_applicable、current_state_only、transition_full、transition_partial、stale_as_current、transition_contradicted。
11. 只有摘要明确且正确表达过去状态到当前状态的变化时才标 transition_full；只表达正确当前状态时标 current_state_only；把旧状态写成当前状态时标 stale_as_current。
12. matched_relation_ids 只能引用输入关系 ID；没有相关关系时返回空数组。
13. reason 只写一句不超过 30 个中文字符的可审计说明，不输出思维链。
14. 对能判断的 summary_fact_id 返回一次；若某条在候选集合中无法判断，可以省略，系统会保守标记为 unsupported。不要为了凑数强行匹配。

输出格式：
{{
  "matches": [
    {{
      "summary_fact_id": "SF...",
      "status": "supported_full",
      "matched_source_fact_ids": ["EF..."],
      "primary_source_fact_id": "EF...",
      "matched_relation_ids": ["EFR..."],
      "transition_assessment": "transition_full",
      "stale_history": false,
      "reason_code": "same_fact_and_state",
      "reason": "简短说明"
    }}
  ]
}}

source_scope: {source_scope}

<evaluation_summary_facts>
{_json(claims)}
</evaluation_summary_facts>

<evaluation_source_facts>
{_json(sources)}
</evaluation_source_facts>

<evaluation_source_relations>
{_json(relations)}
</evaluation_source_relations>

<authorized_temporal_worker_support>
{_json(temporal_support)}
</authorized_temporal_worker_support>
"""


def build_comparison_scope_match_prompt(
    summary_facts: tuple[EvaluationSummaryFact, ...],
    source_facts: tuple[EvaluationSourceFact, ...],
    *,
    source_scope: str,
    temporal_support: dict[str, Any],
    fact_relations: tuple[EvaluationFactRelation, ...] = (),
) -> str:
    """仅为独立横评追加双范围语义，不改变在线 Evaluation Prompt。"""
    base_prompt = build_match_prompt(
        summary_facts,
        source_facts,
        source_scope=source_scope,
        temporal_support=temporal_support,
        fact_relations=fact_relations,
    )
    if source_scope == "all_sources":
        rule = (
            "本批同时包含当前与历史事实。允许多条跨材料证据共同支持一个 Claim；"
            "当前材料明确给出的最终状态优先，旧状态不能覆盖当前状态；"
            "未决冲突不得自行解释。"
        )
    elif source_scope == "current_only":
        rule = (
            "本批只允许使用当前材料事实。即使历史材料可能支持该 Claim，"
            "也不能在本范围内计为支持。"
        )
    else:
        raise ValueError(f"unsupported comparison scope: {source_scope}")
    return (
        base_prompt
        + "\n<evaluation_comparison_scope_rule>\n"
        + rule
        + "\n</evaluation_comparison_scope_rule>\n"
    )


def build_personalization_source_prompt(
    target_entity: str,
    source_facts: tuple[EvaluationSourceFact, ...],
) -> str:
    """构造目标实体与已 grounding 源事实之间的价值标注 Prompt。"""
    facts = [
        {
            "source_fact_id": item.fact_id,
            "source_scope": item.source_scope.value,
            "fact": item.fact,
            "fact_type": item.fact_type,
            "state": item.state.value,
            "history_disposition": (
                item.history_disposition.value
                if item.history_disposition is not None
                else None
            ),
            "entities": list(item.entities),
            "evidence": [evidence.evidence_text for evidence in item.evidence],
        }
        for item in source_facts
    ]
    return f"""你是 Summary Multi-Agent 的 Target Personalization Fact Classifier。

请针对一个显式目标实体，逐条判断已完成 grounding 的源事实是否应进入该实体的个性化纪要。你只做分类，不重新抽取事实、不修改事实、不打分。

规则：
1. relevance 只能是 target_must_include、target_useful_optional、not_target_relevant、uncertain。
2. target_must_include 表示遗漏会明显损害目标实体获取其责任、决定、风险、承诺、偏好或直接影响事项。
3. target_useful_optional 表示对目标实体有帮助，但不是完成其工作所必需。
4. 不得仅因事实属于共同背景就标成目标相关；必须存在输入证据支持的直接关系。
5. 不得从姓名相似、职位常识或外部知识推断角色。历史事实的当前有效性以输入 disposition 为准。
6. relation_types 只能从 actor、owner_assignee、decision_authority、affected_party、speaker、preference、commitment、risk_owner、direct_mention、role_context 中选择。
7. 必须对输入中的每个 source_fact_id 恰好返回一次，不得返回未知或重复 ID。
8. reason 最多 30 个中文字符，不输出思维链。

输出格式：
{{
  "target_entity": "目标实体",
  "annotations": [
    {{
      "source_fact_id": "EF...",
      "relevance": "target_must_include",
      "relation_types": ["owner_assignee"],
      "reason": "目标实体是任务负责人"
    }}
  ]
}}

target_entity: {target_entity}

<personalization_source_facts>
{_json(facts)}
</personalization_source_facts>
"""


def build_personalization_claim_prompt(
    target_entity: str,
    summary_facts: tuple[EvaluationSummaryFact, ...],
    *,
    matches: tuple[EvaluationFactMatch, ...] = (),
    source_facts: tuple[EvaluationSourceFact, ...] = (),
    source_annotations: tuple[PersonalizationSourceAnnotation, ...] = (),
    fact_relations: tuple[EvaluationFactRelation, ...] = (),
) -> str:
    """构造正式摘要 claim 对目标实体指向性的标注 Prompt。"""
    facts = [
        {
            "summary_fact_id": item.summary_fact_id,
            "fact": item.fact,
            "evidence_quote": item.evidence_quote,
            "claimed_scope": item.claimed_scope.value,
            "entities": list(item.entities),
        }
        for item in summary_facts
    ]
    summary_ids = {item.summary_fact_id for item in summary_facts}
    source_by_id = {item.fact_id: item for item in source_facts}
    match_context = []
    for item in matches:
        if item.summary_fact_id not in summary_ids:
            continue
        match_context.append(
            {
                "summary_fact_id": item.summary_fact_id,
                "match_status": item.match_status.value,
                "matched_source_facts": [
                    {
                        "source_fact_id": source_id,
                        "fact": source_by_id[source_id].fact,
                        "entities": list(source_by_id[source_id].entities),
                    }
                    for source_id in item.matched_source_fact_ids
                    if source_id in source_by_id
                ],
                "matched_relation_ids": list(item.matched_relation_ids),
            }
        )
    target_source_context = [
        item.model_dump(mode="json")
        for item in source_annotations
        if item.target_entity == target_entity
    ]
    relation_context = [item.model_dump(mode="json") for item in fact_relations]
    return f"""你是 Summary Multi-Agent 的 Target Personalization Claim Classifier。

请针对一个显式目标实体，逐条判断正式 candidate 0 中已经抽取的摘要 claim 是否在面向该实体表达。你只做分类，不判断真假、不重新匹配源事实、不打分。

规则：
1. relevance 只能是 target_directed、target_contextual、not_target_directed、uncertain。
2. target_directed 表示 claim 直接陈述目标实体的任务、决定、风险、承诺、偏好、发言或状态。
3. target_contextual 表示 claim 虽非目标实体行为，但会直接影响其责任或后续行动。
4. 共同背景、一般结论或仅与其他实体有关的内容标为 not_target_directed。
5. 可以使用 matched_source_context 和 target_source_annotations 识别省略主语但语义明确的目标内容；不得重新判断事实真假。
6. 不得从姓名相似、职位常识或外部知识推断关系。unsupported claim 也必须按其文本实际指向分类，不能因为可能错误就排除。
7. relation_types 只能从 actor、owner_assignee、decision_authority、affected_party、speaker、preference、commitment、risk_owner、direct_mention、role_context 中选择。
8. 必须对输入中的每个 summary_fact_id 恰好返回一次，不得返回未知或重复 ID。
9. reason 最多 30 个中文字符，不输出思维链。

输出格式：
{{
  "target_entity": "目标实体",
  "annotations": [
    {{
      "summary_fact_id": "SF...",
      "relevance": "target_directed",
      "relation_types": ["direct_mention"],
      "reason": "摘要直接陈述其任务"
    }}
  ]
}}

target_entity: {target_entity}

<personalization_summary_facts>
{_json(facts)}
</personalization_summary_facts>

<matched_source_context>
{_json(match_context)}
</matched_source_context>

<target_source_annotations>
{_json(target_source_context)}
</target_source_annotations>

<source_fact_relations>
{_json(relation_context)}
</source_fact_relations>
"""
