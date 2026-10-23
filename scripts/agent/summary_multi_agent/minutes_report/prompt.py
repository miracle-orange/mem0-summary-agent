"""Prompt construction for one-call structured rich-minutes synthesis."""

from __future__ import annotations

import json
from collections.abc import Mapping


RICH_SYNTHESIS_PROMPT_VERSION = "summary_multi_agent.rich_synthesis_prompt.v1"


def build_rich_synthesis_prompt(sections: Mapping[str, str]) -> str:
    """Build the rich prompt from the same five projected input partitions."""

    output_contract = {
        "title": "会议纪要标题",
        "overview": ["2至6条重要结论或执行概览，每条为完整短句"],
        "sections": [
            {
                "heading": "按所选场景组织的业务议题或栏目",
                "entries": [
                    {
                        "kind": "fact|discussion|decision|action|risk|open_question|history_update",
                        "title": "可选的条目标题，无则空字符串",
                        "content": "本条唯一、完整的业务陈述",
                        "owner": "仅 action 使用；未明确则空字符串",
                        "time_requirement": "仅 action 使用；保留原表达或已提供解析结果",
                        "dependency": "仅 action 使用；无则空字符串",
                        "recorded_state": "history_update 的历史记录时情况",
                        "current_state": "history_update 的本次情况",
                        "relation": "history_update 的延续、完成、取消、变更、冲突或待确认关系",
                        "current_evidence_quotes": ["当前原文中的1至3段简短逐字引文"],
                        "history_refs": ["historical_context 中的 H001 等 source_ref"],
                    }
                ],
            }
        ],
    }
    contract = json.dumps(output_contract, ensure_ascii=False, indent=2)
    return f"""你是 Summary Multi-Agent V2 的 Rich Minutes Summarizer。

请依据以下五个输入分区，一次生成可供程序渲染的结构化中文会议纪要。只返回严格 JSON，不输出 Markdown、HTML、代码围栏、分析过程或生成说明。

事实优先级：
1. current_material 是本轮完整当前原文，也是当前事实的直接来源；
2. current_temporal_results 只为当前材料中需要推算的日期提供结果；
3. historical_context 是本次选中的历史背景，每条 source_ref 只用于引用该条历史记录；
4. output_template 和显式偏好只控制表达，不得改变事实。

写作与结构规则：
- 按业务议题组织，不按发言人罗列流水账。保留影响结论或执行的对象、数字、单位、范围、对照、条件、依赖和不确定语气。
- 区分事实、讨论或方案、决定、行动、风险、未决问题和历史变化；不要把建议提升为决定，把计划提升为完成，或扩大任务范围。
- 同一业务陈述只放入一个 section entry。overview 可以概括最重要结论，但不得引入正文没有的业务事实。
- action 的 content 只写事项本身；owner、time_requirement、dependency 分字段填写。缺失字段使用空字符串，不推断负责人或日期。
- history_update 必须引用至少一个有效 history_ref。recorded_state 写历史记录时情况，current_state 只写当前原文明示的本次情况，relation 说明二者关系。
- 历史记录未被当前材料更新时，不得推断完成、取消或仍有效；可以作为背景条目引用，但不要伪造状态变化。
- 当前事实条目提供1至3段简短 current_evidence_quotes，每段必须逐字来自 current_material。历史背景使用 history_refs，不要复制或改写成所谓原文引文。
- 声称历史事项完成、取消、变更或冲突的 history_update，必须同时提供 history_refs 和能支持本次状态的 current_evidence_quotes。
- 原文明示的绝对日期可直接使用；推算日期仅使用 current_temporal_results 中同一事项的结果。未解析或有歧义时保留原表达并注明待确认。
- 遵循 output_template 的场景和组织顺序；没有依据的章节或条目省略，不为空模板编造内容。
- 用户在 summary_task 中明确的格式、详略和关注点优先，但不能突破事实边界。
- 输出内容应适度详实、可执行、无跨章节重复；优先完整写清事实，不用套话增加篇幅。

输出 JSON 契约：
{contract}

<summary_task>
{sections["summary_task"]}
</summary_task>

<current_material>
{sections["current_material"]}
</current_material>

<historical_context>
{sections["historical_context"]}
</historical_context>

<current_temporal_results>
{sections["current_temporal_results"]}
</current_temporal_results>

<output_template>
{sections["output_template"]}
</output_template>
"""
