"""根据摘要风格和请求生成无副作用的模板适配结果。"""

from __future__ import annotations

from typing import Any, Literal, Optional, get_args

from pydantic import Field

from ..schemas import FrozenStrictModel, TaskType
from .registry import BaseTool, ToolContext


MeetingScene = Literal["general", "project_sync", "decision_review", "focused_discussion"]
MEETING_SCENES = get_args(MeetingScene)
SCENE_DESCRIPTIONS = {
    "general": "通用会议：多类事项或场景不明确",
    "project_sync": "项目例会：工作项进展、延续任务、风险与协调",
    "decision_review": "决策评审：评审范围、方案取舍、结论及成立条件",
    "focused_discussion": "专项讨论：问题边界、事实分析、方案与验证",
}
SCENE_SECTIONS = {
    "general": ("会议概览", "主要议题与结论", "历史延续与状态变化", "风险与待确认", "行动项"),
    "project_sync": ("本期概览", "各工作项进展", "历史任务与状态变化", "风险与协调", "后续安排"),
    "decision_review": ("评审对象与范围", "方案依据与取舍", "结论与成立条件", "历史决定与变更", "遗留问题与整改"),
    "focused_discussion": ("问题与边界", "事实分析与方案", "结论与未决问题", "相关历史约束", "验证与行动"),
}
SCENE_INSTRUCTIONS = {
    "general": "围绕业务议题写清事实、讨论或方案及当前结论；议题数量随材料变化，不按发言人罗列流水账。",
    "project_sync": "按工作项保留已完成的具体范围、未完成部分、偏差、阻塞和协调需求；区分本期进展、后续计划与历史记录状态。",
    "decision_review": "保留评审对象、候选方案的关键差异、取舍依据及前提；只有明确表态才能写通过、否决或有条件通过，未定论时如实说明。",
    "focused_discussion": "围绕同一问题的边界、已知事实、分析和备选方案展开，区分已验证结论与待验证假设，保留验证条件。",
}


class TemplateAdaptationInput(FrozenStrictModel):
    summary_request: str = ""
    requested_style: Optional[str] = None
    requested_scene: Optional[MeetingScene] = None
    preferences: dict[str, Any] = Field(default_factory=dict)


# 模板 Tool 的结构化输出；只影响表达结构，不改变事实覆盖要求。
class TemplateAdaptationOutput(FrozenStrictModel):
    template_name: str = Field(min_length=1)
    format_kind: str = Field(min_length=1)
    meeting_scene: MeetingScene = "general"
    sections: tuple[str, ...] = ()
    instructions: tuple[str, ...] = Field(min_length=1)
    used_default: bool = False
    warnings: tuple[str, ...] = ()


STRUCTURED_SECTIONS = SCENE_SECTIONS["general"]


def resolve_template(arguments: TemplateAdaptationInput) -> TemplateAdaptationOutput:
    """纯函数：显式场景优先于模型/计划建议，缺少选择时使用通用模板。"""
    preference_style = str(arguments.preferences.get("summary_style") or "").strip()
    requested = (arguments.requested_style or preference_style).strip().lower()
    natural = requested in {"natural", "自然", "自然版"} or "自然段" in arguments.summary_request
    format_kind = "natural" if natural else "structured"
    style_default = not natural and requested not in {"structured", "结构化", "结构化版"}
    warnings = []
    if style_default and requested:
        warnings.append(f"unsupported style {requested!r}; used structured default")

    explicit_scene = str(arguments.preferences.get("meeting_scene") or "auto").strip()
    if explicit_scene in MEETING_SCENES:
        scene = explicit_scene
        scene_default = False
    elif explicit_scene == "auto":
        scene = arguments.requested_scene or "general"
        scene_default = arguments.requested_scene is None
    else:
        scene = "general"
        scene_default = True
        warnings.append(f"unsupported meeting_scene {explicit_scene!r}; used general default")

    common = (
        SCENE_INSTRUCTIONS[scene],
        "只呈现有材料依据的内容；无依据的历史、风险、决定或行动项整节省略，不为填满栏目编造事实或反复写‘未提及’。",
        "用户明确的格式和详略要求优先于此预置样式；概览简述结论，正文展开必要依据，不跨章节重复整段事实。",
    )
    if natural:
        instructions = common + (
            "按以下组织顺序写连贯自然段：" + "、".join(SCENE_SECTIONS[scene]) + "。",
            "每段围绕一个主题，段数随材料变化，不使用固定栏目、项目列表或 Markdown 表格。",
            "在自然句中写清行动事项、负责人、时间要求及前提；明确区分当前事实与历史记录时状态。",
        )
        sections = ()
    else:
        instructions = common + (
            "使用 Markdown 二级标题组织栏目，主要议题或工作项可用三级标题，具体内容使用简洁条目；小节只在有内容时展开。",
            "历史引用标明来源日期及记录时状态；没有来源日期时说明来源日期未明确，不用当前会议日期替代。",
            "行动项使用 Markdown 表格，列为：事项｜负责人｜时间要求｜前提或依赖。仅纳入有依据的后续任务；缺失字段写‘未明确’。",
        )
        sections = SCENE_SECTIONS[scene]
    return TemplateAdaptationOutput(
        template_name=f"{format_kind}_{scene}_v3",
        format_kind=format_kind,
        meeting_scene=scene,
        sections=sections,
        instructions=instructions,
        used_default=style_default or scene_default,
        warnings=tuple(warnings),
    )


def project_template(template: dict[str, Any]) -> dict[str, Any]:
    """只向写作/审核模型提供选中的格式要求，省略选择与执行审计。"""
    return {
        key: template[key]
        for key in ("template_name", "format_kind", "meeting_scene", "sections", "instructions")
        if key in template and template[key] not in (None, "", [], {}, ())
    }


class TemplateAdaptationTool(BaseTool):
    # 该具体 Tool 的 __doc__ 会被 ToolRegistry 放入 Planner/Worker Prompt；保持为空。
    name = "template_adaptation"
    input_model = TemplateAdaptationInput
    output_model = TemplateAdaptationOutput
    allowed_task_types = frozenset({TaskType.TEMPLATE_ADAPTATION})

    async def execute(
        self, arguments: TemplateAdaptationInput, context: ToolContext
    ) -> TemplateAdaptationOutput:
        """根据显式样式和偏好选择模板，不修改事实、不调用模型或长期 Memory。"""
        del context
        return resolve_template(arguments)
