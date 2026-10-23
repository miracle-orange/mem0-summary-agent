"""将只读可视化模型渲染为适配 Gradio 3.50 的静态 HTML/SVG。"""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from typing import Any

from .schemas import AgentGraphNode, WebRunVisualization


WEB_VISUALIZATION_CSS = r"""
.smw-view,.smw-view *,.smw-panel,.smw-panel *,.smw-error,.smw-error *{box-sizing:border-box}
.smw-view{font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif;color:#182230!important;background:#f4f6f8!important;border:1px solid #cbd5df;border-radius:6px;padding:14px;font-size:14px;line-height:1.55;letter-spacing:0}
.smw-view strong,.smw-panel strong{color:#111827!important;opacity:1!important}.smw-view p,.smw-panel p{color:#243244!important;line-height:1.65;overflow-wrap:anywhere;opacity:1!important}
.smw-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:8px 0 16px}
.smw-metrics-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:8px 0 16px}
.smw-stat,.smw-panel{min-width:0;border:1px solid #c5cfdb;border-radius:6px;background:#fff!important;color:#182230!important;padding:12px;box-shadow:0 1px 2px rgba(15,23,42,.06)}
.smw-stat strong{display:block;font-size:18px;line-height:1.35;margin-top:5px;overflow-wrap:anywhere}
.smw-muted{color:#465568!important;font-size:12px;line-height:1.55;opacity:1!important;overflow-wrap:anywhere}.smw-title{color:#111827!important;font-size:15px;font-weight:700;line-height:1.45;margin:16px 0 9px;opacity:1!important;overflow-wrap:anywhere}
.smw-stage-flow{display:flex;align-items:stretch;gap:8px;overflow-x:auto;padding:4px 2px 13px;scrollbar-gutter:stable}
.smw-stage{flex:0 0 188px;min-width:188px;border:1px solid #c5cfdb;border-top:4px solid #64748b;border-radius:6px;padding:10px;background:#fff!important;color:#182230!important;box-shadow:0 1px 2px rgba(15,23,42,.06)}
.smw-stage strong{display:block;color:#111827!important;font-size:13px;line-height:1.4;margin-bottom:3px;overflow-wrap:anywhere}
.smw-stage.success{border-top-color:#087a55}.smw-stage.warning{border-top-color:#a15c00}.smw-stage.failed{border-top-color:#bd2c2c}.smw-stage.running{border-top-color:#1769aa}.smw-stage.skipped{border-top-color:#64748b;background:#eef2f6!important}
.smw-arrow{display:flex;flex:0 0 18px;align-items:center;justify-content:center;color:#465568!important;font-size:20px;font-weight:700;opacity:1!important}.smw-badge{display:inline-block;max-width:100%;padding:2px 6px;border:1px solid #cbd5df;border-radius:4px;background:#e9eef4!important;color:#26374a!important;font-size:11px;line-height:1.45;margin:4px 3px 0 0;opacity:1!important;overflow-wrap:anywhere}
.smw-status-success{color:#066746!important;font-weight:700;opacity:1!important}.smw-status-warning{color:#7a4300!important;font-weight:700;opacity:1!important}.smw-status-failed{color:#9f2020!important;font-weight:700;opacity:1!important}.smw-status-running{color:#125b91!important;font-weight:700;opacity:1!important}.smw-status-skipped,.smw-status-pending{color:#3f4c5d!important;font-weight:700;opacity:1!important}
.smw-scroll{max-width:100%;overflow-x:auto;overflow-y:hidden;padding-bottom:4px;scrollbar-gutter:stable}.smw-graph-canvas{display:flex;min-width:100%;justify-content:center}.smw-svg{display:block;flex:none;max-width:none;border:1px solid #c5cfdb;border-radius:6px;background:#fff!important}
.smw-node-card{width:100%;height:100%;display:flex;flex-direction:column;overflow:hidden;padding:10px 12px;color:#182230!important;font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.35}
.smw-node-id{max-height:34px;overflow:hidden;color:#111827!important;font-size:13px;font-weight:750;opacity:1!important;overflow-wrap:anywhere;word-break:break-word}
.smw-node-meta{margin-top:3px;color:#334155!important;font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;opacity:1!important}
.smw-node-description{display:-webkit-box;min-height:32px;max-height:49px;margin-top:6px;overflow:hidden;color:#1f2d3d!important;font-size:12px;line-height:1.35;opacity:1!important;overflow-wrap:anywhere;word-break:break-word;-webkit-box-orient:vertical;-webkit-line-clamp:3}
.smw-node-tags{margin-top:auto;color:#3f4c5d!important;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;opacity:1!important}
.smw-table{width:100%;border-collapse:collapse;background:#fff!important;color:#182230!important;font-size:13px;line-height:1.5}.smw-table th,.smw-table td{border-bottom:1px solid #d7dfe8;text-align:left;padding:8px 9px;vertical-align:top;color:#243244!important;opacity:1!important;overflow-wrap:anywhere;word-break:break-word}.smw-table th{background:#e5ebf1!important;color:#172033!important;font-weight:700;white-space:nowrap}.smw-table tbody tr:nth-child(even) td{background:#f6f8fa!important}.smw-table tbody tr:last-child td,.smw-table tbody tr:last-child th{border-bottom:0}
.smw-metric-table{min-width:560px;table-layout:fixed}.smw-metric-table th:first-child,.smw-metric-table td:first-child{width:31%}.smw-metric-table th:nth-child(2),.smw-metric-table td:nth-child(2){width:92px}.smw-metric-name{display:block;color:#111827!important;font-weight:700;white-space:normal;opacity:1!important;overflow-wrap:anywhere}.smw-metric-code{display:inline-block;max-width:100%;margin-top:3px;padding:1px 4px;border:1px solid #cbd5df;border-radius:3px;background:#e9eef4!important;color:#26374a!important;font-family:ui-monospace,Consolas,monospace;font-size:10px;font-weight:600;line-height:1.45;white-space:normal;opacity:1!important;overflow-wrap:anywhere}.smw-metric-value{color:#111827!important;font-size:14px;font-weight:750;white-space:nowrap;opacity:1!important}.smw-metric-fraction{margin-top:2px;color:#465568!important;font-size:11px;white-space:nowrap;opacity:1!important}.smw-metric-description{color:#26374a!important;font-size:12px;line-height:1.62;opacity:1!important}
.smw-comparison-section{min-width:0;margin:12px 0 16px;padding:12px;border:1px solid #c5cfdb;border-radius:6px;background:#fff!important;box-shadow:0 1px 2px rgba(15,23,42,.04)}.smw-comparison-section>.smw-title{margin:0 0 9px}.smw-comparison-table{min-width:980px;table-layout:auto}.smw-comparison-table th:first-child,.smw-comparison-table td:first-child{position:sticky;left:0;z-index:2;min-width:150px;max-width:210px}.smw-comparison-table th:nth-child(2),.smw-comparison-table td:nth-child(2){min-width:96px}.smw-comparison-table th.smw-comparison-metric,.smw-comparison-table td.smw-comparison-value,.smw-comparison-table td.smw-comparison-delta{min-width:154px}.smw-comparison-summary{color:#111827!important;font-weight:700;opacity:1!important}.smw-comparison-status{color:#26374a!important;font-weight:650;white-space:nowrap;opacity:1!important}.smw-comparison-value{color:#111827!important;font-size:13px;font-weight:750;white-space:nowrap;opacity:1!important}.smw-comparison-delta-label{color:#3f4c5d!important;font-size:11px;font-weight:650;opacity:1!important}.smw-comparison-delta{color:#334155!important;font-family:ui-monospace,Consolas,monospace;font-size:12px;font-weight:700;white-space:nowrap;opacity:1!important}.smw-comparison-delta-row td{background:#eef2f6!important;border-bottom:1px solid #cbd5df}
.smw-diff-summary{margin:8px 0;padding:8px 10px;border-left:4px solid #1769aa;background:#edf5fb!important;color:#17324a!important;font-weight:650;opacity:1!important}.smw-diff-table{min-width:680px}.smw-diff-kind{font-weight:700;white-space:nowrap}.smw-diff-kind.gain{color:#066746!important}.smw-diff-kind.loss{color:#8a4b00!important}.smw-diagnostics{margin:18px 0 4px;border:1px solid #c5cfdb;border-radius:6px;background:#fff!important;color:#182230!important;overflow:hidden}.smw-diagnostics>summary{cursor:pointer;padding:11px 13px;background:#e9eef4!important;color:#26374a!important;font-weight:700;opacity:1!important}.smw-diagnostics-body{padding:12px}.smw-diagnostics-note{margin:0 0 10px;color:#465568!important;font-size:12px;opacity:1!important}
.smw-interpretation-overview{margin:8px 0 12px;padding:9px 11px;border-left:4px solid #1769aa;background:#edf5fb!important;color:#17324a!important;line-height:1.65;overflow-wrap:anywhere}.smw-interpretation-item{padding:10px 0;border-top:1px solid #d7dfe8;color:#243244!important}.smw-interpretation-item:first-of-type{border-top:0}.smw-interpretation-item ul{margin:6px 0 0;padding-left:20px}.smw-interpretation-item li{margin:5px 0;color:#243244!important;overflow-wrap:anywhere}.smw-quote{display:block;margin-top:2px;color:#465568!important;font-size:12px;line-height:1.55}
.smw-metric-guide{display:grid;gap:0;margin:10px 0 0;border:1px solid #cbd5df;border-radius:5px;background:#f1f4f7!important;overflow:hidden}.smw-metric-guide-item{display:grid;grid-template-columns:minmax(230px,310px) minmax(320px,1fr);gap:12px;padding:9px 10px;border-bottom:1px solid #d7dfe8;background:#f8fafc!important;font-size:12px;line-height:1.6}.smw-metric-guide-item:last-child{border-bottom:0}.smw-metric-guide-label{min-width:0;color:#111827!important;font-weight:650;opacity:1!important;overflow-wrap:anywhere}.smw-metric-guide-label strong{color:#111827!important;opacity:1!important}.smw-metric-guide-description{min-width:0;color:#26374a!important;font-weight:450;opacity:1!important;overflow-wrap:anywhere;word-break:break-word}
.smw-timeline{display:grid;gap:7px;margin-top:9px}.smw-time-row{display:grid;grid-template-columns:minmax(180px,240px) minmax(240px,1fr) minmax(76px,105px);gap:11px;align-items:center;border-bottom:1px solid #d7dfe8;padding:6px 0}.smw-time-row:last-child{border-bottom:0}.smw-time-row>div:first-child{min-width:0;color:#182230!important;opacity:1!important;overflow-wrap:anywhere}.smw-track{height:20px;background:#dbe3eb!important;position:relative;border:1px solid #bdc8d5;border-radius:4px;overflow:hidden}.smw-bar{position:absolute;min-width:5px;height:100%;background:#1769aa}.smw-bar.success{background:#087a55}.smw-bar.warning{background:#a15c00}.smw-bar.failed{background:#bd2c2c}.smw-bar.running{background:#1769aa}
.smw-funnel{display:flex;align-items:stretch;gap:8px;overflow-x:auto;padding:2px 2px 8px;scrollbar-gutter:stable}.smw-funnel-item{flex:0 0 176px;min-width:176px;border:1px solid #c5cfdb;border-radius:6px;padding:11px;background:#fff!important;color:#182230!important;box-shadow:0 1px 2px rgba(15,23,42,.05)}.smw-funnel-item strong{font-size:20px;line-height:1.3;display:block;margin:5px 0;color:#111827!important}
.smw-warning{border:1px solid #d6a94c;border-left:4px solid #925300;border-radius:4px;background:#fff4d6!important;color:#542f00!important;padding:9px 11px;margin:7px 0;opacity:1!important;overflow-wrap:anywhere}.smw-error{border:1px solid #dc9696;border-left:4px solid #ad2525;border-radius:4px;background:#ffeded!important;color:#721c1c!important;padding:9px 11px;margin:7px 0;opacity:1!important;overflow-wrap:anywhere}
.smw-pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#182230!important;font-family:ui-monospace,Consolas,monospace;font-size:12px;line-height:1.55;margin:0;opacity:1!important}
@media(max-width:980px){.smw-metrics-grid{grid-template-columns:1fr}.smw-metric-guide-item{grid-template-columns:minmax(210px,280px) 1fr}}
@media(max-width:760px){.smw-view{padding:9px}.smw-grid{grid-template-columns:repeat(auto-fit,minmax(150px,1fr))}.smw-stage{flex-basis:172px;min-width:172px}.smw-time-row{grid-template-columns:1fr;gap:5px}.smw-track{min-width:420px}.smw-table{min-width:680px}.smw-comparison-section{padding:9px}.smw-comparison-table{min-width:940px}.smw-metric-guide-item{grid-template-columns:1fr;gap:3px}.smw-metric-guide-description{font-size:12px;line-height:1.65}}
"""

_STYLE = f"<style>{WEB_VISUALIZATION_CSS}</style>"


# 指标名称和解释严格对应 evaluation/metrics.py 与 personalization.py 的计算口径。
METRIC_DISPLAY_INFO: dict[str, tuple[str, str]] = {
    "must_fact_recall": (
        "必要事实召回率",
        "摘要严格覆盖的必要评测单位占全部必要评测单位的比例；包含当前必要事实、仅历史可得的必要事实及必要状态变化。越高越好。",
    ),
    "current_must_fact_recall": (
        "当前必要事实召回率",
        "摘要严格覆盖的当前材料必要事实占全部当前必要事实的比例，用于判断本次会议核心内容是否遗漏。越高越好。",
    ),
    "historical_useful_recall": (
        "历史有用事实召回率（兼容口径）",
        "旧版兼容诊断：严格覆盖的旧口径历史必要事实比例；正式分析应优先使用“历史语义召回率”。",
    ),
    "useful_fact_precision": (
        "有用事实精确率",
        "全部摘要原子 Claim 中，严格正确、匹配有用源事实且非重复的比例，反映纪要内容选择是否有效。越高越好。",
    ),
    "fact_support_precision": (
        "事实支持精确率",
        "全部摘要原子 Claim 中获得源材料严格支持的比例，反映纪要整体事实可靠性。越高越好。",
    ),
    "minutes_fact_f1": (
        "纪要事实 F1",
        "必要事实召回率与有用事实精确率的调和平均，同时衡量核心事实覆盖和内容有效性。越高越好。",
    ),
    "weighted_must_recall": (
        "加权必要事实召回率",
        "诊断指标：完整支持计 1，部分支持计 0.5；不参与纪要事实 F1。越高表示必要事实表达越完整。",
    ),
    "historical_semantic_recall": (
        "历史语义召回率",
        "正确表达的有用历史事实族占全部有用历史事实族的比例；即使当前材料再次确认该事实也可计入。越高越好。",
    ),
    "historical_useful_precision": (
        "历史有用事实精确率",
        "全部历史相关 Claim 中，严格正确且有用的历史 Claim 比例，用于识别历史信息滥用或冗余。越高越好。",
    ),
    "historical_fact_f1": (
        "历史事实 F1",
        "历史语义召回率与历史有用事实精确率的调和平均，综合衡量历史信息覆盖与使用质量。越高越好。",
    ),
    "historical_unique_recall": (
        "历史独有事实召回率",
        "正确覆盖的 history-only 有用事实族占全部 history-only 有用事实族的比例，衡量只靠历史材料才能补充的信息。越高越好。",
    ),
    "historical_unique_contribution_rate": (
        "历史独有贡献率",
        "已正确覆盖的全部有用事实族中，history-only 事实族所占比例；表示历史知识对本份纪要的实际增量，不宜脱离任务单独追求高值。",
    ),
    "current_confirmed_history_recall": (
        "当前确认历史召回率",
        "被当前材料再次确认且由摘要正确表达的历史事实族，占全部 confirmed 历史事实族的比例。越高越好。",
    ),
    "state_transition_recall": (
        "状态变化召回率",
        "摘要正确完整表达的有用状态变化占全部有用状态变化的比例，例如从待办到完成。越高越好。",
    ),
    "state_transition_precision": (
        "状态变化精确率",
        "摘要声称的状态变化中，得到源材料严格支持且表达完整的比例。越高越好。",
    ),
    "state_transition_f1": (
        "状态变化 F1",
        "状态变化召回率与精确率的调和平均，综合衡量变化覆盖和变化表达可靠性。越高越好。",
    ),
    "current_state_accuracy": (
        "当前状态准确率",
        "全部状态相关 Claim 中，最终当前状态表达正确的比例。越高越好。",
    ),
    "stale_state_error_rate": (
        "陈旧状态错误率",
        "全部状态相关 Claim 中，把历史旧状态误写成当前状态的比例。越低越好。",
    ),
    "unsupported_rate": (
        "无依据事实率",
        "全部摘要 Claim 中无法从当前或可见历史材料获得支持的比例。越低越好。",
    ),
    "contradiction_rate": (
        "事实矛盾率",
        "全部摘要 Claim 中与源事实明确矛盾的比例。越低越好。",
    ),
    "partial_rate": (
        "部分支持率",
        "全部摘要 Claim 中仅部分正确，或状态、负责人、时间、数值等关键槽位不完整的比例。越低越好。",
    ),
    "stale_history_rate": (
        "陈旧历史使用率",
        "全部摘要 Claim 中错误沿用已失效历史信息的比例。越低越好。",
    ),
    "redundancy_rate": (
        "事实冗余率",
        "全部摘要 Claim 中重复表达同一事实的比例。越低越好。",
    ),
    "information_density": (
        "信息密度",
        "每 100 个非空白摘要字符覆盖的有用唯一事实数；用于描述紧凑度，需结合召回率和可读性解释。",
    ),
    "compression_ratio": (
        "压缩比",
        "摘要非空白字符数与当前材料加可见历史材料非空白字符数的比值；只描述压缩程度，不直接代表质量高低。",
    ),
    "target_content_must_recall": (
        "目标内容必要事实召回率",
        "摘要覆盖的目标实体必要事实占全部目标实体必要事实的比例；事实正确即可，不要求再次点名目标实体。越高越好。",
    ),
    "target_content_useful_recall": (
        "目标内容有用事实召回率",
        "摘要覆盖的目标实体有用事实占全部目标实体有用事实的比例，不以表达是否显式面向目标为门槛。越高越好。",
    ),
    "target_content_useful_precision": (
        "目标内容有用事实精确率",
        "目标内容相关 Claim 中，严格正确且属于目标有用事实的比例。越高越好。",
    ),
    "target_content_f1": (
        "目标内容 F1",
        "目标内容必要事实召回率与目标内容有用事实精确率的调和平均，衡量与目标相关的内容是否覆盖且有效。越高越好。",
    ),
    "target_directed_must_recall": (
        "目标导向必要事实召回率",
        "通过显式面向目标或目标语境表达而覆盖的目标必要事实比例，衡量个性化表达是否足够明确。越高越好。",
    ),
    "target_directed_useful_recall": (
        "目标导向有用事实召回率",
        "通过目标导向表达覆盖的目标有用事实占全部目标有用事实的比例。越高越好。",
    ),
    "target_directed_useful_precision": (
        "目标导向有用事实精确率",
        "全部目标导向或目标语境 Claim 中，严格正确且有用的比例。越高越好。",
    ),
    "target_directed_f1": (
        "目标导向 F1",
        "目标导向必要事实召回率与目标导向有用事实精确率的调和平均，衡量明确个性化表达的覆盖与可靠性。越高越好。",
    ),
    "target_support_precision": (
        "目标内容支持精确率",
        "全部目标内容相关 Claim 中获得源材料严格支持的比例。越高越好。",
    ),
    "target_current_recall": (
        "目标当前事实召回率",
        "摘要严格覆盖的目标相关当前有用事实占全部目标相关当前有用事实的比例。越高越好。",
    ),
    "target_historical_recall": (
        "目标历史事实召回率",
        "摘要严格覆盖的目标相关历史有用事实占全部目标相关历史有用事实的比例。越高越好。",
    ),
    "target_must_fact_recall": (
        "目标必要事实召回率（兼容口径）",
        "P7.2 兼容别名，使用旧版目标 Claim 门控口径；正式分析应优先使用目标内容与目标导向两组 V2 指标。",
    ),
    "target_useful_recall": (
        "目标有用事实召回率（兼容口径）",
        "P7.2 兼容别名，表示旧口径下目标有用事实覆盖比例。",
    ),
    "target_useful_precision": (
        "目标有用事实精确率（兼容口径）",
        "P7.2 兼容别名，表示旧口径下目标相关 Claim 的有用事实精确率。",
    ),
    "target_fact_f1": (
        "目标事实 F1（兼容口径）",
        "P7.2 兼容别名，由旧口径目标必要事实召回率与目标有用事实精确率计算。",
    ),
    "target_unsupported_attribution_rate": (
        "目标无依据归因率",
        "目标导向 Claim 中，内容或归因无法由源材料支持的比例。越低越好。",
    ),
    "target_contradiction_rate": (
        "目标事实矛盾率",
        "目标导向 Claim 中与源事实明确矛盾的比例。越低越好。",
    ),
    "target_partial_rate": (
        "目标事实部分支持率",
        "目标导向 Claim 中仅获得部分支持或关键槽位不完整的比例。越低越好。",
    ),
    "target_stale_history_rate": (
        "目标陈旧历史使用率",
        "目标导向 Claim 中错误沿用已失效历史状态的比例。越低越好。",
    ),
    "target_redundancy_rate": (
        "目标表达冗余率",
        "目标导向 Claim 中重复表达同一事实的比例。越低越好。",
    ),
    "target_focus_share": (
        "目标关注占比",
        "目标导向 Claim 占全部摘要 Claim 的比例，用于描述个性化篇幅倾向，不直接代表质量高低。",
    ),
}


def _e(value: Any) -> str:
    return html.escape(str(value if value is not None else ""), quote=True)


def _fmt_count(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value)


def _fmt_duration(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "N/A"
    seconds = float(value) / 1000.0
    if seconds < 1:
        return f"{int(value)} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{int(seconds // 60)}m {seconds % 60:.0f}s"


def _metric_value(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, dict):
        nested = value.get("value")
        if isinstance(nested, (int, float)) and not isinstance(nested, bool):
            return float(nested)
    return None


def _fmt_metric(value: Any, *, signed: bool = False) -> str:
    metric = _metric_value(value)
    if metric is None:
        return "N/A"
    if signed:
        return f"{metric:+.4f}"
    return f"{metric:.4f}"


def _metric_display(name: str) -> tuple[str, str]:
    return METRIC_DISPLAY_INFO.get(
        name,
        (
            name,
            "当前版本未登记该扩展指标的中文解释，请结合原始评测产物中的字段定义使用。",
        ),
    )


def _metric_value_html(value: Any) -> str:
    rendered = f'<div class="smw-metric-value">{_e(_fmt_metric(value))}</div>'
    if not isinstance(value, dict):
        return rendered
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if numerator is None or denominator is None:
        return rendered
    return (
        rendered
        + f'<div class="smw-metric-fraction">{_e(numerator)} / {_e(denominator)}</div>'
    )


def _metric_header_html(name: str) -> str:
    label, _ = _metric_display(name)
    return (
        f'<span class="smw-metric-name">{_e(label)}</span>'
        f'<code class="smw-metric-code">{_e(name)}</code>'
    )


def _metric_guide_html(
    names: tuple[str, ...],
    *,
    description_overrides: dict[str, str] | None = None,
) -> str:
    items: list[str] = []
    for name in names:
        label, description = _metric_display(name)
        if description_overrides and name in description_overrides:
            description = description_overrides[name]
        items.append(
            '<div class="smw-metric-guide-item">'
            f'<div class="smw-metric-guide-label"><strong>{_e(label)}</strong> '
            f'<code class="smw-metric-code">{_e(name)}</code></div>'
            f'<div class="smw-metric-guide-description">{_e(description)}</div></div>'
        )
    return '<div class="smw-metric-guide">' + "".join(items) + "</div>"


def _status(value: Any) -> str:
    return str(value or "pending").lower()


def _status_text(value: Any) -> str:
    names = {
        "success": "成功",
        "completed": "完成",
        "completed_with_warnings": "完成（有警告）",
        "warning": "警告",
        "failed": "失败",
        "running": "运行中",
        "pending": "待执行",
        "skipped": "不适用/已跳过",
        "disabled": "未启用",
    }
    status = _status(value)
    return names.get(status, status or "未知")


def _notices(view: WebRunVisualization) -> str:
    blocks = [f'<div class="smw-warning">{_e(item)}</div>' for item in view.warnings]
    blocks.extend(
        f'<div class="smw-error"><strong>{_e(name)}</strong>：{_e(error)}</div>'
        for name, error in view.errors.items()
    )
    return "".join(blocks)


def render_stage_flow_html(view: WebRunVisualization) -> str:
    if not view.stages:
        return '<div class="smw-muted">该运行暂无可识别的阶段信息。</div>'
    items: list[str] = []
    for index, stage in enumerate(view.stages):
        if index:
            items.append('<div class="smw-arrow">›</div>')
        counters = "".join(
            f'<span class="smw-badge">{_e(key)}: {_e(_fmt_count(value))}</span>'
            for key, value in stage.counters.items()
        )
        timing = (
            f'<div class="smw-muted">开始：{_e(stage.started_at or "N/A")}</div>'
            f'<div class="smw-muted">结束：{_e(stage.finished_at or "N/A")}</div>'
            f'<div class="smw-muted">耗时：{_e(_fmt_duration(stage.duration_ms))}</div>'
        )
        status = _status(stage.status)
        items.append(
            f'<div class="smw-stage {_e(status)}">'
            f"<strong>{_e(stage.label)}</strong>"
            f'<div class="smw-status-{_e(status)}">{_e(_status_text(status))}</div>'
            f"{timing}{counters}</div>"
        )
    return '<div class="smw-stage-flow">' + "".join(items) + "</div>"


def render_overview_html(view: WebRunVisualization) -> str:
    data = view.overview
    cards = (
        ("运行状态", _status_text(data.get("status"))),
        ("当前阶段", data.get("stage") or "N/A"),
        ("总耗时", _fmt_duration(data.get("duration_ms"))),
        (
            "任务完成",
            f"{_fmt_count(data.get('successful_task_count'))} / {_fmt_count(data.get('planned_task_count'))}",
        ),
        ("模型调用（已审计）", _fmt_count(data.get("model_call_count"))),
        ("Worker 工具调用", _fmt_count(data.get("tool_call_count"))),
        ("重规划次数", _fmt_count(data.get("replan_count"))),
        ("Coverage 完整", _fmt_count(data.get("coverage_complete"))),
        ("Review", data.get("review_status") or "disabled"),
        ("Evaluation", data.get("evaluation_status") or "disabled"),
    )
    card_html = "".join(
        f'<div class="smw-stat"><span class="smw-muted">{_e(label)}</span><strong>{_e(value)}</strong></div>'
        for label, value in cards
    )
    return (
        _STYLE
        + '<div class="smw-view">'
        + f'<div class="smw-title">{_e(view.run_type)} · {_e(view.run_id or "未记录 run_id")}</div>'
        + f'<div class="smw-grid">{card_html}</div>'
        + render_stage_flow_html(view)
        + _notices(view)
        + "</div>"
    )


def _topological_layers(view: WebRunVisualization) -> list[list[str]]:
    node_ids = {node.task_id for node in view.graph_nodes}
    predecessors = {node_id: set() for node_id in node_ids}
    successors = {node_id: set() for node_id in node_ids}
    for edge in view.graph_edges:
        if edge.source_task_id in node_ids and edge.destination_task_id in node_ids:
            predecessors[edge.destination_task_id].add(edge.source_task_id)
            successors[edge.source_task_id].add(edge.destination_task_id)
    remaining = set(node_ids)
    layers: list[list[str]] = []
    while remaining:
        layer = sorted(
            node_id for node_id in remaining if not (predecessors[node_id] & remaining)
        )
        if not layer:
            layer = [sorted(remaining)[0]]
        layers.append(layer)
        remaining.difference_update(layer)
    return layers


def _dag_svg(view: WebRunVisualization) -> str:
    if not view.graph_nodes:
        return '<div class="smw-panel">本运行类型不产生 Agent DAG，阶段进度见上方流程图。</div>'
    layers = _topological_layers(view)
    node_by_id = {node.task_id: node for node in view.graph_nodes}
    # 同一拓扑层横向排列、依赖层纵向推进，避免并行 Worker 被压成狭长竖图。
    node_w, node_h, gap_x, gap_y, pad = 252, 148, 26, 86, 32
    max_columns = max(len(layer) for layer in layers)
    content_width = max_columns * node_w + max(0, max_columns - 1) * gap_x + pad * 2
    width = max(820, content_width)
    height = pad * 2 + len(layers) * node_h + max(0, len(layers) - 1) * gap_y
    positions: dict[str, tuple[float, float]] = {}
    for layer_index, layer in enumerate(layers):
        row_width = len(layer) * node_w + max(0, len(layer) - 1) * gap_x
        x_start = (width - row_width) / 2
        for column_index, task_id in enumerate(layer):
            positions[task_id] = (
                x_start + column_index * (node_w + gap_x),
                pad + layer_index * (node_h + gap_y),
            )
    parts = [
        f'<div class="smw-graph-canvas" style="width:{width}px">',
        f'<svg class="smw-svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Agent DAG">',
        '<defs><marker id="smw-arrow-declared" markerWidth="9" markerHeight="9" refX="4" refY="8" orient="auto"><path d="M0,0 L8,0 L4,8 z" fill="#64748b"/></marker><marker id="smw-arrow-consumed" markerWidth="9" markerHeight="9" refX="4" refY="8" orient="auto"><path d="M0,0 L8,0 L4,8 z" fill="#087a55"/></marker></defs>',
    ]
    for edge in view.graph_edges:
        if (
            edge.source_task_id not in positions
            or edge.destination_task_id not in positions
        ):
            continue
        sx, sy = positions[edge.source_task_id]
        dx, dy = positions[edge.destination_task_id]
        x1, y1 = sx + node_w / 2, sy + node_h
        x2, y2 = dx + node_w / 2, dy
        mid = (y1 + y2) / 2
        color = "#087a55" if edge.consumed else "#64748b"
        dash = "" if edge.consumed else ' stroke-dasharray="6 5"'
        marker = "smw-arrow-consumed" if edge.consumed else "smw-arrow-declared"
        parts.append(
            f'<path d="M{x1},{y1} C{x1},{mid} {x2},{mid} {x2},{y2 - 7}" fill="none" stroke="{color}" stroke-width="{3 if edge.consumed else 1.7}"{dash} marker-end="url(#{marker})"/>'
        )
    colors = {
        "success": ("#e7f5ef", "#087a55"),
        "warning": ("#fff3d6", "#a15c00"),
        "failed": ("#ffeded", "#bd2c2c"),
        "running": ("#e8f2fb", "#1769aa"),
        "skipped": ("#edf1f5", "#64748b"),
        "pending": ("#f4f6f8", "#64748b"),
    }
    for task_id, (x, y) in positions.items():
        node = node_by_id[task_id]
        status = _status(node.status)
        fill, stroke = colors.get(status, colors["pending"])
        parts.append(
            f'<g><rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        badges = []
        if node.critical:
            badges.append("critical")
        if node.preserved:
            badges.append("preserved")
        if node.replanned:
            badges.append("replanned")
        if node.fallback_reason:
            badges.append("fallback")
        node_meta = f"{node.task_type} · {_status_text(status)}"
        node_tags = " · ".join(badges) or "planned task"
        parts.append(
            f'<foreignObject x="{x + 2}" y="{y + 2}" width="{node_w - 4}" height="{node_h - 4}">'
            '<div xmlns="http://www.w3.org/1999/xhtml" class="smw-node-card">'
            f'<div class="smw-node-id" title="{_e(task_id)}">{_e(task_id)}</div>'
            f'<div class="smw-node-meta" title="{_e(node_meta)}">{_e(node_meta)}</div>'
            f'<div class="smw-node-description" title="{_e(node.description)}">{_e(node.description)}</div>'
            f'<div class="smw-node-tags" title="{_e(node_tags)}">{_e(node_tags)}</div>'
            "</div></foreignObject></g>"
        )
    parts.append("</svg></div>")
    return "".join(parts)


def render_agent_graph_html(view: WebRunVisualization) -> str:
    history_rows = "".join(
        "<tr>"
        f"<td>{_e(item.get('plan_version'))}</td>"
        f"<td>{_e(item.get('source'))}</td>"
        f"<td>{_e(item.get('replan_round'))}</td>"
        f"<td>{_e(item.get('task_count'))}</td>"
        f"<td>{_e(', '.join(item.get('preserved_task_ids', [])))}</td>"
        f"<td>{_e(item.get('replan_reason'))}</td>"
        "</tr>"
        for item in view.plan_history
    )
    history = (
        '<div class="smw-title">计划版本与重规划</div><div class="smw-scroll"><table class="smw-table"><thead><tr><th>版本</th><th>来源</th><th>轮次</th><th>任务数</th><th>保留任务</th><th>原因</th></tr></thead><tbody>'
        + history_rows
        + "</tbody></table></div>"
        if history_rows
        else '<div class="smw-muted">没有可用的 plan history。</div>'
    )
    return (
        _STYLE
        + '<div class="smw-view">'
        + render_stage_flow_html(view)
        + '<div class="smw-muted">虚线灰边：Planner 声明依赖；实线绿边：后序 Agent 的 consumed_result_ids 证明实际消费。</div>'
        + f'<div class="smw-scroll">{_dag_svg(view)}</div>'
        + history
        + "</div>"
    )


def task_choices(view: WebRunVisualization) -> list[tuple[str, str]]:
    return [
        (
            f"{node.task_id} | {node.task_type} | {_status_text(node.status)}",
            node.task_id,
        )
        for node in view.graph_nodes
    ]


def render_task_detail_html(view: WebRunVisualization, task_id: str) -> str:
    node = next((item for item in view.graph_nodes if item.task_id == task_id), None)
    if node is None:
        return (
            _STYLE
            + '<div class="smw-panel smw-muted">请选择一个 Agent 任务查看详情。</div>'
        )
    rows = (
        ("Task ID", node.task_id),
        ("Agent 类型", node.task_type),
        ("状态", _status_text(node.status)),
        ("关键任务", "是" if node.critical else "否"),
        ("耗时", _fmt_duration(node.duration_ms)),
        ("模型调用", _fmt_count(node.model_call_count)),
        ("工具调用", _fmt_count(node.tool_call_count)),
        ("Evidence 数", _fmt_count(node.evidence_count)),
        ("Warning 数", _fmt_count(node.warning_count)),
        ("允许工具", ", ".join(node.allowed_tools) or "无"),
        ("实际消费结果", ", ".join(node.consumed_result_ids) or "无"),
        ("Fallback", node.fallback_reason or "无"),
        ("错误", node.error or "无"),
    )
    table = "".join(
        f"<tr><th>{_e(label)}</th><td>{_e(value)}</td></tr>" for label, value in rows
    )
    return (
        _STYLE
        + '<div class="smw-panel"><div class="smw-title">任务详情</div>'
        + f'<p>{_e(node.description)}</p><table class="smw-table">{table}</table></div>'
    )


def _parse_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def render_timeline_html(view: WebRunVisualization) -> str:
    timed = [
        (_parse_time(item.started_at), _parse_time(item.finished_at), item)
        for item in view.timeline
        if _parse_time(item.started_at)
    ]
    if not timed:
        return (
            _STYLE
            + '<div class="smw-panel smw-muted">暂无可用 execution trace；旧运行仍可通过原始审计查看已有产物。</div>'
        )
    start = min(row[0] for row in timed if row[0] is not None)
    ends = [row[1] or row[0] for row in timed if row[0] is not None]
    finish = max(row for row in ends if row is not None)
    total = max((finish - start).total_seconds(), 0.001)
    rows: list[str] = []
    for item_start, item_finish, item in timed:
        assert item_start is not None
        effective_finish = item_finish or item_start
        left = max(0.0, min(100.0, (item_start - start).total_seconds() / total * 100))
        width = max(
            0.7,
            min(
                100.0 - left,
                (effective_finish - item_start).total_seconds() / total * 100,
            ),
        )
        status = _status(item.status)
        rows.append(
            '<div class="smw-time-row">'
            f'<div><strong>{_e(item.label)}</strong><div class="smw-muted">{_e(item.started_at)}</div></div>'
            f'<div class="smw-track"><div class="smw-bar {_e(status)}" style="left:{left:.2f}%;width:{width:.2f}%"></div></div>'
            f'<div class="smw-status-{_e(status)}">{_e(_fmt_duration(item.duration_ms))}</div></div>'
        )
    return (
        _STYLE
        + '<div class="smw-view"><div class="smw-title">阶段事件与 Worker 并发区间</div>'
        + '<div class="smw-muted">横轴按 execution_trace.jsonl 的首末白名单事件归一化。</div>'
        + '<div class="smw-timeline">'
        + "".join(rows)
        + "</div></div>"
    )


def render_evidence_memory_html(view: WebRunVisualization) -> str:
    funnel: list[str] = []
    for index, item in enumerate(view.memory_flow):
        if index:
            funnel.append('<div class="smw-arrow">›</div>')
        funnel.append(
            '<div class="smw-funnel-item">'
            f'<span class="smw-muted">{_e(item.label)}</span>'
            f"<strong>{_e(_fmt_count(item.value))}</strong>"
            f'<span class="smw-muted">{_e(item.detail)}</span></div>'
        )
    rows = "".join(
        "<tr>"
        f"<td>{_e(item.get('evidence_id'))}</td>"
        f"<td>{_e(item.get('source_kind'))}</td>"
        f"<td>{_e(item.get('source_time'))}</td>"
        f"<td>{_e(item.get('segment_id'))}</td>"
        f"<td>{_e(item.get('offset'))}</td>"
        f"<td>{_e(item.get('worker_id'))}</td>"
        f"<td>{_e(item.get('memory_id'))}</td>"
        "</tr>"
        for item in view.evidence_rows[:200]
    )
    table = (
        '<div class="smw-scroll"><table class="smw-table"><thead><tr><th>Evidence ID</th><th>来源</th><th>Source Time</th><th>Segment</th><th>Offset</th><th>Worker</th><th>Memory ID</th></tr></thead><tbody>'
        + rows
        + "</tbody></table></div>"
        if rows
        else '<div class="smw-panel smw-muted">本运行没有可展示的 Evidence 索引。</div>'
    )
    truncated = (
        '<div class="smw-warning">Evidence 超过 200 条，页面仅展示前 200 条。</div>'
        if len(view.evidence_rows) > 200
        else ""
    )
    return (
        _STYLE
        + '<div class="smw-view"><div class="smw-title">历史记忆到最终 Evidence 的检索漏斗</div>'
        + '<div class="smw-funnel">'
        + "".join(funnel)
        + '</div><div class="smw-title">Evidence 索引</div>'
        + table
        + truncated
        + "</div>"
    )


def _metrics_table(title: str, metrics: dict[str, Any]) -> str:
    if not metrics:
        return f'<div class="smw-panel"><strong>{_e(title)}</strong><div class="smw-muted">N/A</div></div>'
    rows: list[str] = []
    for name, value in metrics.items():
        label, description = _metric_display(name)
        rows.append(
            "<tr>"
            f'<td><span class="smw-metric-name">{_e(label)}</span>'
            f'<code class="smw-metric-code">{_e(name)}</code></td>'
            f"<td>{_metric_value_html(value)}</td>"
            f'<td class="smw-metric-description">{_e(description)}</td>'
            "</tr>"
        )
    return (
        f'<div class="smw-panel"><strong>{_e(title)}</strong>'
        '<div class="smw-scroll"><table class="smw-table smw-metric-table"><thead><tr>'
        "<th>指标</th><th>得分</th><th>说明</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></div>"
    )


def _render_comparison(evaluation: dict[str, Any]) -> str:
    rows = [item for item in evaluation.get("rows", []) if isinstance(item, dict)]
    deltas = evaluation.get("deltas_from_first_summary", {})
    scope_deltas = evaluation.get("scope_deltas_from_first_summary", {})
    scope_problem_claims = evaluation.get("scope_problem_claims", {})
    scope_content_differences = evaluation.get("scope_content_differences", {})
    scope_metric_names = (
        "must_fact_recall",
        "useful_fact_precision",
        "fact_support_precision",
        "minutes_fact_f1",
        "unsupported_rate",
        "contradiction_rate",
        "partial_rate",
        "redundancy_rate",
    )
    metric_groups = (
        (
            "历史",
            (
                "historical_semantic_recall",
                "historical_useful_precision",
                "historical_fact_f1",
                "historical_unique_recall",
                "historical_unique_contribution_rate",
                "current_confirmed_history_recall",
            ),
        ),
        (
            "状态",
            (
                "state_transition_recall",
                "state_transition_precision",
                "state_transition_f1",
                "current_state_accuracy",
                "stale_state_error_rate",
            ),
        ),
        (
            "目标内容",
            (
                "target_content_must_recall",
                "target_content_useful_precision",
                "target_content_f1",
            ),
        ),
        (
            "目标导向",
            (
                "target_directed_must_recall",
                "target_directed_useful_precision",
                "target_directed_f1",
            ),
        ),
    )
    parts = [
        '<div class="smw-panel"><strong>多纪要横向评测</strong>',
        f'<div class="smw-muted">差值基线：{_e(evaluation.get("baseline_summary_id") or "N/A")}；仅并列展示，不自动推荐赢家。</div></div>',
    ]

    def render_group(
        *,
        title: str,
        names: tuple[str, ...],
        scope: str | None = None,
    ) -> None:
        """渲染一组横评指标；新双范围从 scope_metrics 读取。"""
        header = "".join(
            f'<th class="smw-comparison-metric">{_metric_header_html(name)}</th>'
            for name in names
        )
        body: list[str] = []
        scope_note = ""
        description_overrides: dict[str, str] | None = None
        if scope == "all_sources":
            scope_note = (
                "当前与历史材料统一形成事实单位；等价重复只计一次，"
                "状态关系不额外增加整体分母。"
            )
            description_overrides = {
                "must_fact_recall": "严格覆盖的全材料必要事实单位占全部全材料必要事实单位的比例；当前与历史等价事实合并计数。越高越好。",
                "useful_fact_precision": "全部纪要 Claim 中，得到当前或有效历史事实完整支持、属于有用事实单位且非重复的比例。越高越好。",
                "fact_support_precision": "全部纪要 Claim 中，得到当前或历史材料完整支持且时间、状态正确的比例。越高越好。",
                "minutes_fact_f1": "全材料必要事实召回率与全材料有用事实精确率的调和平均。越高越好。",
                "unsupported_rate": "全部纪要 Claim 中，当前和历史材料均无法支持的比例。越低越好。",
                "contradiction_rate": "全部纪要 Claim 中，与全材料事实及当前最终状态矛盾的比例。越低越好。",
                "partial_rate": "全部纪要 Claim 中只能由全材料部分支持，或关键限定不完整的比例。越低越好。",
                "redundancy_rate": "全部纪要 Claim 中重复表达同一事实的比例。越低越好。",
            }
        elif scope == "current_only":
            scope_note = (
                "只允许当前材料提供事实支持；仅由历史材料支持的 Claim 会标为"
                "“当前材料未提供支持，全材料已支持”，不直接称为幻觉。"
            )
            description_overrides = {
                "must_fact_recall": "严格覆盖的当前材料必要事实单位占全部当前必要事实单位的比例。越高越好。",
                "useful_fact_precision": "全部纪要 Claim 中，得到当前材料完整支持、属于当前有用事实且非重复的比例。越高越好。",
                "fact_support_precision": "全部纪要 Claim 中，仅依据当前材料即可获得完整事实支持且时间、状态正确的比例。越高越好。",
                "minutes_fact_f1": "当前材料必要事实召回率与当前材料有用事实精确率的调和平均。越高越好。",
                "unsupported_rate": "全部纪要 Claim 中，当前材料范围内没有支持的比例；其中可能包含由历史材料合法补充的内容。越低通常越好，需结合全材料结果解释。",
                "contradiction_rate": "全部纪要 Claim 中，与当前材料明确矛盾的比例。越低越好。",
                "partial_rate": "全部纪要 Claim 中只能由当前材料部分支持，或关键限定不完整的比例。越低越好。",
                "redundancy_rate": "全部纪要 Claim 中重复表达同一事实的比例。越低越好。",
            }
        for row in rows:
            summary_id = str(row.get("summary_id") or "")
            if scope is None:
                metrics = row.get("metrics", {})
                metric_deltas = (
                    deltas.get(summary_id, {}) if isinstance(deltas, dict) else {}
                )
            else:
                raw_scopes = row.get("scope_metrics", {})
                raw_scope_details = row.get("scope_metric_details", {})
                raw_scope_statuses = row.get("scope_statuses", {})
                raw_scope_deltas = (
                    scope_deltas.get(summary_id, {})
                    if isinstance(scope_deltas, dict)
                    else {}
                )
                scalar_metrics = (
                    raw_scopes.get(scope, {}) if isinstance(raw_scopes, dict) else {}
                )
                metrics = (
                    raw_scope_details.get(scope, {})
                    if isinstance(raw_scope_details, dict)
                    else {}
                )
                if not metrics:
                    metrics = scalar_metrics
                metric_deltas = (
                    raw_scope_deltas.get(scope, {})
                    if isinstance(raw_scope_deltas, dict)
                    else {}
                )
                row_status = (
                    raw_scope_statuses.get(scope)
                    if isinstance(raw_scope_statuses, dict)
                    else None
                )
            metrics = metrics if isinstance(metrics, dict) else {}
            metric_deltas = metric_deltas if isinstance(metric_deltas, dict) else {}
            if scope is None:
                row_status = row.get("status")
            values = "".join(
                f'<td class="smw-comparison-value">{_metric_value_html(metrics.get(name))}</td>'
                for name in names
            )
            delta_cells = "".join(
                f'<td class="smw-comparison-delta">{_e(_fmt_metric(metric_deltas.get(name), signed=True))}</td>'
                for name in names
            )
            body.append(
                f'<tr><td class="smw-comparison-summary">{_e(row.get("display_name") or summary_id)}</td>'
                f'<td class="smw-comparison-status">{_e(_status_text(row_status))}</td>{values}</tr>'
                '<tr class="smw-comparison-delta-row">'
                '<td class="smw-comparison-delta-label">相对首份差值</td><td></td>'
                f"{delta_cells}</tr>"
            )
        parts.append(
            '<section class="smw-comparison-section">'
            f'<div class="smw-title">{_e(title)}指标</div>'
            + (f'<div class="smw-muted">{_e(scope_note)}</div>' if scope_note else "")
            + '<div class="smw-scroll"><table class="smw-table smw-comparison-table">'
            f"<thead><tr><th>纪要</th><th>状态</th>{header}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>"
            + _metric_guide_html(
                names,
                description_overrides=description_overrides,
            )
            + "</section>"
        )

    has_scope_metrics = any(
        isinstance(row.get("scope_metrics"), dict) and row.get("scope_metrics")
        for row in rows
    )
    if has_scope_metrics:
        render_group(
            title="全材料整体",
            names=scope_metric_names,
            scope="all_sources",
        )
        render_group(
            title="当前材料整体",
            names=scope_metric_names,
            scope="current_only",
        )
    else:
        render_group(
            title="整体（旧运行口径）",
            names=(
                "must_fact_recall",
                "current_must_fact_recall",
                "useful_fact_precision",
                "fact_support_precision",
                "minutes_fact_f1",
            ),
        )
    for title, names in metric_groups:
        render_group(title=title, names=names)
    row_by_id = {str(item.get("summary_id") or ""): item for item in rows}
    qualitative = evaluation.get("qualitative_comparison", {})
    scope_labels = {
        "all_sources": "全材料",
        "current_only": "当前材料",
    }
    parts.append(
        '<div class="smw-title">LLM 直接定性对比</div>'
        '<div class="smw-muted">解释层直接阅读对应范围的原始材料与全部纪要；'
        "仅帮助理解差异，不参与数值评分，也不自动推荐赢家。</div>"
    )
    for scope in ("all_sources", "current_only"):
        result = qualitative.get(scope, {}) if isinstance(qualitative, dict) else {}
        section = [
            '<section class="smw-comparison-section">',
            f'<div class="smw-title">{_e(scope_labels[scope])}直接对比</div>',
        ]
        if not isinstance(result, dict) or result.get("status") not in {
            "completed",
            "completed_with_warnings",
        }:
            message = (
                result.get("error") or "该运行未生成直接定性对比。"
                if isinstance(result, dict)
                else "该运行未生成直接定性对比。"
            )
            section.append(f'<div class="smw-muted">{_e(message)}</div>')
        else:
            if result.get("status") == "completed_with_warnings":
                section.append(
                    '<div class="smw-warning">部分定性观察因证据 Span 无效而被隔离；'
                    f"未核验 {_e(result.get('unverified_note_count', 0))} 条。</div>"
                )
            section.append(
                f'<div class="smw-interpretation-overview">{_e(result.get("overview") or "N/A")}</div>'
            )
            for item in result.get("summaries", []):
                if not isinstance(item, dict):
                    continue
                summary_id = str(item.get("summary_id") or "")
                display_name = row_by_id.get(summary_id, {}).get(
                    "display_name", summary_id
                )
                detail_rows: list[str] = []
                for label, notes in (
                    ("优势", item.get("strengths", [])),
                    ("局限", item.get("limitations", [])),
                ):
                    for note in notes if isinstance(notes, list) else []:
                        if not isinstance(note, dict):
                            continue
                        quote = note.get("summary_quote") or note.get("source_quote")
                        detail_rows.append(
                            f"<li><strong>{_e(label)}：</strong>{_e(note.get('description'))}"
                            f'<span class="smw-quote">依据：{_e(quote)}</span></li>'
                        )
                section.append(
                    '<div class="smw-interpretation-item">'
                    f"<strong>{_e(display_name)}</strong>"
                    f'<div class="smw-muted">{_e(item.get("relative_to_baseline") or "N/A")}</div>'
                    f"<ul>{''.join(detail_rows)}</ul></div>"
                )
        section.append("</section>")
        parts.extend(section)

    verified = evaluation.get("verified_content_differences", {})
    baseline_id = str(evaluation.get("baseline_summary_id") or "")
    baseline_name = row_by_id.get(baseline_id, {}).get(
        "display_name", baseline_id or "第一份纪要"
    )
    verified_sections: list[str] = []
    for scope in ("all_sources", "current_only"):
        by_summary = verified.get(scope, {}) if isinstance(verified, dict) else {}
        verified_sections.append(
            f'<div class="smw-title">{_e(scope_labels[scope])}</div>'
        )
        if not isinstance(by_summary, dict) or not by_summary:
            verified_sections.append(
                '<div class="smw-muted">该运行没有可用的二次核验结果。</div>'
            )
            continue
        for summary_id, difference in by_summary.items():
            if not isinstance(difference, dict):
                continue
            display_name = row_by_id.get(str(summary_id), {}).get(
                "display_name", summary_id
            )
            if difference.get("status") not in {"completed", "completed_with_warnings"}:
                verified_sections.append(
                    f'<div class="smw-warning">{_e(display_name)} 相对 {_e(baseline_name)}：'
                    f"{_e(difference.get('error') or difference.get('reason') or '核验未完成')}</div>"
                )
                continue
            verified_sections.append(
                '<div class="smw-diff-summary">'
                f"{_e(display_name)} 相对 {_e(baseline_name)}：确认新增 "
                f"{_e(difference.get('confirmed_added_count', 0))} 条，确认减少 "
                f"{_e(difference.get('confirmed_reduced_count', 0))} 条；未确认 "
                f"{_e(difference.get('not_confirmed_count', 0))} 条，不确定 "
                f"{_e(difference.get('uncertain_count', 0))} 条，未核验 "
                f"{_e(difference.get('unverified_count', 0))} 条。</div>"
            )
            difference_rows: list[str] = []
            for item in difference.get("items", []):
                if not isinstance(item, dict) or item.get(
                    "verification_result"
                ) not in {"confirmed_added", "confirmed_reduced"}:
                    continue
                is_added = item.get("verification_result") == "confirmed_added"
                difference_rows.append(
                    "<tr>"
                    f'<td class="smw-diff-kind {"gain" if is_added else "loss"}">'
                    f"{'确认新增' if is_added else '确认减少'}</td>"
                    f"<td>{_e(item.get('fact') or item.get('unit_id'))}</td>"
                    f"<td>{_e(item.get('baseline_quote') or '无')}</td>"
                    f"<td>{_e(item.get('candidate_quote') or '无')}</td>"
                    f"<td>{_e(item.get('reason') or '')}</td></tr>"
                )
            if difference_rows:
                verified_sections.append(
                    '<div class="smw-scroll"><table class="smw-table smw-diff-table">'
                    "<thead><tr><th>结论</th><th>事实</th><th>基线证据</th>"
                    "<th>候选证据</th><th>说明</th></tr></thead>"
                    f"<tbody>{''.join(difference_rows)}</tbody></table></div>"
                )
    parts.append(
        '<details class="smw-diagnostics">'
        "<summary>事实覆盖差异（二次核验，仅供核查）</summary>"
        '<div class="smw-diagnostics-body">'
        '<div class="smw-diagnostics-note">原始候选来自 Claim 集合差异；这里只展示模型重新阅读两份纪要后确认的新增或减少。'
        "核验结果不回写数值指标。</div>"
        f"{''.join(verified_sections)}</div></details>"
    )
    diagnostic_rows: list[str] = []
    added_problem_keys: set[tuple[str, str, str, str]] = set()
    if isinstance(scope_content_differences, dict):
        for scope, differences in scope_content_differences.items():
            if not isinstance(differences, dict):
                continue
            for summary_id, difference in differences.items():
                if not isinstance(difference, dict):
                    continue
                for claim in difference.get("added_problem_claims", []):
                    if not isinstance(claim, dict):
                        continue
                    added_problem_keys.add(
                        (
                            str(scope),
                            str(summary_id),
                            str(claim.get("summary_fact_id") or ""),
                            str(claim.get("status") or ""),
                        )
                    )
    if isinstance(scope_problem_claims, dict):
        for scope, by_summary in scope_problem_claims.items():
            if not isinstance(by_summary, dict):
                continue
            for summary_id, claims in by_summary.items():
                if not isinstance(claims, list):
                    continue
                for claim in claims:
                    if not isinstance(claim, dict):
                        continue
                    key = (
                        str(scope),
                        str(summary_id),
                        str(claim.get("summary_fact_id") or ""),
                        str(claim.get("status") or ""),
                    )
                    interpretation = claim.get("scope_interpretation")
                    note = (
                        "当前材料未提供支持，全材料已支持"
                        if interpretation == "supported_by_all_sources_not_current"
                        else str(claim.get("reason") or "")
                    )
                    diagnostic_rows.append(
                        "<tr>"
                        f"<td>{_e(scope_labels.get(str(scope), scope))}</td>"
                        f'<td class="smw-comparison-summary">{_e(row_by_id.get(str(summary_id), {}).get("display_name", summary_id))}</td>'
                        f"<td>{_e('相对基线新增' if key in added_problem_keys else '问题 Claim')}</td>"
                        f"<td>{_e(claim.get('claim') or claim.get('summary_fact_id'))}</td>"
                        f"<td>{_e(claim.get('status') or 'N/A')}</td>"
                        f"<td>{_e(note)}</td></tr>"
                    )
    diagnostics_body = (
        '<div class="smw-scroll"><table class="smw-table">'
        "<thead><tr><th>范围</th><th>纪要</th><th>类型</th><th>Claim</th><th>判定</th><th>说明</th></tr></thead>"
        f"<tbody>{''.join(diagnostic_rows)}</tbody></table></div>"
        if diagnostic_rows
        else '<div class="smw-muted">本次没有问题诊断记录。</div>'
    )
    parts.append(
        '<details class="smw-diagnostics">'
        "<summary>问题诊断（模型判定，仅供核查）</summary>"
        '<div class="smw-diagnostics-body">'
        '<div class="smw-diagnostics-note">这些结果可能存在模型误判，不作为必须接受的结论。</div>'
        f"{diagnostics_body}</div></details>"
    )
    return "".join(parts)


def _render_lite_evaluation(evaluation: dict[str, Any]) -> str:
    rows = [item for item in evaluation.get("rows", []) if isinstance(item, dict)]
    body: list[str] = []
    for item in rows:
        metrics = (
            item.get("metrics", {}) if isinstance(item.get("metrics"), dict) else {}
        )
        body.append(
            "<tr>"
            f'<td class="smw-comparison-summary">{_e(item.get("display_name") or item.get("summary_id"))}</td>'
            f"<td>{_e(_status_text(item.get('status')))}</td>"
            f"<td>{_metric_value_html(metrics.get('current_key_fact_coverage'))}</td>"
            f"<td>{_metric_value_html(metrics.get('material_support_rate'))}</td>"
            f"<td>{_metric_value_html(metrics.get('key_fact_relevance_rate'))}</td>"
            f"<td>{_e(item.get('claim_count', 0))}</td>"
            f"<td>{_e(len(item.get('missing_reference_fact_ids', [])))}</td>"
            f"<td>{_e(len(item.get('unsupported_claim_ids', [])))}</td>"
            "</tr>"
        )
    stats = evaluation.get("execution_stats", {})
    if not isinstance(stats, dict):
        stats = {}
    table = (
        '<div class="smw-scroll"><table class="smw-table smw-comparison-table">'
        "<thead><tr><th>纪要</th><th>状态</th><th>当前关键事实覆盖率</th>"
        "<th>材料支持率</th><th>关键事实关联率</th><th>原子事实</th>"
        "<th>遗漏关键事实</th><th>未支持事实</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table></div>"
        if body
        else '<div class="smw-muted">没有可展示的精简评测候选结果。</div>'
    )
    return (
        '<div class="smw-title">精简 Evaluation · '
        + _e(evaluation.get("status") or "disabled")
        + "</div>"
        + '<div class="smw-panel"><div class="smw-muted">'
        + "参考事实由模型从当前材料提取，并非人工金标准；三项指标与严格模式口径不同。"
        + "历史材料只参与支持性判断，不进入当前关键事实覆盖率分母。"
        + "</div>"
        + f'<div class="smw-muted">参考事实 {_e(evaluation.get("reference_fact_count", 0))} 条 · '
        + f"模型调用 {_e(stats.get('model_call_count', 0))} 次 · "
        + f"耗时 {_e(_fmt_duration(stats.get('elapsed_ms')))}</div></div>"
        + table
    )


def render_review_evaluation_html(view: WebRunVisualization) -> str:
    review = view.review
    review_rows = "".join(
        "<tr>"
        f"<td>{_e(item.get('round_index'))}</td>"
        f"<td>{_e(item.get('red_candidate_issue_count'))}</td>"
        f"<td>{_e(item.get('red_kept_count'))}</td>"
        f"<td>{_e(item.get('red_dismissed_count'))}</td>"
        f"<td>{_e(item.get('blue_applied_count'))}</td>"
        f"<td>{_e(item.get('judge_accept_count'))}</td>"
        f"<td>{_e(item.get('judge_non_accept_count'))}</td>"
        f"<td>{_e(item.get('deferred_count'))}</td>"
        f"<td>{_e(item.get('remaining_issue_count'))}</td>"
        f"<td>{_e(item.get('judge_disposition'))}</td>"
        "</tr>"
        for item in review.get("rounds", [])
        if isinstance(item, dict)
    )
    review_html = (
        '<div class="smw-panel"><strong>Shadow Review</strong>'
        f'<div class="smw-muted">终止：{_e(review.get("termination") or "disabled")} · 推荐 candidate：{_e(_fmt_count(review.get("recommended_candidate_index")))}</div></div>'
    )
    if review_rows:
        review_html += (
            '<div class="smw-scroll"><table class="smw-table"><thead><tr><th>轮次</th><th>Red 候选</th><th>Red 保留</th><th>Red 驳回</th><th>Blue 修改</th><th>Judge 接受</th><th>Judge 未接受</th><th>Deferred</th><th>Remaining</th><th>Judge 裁决</th></tr></thead><tbody>'
            + review_rows
            + "</tbody></table></div>"
        )
    evaluation = view.evaluation
    if evaluation.get("mode") == "lite":
        evaluation_html = _render_lite_evaluation(evaluation)
    elif evaluation.get("comparison"):
        evaluation_html = _render_comparison(evaluation)
    else:
        overall = (
            evaluation.get("overall", {})
            if isinstance(evaluation.get("overall"), dict)
            else {}
        )
        personalization = (
            evaluation.get("personalization", {})
            if isinstance(evaluation.get("personalization"), dict)
            else {}
        )
        evaluation_html = (
            f'<div class="smw-title">Evaluation · {_e(evaluation.get("status") or "disabled")}</div>'
            + '<div class="smw-metrics-grid">'
            + _metrics_table("整体事实", overall.get("core", {}))
            + _metrics_table("历史事实", overall.get("history", {}))
            + _metrics_table("状态变化", overall.get("transition", {}))
            + _metrics_table("安全性", overall.get("safety", {}))
            + _metrics_table("简洁性", overall.get("conciseness", {}))
            + _metrics_table("目标内容与目标导向", personalization.get("macro", {}))
            + "</div>"
        )
    return _STYLE + '<div class="smw-view">' + review_html + evaluation_html + "</div>"


def visualization_json(view: WebRunVisualization) -> str:
    """为测试和受控调试返回不含原文/Prompt 的 ViewModel JSON。"""
    return json.dumps(view.model_dump(mode="json"), ensure_ascii=False, indent=2)
