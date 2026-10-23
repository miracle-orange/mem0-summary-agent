# Summary Multi-Agent V2 权威架构

## 1. 文档地位

本文是 Summary Multi-Agent V2 的当前权威架构说明。设计依据按以下顺序解释：

1. `design_inputs/SUMMARY_MULTI_AGENT_V2_DESIGN_INPUTS/` 中的最终流程图、模块说明和验收约束；
2. 当前 `scripts/agent/` 中已经验证可运行的能力与兼容要求；
3. `qiqihezh/deepresearch-agent` 仅提供自研编排、DAG 和 AgentPool 思路，不决定本项目业务流程，也不复制其代码。

`SUMMARY_MULTI_AGENT_PLAN.md` 是 V1 历史设计，不再决定 V2 的 Agent 划分和运行流程。

## 2. 已确认的架构决策

1. Typed Memory 继续使用固定六类：`time`、`decision`、`constraint`、`task`、`risk`、`preference`；设计材料中的 `action_item` 映射为 `task`。
2. V2 首先改造 Summarize；当前 Ingest、跨 chunk 去重和长期 Memory 写入流程保持可用。
3. Baseline Summarize 与 Multi-Agent V2 并存；目标机器验收前 baseline 仍是默认模式。
4. Artifact Board 保留为历史检索证据的派生审计视图，不等同于 M4 Shared Working Memory。
5. 第一阶段不迁移 Qdrant 存量数据；Memory Adapter 将旧 metadata 映射为 V2 `EvidenceRef`。
6. 超长当前材料采用全部分段参与的结构化 map-reduce，并以 Coverage Ledger 强制完整覆盖。
7. M5 初期使用 shadow mode；目标机器验证通过后才允许审核结果自动替换 baseline 输出。
8. P5 之后先实现模型驱动 Planner、动态 DAG、真实 Replanning、M4 协作和辅助 Context Compressor，再开发 Web V2。
9. Context Compressor 永不压缩、截断或替代当前原始材料；它只处理历史召回、重复 Worker/Tool 结果和其他辅助上下文。
10. 自主性受 Worker/Tool Registry、Schema、DAG、权限和预算约束；模型不能动态导入代码、创建任意工具或写长期 Memory。
11. 自主性同时按语义自主性和运行自主性评价；Orchestrator/AgentPool/Tool/M4/Compressor 不因具备异步或规则能力而被表述为语义 Agent。
12. Model Planner 负责全局任务自主拆解；四类 Worker 负责子任务内受控工具行动；Summarizer 保持证据合成组件；详细边界见 `SUMMARY_MULTI_AGENT_WORKER_AUTONOMY_ZH.md`。

## 3. 生命周期边界

### 3.1 Ingest

```text
历史原始资料
-> 分块 typed memory 抽取
-> 同一历史资料跨 chunk 去重
-> memory.add(..., infer=False)
-> mem0 + Qdrant 长期知识库
```

Ingest 是唯一允许写入长期 Memory 的在线入口。V2 不把最终摘要、Worker 结果、审核意见或运行状态自动写入长期 Memory。

### 3.2 Summarize V2

```text
SummaryRequest + RawMaterialBundle
-> M1 Orchestrator 初始化 M4
-> M2 Planner 生成并校验 JSON DAG
-> 拓扑调度 + AgentPool + 四类受控自主 Worker + Tool Registry
-> AgentResult 收集、失败统计和有限 Replanning
-> M3 仅压缩辅助上下文
-> Summarizer 生成候选摘要
-> 可选 M5 Red-Blue-Judge
-> 保存摘要、证据索引、状态与 Trace
-> 可选事实级 Evaluation Sidecar
```

四类 Worker 固定为：

- `semantic_memory`：检索和筛选历史 Typed Memory；
- `entity_personalization`：处理目标实体、相关人物和一跳关系；
- `template_adaptation`：根据摘要要求、用途和偏好选择或构造输出模板；
- `temporal_reasoning`：识别相对时间，调用确定性时间工具完成规范化；无可靠过去/未来方向的裸星期保持歧义，不能默认猜成未来日期。

风险、任务和决策不会因为没有同名 Worker 而消失：历史信息由 Semantic Memory Worker 召回，当前材料事实由完整分段分析和 Summarizer 处理。

P6.0 已完成 PlannerInput、Worker/Tool Capability、计划预算、来源/版本/父计划/Replan 原因/成功任务保留契约和 `plan_history.json`。P6.1 已在不删除 fallback 的前提下接入模型驱动初始 Planner：模型可以根据请求动态选择 Registry 中的能力，并为同一能力创建一个或多个不同目标的子任务，但不得发明未注册 Worker/Tool。模型只输出 `ModelPlanDraft`，`plan_id/source/lineage` 由系统生成；系统从完整材料建立确定性工作量下界，防止任务存在但预算不足。P6.3 已允许模型 Worker 任务声明最小数据依赖；规则 Worker 仍保持空 dependencies。P6.5 已接入模型驱动 Replanner：只在 Orchestrator 的有限关键失败循环中调用，读取成功结果/Evidence、失败和剩余预算，输出 `ModelReplanDraft`；Runtime 继续掌握计划身份、权限、预算、执行和终止控制。

P6.2 将固定逻辑 Worker 升级为子任务范围内的行动型 Agent。P6.2.0 已实现通用 WorkerAction/Observation/预算/终止协议和模型行动循环；P6.2.1 已把四类生产 Worker 接入该循环，并由专业结果组装器把模型选中的成功 Observation 转换为既有 payload/Evidence。Worker 可以根据目标和 Tool Observation 决定调用哪个授权工具、是否补充行动以及何时完成，但不能修改 DAG、调用其他 Agent、直接写 M4 或长期 Memory。四类 capability 保持专业边界，Planner 可以选择子集或创建多个实例，不要求每轮机械执行全部四类。

P6.3 在该基础上接入真实数据依赖。Orchestrator 只把声明依赖且成功的 `AgentResult` 以最小受控投影交给下游 Worker；模型动作必须通过 `referenced_prior_result_ids` 标注使用点，具体上游证据通过 `referenced_prior_evidence_ids` 精确选择，成功结果以 `consumed_result_ids` 和 `consumed_evidence_ids` 保存 lineage。未声明、缺失、失败或最终未消费的依赖会被拒绝。上游原始 ToolResult、模型 raw output 和完整执行审计不进入下游 Prompt。M4 checkpoint 可严格加载并保留这些关系。目标机器运行 `20260810-141730` 已验证非空依赖、显式消费、`lineage_ok=true` 且无 fallback。

P6.4 在 M4 与 Summarizer/Red 之间增加确定性辅助 Context Compressor。完整 `AgentResult`、ToolResult、Worker execution 和 raw trace 仍保留在 M4 与审计产物；合成视图把跨 Worker 同一 `memory_id` 归并为一份 canonical `historical_records`，各 Worker 通过 `record_ids` 保留来源关联。time/decision/constraint/task/risk 唯一记录全部保留，非保护类型才受独立 top-k；同一身份的冲突内容以 `content_variants` 保留。压缩前后校验当前材料哈希、segment manifest、Coverage Ledger 和当前原文 Evidence offset，任一变化直接失败。目标机器 `deterministic|disabled` 配对验收已确认四组原文不变量保持。

### 3.3 Agent、Runtime 与基础设施边界

| 模块 | 定位 |
|---|---|
| Planner/Replanner | 受约束的全局规划 Agent |
| 四类 Worker | 子任务内有限 action-observation 循环的专业 Agent |
| Red/Blue/Judge | 问题发现、局部修复和质量决策 Agent |
| Summarizer | LLM 驱动的证据合成组件，可报告证据不足但不直接取证 |
| Orchestrator | 高运行自治的 Agent Runtime，不理解并改写业务目标 |
| AgentPool | Worker Registry、复用和生命周期管理器 |
| Tool Registry/Tool | 被动、权限化、Schema 化能力 |
| M4/Artifact Board/Compressor | 状态、审计视图和辅助上下文处理基础设施 |

### 3.4 asyncio 编排模型

V2 继续使用自研 `asyncio + DAG + AgentPool`。Orchestrator 按拓扑层并发无依赖任务，分别限制 Worker、模型和工具并发；单 Worker 内依赖 Observation 的行动默认串行。timeout/cancel 必须向下传播并释放 Pool lease，M4/checkpoint 写入受锁保护，异步路径内禁止嵌套 `asyncio.run()`。

DeepSeek WebSocket 初期模型全局并发保持 `1`，不能用 `max_concurrent_workers` 同时放大模型调用；Worker 可以并发执行非模型准备和独立工具。确认目标机器网关稳定后再单独调整模型并发。

## 4. M4 Shared Working Memory

M4 是单次运行内的受控状态，不是新的长期向量库。它在 Orchestrator 启动时创建，至少包含：

- 不可变 `SummaryRequest` 和 `RawMaterialBundle`；
- 当前 DAG、任务状态、AgentResult 和 ToolResult；
- 每个任务可用与实际消费的前序 AgentResult/Evidence lineage；
- Evidence Index 与原文 Coverage Ledger；
- Artifact Board 历史证据视图；
- Replanning、降级、错误、审核轮次和运行指标；
- 追加式 Trace，以及输出目录中的可选原子 checkpoint。

Summarize 中的 Memory Adapter 默认只读。状态更新必须经 Orchestrator，不允许 Worker 任意修改全局状态。

## 5. 当前原始资料保护

当前原始资料是本轮第一事实来源，必须满足：

1. 不执行首尾截断、相似度过滤、Top-K 选段或 LLM 短摘要替换；
2. 长材料按稳定字符偏移生成 `RawSegment`，每段保留 ID、起止位置和哈希；
3. 所有分段都必须进入规定的当前材料分析阶段，并写入 Coverage Ledger；
4. Evidence Index 必须能回查当前原文 segment/offset 或长期 Memory ID；
5. 任一必需分段未成功处理时不得返回无警告 `COMPLETED`；
6. M3 只处理历史召回、冗余 Worker 结果和工具轨迹等辅助上下文。

M3 可以读取摘要要求和当前材料的结构化索引来判断辅助信息相关性，但输出只能替换辅助上下文，不能替换 `current_material`、`RawMaterialBundle`、`RawSegment`、Coverage Ledger、Current Fact Map 或原文 Evidence。压缩前后必须校验当前材料哈希、segment coverage 和 evidence offset 不变，并在产物中保持 `current_material_compressed=false`。

现有 `prepare_current_material_for_agents()` 的首尾预算行为只允许继续服务 baseline，V2 不得调用它。

## 6. Artifact Board 定位

现有 `artifact_board.json` 继续用于展示本轮检索到的六类历史 Typed Memory，并帮助兼容现有摘要 prompt 和 Web UI。V2 中它是：

```text
M4.historical_evidence_board
```

它不是 DAG 状态、Worker 通信总线或当前材料事实板。V2 的核心产物包括 `plan.json`、`plan_history.json`、`working_state.json`、`agent_results.json`、`evidence_index.json`、`coverage_ledger.json`、`synthesis_context.json`、Review 阶段产物、`execution_trace.jsonl` 和 `summary_run_result.json`。

## 7. 兼容与切换

后续入口使用显式模式：

```text
baseline
multi_agent_v2
```

在 V2 通过本机 mock、目标机器 Qwen/DeepSeek E2E 和 baseline 回归前，默认值保持 `baseline`。新增功能严格位于 `scripts/agent/`，不改写根目录 `scripts/` 稳定 demo。

## 8. 非目标

当前阶段不做以下事项：

- 不迁移或批量重写已有 Qdrant Typed Memory；
- 不引入 LangGraph、CrewAI、AutoGen 等编排框架；
- 不让 Agent 自动决定长期 Memory 写入；
- 不把 M6 Evolution Engine 放到在线关键路径；
- 不在未验证前删除 baseline、Artifact Board、现有 Critic 或 Rewrite Draft。

## 9. 自主化与 Web 准入

P6 的权威实施门槛见 `SUMMARY_MULTI_AGENT_AUTONOMY_ROADMAP_ZH.md`。模型 Planner、动态 DAG、携带成功结果的有限 Replanning、M4 跨 Agent 协作、辅助 Context Compressor 和跨领域回归完成前，不进入 Web V2。Web 是已验证能力的使用与审计入口，不负责弥补核心自主编排能力。

## 10. 事实级 Evaluation Sidecar

P7.1 在正式摘要保存后提供可选、只读的事实级评测。它从完整当前材料独立构建带 quote/offset 的源事实，并只读取本轮 `SynthesisContext` 实际可见的历史记录；随后从正式摘要 candidate 0 抽取事实、执行当前事实优先的语义匹配，再由 Python 计算 Recall、Precision、F1、安全性和简洁性指标。

Evaluation 不属于生成或审核循环：不改变 DAG、M4、Review 选择或正式摘要，不扫描完整 Qdrant，不写长期 Memory，也不自行重算相对日期。Temporal Worker 的成功结果只作为匹配证据。评测失败只把已成功的摘要运行降为 `completed_with_warnings`，并保留独立失败产物。完整契约见 `SUMMARY_MULTI_AGENT_EVALUATION_DESIGN_ZH.md`。

P7.2 在显式 `target_entities` 非空时追加目标实体评测分支。它复用 P7.1 已完成 grounding 的源事实、正式摘要事实和 current-first 匹配结果，只由模型分类目标相关性，再由 Python 计算 per-target/macro/micro 指标。该分支不依赖 Planner 是否生成 Entity Worker，也不从自然语言 Request 猜测目标；独立预算或 timeout 失败不会把整体 `evaluation_status` 改为失败。
