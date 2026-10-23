# 精简评测：流程、指标与运行边界

## 两个入口，一套核心

精简评测同时用于生成后 candidate 0 评测和独立 2–5 份纪要横评。两者复用 `evaluation/lite/`，只由适配器决定证据范围：

- 在线：完整当前原文，加本次最终可见的历史 Memory 业务内容；
- 独立横评：完整当前原文，加用户明确提供的历史原文。

旧请求默认 `evaluation_mode=strict`。精简模式必须显式选择 `lite`，不会调用严格模式的 Source Snapshot、双 scope Matcher、个性化、定性解释或差异事实复核。

## 四类模型调用

1. **参考事实抽取**：只看当前材料、材料日期和共同摘要要求，返回 `{"key_facts":[...]}`。超长当前材料按自然边界切为约 3000 字、重叠 100 字；只对完全相同的事实文字做确定性去重。
2. **纪要事实拆分**：每份纪要各调用一次，只看该纪要，返回 `{"atomic_facts":[...]}`。它抽取所有主张，不判断正确性。
3. **覆盖匹配**：输入编号后的参考事实和候选原子事实，返回每条参考事实对应的单条候选编号或 `null`。主题相近、部分提及、主体/数字/否定/状态不一致均不算完整覆盖。
4. **材料支持判断**：输入允许证据和候选原子事实，逐条返回 `supported` 或 `not supported`。历史 Memory 明确标为抽取记录，不能自动证明当前已完成、取消或作出决定。

正常且无需分批、修复时，调用数为 `K + 3N`。匹配和支持判断职责独立：遗漏关键事实不等于纪要内容错误，覆盖成功也不能替代材料支持判断。

## 指标

| 指标 | 计算 |
| --- | --- |
| 当前关键事实覆盖率 | 匹配成功的当前参考事实数 / 当前参考事实总数 |
| 材料支持率 | 判为 supported 的候选原子事实数 / 候选原子事实总数 |
| 关键事实关联率 | 被任一参考事实匹配的不同候选事实数 / 候选原子事实总数 |

参考事实是模型抽取结果，不是人工金标准。三项指标不冒充严格模式的 precision、recall 或 F1，也不合成总分或推荐最佳纪要。零分母为 `N/A`。

## 失败、预算与审计

- JSON/编号契约最多修复一次；上下文超限、超时、取消和预算耗尽不进入格式修复。
- 覆盖或支持其中一侧失败时，另一侧结果继续保留；失败侧指标为 `N/A`，不能用缩小后的分母发布正常分数。
- 默认整轮 1200 秒、单次调用 180 秒、最多 64 次调用、并发 1；可独立配置为并发 1–4。
- 模型窗口已知且本地匹配 tokenizer 可可靠计量时执行请求前阻断；只有估算时只记录风险。服务端明确超限统一归类为 `context_overflow`。
- 运行内缓存键包含实际 Prompt、阶段、输出契约和输出预算；不建立跨项目隐式缓存。

业务产物为 `evaluation_lite_reference.json`、`evaluation_lite_result.json`、Markdown/HTML 报告。`evaluation_lite_model_trace.json` 保存 Prompt 和原始响应，仅供本机审计，不进入 Web 安全下载包。

## 命令行运行

生成后评测默认仍关闭；同时显式开启评测并选择 `lite`：

```bash
python scripts/agent/summary_multi_agent_demo.py \
  --current-file path/to/current.md \
  --user-id example_user \
  --project example_project \
  --source-time 2026-10-15 \
  --summary-request "生成结构化项目会议纪要" \
  --enable-evaluation \
  --evaluation-mode lite \
  --output-dir outputs/agent_multi_lite_online
```

独立比较 2–5 份纪要：

```bash
python scripts/agent/summary_multi_agent_evaluation_demo.py \
  --project example_project \
  --summary-request "生成结构化项目会议纪要" \
  --current-file path/to/current.md \
  --current-source-time 2026-10-15 \
  --history "history_1|上次会议|2026-10-08|path/to/history.md" \
  --summary "direct|直接生成|path/to/direct.md" \
  --summary "multi_agent|多代理|path/to/multi_agent.md" \
  --evaluation-mode lite \
  --lite-model-context-window-tokens 65536 \
  --output-dir outputs/agent_multi_lite_compare
```

Web 的“生成纪要”和“纪要评测”两个入口均可选择“严格评测／精简评测”。精简模式不会读取严格模式的个性化、定性解释、差异复核或小批匹配参数；切回严格模式时原参数和原产物命名继续生效。
