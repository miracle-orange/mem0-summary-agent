from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent_web.visualization import build_run_visualization  # noqa: E402
from summary_multi_agent_web.visualization_render import (  # noqa: E402
    WEB_VISUALIZATION_CSS,
    render_agent_graph_html,
    render_evidence_memory_html,
    render_overview_html,
    render_review_evaluation_html,
    render_task_detail_html,
    render_timeline_html,
    task_choices,
)


def _write_json(root: Path, name: str, value: object) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class WebRunVisualizationTests(unittest.TestCase):
    def test_summarize_projects_declared_and_consumed_dependencies_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_json(
                root,
                "web_run_info.json",
                {
                    "run_id": "run-vis",
                    "run_type": "multi_agent_summarize",
                    "status": "completed_with_warnings",
                    "stage": "completed_with_warnings",
                    "started_at": "2026-09-07T01:00:00+00:00",
                    "finished_at": "2026-09-07T01:00:06+00:00",
                },
            )
            _write_json(
                root,
                "summary_run_result.json",
                {
                    "run_id": "run-vis",
                    "status": "completed_with_warnings",
                    "evaluation_enabled": True,
                    "evaluation_status": "completed",
                    "warnings": ["<script>alert('warning')</script>"],
                },
            )
            _write_json(
                root,
                "plan.json",
                {
                    "plan_id": "plan-2",
                    "plan_version": 2,
                    "source": "model_replan",
                    "parent_plan_id": "plan-1",
                    "replan_round": 1,
                    "preserved_task_ids": ["memory"],
                    "tasks": [
                        {
                            "task_id": "memory",
                            "task_type": "semantic_memory",
                            "description": "检索历史 <script>alert(1)</script>",
                            "dependencies": [],
                            "critical": False,
                            "allowed_tools": ["typed_memory_search"],
                        },
                        {
                            "task_id": "entity",
                            "task_type": "entity_personalization",
                            "description": "形成目标实体证据",
                            "dependencies": ["memory"],
                            "critical": True,
                            "allowed_tools": ["entity_query"],
                        },
                        {
                            "task_id": "temporal",
                            "task_type": "temporal_reasoning",
                            "description": "解析时间",
                            "dependencies": ["memory"],
                            "critical": False,
                            "allowed_tools": ["calendar_resolve"],
                        },
                    ],
                },
            )
            _write_json(
                root,
                "plan_history.json",
                [
                    {
                        "plan_id": "plan-1",
                        "plan_version": 1,
                        "source": "model",
                        "replan_round": 0,
                        "tasks": [{"task_id": "memory"}],
                    },
                    json.loads((root / "plan.json").read_text("utf-8")),
                ],
            )
            _write_json(
                root,
                "agent_results.json",
                {
                    "memory": {
                        "task_type": "semantic_memory",
                        "status": "success",
                        "duration_ms": 1500,
                        "evidence": [{"evidence_id": "ev-memory"}],
                        "warnings": [],
                        "tool_results": [
                            {
                                "tool_name": "typed_memory_search",
                                "status": "success",
                                "payload": {
                                    "diagnostics": {
                                        "backend_raw_count": 8,
                                        "accepted_count": 5,
                                    }
                                },
                            }
                        ],
                        "worker_execution": {
                            "model_attempts": [{"raw_output": "must not render"}],
                            "final_budget": {"model_calls_used": 2, "tool_calls_used": 1},
                        },
                    },
                    "entity": {
                        "task_type": "entity_personalization",
                        "status": "success",
                        "duration_ms": 900,
                        "evidence": [],
                        "warnings": ["one"],
                        "tool_results": [],
                        "consumed_result_ids": ["memory"],
                        "worker_execution": {
                            "final_budget": {"model_calls_used": 1, "tool_calls_used": 0}
                        },
                    },
                    "temporal": {
                        "task_type": "temporal_reasoning",
                        "status": "success",
                        "duration_ms": 700,
                        "evidence": [],
                        "warnings": [],
                        "tool_results": [],
                        "worker_execution": {},
                    },
                },
            )
            _write_json(root, "working_state.json", {"task_states": {}})
            _write_json(
                root,
                "memory_store_probe.json",
                {"status": "ready", "matching_count": 13},
            )
            _write_json(
                root,
                "context_compression_report.json",
                {"retained_record_count": 6},
            )
            _write_json(
                root,
                "evidence_index.json",
                {
                    "ev-memory": {
                        "evidence_id": "ev-memory",
                        "source_kind": "long_term_memory",
                        "source_time": "2026-03-12",
                        "segment_id": "seg-1",
                        "start_offset": 10,
                        "end_offset": 20,
                        "task_id": "memory",
                        "memory_id": "mem-1",
                    }
                },
            )
            _write_json(root, "coverage_ledger.json", {"coverage_complete": True})
            _write_json(
                root,
                "evaluation_metrics.json",
                {
                    "core_metrics": {
                        "current_must_fact_recall": {
                            "numerator": 8,
                            "denominator": 10,
                            "value": 0.8,
                        },
                        "minutes_fact_f1": {"value": 0.9},
                    },
                    "history_metrics": {"historical_fact_f1": {"value": 0.8}},
                    "transition_metrics": {"state_transition_f1": {"value": 0.7}},
                },
            )
            _write_json(
                root,
                "review_run_result.json",
                {
                    "enabled": True,
                    "shadow_mode": True,
                    "termination": "accepted",
                    "recommended_candidate_index": 1,
                    "rounds": [
                        {
                            "round_index": 1,
                            "red_report": {
                                "candidate_issue_count": 3,
                                "issues": [{"issue_id": "i1"}],
                                "deferred_issues": [{"issue_id": "i2"}],
                                "issue_dispositions": [
                                    {"disposition": "kept"},
                                    {"disposition": "dismissed"},
                                ],
                            },
                            "blue_result": {
                                "repairs": [{"issue_id": "i1", "applied": True}]
                            },
                            "judge_decision": {
                                "disposition": "accept",
                                "score": 0.92,
                                "remaining_issue_ids": [],
                            },
                        }
                    ],
                },
            )
            (root / "summary_multi_agent.md").write_text("summary", encoding="utf-8")
            trace_lines = [
                {"timestamp": "2026-09-07T01:00:00+00:00", "event_type": "pipeline.started", "run_id": "run-vis", "payload": {}},
                {"timestamp": "2026-09-07T01:00:01+00:00", "event_type": "plan.created", "run_id": "run-vis", "payload": {}},
                {"timestamp": "2026-09-07T01:00:02+00:00", "event_type": "task.started", "run_id": "run-vis", "payload": {"task_id": "memory"}},
                {"timestamp": "2026-09-07T01:00:04+00:00", "event_type": "task.finished", "run_id": "run-vis", "payload": {"task_id": "memory", "status": "success"}},
                {"timestamp": "2026-09-07T01:00:06+00:00", "event_type": "pipeline.finished", "run_id": "run-vis", "payload": {"status": "completed_with_warnings"}},
            ]
            trace_path = root / "execution_trace.jsonl"
            trace_path.write_text(
                "\n".join(json.dumps(item) for item in trace_lines) + "\n{broken",
                encoding="utf-8",
            )
            before = {
                path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                for path in root.iterdir()
                if path.is_file()
            }

            view = build_run_visualization(root)

            after = {
                path.name: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                for path in root.iterdir()
                if path.is_file()
            }
            self.assertEqual(before, after)
            self.assertEqual(view.run_type, "multi_agent_summarize")
            self.assertEqual(len(view.graph_nodes), 3)
            edge_by_target = {
                (item.source_task_id, item.destination_task_id): item.consumed
                for item in view.graph_edges
            }
            self.assertTrue(edge_by_target[("memory", "entity")])
            self.assertFalse(edge_by_target[("memory", "temporal")])
            self.assertTrue(next(item for item in view.graph_nodes if item.task_id == "memory").preserved)
            self.assertTrue(next(item for item in view.graph_nodes if item.task_id == "entity").replanned)
            self.assertEqual(view.overview["model_call_count"], 3)
            self.assertEqual(view.overview["tool_call_count"], 1)
            self.assertEqual([item.value for item in view.memory_flow], [13, 8, 5, 6, 1])
            self.assertEqual(len(view.timeline), 4)
            self.assertTrue(any("损坏" in item for item in view.warnings))

            graph_html = render_agent_graph_html(view)
            overview_html = render_overview_html(view)
            self.assertIn("stroke-dasharray", graph_html)
            self.assertIn("#087a55", graph_html)
            self.assertIn("foreignObject", graph_html)
            self.assertIn("preserveAspectRatio=\"xMidYMid meet\"", graph_html)
            self.assertIn("smw-node-description", graph_html)
            self.assertNotIn("<script>alert(1)</script>", graph_html)
            self.assertIn("&lt;script&gt;alert(1)&lt;", graph_html)
            self.assertIn("/script&gt;", graph_html)
            self.assertNotIn("<script>alert('warning')</script>", overview_html)
            self.assertNotIn("must not render", graph_html)
            self.assertIn("memory", [value for _, value in task_choices(view)])
            self.assertIn("typed_memory_search", render_task_detail_html(view, "memory"))
            self.assertIn("Worker · memory", render_timeline_html(view))
            self.assertIn("ev-memory", render_evidence_memory_html(view))
            review_html = render_review_evaluation_html(view)
            self.assertIn("minutes_fact_f1", review_html)
            self.assertIn("纪要事实 F1", review_html)
            self.assertIn("必要事实召回率与有用事实精确率的调和平均", review_html)
            self.assertIn("当前必要事实召回率", review_html)
            self.assertIn("8 / 10", review_html)
            self.assertIn("Red 保留", review_html)
            self.assertIn("accepted", review_html)

    def test_ingest_stage_flow_has_no_agent_dag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_json(
                root,
                "web_run_info.json",
                {
                    "run_id": "ingest-1",
                    "run_type": "ingest",
                    "status": "completed",
                    "stage": "completed",
                },
            )
            _write_json(root, "ingest_run_result.json", {"status": "completed"})
            _write_json(
                root,
                "ingest_stage_report.json",
                {"status": "completed", "stage": "verifying"},
            )
            for name in (
                "e2e_preprocess_info.json",
                "typed_memories.json",
                "typed_memory_dedup_report.json",
                "ingest_write_journal.json",
                "ingest_readback_report.json",
            ):
                _write_json(root, name, {})

            view = build_run_visualization(root)

            self.assertEqual([item.stage_id for item in view.stages], [
                "validation", "preprocessing", "extracting", "deduplicating", "writing", "verifying"
            ])
            self.assertFalse(view.graph_nodes)
            self.assertIn("不产生 Agent DAG", render_agent_graph_html(view))

    def test_comparison_renders_side_by_side_metrics_and_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_json(
                root,
                "web_run_info.json",
                {
                    "run_id": "compare-1",
                    "run_type": "evaluation_compare",
                    "status": "completed_with_warnings",
                    "stage": "completed_with_warnings",
                    "target_entities": ["周明远"],
                },
            )
            _write_json(
                root,
                "evaluation_compare_run_result.json",
                {
                    "status": "completed_with_warnings",
                    "summaries": [
                        {"summary_id": "a", "status": "completed"},
                        {"summary_id": "b", "status": "failed"},
                    ],
                    "warnings": [],
                },
            )
            _write_json(root, "evaluation_source_snapshot.json", {"snapshot_id": "snap"})
            _write_json(root, "evaluation_personalization_source_annotations.json", [])
            _write_json(
                root,
                "evaluation_comparison_metrics.json",
                {
                    "baseline_summary_id": "a",
                    "successful_summary_count": 1,
                    "failed_summary_count": 1,
                    "rows": [
                        {
                            "summary_id": "a",
                            "display_name": "纪要 A",
                            "status": "completed",
                            "metrics": {"minutes_fact_f1": 0.8, "target_content_f1": 0.7},
                            "scope_metrics": {
                                "all_sources": {
                                    "must_fact_recall": 0.9,
                                    "useful_fact_precision": 0.8,
                                },
                                "current_only": {
                                    "must_fact_recall": 0.7,
                                    "useful_fact_precision": 0.6,
                                },
                            },
                            "scope_metric_details": {
                                "all_sources": {
                                    "must_fact_recall": {
                                        "numerator": 9,
                                        "denominator": 10,
                                        "value": 0.9,
                                        "unit": "ratio",
                                    }
                                },
                                "current_only": {
                                    "must_fact_recall": {
                                        "numerator": 7,
                                        "denominator": 10,
                                        "value": 0.7,
                                        "unit": "ratio",
                                    }
                                },
                            },
                        },
                        {
                            "summary_id": "b",
                            "display_name": "<script>B</script>",
                            "status": "failed",
                            "metrics": {},
                        },
                    ],
                    "deltas_from_first_summary": {
                        "a": {"minutes_fact_f1": 0.0},
                        "b": {"minutes_fact_f1": None},
                    },
                    "scope_deltas_from_first_summary": {
                        "a": {
                            "all_sources": {"must_fact_recall": 0.0},
                            "current_only": {"must_fact_recall": 0.0},
                        }
                    },
                    "scope_problem_claims": {
                        "all_sources": {
                            "a": [],
                            "b": [
                                {
                                    "summary_fact_id": "S2",
                                    "claim": "无来源推断",
                                    "status": "unsupported",
                                    "reason": "源材料没有对应事实",
                                }
                            ],
                        },
                        "current_only": {
                            "a": [
                                {
                                    "summary_fact_id": "S1",
                                    "claim": "历史材料补充事实",
                                    "status": "unsupported",
                                    "reason": "当前范围无支持",
                                    "scope_interpretation": "supported_by_all_sources_not_current",
                                }
                            ]
                        },
                    },
                    "scope_content_differences": {
                        "all_sources": {
                            "b": {
                                "status": "completed",
                                "baseline_summary_id": "a",
                                "added_covered_count": 1,
                                "reduced_covered_count": 1,
                                "added_problem_count": 1,
                                "added_covered_units": [
                                    {
                                        "unit_id": "U2",
                                        "fact": "<script>历史补充事实</script>",
                                        "salience": "must_include",
                                        "source_scopes": ["historical"],
                                    }
                                ],
                                "reduced_covered_units": [
                                    {
                                        "unit_id": "U1",
                                        "fact": "当前关键事实",
                                        "salience": "must_include",
                                        "source_scopes": ["current"],
                                    }
                                ],
                                "added_problem_claims": [
                                    {
                                        "summary_fact_id": "S2",
                                        "claim": "无来源推断",
                                        "status": "unsupported",
                                    }
                                ],
                            }
                        },
                        "current_only": {
                            "b": {
                                "status": "unavailable",
                                "baseline_summary_id": "a",
                                "reason": "当前纪要范围评测未完成",
                            }
                        },
                    },
                    "qualitative_comparison": {
                        "all_sources": {
                            "status": "completed_with_warnings",
                            "unverified_note_count": 1,
                            "overview": "纪要 B 增加历史信息但当前范围评测未完成。",
                            "summaries": [
                                {
                                    "summary_id": "a",
                                    "strengths": [
                                        {
                                            "description": "覆盖当前重点",
                                            "source_id": "current",
                                            "source_quote": "当前关键事实",
                                            "summary_quote": "当前关键事实",
                                        }
                                    ],
                                    "limitations": [],
                                    "relative_to_baseline": "作为基线。",
                                },
                                {
                                    "summary_id": "b",
                                    "strengths": [],
                                    "limitations": [],
                                    "relative_to_baseline": "增加历史背景。",
                                },
                            ],
                        },
                        "current_only": {
                            "status": "failed",
                            "error": "当前范围直接对比未完成",
                        },
                    },
                    "verified_content_differences": {
                        "all_sources": {
                            "b": {
                                "status": "completed",
                                "confirmed_added_count": 1,
                                "confirmed_reduced_count": 0,
                                "not_confirmed_count": 1,
                                "uncertain_count": 0,
                                "unverified_count": 0,
                                "items": [
                                    {
                                        "unit_id": "U2",
                                        "fact": "<script>历史补充事实</script>",
                                        "verification_result": "confirmed_added",
                                        "baseline_quote": None,
                                        "candidate_quote": "历史补充事实",
                                        "reason": "仅候选纪要表达",
                                    }
                                ],
                            }
                        },
                        "current_only": {
                            "b": {
                                "status": "unavailable",
                                "reason": "当前纪要范围评测未完成",
                                "items": [],
                            }
                        },
                    },
                },
            )

            view = build_run_visualization(root)
            rendered = render_review_evaluation_html(view)

            self.assertEqual(view.run_type, "evaluation_compare")
            self.assertFalse(view.graph_nodes)
            self.assertIn("纪要 A", rendered)
            self.assertIn("+0.0000", rendered)
            self.assertIn("全材料整体指标", rendered)
            self.assertIn("当前材料整体指标", rendered)
            self.assertIn("9 / 10", rendered)
            self.assertIn("7 / 10", rendered)
            self.assertIn("当前材料未提供支持，全材料已支持", rendered)
            self.assertIn("等价重复只计一次", rendered)
            self.assertIn("历史有用事实精确率", rendered)
            self.assertIn("状态变化精确率", rendered)
            self.assertIn("陈旧状态错误率", rendered)
            self.assertIn("越低越好", rendered)
            self.assertIn('class="smw-comparison-section"', rendered)
            self.assertIn('class="smw-table smw-comparison-table"', rendered)
            self.assertIn('class="smw-comparison-value"', rendered)
            self.assertIn('class="smw-comparison-delta"', rendered)
            self.assertIn('class="smw-metric-guide-label"', rendered)
            self.assertIn('class="smw-metric-guide-description"', rendered)
            self.assertNotIn("<script>B</script>", rendered)
            self.assertIn("&lt;script&gt;B&lt;/script&gt;", rendered)
            self.assertIn("不自动推荐赢家", rendered)
            self.assertIn("LLM 直接定性对比", rendered)
            self.assertIn("覆盖当前重点", rendered)
            self.assertIn("部分定性观察因证据 Span 无效而被隔离", rendered)
            self.assertIn("事实覆盖差异（二次核验，仅供核查）", rendered)
            self.assertIn("确认新增 1 条", rendered)
            self.assertNotIn("新增覆盖 1 条", rendered)
            self.assertNotIn("减少覆盖 1 条", rendered)
            self.assertNotIn("新增问题 Claim 1 条", rendered)
            self.assertIn("历史补充事实", rendered)
            self.assertNotIn("<script>历史补充事实</script>", rendered)
            self.assertIn(
                "&lt;script&gt;历史补充事实&lt;/script&gt;", rendered
            )
            self.assertIn("当前纪要范围评测未完成", rendered)
            self.assertIn('<details class="smw-diagnostics">', rendered)
            self.assertIn("问题诊断（模型判定，仅供核查）", rendered)
            self.assertIn("相对基线新增", rendered)
            self.assertIn("可能存在模型误判", rendered)
            self.assertGreater(
                rendered.index("问题诊断（模型判定，仅供核查）"),
                rendered.index("目标导向指标"),
            )

    def test_missing_and_corrupt_optional_artifacts_do_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _write_json(
                root,
                "web_run_info.json",
                {
                    "run_id": "partial",
                    "run_type": "multi_agent_summarize",
                    "status": "executing",
                    "stage": "executing",
                },
            )
            (root / "plan.json").write_text("{not-json", encoding="utf-8")

            view = build_run_visualization(root)

            self.assertEqual(view.run_id, "partial")
            self.assertIn("plan.json", view.errors)
            self.assertEqual(
                next(item for item in view.stages if item.stage_id == "agent_execution").status,
                "running",
            )
            self.assertTrue(all(item.value is None for item in view.memory_flow))
            self.assertIn("N/A", render_evidence_memory_html(view))
            self.assertIn("运行状态", render_overview_html(view))

    def test_web_source_groups_observability_under_result_details(self) -> None:
        source = (AGENT_ROOT / "summary_multi_agent_web_demo.py").read_text(
            encoding="utf-8"
        )
        positions = [
            source.index(
                f'with gr.Tab("{label}")',
                source.index('with gr.Tab("处理详情")'),
            )
            for label in (
                "运行概况",
                "任务图",
                "时间线",
                "证据与记忆",
                "审核与评测",
                "原始审计",
            )
        ]
        self.assertEqual(sorted(positions), positions)
        self.assertLess(source.index('with gr.Tab("完整纪要")'),
                        source.index('with gr.Tab("处理详情")'))
        self.assertIn("agent_task_selector.change", source)
        self.assertIn("demo.load(refresh_runs", source)

    def test_visualization_css_and_dag_constrain_long_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            long_task_id = "retrieve_extremely_long_target_entity_historical_evidence"
            long_description = (
                "读取目标实体跨越多次会议的历史证据，并核对负责人、时间、数值和状态变化，"
                "这是一段刻意加长的中文任务描述，用于验证节点正文不会越过图形边界。"
            )
            _write_json(
                root,
                "web_run_info.json",
                {
                    "run_id": "long-text",
                    "run_type": "multi_agent_summarize",
                    "status": "completed",
                    "stage": "completed",
                },
            )
            _write_json(
                root,
                "plan.json",
                {
                    "plan_id": "plan-long",
                    "plan_version": 1,
                    "tasks": [
                        {
                            "task_id": long_task_id,
                            "task_type": "entity_personalization",
                            "description": long_description,
                            "dependencies": [],
                            "critical": True,
                            "allowed_tools": ["entity_query"],
                        }
                    ],
                },
            )

            rendered = render_agent_graph_html(build_run_visualization(root))

            self.assertIn(f'title="{long_task_id}"', rendered)
            self.assertIn(f'title="{long_description}"', rendered)
            self.assertIn('width="820"', rendered)
            self.assertIn('height="212"', rendered)
            self.assertIn("overflow-wrap:anywhere", WEB_VISUALIZATION_CSS)
            self.assertIn("-webkit-line-clamp:3", WEB_VISUALIZATION_CSS)
            self.assertIn("overflow:hidden", WEB_VISUALIZATION_CSS)
            self.assertIn("color:#182230!important", WEB_VISUALIZATION_CSS)
            self.assertIn("background:#fff!important", WEB_VISUALIZATION_CSS)

    def test_visualization_css_keeps_auditing_text_readable_under_theme_overrides(self) -> None:
        required_rules = (
            ".smw-muted{color:#465568!important",
            ".smw-table th,.smw-table td{",
            "color:#243244!important",
            ".smw-metric-description{color:#26374a!important",
            ".smw-metric-guide-label{",
            "color:#111827!important",
            ".smw-metric-guide-description{",
            "color:#26374a!important",
            ".smw-comparison-value{color:#111827!important",
            ".smw-comparison-delta{color:#334155!important",
            "opacity:1!important",
        )
        for rule in required_rules:
            self.assertIn(rule, WEB_VISUALIZATION_CSS)

        def contrast_ratio(foreground: str, background: str) -> float:
            def luminance(color: str) -> float:
                channels = [int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
                linear = [
                    channel / 12.92
                    if channel <= 0.04045
                    else ((channel + 0.055) / 1.055) ** 2.4
                    for channel in channels
                ]
                return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

            lighter, darker = sorted(
                (luminance(foreground), luminance(background)), reverse=True
            )
            return (lighter + 0.05) / (darker + 0.05)

        for foreground, background in (
            ("#26374a", "#f8fafc"),
            ("#465568", "#ffffff"),
            ("#111827", "#e5ebf1"),
            ("#542f00", "#fff4d6"),
            ("#721c1c", "#ffeded"),
        ):
            self.assertGreaterEqual(contrast_ratio(foreground, background), 4.5)


if __name__ == "__main__":
    unittest.main()
