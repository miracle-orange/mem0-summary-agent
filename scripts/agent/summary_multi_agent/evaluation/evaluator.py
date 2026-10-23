"""组织正式摘要后的独立事实级评测。"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ..adapters.model import ModelClient
from ..schemas import RunConfig, SummaryWorkingState, TaskType
from .model_io import SharedEvaluationModelClient
from .reporting import build_evaluation_report
from .scorer import EvaluationSummaryScorer
from .schemas import EvaluationArtifacts
from .snapshot import EvaluationSourceSnapshotBuilder


def _stable_json(value: Any) -> str:
    """用稳定键序列化可见历史记录，供长度统计和报告 fallback。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _visible_history_text(records: dict[str, dict[str, Any]]) -> str:
    """将本轮可见历史记录投影为评测上下文长度统计文本。"""
    values: list[str] = []
    for record_id, record in sorted(records.items()):
        content = str(record.get("content") or "").strip()
        variants = record.get("content_variants")
        if isinstance(variants, list):
            variant_text = " / ".join(
                str(item).strip() for item in variants if str(item).strip()
            )
            content = variant_text or content
        values.append(f"{record_id}: {content or _stable_json(record)}")
    return "\n".join(values)


def _visible_historical_records(context: Any) -> dict[str, dict[str, Any]]:
    """返回 Synthesis 实际可见的 canonical 或 passthrough 历史记录。"""

    records = {
        str(record_id): dict(record)
        for record_id, record in context.historical_records.items()
        if isinstance(record, dict)
    }
    for task_id, result in context.worker_results.items():
        if not isinstance(result, dict):
            continue
        payload = result.get("payload")
        if not isinstance(payload, dict):
            continue
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            continue
        evidence_ids = tuple(
            str(item)
            for item in result.get("evidence_ids", [])
            if str(item).strip()
        )
        for index, record in enumerate(raw_records):
            if not isinstance(record, dict):
                continue
            record_id = str(
                record.get("memory_id")
                or record.get("record_ref_id")
                or f"{task_id}:record:{index}"
            )
            projected = dict(record)
            projected.setdefault("record_ref_id", record_id)
            projected.setdefault("source_task_ids", [task_id])
            projected.setdefault("evidence_ids", list(evidence_ids))
            records.setdefault(record_id, projected)
    return records


def _temporal_support(worker_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """提取成功 Temporal Worker 结果，作为匹配的授权日期辅助证据。"""
    return {
        task_id: result
        for task_id, result in worker_results.items()
        if result.get("task_type") == TaskType.TEMPORAL_REASONING.value
        and result.get("status") == "success"
    }


class SummaryEvaluator:
    """编排只读事实级 Evaluation Sidecar，不改变正式摘要或 M4。"""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        config: RunConfig,
    ) -> None:
        """初始化 fresh Snapshot Builder 和 candidate 0 Scorer。"""
        self._config = config
        self._shared_overall_client = SharedEvaluationModelClient(
            model_client,
            max_calls=config.evaluation_max_model_calls,
        )
        self._shared_personalization_client = SharedEvaluationModelClient(
            model_client,
            max_calls=config.evaluation_personalization_max_model_calls,
        )
        self._snapshot_builder = EvaluationSourceSnapshotBuilder(
            model_client=self._shared_overall_client,
            personalization_model_client=self._shared_personalization_client,
            config=config,
        )
        self._scorer = EvaluationSummaryScorer(
            model_client=self._shared_overall_client,
            personalization_model_client=self._shared_personalization_client,
            config=config,
        )
        self.source_snapshot = None

    @property
    def model_trace(self) -> tuple[dict[str, Any], ...]:
        """返回评测模型调用与修复 trace。"""
        return (
            *self._snapshot_builder.model_trace,
            *self._scorer.model_trace,
        )

    @property
    def execution_stats(self) -> dict[str, int]:
        """返回当前 Sidecar 的模型调用和 repair 统计。"""
        source = self._snapshot_builder.execution_stats
        score = self._scorer.execution_stats
        return {
            "model_call_count": self._shared_overall_client.model_call_count,
            "repair_call_count": (
                source.get("repair_call_count", 0)
                + score.get("repair_call_count", 0)
            ),
        }

    @property
    def personalization_model_trace(self) -> tuple[dict[str, Any], ...]:
        """返回个性化分支自己的模型调用 trace。"""
        return (
            *self._snapshot_builder.personalization_model_trace,
            *self._scorer.personalization_model_trace,
        )

    @property
    def personalization_execution_stats(self) -> dict[str, int]:
        """返回个性化分支的独立调用预算统计。"""
        source = self._snapshot_builder.execution_stats
        score = self._scorer.personalization_execution_stats
        return {
            "model_call_count": (
                self._shared_personalization_client.model_call_count
            ),
            "repair_call_count": (
                source.get("personalization_repair_call_count", 0)
                + score.get("repair_call_count", 0)
            ),
            "source_annotation_batch_count": source.get(
                "personalization_source_annotation_batch_count", 0
            ),
            "claim_annotation_batch_count": score.get(
                "claim_annotation_batch_count", 0
            ),
        }

    async def evaluate(
        self,
        state: SummaryWorkingState,
        final_summary: str,
    ) -> EvaluationArtifacts:
        """fresh 构建一次 Snapshot，并只评测正式 candidate 0。"""
        return await asyncio.wait_for(
            self._evaluate_fresh(state, final_summary),
            timeout=self._config.evaluation_timeout_seconds,
        )

    async def _evaluate_fresh(
        self,
        state: SummaryWorkingState,
        final_summary: str,
    ) -> EvaluationArtifacts:
        """运行在线 fresh Snapshot 与固定 Snapshot Scorer。"""
        target_entities = tuple(
            dict.fromkeys(
                item.strip() for item in state.request.target_entities if item.strip()
            )
        )
        context = state.synthesis_context
        if context is None:
            raise ValueError("evaluation requires an existing SynthesisContext")
        visible_historical_records = _visible_historical_records(context)
        snapshot_bundle = await self._snapshot_builder.build_from_visible_records(
            request=state.request,
            raw_material=state.raw_material,
            historical_records=visible_historical_records,
            evidence_index=context.evidence_index,
            visible_history_text=_visible_history_text(visible_historical_records),
            target_entities=target_entities,
        )
        self.source_snapshot = snapshot_bundle.snapshot
        result = await self._scorer.score(
            snapshot_bundle=snapshot_bundle,
            summary=final_summary,
            temporal_support=_temporal_support(context.worker_results),
            worker_results=context.worker_results,
        )
        score_stats = dict(result.metrics.execution_stats)
        source_stats = dict(snapshot_bundle.snapshot.execution_stats)
        combined_stats = {**source_stats, **score_stats}
        combined_stats["model_call_count"] = (
            self._shared_overall_client.model_call_count
        )
        combined_stats["repair_call_count"] = (
            source_stats.get("repair_call_count", 0)
            + score_stats.get("repair_call_count", 0)
        )
        result = result.model_copy(
            update={
                "metrics": result.metrics.model_copy(
                    update={"execution_stats": combined_stats}
                )
            }
        )
        personalization = result.personalization
        if personalization is not None:
            personalization = personalization.model_copy(
                update={"model_trace": self.personalization_model_trace}
            )
        return result.model_copy(
            update={
                "model_trace": self.model_trace,
                "personalization": personalization,
            }
        )
