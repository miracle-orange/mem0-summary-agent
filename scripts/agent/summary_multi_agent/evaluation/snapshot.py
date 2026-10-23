"""构建与待评测纪要无关、可在同一运行内复用的源事实快照。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..adapters.model import ModelClient
from ..raw_material import build_raw_material_bundle
from ..schemas import RawMaterialBundle, RunConfig, SummaryRequest
from .model_io import EvaluationModelIO
from .personalization import TargetPersonalizationEvaluator
from .schemas import (
    EvaluationSourceDocument,
    EvaluationSourceSnapshot,
    PersonalizationSourceAnnotation,
)
from .source_fact_builder import EvaluationSourceFactBuilder


def _stable_json(value: Any) -> str:
    """生成 Snapshot ID 使用的稳定 JSON。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _snapshot_id(payload: dict[str, Any]) -> str:
    """根据完整源事实契约生成稳定快照 ID。"""
    digest = hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()
    return f"evaluation_snapshot_{digest[:24]}"


@dataclass(frozen=True)
class EvaluationSourceSnapshotBundle:
    """持有可落盘快照和仅供本次计算长度的原始文本。"""

    snapshot: EvaluationSourceSnapshot
    current_material: str
    visible_history_text: str


class EvaluationSourceSnapshotBuilder:
    """独立于摘要构建当前/历史事实、关系和目标源标注。"""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        personalization_model_client: Optional[ModelClient] = None,
        config: RunConfig,
        max_model_calls: Optional[int] = None,
        personalization_max_model_calls: Optional[int] = None,
    ) -> None:
        """初始化源事实和目标源标注各自的有界模型 I/O。"""
        self._config = config
        self._model_io = EvaluationModelIO(
            model_client,
            max_model_calls=(
                max_model_calls
                if max_model_calls is not None
                else config.evaluation_max_model_calls
            ),
        )
        personalization_config = config.model_copy(
            update={
                "evaluation_personalization_max_model_calls": (
                    personalization_max_model_calls
                    if personalization_max_model_calls is not None
                    else config.evaluation_personalization_max_model_calls
                )
            }
        )
        self._personalization = TargetPersonalizationEvaluator(
            model_client=personalization_model_client or model_client,
            config=personalization_config,
        )
        self.source_builder: Optional[EvaluationSourceFactBuilder] = None

    @property
    def model_trace(self) -> tuple[dict[str, Any], ...]:
        """返回源事实阶段模型 trace。"""
        return tuple(self._model_io.traces)

    @property
    def personalization_model_trace(self) -> tuple[dict[str, Any], ...]:
        """返回目标源事实标注 trace。"""
        return self._personalization.model_trace

    @property
    def execution_stats(self) -> dict[str, int]:
        """返回 Snapshot 阶段模型与分段统计。"""
        builder = self.source_builder
        return {
            "model_call_count": self._model_io.model_call_count,
            "repair_call_count": self._model_io.repair_call_count,
            "evaluation_current_segment_count": (
                builder.current_segment_count if builder else 0
            ),
            "evaluation_history_batch_count": (
                builder.historical_batch_count if builder else 0
            ),
            "evaluation_adaptive_split_count": (
                builder.adaptive_split_count if builder else 0
            ),
            "evaluation_adaptive_split_child_count": (
                builder.adaptive_split_child_count if builder else 0
            ),
            "personalization_model_call_count": (
                self._personalization.execution_stats["model_call_count"]
            ),
            "personalization_repair_call_count": (
                self._personalization.execution_stats["repair_call_count"]
            ),
            "personalization_source_annotation_batch_count": (
                self._personalization.execution_stats[
                    "source_annotation_batch_count"
                ]
            ),
        }

    async def build_from_documents(
        self,
        *,
        summary_request: str,
        current_source: EvaluationSourceDocument,
        historical_sources: tuple[EvaluationSourceDocument, ...] = (),
        target_entities: tuple[str, ...] = (),
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> EvaluationSourceSnapshotBundle:
        """从显式原文构建横评 Snapshot，历史 Evidence 逐条 grounding。"""
        request = SummaryRequest(
            current_material=current_source.text,
            source_time=current_source.source_time,
            summary_request=summary_request,
            target_entities=[],
        )
        raw_material = build_raw_material_bundle(
            current_source.text,
            source_id=current_source.source_id,
            max_segment_chars=self._config.raw_segment_max_chars,
            overlap_ratio=self._config.raw_segment_overlap_ratio,
        )
        builder = EvaluationSourceFactBuilder(
            model_io=self._model_io,
            config=self._config,
        )
        self.source_builder = builder
        current_facts = await builder.build_current_facts(request, raw_material)
        historical_facts = await builder.build_raw_historical_facts(
            summary_request=summary_request,
            current_facts=current_facts,
            historical_sources=historical_sources,
        )
        if target_entities and status_callback is not None:
            status_callback("annotating_target_sources")
        return await self._finalize(
            summary_request=summary_request,
            current_source_id=current_source.source_id,
            current_source_time=current_source.source_time,
            current_material=current_source.text,
            current_material_sha256=raw_material.content_sha256,
            historical_source_sha256={
                item.source_id: hashlib.sha256(item.text.encode("utf-8")).hexdigest()
                for item in historical_sources
            },
            historical_source_times={
                item.source_id: item.source_time for item in historical_sources
            },
            visible_history_text="\n".join(item.text for item in historical_sources),
            source_facts=(*current_facts, *historical_facts),
            target_entities=target_entities,
        )

    async def build_from_visible_records(
        self,
        *,
        request: SummaryRequest,
        raw_material: RawMaterialBundle,
        historical_records: dict[str, dict[str, Any]],
        evidence_index: dict[str, object],
        visible_history_text: str,
        target_entities: tuple[str, ...],
    ) -> EvaluationSourceSnapshotBundle:
        """从在线 Summarize 实际可见记录构建 fresh Snapshot。"""
        builder = EvaluationSourceFactBuilder(
            model_io=self._model_io,
            config=self._config,
        )
        self.source_builder = builder
        current_facts = await builder.build_current_facts(request, raw_material)
        historical_facts = await builder.build_historical_facts(
            request,
            current_facts,
            historical_records,
            evidence_index,
        )
        history_hashes: dict[str, str] = {}
        history_times: dict[str, str] = {}
        for record_id, record in historical_records.items():
            history_hashes[record_id] = hashlib.sha256(
                _stable_json(record).encode("utf-8")
            ).hexdigest()
            metadata = record.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            history_times[record_id] = str(metadata.get("source_time") or "unknown")
        return await self._finalize(
            summary_request=request.summary_request,
            current_source_id=raw_material.source_id,
            current_source_time=request.source_time or "unknown",
            current_material=request.current_material,
            current_material_sha256=raw_material.content_sha256,
            historical_source_sha256=history_hashes,
            historical_source_times=history_times,
            visible_history_text=visible_history_text,
            source_facts=(*current_facts, *historical_facts),
            target_entities=target_entities,
        )

    async def _finalize(
        self,
        *,
        summary_request: str,
        current_source_id: str,
        current_source_time: str,
        current_material: str,
        current_material_sha256: str,
        historical_source_sha256: dict[str, str],
        historical_source_times: dict[str, str],
        visible_history_text: str,
        source_facts: tuple[Any, ...],
        target_entities: tuple[str, ...],
    ) -> EvaluationSourceSnapshotBundle:
        """完成目标源标注并生成可复用的不可变 Snapshot。"""
        builder = self.source_builder
        assert builder is not None
        relations = builder.relations
        source_annotations: tuple[PersonalizationSourceAnnotation, ...] = ()
        snapshot_warnings = list(builder.warnings)
        if target_entities:
            try:
                source_annotations = await self._personalization.annotate_sources(
                    target_entities=target_entities,
                    source_facts=source_facts,
                )
            except Exception as exc:
                snapshot_warnings.append(
                    "personalization source annotation failed: "
                    f"{type(exc).__name__}: {exc}"
                )
        payload = {
            "current_source_id": current_source_id,
            "current_material_sha256": current_material_sha256,
            "historical_source_sha256": historical_source_sha256,
            "summary_request": summary_request,
            "source_facts": [item.model_dump(mode="json") for item in source_facts],
            "fact_relations": [item.model_dump(mode="json") for item in relations],
            "target_entities": list(target_entities),
            "target_source_annotations": [
                item.model_dump(mode="json") for item in source_annotations
            ],
        }
        snapshot = EvaluationSourceSnapshot(
            snapshot_id=_snapshot_id(payload),
            current_source_id=current_source_id,
            current_source_time=current_source_time,
            current_material_sha256=current_material_sha256,
            historical_source_sha256=historical_source_sha256,
            historical_source_times=historical_source_times,
            summary_request=summary_request,
            source_facts=source_facts,
            fact_relations=relations,
            target_entities=target_entities,
            target_source_annotations=source_annotations,
            config={
                "mode": "fresh",
                "segment_max_chars": self._config.evaluation_segment_max_chars,
                "source_fact_max_tokens": (
                    self._config.evaluation_source_fact_max_tokens
                ),
                "history_batch_size": self._config.evaluation_history_batch_size,
                "adaptive_split_enabled": (
                    self._config.evaluation_adaptive_split_enabled
                ),
                "adaptive_split_min_chars": (
                    self._config.evaluation_adaptive_split_min_chars
                ),
                "adaptive_split_max_depth": (
                    self._config.evaluation_adaptive_split_max_depth
                ),
            },
            execution_stats=self.execution_stats,
            warnings=tuple(dict.fromkeys(snapshot_warnings)),
        )
        return EvaluationSourceSnapshotBundle(
            snapshot=snapshot,
            current_material=current_material,
            visible_history_text=visible_history_text,
        )
