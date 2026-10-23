"""精简评测共享 Runner，以及生成后 candidate 0 的在线适配器。"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from ...adapters.model import ModelClient
from ...model_input_audit import _TokenCounter
from ...schemas import LiteEvaluationConfig, RunConfig, SummaryWorkingState
from ..evaluator import _visible_historical_records
from ..schemas import EvaluationCompareRequest
from .model_io import LiteEvaluationModelIO
from .prompts import (
    build_decompose_prompt,
    build_match_prompt,
    build_reference_prompt,
    build_verify_prompt,
)
from .schemas import (
    LiteAtomicClaim,
    LiteCandidateInput,
    LiteCandidateMetrics,
    LiteCandidateResult,
    LiteClaimVerification,
    LiteDecomposeResponse,
    LiteEvaluationResult,
    LiteEvidenceSource,
    LiteExtractResponse,
    LiteFactMatch,
    LiteMatchResponse,
    LiteMetricValue,
    LiteReferenceBundle,
    LiteReferenceFact,
    LiteVerifyResponse,
)

StatusCallback = Callable[[str], None] | None


def _ratio(numerator: int, denominator: int) -> LiteMetricValue:
    return LiteMetricValue(
        numerator=numerator,
        denominator=denominator,
        value=(numerator / denominator if denominator else None),
    )


def _natural_segments(
    text: str, *, max_chars: int, overlap_chars: int
) -> tuple[tuple[str, str], ...]:
    """按自然边界优先切分，并保留固定重叠供跨边界事实抽取。"""
    segments: list[tuple[str, str]] = []
    start = 0
    ordinal = 0
    while start < len(text):
        hard_end = min(start + max_chars, len(text))
        end = hard_end
        if hard_end < len(text):
            search_start = start + max(max_chars // 2, 1)
            best = -1
            for marker in ("\n", "。", "！", "？", ";", "；", ". "):
                position = text.rfind(marker, search_start, hard_end)
                if position >= 0:
                    best = max(best, position + len(marker))
            if best > start:
                end = best
        segment_text = text[start:end]
        segment_id = f"segment_{ordinal:04d}_{start}_{end}"
        segments.append((segment_id, segment_text))
        if end >= len(text):
            break
        next_start = end - overlap_chars
        if next_start <= start:
            next_start = end
        start = next_start
        ordinal += 1
    return tuple(segments)


def _validate_match(
    response: LiteMatchResponse, *, reference_count: int, claim_count: int
) -> None:
    if len(response.matches) != reference_count:
        raise ValueError(f"expected {reference_count} lite matches")
    for index, item in enumerate(response.matches):
        if item.fact_idx != index:
            raise ValueError("lite matches must be complete, ordered, and zero-based")
        if item.align_unit_idx is not None and item.align_unit_idx >= claim_count:
            raise ValueError("lite match references an unknown candidate claim")


def _validate_verifications(response: LiteVerifyResponse, *, claim_count: int) -> None:
    if len(response.verifications) != claim_count:
        raise ValueError(f"expected {claim_count} lite verifications")
    for index, item in enumerate(response.verifications):
        if item.fact_idx != index:
            raise ValueError(
                "lite verifications must be complete, ordered, and zero-based"
            )


class LiteEvaluationRunner:
    """执行共享参考事实和每候选三阶段精简评测。"""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        config: LiteEvaluationConfig,
        trace_path: Path | None = None,
    ) -> None:
        self._config = config
        self._io = LiteEvaluationModelIO(model_client, config, trace_path=trace_path)
        self.last_stage = "not_started"
        self.last_reference: LiteReferenceBundle | None = None
        self.started_at = 0.0

    @property
    def model_trace(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._io.traces)

    @property
    def execution_stats(self) -> dict[str, int]:
        return {
            "model_call_count": self._io.model_call_count,
            "repair_call_count": self._io.repair_call_count,
            "cache_hit_count": self._io.cache_hit_count,
        }

    async def run(
        self,
        *,
        current_source: LiteEvidenceSource,
        historical_sources: tuple[LiteEvidenceSource, ...],
        summary_request: str,
        candidates: tuple[LiteCandidateInput, ...],
        status_callback: StatusCallback = None,
    ) -> LiteEvaluationResult:
        if current_source.source_kind != "current":
            raise ValueError(
                "lite evaluation current source must use source_kind=current"
            )
        if not candidates:
            raise ValueError("lite evaluation requires at least one candidate")
        candidate_ids = [item.summary_id for item in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("lite evaluation candidate IDs must be unique")
        self.started_at = time.monotonic()
        return await asyncio.wait_for(
            self._run_impl(
                current_source=current_source,
                historical_sources=historical_sources,
                summary_request=summary_request,
                candidates=candidates,
                status_callback=status_callback,
            ),
            timeout=self._config.timeout_seconds,
        )

    async def _run_impl(
        self,
        *,
        current_source: LiteEvidenceSource,
        historical_sources: tuple[LiteEvidenceSource, ...],
        summary_request: str,
        candidates: tuple[LiteCandidateInput, ...],
        status_callback: StatusCallback,
    ) -> LiteEvaluationResult:
        def update(stage: str) -> None:
            self.last_stage = stage
            if status_callback is not None:
                status_callback(stage)

        update("lite_reference_facts")
        reference = await self._build_reference(
            current_source=current_source,
            summary_request=summary_request,
        )
        self.last_reference = reference
        update("lite_scoring")
        evidence = (current_source, *historical_sources)
        candidate_results = await asyncio.gather(
            *(
                self._evaluate_candidate(
                    candidate=candidate,
                    reference=reference,
                    evidence_sources=evidence,
                )
                for candidate in candidates
            )
        )
        update("lite_aggregating")
        warnings = tuple(
            warning for result in candidate_results for warning in result.warnings
        )
        success_count = sum(item.status != "failed" for item in candidate_results)
        status = (
            "failed"
            if success_count == 0
            else (
                "completed_with_warnings"
                if warnings or success_count != len(candidate_results)
                else "completed"
            )
        )
        baseline = candidate_results[0]
        deltas = self._deltas(baseline, tuple(candidate_results))
        elapsed_ms = int((time.monotonic() - self.started_at) * 1000)
        stats = {
            **self.execution_stats,
            "reference_segment_count": reference.segment_count,
            "candidate_count": len(candidates),
            "successful_candidate_count": success_count,
            "failed_candidate_count": len(candidates) - success_count,
            "elapsed_ms": elapsed_ms,
        }
        return LiteEvaluationResult(
            status=status,
            reference=reference,
            candidates=tuple(candidate_results),
            baseline_summary_id=baseline.summary_id,
            deltas_from_baseline=deltas,
            execution_stats=stats,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    async def _build_reference(
        self,
        *,
        current_source: LiteEvidenceSource,
        summary_request: str,
    ) -> LiteReferenceBundle:
        fits_single = False
        if self._config.reference_chunking == "auto" and len(current_source.text) <= 24000:
            candidate_prompt, _ = build_reference_prompt(
                segment_text=current_source.text,
                source_time=current_source.source_time,
                summary_request=summary_request,
            )
            window = self._config.model_context_window_tokens
            fits_single = window is None or (
                _TokenCounter().measure(candidate_prompt)["tokens"]
                + self._config.reference_max_tokens
                + self._config.context_safety_margin_tokens <= window
            )
        if fits_single:
            segments = (("segment_0001", current_source.text),)
        else:
            segments = _natural_segments(
                current_source.text,
                max_chars=self._config.segment_max_chars,
                overlap_chars=self._config.segment_overlap_chars,
            )

        async def extract(
            segment_id: str, text: str
        ) -> tuple[str, LiteExtractResponse]:
            prompt, sections = build_reference_prompt(
                segment_text=text,
                source_time=current_source.source_time,
                summary_request=summary_request,
            )
            response = await self._io.complete_json(
                stage="lite_reference_extract",
                prompt=prompt,
                sections=sections,
                response_model=LiteExtractResponse,
                max_tokens=self._config.reference_max_tokens,
                metadata={"segment_id": segment_id},
            )
            return segment_id, response

        extracted = await asyncio.gather(
            *(extract(segment_id, text) for segment_id, text in segments)
        )
        deduplicated: OrderedDict[str, list[str]] = OrderedDict()
        for segment_id, response in extracted:
            for raw_fact in response.key_facts:
                fact = raw_fact.strip()
                if not fact:
                    continue
                deduplicated.setdefault(fact, []).append(segment_id)
        if not deduplicated:
            raise ValueError("lite reference extraction returned no key facts")
        facts = tuple(
            LiteReferenceFact(
                fact_id=f"K{index:04d}",
                text=fact,
                source_segment_ids=tuple(dict.fromkeys(segment_ids)),
            )
            for index, (fact, segment_ids) in enumerate(deduplicated.items(), start=1)
        )
        return LiteReferenceBundle(
            current_source_id=current_source.source_id,
            current_source_time=current_source.source_time,
            summary_request=summary_request,
            current_material_sha256=hashlib.sha256(
                current_source.text.encode("utf-8")
            ).hexdigest(),
            facts=facts,
            segment_count=len(segments),
        )

    async def _evaluate_candidate(
        self,
        *,
        candidate: LiteCandidateInput,
        reference: LiteReferenceBundle,
        evidence_sources: tuple[LiteEvidenceSource, ...],
    ) -> LiteCandidateResult:
        summary_hash = hashlib.sha256(candidate.text.encode("utf-8")).hexdigest()
        support_scope = self._support_scope(evidence_sources)
        prompt, sections = build_decompose_prompt(candidate.text)
        try:
            decomposition = await self._io.complete_json(
                stage="lite_candidate_decompose",
                prompt=prompt,
                sections=sections,
                response_model=LiteDecomposeResponse,
                max_tokens=self._config.claim_max_tokens,
                metadata={"summary_id": candidate.summary_id},
            )
        except Exception as exc:  # noqa: BLE001 - isolate one candidate failure
            return LiteCandidateResult(
                summary_id=candidate.summary_id,
                display_name=candidate.display_name,
                status="failed",
                summary_sha256=summary_hash,
                support_scope=support_scope,
                error=f"{type(exc).__name__}: {exc}",
                warnings=(
                    f"candidate decomposition failed: {type(exc).__name__}: {exc}",
                ),
            )
        claims = tuple(
            LiteAtomicClaim(claim_id=f"C{index:04d}", text=text.strip())
            for index, text in enumerate(
                dict.fromkeys(
                    item.strip() for item in decomposition.atomic_facts if item.strip()
                ),
                start=1,
            )
        )
        if not claims:
            metrics = LiteCandidateMetrics(
                current_key_fact_coverage=_ratio(0, len(reference.facts)),
                material_support_rate=_ratio(0, 0),
                key_fact_relevance_rate=_ratio(0, 0),
            )
            return LiteCandidateResult(
                summary_id=candidate.summary_id,
                display_name=candidate.display_name,
                status="completed_with_warnings",
                summary_sha256=summary_hash,
                support_scope=support_scope,
                metrics=metrics,
                missing_reference_fact_ids=tuple(
                    item.fact_id for item in reference.facts
                ),
                warnings=("candidate decomposition returned no factual claims",),
            )

        match_task = asyncio.create_task(
            self._match(reference.facts, claims, candidate.summary_id)
        )
        verify_task = asyncio.create_task(
            self._verify(evidence_sources, claims, candidate.summary_id)
        )
        match_value, verify_value = await asyncio.gather(
            match_task, verify_task, return_exceptions=True
        )
        for value in (match_value, verify_value):
            if isinstance(value, asyncio.CancelledError):
                raise value
        warnings: list[str] = []
        matches: tuple[LiteFactMatch, ...] = ()
        verifications: tuple[LiteClaimVerification, ...] = ()
        if isinstance(match_value, BaseException):
            warnings.append(
                f"coverage matching failed: {type(match_value).__name__}: {match_value}"
            )
        else:
            matches = match_value
        if isinstance(verify_value, BaseException):
            warnings.append(
                f"support verification failed: {type(verify_value).__name__}: {verify_value}"
            )
        else:
            verifications = verify_value

        matched_reference_ids = {
            item.fact_id for item in matches if item.claim_id is not None
        }
        aligned_claim_ids = {
            item.claim_id for item in matches if item.claim_id is not None
        }
        supported_claim_ids = {
            item.claim_id for item in verifications if item.conclusion == "supported"
        }
        metrics = LiteCandidateMetrics(
            current_key_fact_coverage=(
                _ratio(len(matched_reference_ids), len(reference.facts))
                if not isinstance(match_value, BaseException)
                else _ratio(0, 0)
            ),
            material_support_rate=(
                _ratio(len(supported_claim_ids), len(claims))
                if not isinstance(verify_value, BaseException)
                else _ratio(0, 0)
            ),
            key_fact_relevance_rate=(
                _ratio(len(aligned_claim_ids), len(claims))
                if not isinstance(match_value, BaseException)
                else _ratio(0, 0)
            ),
        )
        both_scoring_stages_failed = isinstance(
            match_value, BaseException
        ) and isinstance(verify_value, BaseException)
        return LiteCandidateResult(
            summary_id=candidate.summary_id,
            display_name=candidate.display_name,
            status=(
                "failed"
                if both_scoring_stages_failed
                else "completed_with_warnings"
                if warnings
                else "completed"
            ),
            summary_sha256=summary_hash,
            support_scope=support_scope,
            claims=claims,
            matches=matches,
            verifications=verifications,
            metrics=metrics,
            missing_reference_fact_ids=tuple(
                item.fact_id
                for item in reference.facts
                if item.fact_id not in matched_reference_ids
            )
            if not isinstance(match_value, BaseException)
            else (),
            unsupported_claim_ids=tuple(
                item.claim_id
                for item in claims
                if item.claim_id not in supported_claim_ids
            )
            if not isinstance(verify_value, BaseException)
            else (),
            warnings=tuple(warnings),
            error=("; ".join(warnings) if both_scoring_stages_failed else None),
        )

    async def _match(
        self,
        reference_facts: tuple[LiteReferenceFact, ...],
        claims: tuple[LiteAtomicClaim, ...],
        summary_id: str,
    ) -> tuple[LiteFactMatch, ...]:
        max_rows = max(1, (self._config.match_max_tokens - 128) // 32)
        combined: list[LiteFactMatch] = []
        for start in range(0, len(reference_facts), max_rows):
            batch = reference_facts[start : start + max_rows]
            batch_size = len(batch)
            claim_count = len(claims)
            prompt, sections = build_match_prompt(batch, claims)
            response = await self._io.complete_json(
                stage="lite_coverage_match",
                prompt=prompt,
                sections=sections,
                response_model=LiteMatchResponse,
                max_tokens=self._config.match_max_tokens,
                validator=partial(
                    _validate_match,
                    reference_count=batch_size,
                    claim_count=claim_count,
                ),
                metadata={"summary_id": summary_id, "batch_start": start},
            )
            for raw in response.matches:
                reference = batch[raw.fact_idx]
                claim = (
                    claims[raw.align_unit_idx]
                    if raw.align_unit_idx is not None
                    else None
                )
                combined.append(
                    LiteFactMatch(
                        fact_idx=start + raw.fact_idx,
                        align_unit_idx=raw.align_unit_idx,
                        fact_id=reference.fact_id,
                        claim_id=claim.claim_id if claim is not None else None,
                    )
                )
        return tuple(combined)

    async def _verify(
        self,
        evidence_sources: tuple[LiteEvidenceSource, ...],
        claims: tuple[LiteAtomicClaim, ...],
        summary_id: str,
    ) -> tuple[LiteClaimVerification, ...]:
        max_rows = max(1, (self._config.verification_max_tokens - 128) // 24)
        combined: list[LiteClaimVerification] = []
        for start in range(0, len(claims), max_rows):
            batch = claims[start : start + max_rows]
            batch_size = len(batch)
            prompt, sections = build_verify_prompt(evidence_sources, batch)
            response = await self._io.complete_json(
                stage="lite_support_verify",
                prompt=prompt,
                sections=sections,
                response_model=LiteVerifyResponse,
                max_tokens=self._config.verification_max_tokens,
                validator=partial(
                    _validate_verifications,
                    claim_count=batch_size,
                ),
                metadata={"summary_id": summary_id, "batch_start": start},
            )
            for raw in response.verifications:
                claim = batch[raw.fact_idx]
                combined.append(
                    LiteClaimVerification(
                        fact_idx=start + raw.fact_idx,
                        conclusion=raw.conclusion,
                        claim_id=claim.claim_id,
                    )
                )
        return tuple(combined)

    @staticmethod
    def _support_scope(evidence_sources: tuple[LiteEvidenceSource, ...]) -> str:
        kinds = {item.source_kind for item in evidence_sources}
        if "historical_memory" in kinds:
            return "current_plus_visible_memory"
        if "historical_raw" in kinds:
            return "current_plus_explicit_history"
        return "current_only"

    @staticmethod
    def _deltas(
        baseline: LiteCandidateResult,
        candidates: tuple[LiteCandidateResult, ...],
    ) -> dict[str, dict[str, float | None]]:
        names = (
            "current_key_fact_coverage",
            "material_support_rate",
            "key_fact_relevance_rate",
        )
        result: dict[str, dict[str, float | None]] = {}
        for candidate in candidates:
            values: dict[str, float | None] = {}
            for name in names:
                base_metric = (
                    getattr(baseline.metrics, name) if baseline.metrics else None
                )
                candidate_metric = (
                    getattr(candidate.metrics, name) if candidate.metrics else None
                )
                values[name] = (
                    candidate_metric.value - base_metric.value
                    if (
                        candidate_metric is not None
                        and base_metric is not None
                        and candidate_metric.value is not None
                        and base_metric.value is not None
                    )
                    else None
                )
            result[candidate.summary_id] = values
        return result


def _memory_sources(context: Any) -> tuple[LiteEvidenceSource, ...]:
    records = _visible_historical_records(context)
    grouped: OrderedDict[tuple[str | None, str], list[str]] = OrderedDict()
    for record_id, record in sorted(records.items()):
        content = str(record.get("content") or "").strip()
        variants = record.get("content_variants")
        if isinstance(variants, list):
            values = [str(item).strip() for item in variants if str(item).strip()]
            if values:
                content = " / ".join(values)
        if not content:
            continue
        metadata = (
            record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        )
        source_time = (
            str(
                record.get("source_time")
                or metadata.get("source_time")
                or metadata.get("source_date")
                or ""
            ).strip()
            or None
        )
        grouped.setdefault((source_time, content), []).append(record_id)
    return tuple(
        LiteEvidenceSource(
            source_id=record_ids[0],
            label="历史记忆记录 " + ", ".join(record_ids),
            source_time=source_time,
            source_kind="historical_memory",
            text=content,
        )
        for (source_time, content), record_ids in grouped.items()
    )


class LiteSummaryEvaluator:
    """将生成后正式 candidate 0 投影为共享 lite Runner 的输入。"""

    def __init__(
        self,
        *,
        model_client: ModelClient,
        config: RunConfig,
        output_dir: Path | None = None,
    ) -> None:
        self._config = config
        lite_config = config.lite_evaluation
        if lite_config.model_context_window_tokens is None:
            lite_config = lite_config.model_copy(
                update={
                    "model_context_window_tokens": (config.model_context_window_tokens),
                    "context_safety_margin_tokens": (
                        config.model_context_safety_margin_tokens
                    ),
                }
            )
        self._runner = LiteEvaluationRunner(
            model_client=model_client,
            config=lite_config,
            trace_path=(
                output_dir / "evaluation_lite_model_trace.json"
                if output_dir is not None
                else None
            ),
        )

    @property
    def model_trace(self) -> tuple[dict[str, Any], ...]:
        return self._runner.model_trace

    @property
    def execution_stats(self) -> dict[str, int]:
        return self._runner.execution_stats

    @property
    def last_stage(self) -> str:
        return self._runner.last_stage

    @property
    def reference(self) -> LiteReferenceBundle | None:
        return self._runner.last_reference

    async def evaluate(
        self, state: SummaryWorkingState, final_summary: str
    ) -> LiteEvaluationResult:
        if state.synthesis_context is None:
            raise ValueError("lite evaluation requires an existing SynthesisContext")
        current = LiteEvidenceSource(
            source_id=state.raw_material.source_id,
            label="当前材料",
            source_time=state.request.source_time,
            source_kind="current",
            text=state.request.current_material,
        )
        return await self._runner.run(
            current_source=current,
            historical_sources=_memory_sources(state.synthesis_context),
            summary_request=state.request.summary_request,
            candidates=(
                LiteCandidateInput(
                    summary_id="candidate_0",
                    display_name="正式纪要 candidate 0",
                    text=final_summary,
                ),
            ),
        )


class LiteEvaluationComparisonRunner:
    """把独立横评请求适配到共享的精简评测 Runner。"""

    def __init__(
        self, *, model_client: ModelClient, output_dir: Path | None = None
    ) -> None:
        self._model_client = model_client
        self._output_dir = output_dir
        self._runner: LiteEvaluationRunner | None = None
        self.last_stage = "not_started"

    @property
    def model_trace(self) -> tuple[dict[str, Any], ...]:
        return self._runner.model_trace if self._runner is not None else ()

    @property
    def execution_stats(self) -> dict[str, int]:
        return self._runner.execution_stats if self._runner is not None else {}

    @property
    def reference(self) -> LiteReferenceBundle | None:
        return self._runner.last_reference if self._runner is not None else None

    async def run(
        self,
        request: EvaluationCompareRequest,
        *,
        status_callback: StatusCallback = None,
    ) -> LiteEvaluationResult:
        self._runner = LiteEvaluationRunner(
            model_client=self._model_client,
            config=request.evaluation_config.lite_evaluation,
            trace_path=(
                self._output_dir / "evaluation_lite_model_trace.json"
                if self._output_dir is not None
                else None
            ),
        )

        def update(stage: str) -> None:
            self.last_stage = stage
            if status_callback is not None:
                status_callback(stage)

        current = LiteEvidenceSource(
            source_id=request.current_source.source_id,
            label=request.current_source.label,
            source_time=request.current_source.source_time,
            source_kind="current",
            text=request.current_source.text,
        )
        history = tuple(
            LiteEvidenceSource(
                source_id=item.source_id,
                label=item.label,
                source_time=item.source_time,
                source_kind="historical_raw",
                text=item.text,
            )
            for item in request.historical_sources
        )
        candidates = tuple(
            LiteCandidateInput(
                summary_id=item.summary_id,
                display_name=item.display_name,
                text=item.text,
            )
            for item in request.summaries
        )
        return await self._runner.run(
            current_source=current,
            historical_sources=history,
            summary_request=request.summary_request,
            candidates=candidates,
            status_callback=update,
        )
