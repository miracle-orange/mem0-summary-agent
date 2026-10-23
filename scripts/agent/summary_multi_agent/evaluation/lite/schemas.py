"""精简评测的版本化输入、判断结果和指标契约。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ...schemas import FrozenStrictModel


class LiteEvidenceSource(FrozenStrictModel):
    source_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    source_time: str | None = None
    source_kind: Literal["current", "historical_raw", "historical_memory"]
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_text(self) -> LiteEvidenceSource:
        if not self.text.strip():
            raise ValueError("lite evidence text must contain non-whitespace text")
        return self


class LiteCandidateInput(FrozenStrictModel):
    summary_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_text(self) -> LiteCandidateInput:
        if not self.text.strip():
            raise ValueError("lite candidate text must contain non-whitespace text")
        return self


class LiteReferenceFact(FrozenStrictModel):
    fact_id: str = Field(pattern=r"^K\d{4}$")
    text: str = Field(min_length=1)
    source_segment_ids: tuple[str, ...] = Field(min_length=1)


class LiteReferenceBundle(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.evaluation_lite_reference.v1"] = (
        "summary_multi_agent.evaluation_lite_reference.v1"
    )
    current_source_id: str = Field(min_length=1)
    current_source_time: str | None = None
    summary_request: str = Field(min_length=1)
    current_material_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    facts: tuple[LiteReferenceFact, ...] = Field(min_length=1)
    segment_count: int = Field(ge=1)
    extraction_complete: Literal[True] = True
    warnings: tuple[str, ...] = ()


class LiteAtomicClaim(FrozenStrictModel):
    claim_id: str = Field(pattern=r"^C\d{4}$")
    text: str = Field(min_length=1)


class LiteFactMatch(FrozenStrictModel):
    fact_idx: int = Field(ge=0)
    align_unit_idx: int | None = Field(default=None, ge=0)
    fact_id: str = Field(pattern=r"^K\d{4}$")
    claim_id: str | None = Field(default=None, pattern=r"^C\d{4}$")


class LiteClaimVerification(FrozenStrictModel):
    fact_idx: int = Field(ge=0)
    conclusion: Literal["supported", "not supported"]
    claim_id: str = Field(pattern=r"^C\d{4}$")


class LiteMetricValue(FrozenStrictModel):
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)
    value: float | None = Field(default=None, ge=0.0, le=1.0)


class LiteCandidateMetrics(FrozenStrictModel):
    current_key_fact_coverage: LiteMetricValue
    material_support_rate: LiteMetricValue
    key_fact_relevance_rate: LiteMetricValue


class LiteCandidateResult(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.evaluation_lite_candidate.v1"] = (
        "summary_multi_agent.evaluation_lite_candidate.v1"
    )
    summary_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    status: Literal["completed", "completed_with_warnings", "failed"]
    summary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    support_scope: Literal[
        "current_only",
        "current_plus_explicit_history",
        "current_plus_visible_memory",
    ]
    claims: tuple[LiteAtomicClaim, ...] = ()
    matches: tuple[LiteFactMatch, ...] = ()
    verifications: tuple[LiteClaimVerification, ...] = ()
    metrics: LiteCandidateMetrics | None = None
    missing_reference_fact_ids: tuple[str, ...] = ()
    unsupported_claim_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: str | None = None


class LiteEvaluationResult(FrozenStrictModel):
    schema_version: Literal["summary_multi_agent.evaluation_lite_result.v1"] = (
        "summary_multi_agent.evaluation_lite_result.v1"
    )
    status: Literal["completed", "completed_with_warnings", "failed"]
    evaluation_mode: Literal["lite"] = "lite"
    reference: LiteReferenceBundle
    candidates: tuple[LiteCandidateResult, ...]
    baseline_summary_id: str = Field(min_length=1)
    deltas_from_baseline: dict[str, dict[str, float | None]] = Field(
        default_factory=dict
    )
    execution_stats: dict[str, int] = Field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class LiteExtractResponse(FrozenStrictModel):
    key_facts: tuple[str, ...] = ()


class LiteDecomposeResponse(FrozenStrictModel):
    atomic_facts: tuple[str, ...] = ()


class LiteRawMatch(FrozenStrictModel):
    fact_idx: int = Field(ge=0)
    align_unit_idx: int | None = Field(default=None, ge=0)


class LiteMatchResponse(FrozenStrictModel):
    matches: tuple[LiteRawMatch, ...]


class LiteRawVerification(FrozenStrictModel):
    fact_idx: int = Field(ge=0)
    conclusion: Literal["supported", "not supported"]


class LiteVerifyResponse(FrozenStrictModel):
    verifications: tuple[LiteRawVerification, ...]
