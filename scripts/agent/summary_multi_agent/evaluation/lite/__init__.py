"""共享的精简会议纪要评测流程。"""

from .artifacts import (
    write_lite_comparison_artifacts,
    write_lite_comparison_failure_artifacts,
    write_lite_evaluation_artifacts,
    write_lite_failure_artifacts,
)
from .evaluator import (
    LiteEvaluationComparisonRunner,
    LiteEvaluationRunner,
    LiteSummaryEvaluator,
)
from .schemas import (
    LiteCandidateInput,
    LiteCandidateResult,
    LiteEvaluationResult,
    LiteEvidenceSource,
    LiteReferenceBundle,
)

__all__ = [
    "LiteCandidateInput",
    "LiteCandidateResult",
    "LiteEvaluationComparisonRunner",
    "LiteEvaluationResult",
    "LiteEvaluationRunner",
    "LiteEvidenceSource",
    "LiteReferenceBundle",
    "LiteSummaryEvaluator",
    "write_lite_comparison_artifacts",
    "write_lite_comparison_failure_artifacts",
    "write_lite_evaluation_artifacts",
    "write_lite_failure_artifacts",
]
