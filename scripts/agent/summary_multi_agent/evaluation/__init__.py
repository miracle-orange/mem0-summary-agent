"""正式摘要生成后的只读事实级评测 Sidecar。"""

from .evaluator import EvaluationArtifacts, SummaryEvaluator
from .comparison import EvaluationComparisonRunner
from .comparison_scope import (
    EvaluationComparisonScopeEvaluator,
    build_comparison_basis,
    calculate_comparison_scope_metrics,
)
from .metrics import calculate_evaluation_metrics
from .personalization import (
    TargetPersonalizationEvaluator,
    calculate_personalization_metrics,
)
from .schemas import (
    EvaluationFactMatch,
    EvaluationFactRelation,
    EvaluationCompareRequest,
    EvaluationComparisonBasis,
    EvaluationComparisonFactUnit,
    EvaluationComparisonScopeBasis,
    EvaluationComparisonScopeMetrics,
    EvaluationComparisonScopeResult,
    EvaluationMetrics,
    EvaluationSourceSnapshot,
    EvaluationSourceFact,
    EvaluationSummaryFact,
    PersonalizationEvaluationMetrics,
)

__all__ = [
    "EvaluationArtifacts",
    "EvaluationComparisonRunner",
    "EvaluationComparisonBasis",
    "EvaluationComparisonFactUnit",
    "EvaluationComparisonScopeBasis",
    "EvaluationComparisonScopeEvaluator",
    "EvaluationComparisonScopeMetrics",
    "EvaluationComparisonScopeResult",
    "EvaluationCompareRequest",
    "EvaluationFactMatch",
    "EvaluationFactRelation",
    "EvaluationMetrics",
    "EvaluationSourceFact",
    "EvaluationSourceSnapshot",
    "EvaluationSummaryFact",
    "PersonalizationEvaluationMetrics",
    "SummaryEvaluator",
    "TargetPersonalizationEvaluator",
    "calculate_evaluation_metrics",
    "build_comparison_basis",
    "calculate_comparison_scope_metrics",
    "calculate_personalization_metrics",
]
