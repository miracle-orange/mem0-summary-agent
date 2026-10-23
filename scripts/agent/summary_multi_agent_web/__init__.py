"""Summary Multi-Agent V2 的 Web 请求、后台运行、E2E 和审计适配器。"""

from .multi_agent_runner import run_multi_agent_summary
from .e2e_ingest import run_web_ingest
from .evaluation_compare import run_web_evaluation_compare
from .run_manager import WebRunManager
from .visualization import build_run_visualization
from .schemas import (
    WebE2ERequest,
    WebE2ERunResult,
    WebIngestConfig,
    WebIngestRequest,
    WebIngestResult,
    WebHistorySource,
    WebRunStatus,
    WebSummarizeRequest,
    build_run_config,
    build_summary_request,
)

__all__ = [
    "WebRunManager",
    "WebE2ERequest",
    "WebE2ERunResult",
    "WebIngestConfig",
    "WebIngestRequest",
    "WebIngestResult",
    "WebHistorySource",
    "WebRunStatus",
    "WebSummarizeRequest",
    "build_run_config",
    "build_run_visualization",
    "build_summary_request",
    "run_multi_agent_summary",
    "run_web_evaluation_compare",
    "run_web_ingest",
]
