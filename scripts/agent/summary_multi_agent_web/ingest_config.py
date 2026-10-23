"""解析 Web Ingest 的请求级配置，不修改进程环境变量。"""

from __future__ import annotations

import os
from typing import Any

from .schemas import WebIngestConfig


def _positive_env(name: str) -> int | None:
    """读取正整数环境变量；缺失、非法或非正值按未提供处理。"""
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _resolved(
    explicit: int | None,
    env_name: str,
    backend: str,
    *,
    deepseek_default: int,
    local_default: int,
) -> int:
    """按请求显式值、环境变量、后端默认值的优先级解析配置。"""
    if explicit is not None:
        return explicit
    env_value = _positive_env(env_name)
    if env_value is not None:
        return env_value
    return deepseek_default if backend == "deepseek_ws" else local_default


def resolve_ingest_config(
    config: WebIngestConfig | None = None,
    *,
    runtime_info: dict[str, Any] | None = None,
    execution_profile: str = "legacy",
) -> dict[str, int]:
    """按请求 > 环境变量 > 后端感知默认值解析 Ingest 配置。"""

    requested = config or WebIngestConfig()
    runtime = runtime_info or {}
    backend = str(runtime.get("llm_backend") or os.getenv("SMW_LLM_BACKEND", "local_qwen"))
    cloud_fast = execution_profile == "cloud_fast"
    return {
        "chunk_char_limit": _resolved(
            requested.chunk_char_limit,
            "SMW_E2E_CHUNK_CHAR_LIMIT",
            backend,
            deepseek_default=8000 if cloud_fast else 1000,
            local_default=8000 if cloud_fast else 5000,
        ),
        "chunk_max_tokens": _resolved(
            requested.chunk_max_tokens,
            "SMW_E2E_CHUNK_MAX_TOKENS",
            backend,
            deepseek_default=1200,
            local_default=1200,
        ),
        "merge_max_tokens": _resolved(
            requested.merge_max_tokens,
            "SMW_E2E_MERGE_MAX_TOKENS",
            backend,
            deepseek_default=1600,
            local_default=1600,
        ),
        "typed_extraction_max_tokens": _resolved(
            requested.typed_extraction_max_tokens,
            "SMW_TYPED_EXTRACTION_MAX_TOKENS",
            backend,
            deepseek_default=8192 if cloud_fast else 3000,
            local_default=8192 if cloud_fast else 8000,
        ),
        "dedup_judge_max_tokens": _resolved(
            requested.dedup_judge_max_tokens,
            "SMW_TYPED_DEDUP_JUDGE_MAX_TOKENS",
            backend,
            deepseek_default=1200,
            local_default=1200,
        ),
        "dedup_judge_max_candidates": _resolved(
            requested.dedup_judge_max_candidates,
            "SMW_TYPED_DEDUP_JUDGE_MAX_CANDIDATES",
            backend,
            deepseek_default=20,
            local_default=20,
        ),
    }
