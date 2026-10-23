"""只读检查 Web 运行使用的 Memory store 和本次写入回读结果。"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable


TYPED_MEMORY_SCHEMA_VERSION = "summary_typed_memory.v1"


def _items(value: Any) -> list[dict[str, Any]]:
    """兼容 Mem0 列表或 ``{"results": [...]}`` 返回形态。"""
    if isinstance(value, dict):
        value = value.get("results", [])
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, dict)]


def _metadata(item: dict[str, Any]) -> dict[str, Any]:
    """安全取得一条 Store 记录的 metadata。"""
    value = item.get("metadata")
    return dict(value) if isinstance(value, dict) else {}


def _memory_id(item: dict[str, Any]) -> str:
    """优先读取 Typed Memory ID，否则回退到底层记录 ID。"""
    metadata = _metadata(item)
    return str(metadata.get("typed_memory_id") or item.get("id") or "").strip()


def _call_get_all(memory: Any, user_id: str, limit: int) -> Any:
    """兼容 Mem0 新旧 ``get_all`` 签名执行有界身份域读取。"""
    get_all = getattr(memory, "get_all", None)
    if not callable(get_all):
        raise RuntimeError("Memory client does not provide get_all for store probing")

    # Mem0 v3 要求身份放入 filters；只有旧版客户端拒绝该关键字时才走兼容签名。
    try:
        return get_all(filters={"user_id": user_id}, limit=limit)
    except TypeError:
        try:
            return get_all(user_id=user_id, limit=limit)
        except TypeError:
            return get_all(user_id=user_id)


def probe_memory_store(
    memory: Any,
    *,
    user_id: str,
    project: str,
    schema_version: str = TYPED_MEMORY_SCHEMA_VERSION,
    limit: int = 10000,
) -> dict[str, Any]:
    """有界列举身份域并分类记录，不返回 Memory 正文内容。"""

    identity = {
        "user_id": user_id,
        "project": project,
        "memory_schema_version": schema_version,
    }
    try:
        rows = _items(_call_get_all(memory, user_id, limit))
    except Exception as exc:
        return {
            "schema_version": "summary_multi_agent_web.store_probe.v1",
            "status": "error",
            "identity": identity,
            "fetched_count": 0,
            "matching_count": 0,
            "type_counts": {},
            "error": f"{type(exc).__name__}: {exc}",
        }

    type_counts: Counter[str] = Counter()
    matching_ids: list[str] = []
    project_mismatch = 0
    schema_mismatch = 0
    for row in rows:
        metadata = _metadata(row)
        if str(metadata.get("project") or "") != project:
            project_mismatch += 1
            continue
        if str(metadata.get("memory_schema_version") or "") != schema_version:
            schema_mismatch += 1
            continue
        memory_id = _memory_id(row)
        if not memory_id:
            continue
        memory_type = str(metadata.get("memory_type") or "unknown").strip()
        type_counts[memory_type] += 1
        matching_ids.append(memory_id)

    return {
        "schema_version": "summary_multi_agent_web.store_probe.v1",
        "status": "ready" if matching_ids else "empty",
        "identity": identity,
        "fetched_count": len(rows),
        "matching_count": len(matching_ids),
        "matching_memory_ids": matching_ids,
        "type_counts": dict(sorted(type_counts.items())),
        "project_mismatch_count": project_mismatch,
        "schema_mismatch_count": schema_mismatch,
        "limit": limit,
    }


def operation_ids_from_add_results(add_results: Any) -> list[str]:
    """从 Mem0 ADD/UPDATE 响应提取持久化 ID，供写后回读使用。"""

    operation_ids: list[str] = []
    for response in add_results if isinstance(add_results, list) else []:
        for item in _items(response):
            event = str(item.get("event") or "").upper()
            if event not in {"ADD", "UPDATE"}:
                continue
            memory_id = str(item.get("id") or item.get("memory_id") or "").strip()
            if memory_id:
                operation_ids.append(memory_id)
    return list(dict.fromkeys(operation_ids))


def verify_operation_readback(
    memory: Any,
    operation_ids: list[str],
    *,
    user_id: str,
    project: str,
    schema_version: str = TYPED_MEMORY_SCHEMA_VERSION,
) -> dict[str, Any]:
    """从重开客户端逐个读取 ADD/UPDATE ID，并核对身份过滤条件。"""

    recovered: list[str] = []
    missing: list[str] = []
    identity_mismatch: list[str] = []
    get_memory = getattr(memory, "get", None)
    if not callable(get_memory):
        return {
            "schema_version": "summary_multi_agent_web.ingest_readback.v1",
            "status": "error",
            "expected_operation_count": len(operation_ids),
            "recovered_operation_count": 0,
            "missing_operation_ids": list(operation_ids),
            "identity_mismatch_ids": [],
            "error": "Memory client does not provide get for write readback",
        }

    for memory_id in operation_ids:
        try:
            try:
                raw_row = get_memory(memory_id)
            except TypeError:
                raw_row = get_memory(memory_id=memory_id)
            rows = _items(raw_row)
            row = rows[0] if rows else raw_row
        except Exception:
            row = None
        if not isinstance(row, dict):
            missing.append(memory_id)
            continue
        metadata = _metadata(row)
        if (
            str(metadata.get("project") or "") != project
            or str(metadata.get("memory_schema_version") or "") != schema_version
            or str(metadata.get("user_id") or user_id) != user_id
        ):
            identity_mismatch.append(memory_id)
            continue
        recovered.append(memory_id)

    status = "verified" if not missing and not identity_mismatch else "partial"
    if not recovered and operation_ids:
        status = "failed"
    return {
        "schema_version": "summary_multi_agent_web.ingest_readback.v1",
        "status": status,
        "expected_operation_count": len(operation_ids),
        "recovered_operation_count": len(recovered),
        "recovered_operation_ids": recovered,
        "missing_operation_ids": missing,
        "identity_mismatch_ids": identity_mismatch,
    }


def reopen_and_verify(
    memory_factory: Callable[[], Any],
    memory_closer: Callable[[Any], None],
    operation_ids: list[str],
    *,
    user_id: str,
    project: str,
    schema_version: str = TYPED_MEMORY_SCHEMA_VERSION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """打开新客户端、执行 Store Probe 和写后回读，最后无论结果如何关闭客户端。"""

    reopened = memory_factory()
    try:
        probe = probe_memory_store(
            reopened,
            user_id=user_id,
            project=project,
            schema_version=schema_version,
        )
        readback = verify_operation_readback(
            reopened,
            operation_ids,
            user_id=user_id,
            project=project,
            schema_version=schema_version,
        )
    finally:
        memory_closer(reopened)
    return probe, readback
