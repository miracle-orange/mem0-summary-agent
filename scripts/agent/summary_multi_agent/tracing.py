"""追加式 JSONL 运行轨迹记录器，供失败诊断和离线审计使用。"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class TraceRecorder:
    """以追加式 JSONL 保存运行事件，并提供进程内审计快照。"""

    def __init__(self, path: Optional[Path] = None) -> None:
        """初始化可选文件路径；路径为空时只保留内存事件。"""
        self._path = path
        self._events: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        """返回当前已记录事件的不可变视图。"""
        return tuple(self._events)

    async def emit(
        self,
        event_type: str,
        *,
        run_id: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """追加一个事件，并在需要时在线程中写入 JSONL。

        锁覆盖内存追加和文件 append，保证并发 Worker 的事件顺序不交错；文件
        I/O 仍可抛出异常，取消会按 asyncio 语义向上传播。
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "run_id": run_id,
            "payload": payload or {},
        }
        encoded = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        async with self._lock:
            self._events.append(event)
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                await asyncio.to_thread(self._append, encoded)
        return event

    def _append(self, encoded: str) -> None:
        """同步追加已编码的一行事件；由 ``emit`` 放入线程执行。"""
        if self._path is None:
            return
        with self._path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
