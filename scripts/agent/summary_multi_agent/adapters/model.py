"""把同步 LLM 回调适配为 V2 异步 ModelClient 接口。"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from pathlib import Path
from typing import Callable, Protocol

from ..model_input_audit import _TokenCounter


_CONTEXT_OVERFLOW_PATTERNS = (
    re.compile(r"maximum context length", re.IGNORECASE),
    re.compile(r"context[_ ]overflow", re.IGNORECASE),
    re.compile(r"requested \d+ tokens", re.IGNORECASE),
    re.compile(r"reduce the length of the messages", re.IGNORECASE),
)


class ContextOverflowError(RuntimeError):
    """The prompt cannot fit a configured or server-declared context window."""


def is_context_overflow_error(exc: BaseException) -> bool:
    """Recognize normalized and common OpenAI-compatible overflow errors."""

    if isinstance(exc, ContextOverflowError):
        return True
    message = str(exc)
    return any(pattern.search(message) for pattern in _CONTEXT_OVERFLOW_PATTERNS)


class ModelClient(Protocol):
    """V2 统一使用的异步模型调用协议。"""

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """异步完成一次模型请求并返回文本结果。"""
        ...


class SyncCallableModelClient:
    """把同步回调放入线程池，避免阻塞 DAG 的 asyncio 事件循环。"""

    def __init__(self, call: Callable[..., str]) -> None:
        """保存一个接受 prompt、temperature 和 max_tokens 的同步回调。"""
        self._call = call

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """在线程中执行同步模型调用，并原样传播回调异常或取消。"""
        # 同步模型客户端放到线程池，避免阻塞 DAG 的 asyncio 事件循环。
        return await asyncio.to_thread(
            self._call,
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class AsyncOpenAICompatibleModelClient:
    """One reusable async connection pool per run, with non-secret call accounting."""

    def __init__(self, *, trace_path: Path | None = None) -> None:
        from openai import AsyncOpenAI
        from summary_memory_llm import _external_settings

        url, key, self._model = _external_settings()
        self._url = url
        self._client = AsyncOpenAI(base_url=url, api_key=key, timeout=180, max_retries=0)
        self._trace_path = trace_path
        self._trace_lock = asyncio.Lock()

    async def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        return await self._request(prompt, temperature, max_tokens, json_output=False)

    async def complete_json(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        return await self._request(prompt, temperature, max_tokens, json_output=True)

    async def _request(self, prompt: str, temperature: float, max_tokens: int,
                       *, json_output: bool) -> str:
        from openai import APIStatusError
        from summary_memory_llm import (
            _external_request_kwargs, _external_response_text, _safe_external_error,
            ExternalStreamAccumulator, _external_error_details,
        )

        call_id = uuid.uuid4().hex
        started = time.monotonic()
        kwargs = _external_request_kwargs(
            self._model, prompt, temperature, max_tokens, json_output,
            base_url=self._url,
        )
        for attempt in range(2):
            stream_state = ExternalStreamAccumulator()
            try:
                response = await self._client.chat.completions.create(**kwargs)
                if kwargs["stream"]:
                    stream = response
                    try:
                        async for chunk in stream:
                            stream_state.add(chunk)
                    finally:
                        await stream.close()
                    response = stream_state.response()
                content = _external_response_text(response)
                usage = response.usage
                await self._record({
                    "call_id": call_id, "status": "completed", "model": self._model,
                    "json_output": json_output, "attempts": attempt + 1,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": usage.model_dump(mode="json") if usage else None,
                    "stream": kwargs["stream"],
                    "first_chunk_ms": stream_state.first_chunk_ms,
                    "chunks_received": stream_state.chunks_received,
                })
                return content
            except APIStatusError as exc:
                if attempt == 0 and not stream_state.chunks_received and exc.status_code in (429, 500, 502, 503, 504):
                    try:
                        delay = min(180.0, max(0.0, float(exc.response.headers.get("retry-after", "1"))))
                    except ValueError:
                        delay = 1.0
                    await asyncio.sleep(delay)
                    continue
                await self._record({
                    "call_id": call_id, "status": "failed", "model": self._model,
                    "attempts": attempt + 1, "http_status": exc.status_code,
                    "stream": kwargs["stream"], **stream_state.metrics(),
                    **_external_error_details(exc),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                })
                raise _safe_external_error(exc) from None
            except asyncio.CancelledError:
                await self._record({"call_id": call_id, "status": "cancelled",
                                    "model": self._model, "stream": kwargs["stream"],
                                    **stream_state.metrics()})
                raise
            except Exception as exc:
                await self._record({
                    "call_id": call_id, "status": "failed", "model": self._model,
                    "attempts": attempt + 1,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "error_type": type(exc).__name__,
                    "stream": kwargs["stream"],
                    **stream_state.metrics(),
                    **_external_error_details(exc),
                })
                raise _safe_external_error(exc) from None
        raise AssertionError("unreachable retry state")

    async def _record(self, entry: dict) -> None:
        if self._trace_path is None:
            return
        async with self._trace_lock:
            self._trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self._trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    async def aclose(self) -> None:
        await self._client.close()


class ContextWindowModelClient:
    """Apply conservative preflight checks and normalize server overflow errors."""

    def __init__(
        self,
        delegate: ModelClient,
        *,
        context_window_tokens: int | None,
        safety_margin_tokens: int = 1024,
    ) -> None:
        self._delegate = delegate
        self._context_window_tokens = context_window_tokens
        self._safety_margin_tokens = safety_margin_tokens
        self._counter = _TokenCounter()

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        measurement = self._counter.measure(prompt)
        if (
            self._context_window_tokens is not None
            and measurement["token_count_measured"]
            and measurement["tokens"]
            + max_tokens
            + self._safety_margin_tokens
            > self._context_window_tokens
        ):
            raise ContextOverflowError(
                "context_overflow: measured prompt tokens plus output reserve and "
                f"safety margin exceed configured window "
                f"({measurement['tokens']} + {max_tokens} + "
                f"{self._safety_margin_tokens} > {self._context_window_tokens})"
            )
        try:
            return await self._delegate.complete(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except BaseException as exc:
            if is_context_overflow_error(exc):
                raise ContextOverflowError(
                    f"context_overflow: {type(exc).__name__}: {exc}"
                ) from exc
            raise

    async def complete_json(
        self, prompt: str, *, temperature: float, max_tokens: int
    ) -> str:
        method = getattr(self._delegate, "complete_json", None)
        if method is None:
            return await self.complete(prompt, temperature=temperature, max_tokens=max_tokens)
        measurement = self._counter.measure(prompt)
        if (
            self._context_window_tokens is not None
            and measurement["token_count_measured"]
            and measurement["tokens"] + max_tokens + self._safety_margin_tokens
            > self._context_window_tokens
        ):
            raise ContextOverflowError("context_overflow: JSON request exceeds configured window")
        try:
            return await method(prompt, temperature=temperature, max_tokens=max_tokens)
        except BaseException as exc:
            if is_context_overflow_error(exc):
                raise ContextOverflowError(f"context_overflow: {type(exc).__name__}: {exc}") from exc
            raise
