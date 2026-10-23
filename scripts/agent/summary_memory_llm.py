"""按环境变量选择模型服务，并隐藏凭据细节。"""

import asyncio
import json
import os
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any
from types import SimpleNamespace
from urllib.parse import urlsplit


LOCAL_QWEN_BACKEND = "local_qwen"
DEEPSEEK_WS_BACKEND = "deepseek_ws"
OPENAI_COMPATIBLE_BACKEND = "openai_compatible"
SUPPORTED_LLM_BACKENDS = (LOCAL_QWEN_BACKEND, DEEPSEEK_WS_BACKEND, OPENAI_COMPATIBLE_BACKEND)

DEFAULT_DEEPSEEK_WS_URL = ""
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
TRUE_ENV_VALUES = {"1", "true", "yes", "y", "on"}
_audit_lock = threading.Lock()


def env_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in TRUE_ENV_VALUES


def load_runtime_env() -> None:
    """Load the repository's optional .env without replacing process settings."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError("Install python-dotenv to use the project .env file.") from exc
    load_dotenv(env_path, override=False)


def _external_settings() -> tuple[str, str, str]:
    load_runtime_env()
    url = (os.getenv("OPENAI_BASE_URL") or "").strip().rstrip("/")
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("OPENAI_MODEL") or "").strip()
    missing = [
        name for name, value in (
            ("OPENAI_BASE_URL", url), ("OPENAI_API_KEY", key), ("OPENAI_MODEL", model)
        ) if not value
    ]
    if missing:
        raise RuntimeError("External model configuration is missing: " + ", ".join(missing))
    if not url.startswith(("https://", "http://")):
        raise RuntimeError("OPENAI_BASE_URL must be an HTTP(S) URL.")
    parsed = urlsplit(url)
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RuntimeError("OPENAI_BASE_URL must not contain credentials or query parameters.")
    return url, key, model


def validate_llm_configuration() -> None:
    """Validate credentials without sending a request or printing the key."""
    if get_llm_backend() == OPENAI_COMPATIBLE_BACKEND:
        _external_settings()


@lru_cache(maxsize=8)
def _sync_external_client(url: str, key: str):
    from openai import OpenAI
    return OpenAI(base_url=url, api_key=key, timeout=180, max_retries=0)


def _external_request_kwargs(model: str, prompt: str, temperature: float,
                             max_tokens: int, json_output: bool,
                             *, base_url: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": env_flag("SMW_OPENAI_STREAM", True),
    }
    if json_output:
        kwargs["response_format"] = {"type": "json_object"}
    if urlsplit(base_url or os.getenv("OPENAI_BASE_URL", "")).hostname == "api.deepseek.com":
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return kwargs


class ExternalResponseError(RuntimeError):
    """Locally generated, non-secret response validation diagnostic."""


class ExternalStreamAccumulator:
    """Collect one streamed response without exposing partial output to callers."""

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.finish_reason = None
        self.usage = None
        self.chunks_received = 0
        self.content_chars = 0
        self.started = time.monotonic()
        self.first_chunk_ms = None

    def add(self, chunk: Any) -> None:
        if self.first_chunk_ms is None:
            self.first_chunk_ms = int((time.monotonic() - self.started) * 1000)
        self.chunks_received += 1
        if getattr(chunk, "usage", None) is not None:
            self.usage = chunk.usage
        for choice in chunk.choices:
            if choice.index != 0:
                continue
            content = getattr(choice.delta, "content", None)
            if content:
                self.parts.append(content)
                self.content_chars += len(content)
            if choice.finish_reason is not None:
                self.finish_reason = choice.finish_reason

    def response(self) -> Any:
        if self.finish_reason is None:
            raise ExternalResponseError("External model request failed (stream_incomplete: no finish reason).")
        return SimpleNamespace(choices=[SimpleNamespace(
            finish_reason=self.finish_reason,
            message=SimpleNamespace(content="".join(self.parts)),
        )], usage=self.usage)

    def metrics(self) -> dict[str, Any]:
        return {"chunks_received": self.chunks_received,
                "received_content_chars": self.content_chars,
                "first_chunk_ms": self.first_chunk_ms,
                "finish_reason": self.finish_reason}


def _external_error_details(exc: BaseException) -> dict[str, Any]:
    # Type names only: exception messages can contain credentials or request bodies.
    chain = []
    seen = set()
    cause = exc
    while cause is not None and id(cause) not in seen and len(chain) < 6:
        seen.add(id(cause))
        chain.append(type(cause).__name__)
        cause = cause.__cause__ or cause.__context__
    return {"error_chain": chain}


def _external_response_text(response: Any) -> str:
    if not response.choices:
        raise RuntimeError("External model returned no choices.")
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise ExternalResponseError("External model request failed (output_truncated: finish_reason=length).")
    content = choice.message.content or ""
    if not content.strip():
        raise RuntimeError("External model returned empty content.")
    return content


def _safe_external_error(exc: Exception) -> RuntimeError:
    """Preserve actionable status without persisting provider response bodies."""
    status = getattr(exc, "status_code", None)
    if isinstance(exc, ExternalResponseError):
        return exc
    message = str(exc).lower()
    if any(marker in message for marker in (
        "maximum context length", "context_overflow", "context overflow",
        "reduce the length of the messages",
    )):
        return RuntimeError("context_overflow: external model rejected the request size")
    if status is not None:
        return RuntimeError(f"External model request failed (HTTP {status}).")
    chain = " -> ".join(_external_error_details(exc)["error_chain"])
    return RuntimeError(f"External model request failed ({chain}).")


def _external_complete(prompt: str, temperature: float, max_tokens: int,
                       *, json_output: bool = False,
                       audit_path: Path | None = None) -> str:
    url, key, model = _external_settings()
    from openai import APIStatusError
    client = _sync_external_client(url, key)
    kwargs = _external_request_kwargs(model, prompt, temperature, max_tokens, json_output)
    call_id = uuid.uuid4().hex
    started = time.monotonic()
    for attempt in range(2):
        stream_state = ExternalStreamAccumulator()
        try:
            response = client.chat.completions.create(**kwargs)
            if kwargs["stream"]:
                stream = response
                try:
                    for chunk in stream:
                        stream_state.add(chunk)
                finally:
                    stream.close()
                response = stream_state.response()
            content = _external_response_text(response)
            if audit_path is not None:
                _write_external_audit(audit_path, {
                    "call_id": call_id, "status": "completed", "model": model,
                    "json_output": json_output, "attempts": attempt + 1,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "finish_reason": response.choices[0].finish_reason,
                    "usage": response.usage.model_dump(mode="json") if response.usage else None,
                    "stream": kwargs["stream"],
                    "first_chunk_ms": stream_state.first_chunk_ms,
                    "chunks_received": stream_state.chunks_received,
                })
            return content
        except APIStatusError as exc:
            if attempt or stream_state.chunks_received or exc.status_code not in (429, 500, 502, 503, 504):
                if audit_path is not None:
                    _write_external_audit(audit_path, {
                        "call_id": call_id, "status": "failed", "model": model,
                        "attempts": attempt + 1, "http_status": exc.status_code,
                        "stream": kwargs["stream"], **stream_state.metrics(),
                        **_external_error_details(exc),
                        "duration_ms": int((time.monotonic() - started) * 1000),
                    })
                raise _safe_external_error(exc) from None
            wait = exc.response.headers.get("retry-after", "1")
            try:
                delay = min(180.0, max(0.0, float(wait)))
            except ValueError:
                delay = 1.0
            time.sleep(delay)
        except Exception as exc:
            if audit_path is not None:
                _write_external_audit(audit_path, {
                    "call_id": call_id, "status": "failed", "model": model,
                    "attempts": attempt + 1, "error_type": type(exc).__name__,
                    "stream": kwargs["stream"],
                    **stream_state.metrics(),
                    **_external_error_details(exc),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                })
            raise _safe_external_error(exc) from None
    raise AssertionError("unreachable retry state")


def _write_external_audit(path: Path, entry: dict[str, Any]) -> None:
    with _audit_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def make_audited_llm_call(path: Path, *, json_output: bool = False):
    def call(prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
        return _external_complete(
            prompt, temperature, max_tokens, json_output=json_output,
            audit_path=path,
        )
    return call


def _required_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} is required when SMW_LLM_BACKEND={DEEPSEEK_WS_BACKEND}.")
    return value


def get_llm_backend() -> str:
    load_runtime_env()
    backend = (os.getenv("SMW_LLM_BACKEND", LOCAL_QWEN_BACKEND) or "").strip().lower()
    if backend not in SUPPORTED_LLM_BACKENDS:
        allowed = ", ".join(SUPPORTED_LLM_BACKENDS)
        raise RuntimeError(f"Unsupported SMW_LLM_BACKEND={backend!r}. Expected one of: {allowed}.")
    return backend


def get_llm_runtime_info() -> dict[str, Any]:
    """Return non-secret settings suitable for logs and run_info.json."""
    backend = get_llm_backend()
    if backend == OPENAI_COMPATIBLE_BACKEND:
        url, _, model = _external_settings()
        return {
            "llm_backend": backend,
            "llm_model": model,
            "llm_transport": "openai_http",
            "llm_endpoint": url,
            "llm_stream": env_flag("SMW_OPENAI_STREAM", True),
            "thinking_enabled": False if urlsplit(url).hostname == "api.deepseek.com" else None,
        }
    if backend == LOCAL_QWEN_BACKEND:
        return {
            "llm_backend": backend,
            "llm_model": os.getenv("VLLM_MODEL", "qwen-local"),
            "llm_transport": "openai_http",
            "llm_endpoint": os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        }

    return {
        "llm_backend": backend,
        "llm_model": os.getenv("SMW_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        "llm_transport": "websocket_openai_compatible",
        "llm_endpoint": os.getenv("SMW_DEEPSEEK_WS_URL", DEFAULT_DEEPSEEK_WS_URL),
        "deepseek_enable_thinking": env_flag("SMW_DEEPSEEK_ENABLE_THINKING", False),
    }


def _call_local_qwen(prompt: str, temperature: float, max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.getenv("VLLM_API_KEY", "local-key"),
    )
    response = client.chat.completions.create(
        model=os.getenv("VLLM_MODEL", "qwen-local"),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def build_deepseek_ws_payload(prompt: str, temperature: float, max_tokens: int) -> dict[str, Any]:
    """Build the company gateway request without exposing credentials in artifacts."""
    payload: dict[str, Any] = {
        "api_key": _required_env("SMW_DEEPSEEK_API_KEY"),
        "model": os.getenv("SMW_DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL),
        "stream": False,
        "extra_body": {"enable_thinking": env_flag("SMW_DEEPSEEK_ENABLE_THINKING", False)},
        "temperature": temperature,
        "top_p": float(os.getenv("SMW_DEEPSEEK_TOP_P", "0.9")),
        "top_k": int(os.getenv("SMW_DEEPSEEK_TOP_K", "1")),
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    payload["user"] = _required_env("SMW_DEEPSEEK_USER")
    return payload


def _websocket_connect_kwargs(sender: str) -> dict[str, Any]:
    import websockets

    major_version = int(websockets.__version__.split(".", 1)[0])
    headers = {"sender": sender}
    if major_version >= 14:
        kwargs: dict[str, Any] = {"additional_headers": headers}
        if major_version >= 15:
            kwargs["proxy"] = None
        return kwargs
    return {"extra_headers": headers}


def _content_from_choice(choice: dict[str, Any]) -> tuple[str, bool]:
    message = choice.get("message")
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message["content"]), True

    delta = choice.get("delta")
    if isinstance(delta, dict) and delta.get("content") is not None:
        return str(delta["content"]), False

    content = choice.get("content")
    if content is not None:
        return str(content), True
    return "", False


async def _call_deepseek_ws_async(prompt: str, temperature: float, max_tokens: int) -> str:
    import websockets

    url = _required_env("SMW_DEEPSEEK_WS_URL")
    sender = _required_env("SMW_DEEPSEEK_SENDER")
    timeout_seconds = max(1.0, float(os.getenv("SMW_DEEPSEEK_WS_TIMEOUT_SECONDS", "120")))
    payload = build_deepseek_ws_payload(prompt, temperature, max_tokens)
    fragments: list[str] = []

    try:
        async with websockets.connect(url, **_websocket_connect_kwargs(sender)) as websocket:
            await asyncio.wait_for(
                websocket.send(json.dumps(payload, ensure_ascii=False)),
                timeout=timeout_seconds,
            )
            while True:
                raw_response = await asyncio.wait_for(websocket.recv(), timeout=timeout_seconds)
                if not raw_response:
                    continue
                response = json.loads(raw_response)
                error = response.get("error")
                if isinstance(error, dict):
                    message = str(error.get("message") or error.get("code") or "unknown gateway error")
                    raise RuntimeError(
                        "DeepSeek WebSocket gateway returned an error: "
                        f"{message} (model={payload['model']!r}, max_tokens={max_tokens})."
                    )
                choices = response.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                content, is_complete_message = _content_from_choice(choice)
                if content:
                    if is_complete_message:
                        fragments = [content]
                    else:
                        fragments.append(content)

                if choice.get("finish_reason"):
                    break
    except asyncio.TimeoutError as exc:
        raise RuntimeError(f"DeepSeek WebSocket request timed out after {timeout_seconds:g} seconds.") from exc

    text = "".join(fragments).strip()
    if not text:
        raise RuntimeError("DeepSeek WebSocket response completed without assistant content.")
    return text


def _run_coroutine(coro: Any) -> str:
    """在 notebook 已有事件循环时用独立线程桥接同步调用。"""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        has_running_loop = False
    else:
        has_running_loop = True

    if not has_running_loop:
        return asyncio.run(coro)

    # Notebook 主线程可能已经运行事件循环，直接 asyncio.run 会触发嵌套循环错误。
    result: dict[str, Any] = {}
    failure: list[BaseException] = []

    def run() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover - defensive notebook bridge.
            failure.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join()
    if failure:
        raise failure[0]
    return str(result["value"])


def call_summary_llm(prompt: str, temperature: float = 0.0, max_tokens: int = 1024) -> str:
    """Call the configured model backend."""
    backend = get_llm_backend()
    if backend == OPENAI_COMPATIBLE_BACKEND:
        return _external_complete(prompt, temperature, max_tokens)
    if backend == LOCAL_QWEN_BACKEND:
        return _call_local_qwen(prompt, temperature, max_tokens)
    return _run_coroutine(_call_deepseek_ws_async(prompt, temperature, max_tokens))
