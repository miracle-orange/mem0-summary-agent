from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from summary_memory_llm import _external_complete, _safe_external_error
from summary_multi_agent.adapters.model import AsyncOpenAICompatibleModelClient


class APIStatusError(Exception):
    pass


def chunk(text=None, finish=None, usage=None):
    return NS(choices=[NS(index=0, delta=NS(content=text), finish_reason=finish)], usage=usage)


class Stream:
    def __init__(self, items):
        self.items = items
        self.closed = False

    def __iter__(self):
        for item in self.items:
            if isinstance(item, BaseException):
                raise item
            yield item

    def close(self):
        self.closed = True


class AsyncStream(Stream):
    async def __aiter__(self):
        for item in self:
            yield item

    async def close(self):
        self.closed = True


ENV = {"OPENAI_BASE_URL": "https://api.siliconflow.cn/v1",
       "OPENAI_API_KEY": "secret-key", "OPENAI_MODEL": "test-model",
       "SMW_OPENAI_STREAM": "true"}


class SyncStreamTests(unittest.TestCase):
    def run_stream(self, items):
        stream = Stream(items)
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return stream

        client = NS(chat=NS(completions=NS(create=create)))
        with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", ENV), \
             patch("summary_memory_llm.load_runtime_env"), \
             patch.dict(sys.modules, {"openai": NS(APIStatusError=APIStatusError)}), \
             patch("summary_memory_llm._sync_external_client", return_value=client):
            path = Path(temp) / "calls.jsonl"
            try:
                result = _external_complete("private-input", 0, 100, json_output=True, audit_path=path)
                error = None
            except RuntimeError as exc:
                result, error = None, str(exc)
            audit = json.loads(path.read_text("utf-8"))
        self.assertTrue(stream.closed)
        self.assertEqual(1, len(calls))
        self.assertTrue(calls[0]["stream"])
        self.assertEqual({"type": "json_object"}, calls[0]["response_format"])
        self.assertNotIn("secret-key", json.dumps(audit))
        self.assertNotIn("private-input", json.dumps(audit))
        return result, error, audit

    def test_success_including_usage_only_tail(self):
        usage = NS(model_dump=lambda **_: {"total_tokens": 50})
        result, error, audit = self.run_stream([
            chunk('{"ok":'), chunk('true}', "stop"), NS(choices=[], usage=usage),
        ])
        self.assertEqual('{"ok":true}', result)
        self.assertIsNone(error)
        self.assertEqual(50, audit["usage"]["total_tokens"])

    def test_incomplete_truncated_empty_and_disconnected_are_not_success(self):
        cases = [([chunk("partial")], "stream_incomplete"),
                 ([chunk("partial", "length")], "output_truncated"),
                 ([chunk("", "stop")], "RuntimeError"),
                 ([chunk("partial"), ConnectionError("secret-key")], "ConnectionError")]
        for items, expected in cases:
            with self.subTest(expected=expected):
                result, error, audit = self.run_stream(items)
                self.assertIsNone(result)
                self.assertIn(expected, error)
                self.assertNotIn("secret-key", error)
                self.assertEqual("failed", audit["status"])

    def test_cause_chain_keeps_types_not_sensitive_messages(self):
        outer = ConnectionError("secret-key")
        outer.__cause__ = TimeoutError("private-input")
        self.assertEqual("External model request failed (ConnectionError -> TimeoutError).",
                         str(_safe_external_error(outer)))


class AsyncStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_disconnect_and_cancel_close_stream(self):
        for failure in (None, ConnectionError("secret-key"), asyncio.CancelledError()):
            with self.subTest(failure=type(failure).__name__):
                stream = AsyncStream([chunk("ok"), failure or chunk(finish="stop")])
                calls = []

                async def create(**kwargs):
                    calls.append(kwargs)
                    return stream

                adapter = AsyncOpenAICompatibleModelClient.__new__(AsyncOpenAICompatibleModelClient)
                adapter._client = NS(chat=NS(completions=NS(create=create)))
                adapter._url = ENV["OPENAI_BASE_URL"]
                adapter._model = "test-model"
                adapter._trace_lock = asyncio.Lock()
                with tempfile.TemporaryDirectory() as temp, patch.dict("os.environ", ENV), \
                     patch.dict(sys.modules, {"openai": NS(APIStatusError=APIStatusError)}):
                    adapter._trace_path = Path(temp) / "calls.jsonl"
                    if failure is None:
                        self.assertEqual("ok", await adapter.complete("private-input", temperature=0, max_tokens=100))
                    else:
                        expected = asyncio.CancelledError if isinstance(failure, asyncio.CancelledError) else RuntimeError
                        with self.assertRaises(expected):
                            await adapter.complete("private-input", temperature=0, max_tokens=100)
                    audit = json.loads(adapter._trace_path.read_text("utf-8"))
                    self.assertEqual("completed" if failure is None else
                                     "cancelled" if isinstance(failure, asyncio.CancelledError) else "failed",
                                     audit["status"])
                    self.assertNotIn("secret-key", json.dumps(audit))
                self.assertTrue(stream.closed)
                self.assertEqual(1, len(calls))


if __name__ == "__main__":
    unittest.main()
