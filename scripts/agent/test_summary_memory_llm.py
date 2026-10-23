import json
import os
import sys
import types
import unittest
from unittest.mock import patch

import summary_memory_llm as llm


class FakeWebSocket:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    async def send(self, content):
        self.sent.append(content)

    async def recv(self):
        return self.responses.pop(0)


class FakeConnection:
    def __init__(self, websocket):
        self.websocket = websocket

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, exc_type, exc, traceback):
        return False


class SummaryMemoryLlmTests(unittest.TestCase):
    def setUp(self):
        self.deepseek_env = {
            "SMW_LLM_BACKEND": "deepseek_ws",
            "SMW_DEEPSEEK_API_KEY": "test-api-key",
            "SMW_DEEPSEEK_USER": "test-user",
            "SMW_DEEPSEEK_SENDER": "test-sender",
            "SMW_DEEPSEEK_WS_URL": "ws://127.0.0.1:18087/v1/chat/completions",
        }

    def test_build_deepseek_payload_uses_stage_parameters(self):
        with patch.dict(os.environ, self.deepseek_env, clear=False):
            payload = llm.build_deepseek_ws_payload("ping", temperature=0.0, max_tokens=64)

        self.assertEqual(payload["model"], llm.DEFAULT_DEEPSEEK_MODEL)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["max_tokens"], 64)
        self.assertEqual(payload["messages"], [{"role": "user", "content": "ping"}])
        self.assertEqual(payload["user"], "test-user")

    def test_deepseek_adapter_joins_delta_chunks_and_sends_sender_header(self):
        websocket = FakeWebSocket(
            [
                json.dumps({"choices": [{"delta": {"content": "hello "}}]}),
                json.dumps({"choices": [{"delta": {"content": "world"}, "finish_reason": "stop"}]}),
            ]
        )
        captured = {}

        def connect(url, **kwargs):
            captured["url"] = url
            captured["kwargs"] = kwargs
            return FakeConnection(websocket)

        fake_websockets = types.SimpleNamespace(__version__="15.0", connect=connect)
        with patch.dict(os.environ, self.deepseek_env, clear=False), patch.dict(
            sys.modules, {"websockets": fake_websockets}
        ):
            response = llm.call_summary_llm("ping", temperature=0.0, max_tokens=64)

        self.assertEqual(response, "hello world")
        self.assertEqual(captured["kwargs"]["additional_headers"], {"sender": "test-sender"})
        self.assertIsNone(captured["kwargs"]["proxy"])
        sent_payload = json.loads(websocket.sent[0])
        self.assertEqual(sent_payload["api_key"], "test-api-key")
        self.assertEqual(sent_payload["messages"][0]["content"], "ping")

    def test_runtime_info_never_contains_deepseek_credentials(self):
        with patch.dict(os.environ, self.deepseek_env, clear=False):
            runtime_info = llm.get_llm_runtime_info()

        self.assertEqual(runtime_info["llm_backend"], llm.DEEPSEEK_WS_BACKEND)
        self.assertNotIn("test-api-key", json.dumps(runtime_info))
        self.assertNotIn("test-user", json.dumps(runtime_info))
        self.assertNotIn("test-sender", json.dumps(runtime_info))


if __name__ == "__main__":
    unittest.main()
