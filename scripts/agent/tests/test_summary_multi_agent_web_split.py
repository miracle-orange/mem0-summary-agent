from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path


AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from summary_multi_agent_web.ingest_config import resolve_ingest_config  # noqa: E402
from summary_multi_agent_web.schemas import WebIngestConfig, WebIngestRequest  # noqa: E402
from summary_multi_agent_web_demo import _optional_int, build_ingest_request  # noqa: E402
from summary_multi_agent_web.store_probe import (  # noqa: E402
    probe_memory_store,
    verify_operation_readback,
)


class ProbeMemoryV3:
    def __init__(self) -> None:
        self.rows = [
            {
                "id": "mem-1",
                "metadata": {
                    "project": "demo",
                    "memory_schema_version": "summary_typed_memory.v1",
                    "memory_type": "task",
                },
            }
        ]
        self.filters = None

    def get_all(self, *, filters: dict[str, str], limit: int = 10000):
        self.filters = filters
        del limit
        return {"results": self.rows}

    def get(self, memory_id: str):
        return next((row for row in self.rows if row["id"] == memory_id), None)


class ProbeMemoryLegacy(ProbeMemoryV3):
    def get_all(self, *, user_id: str, limit: int = 10000):
        self.filters = {"user_id": user_id}
        del limit
        return {"results": self.rows}


class ProbeMemoryFailure(ProbeMemoryV3):
    def get_all(self, *, filters: dict[str, str], limit: int = 10000):
        del filters, limit
        raise ValueError("Qdrant inventory unavailable")


class WebSplitContractTests(unittest.TestCase):
    def test_blank_ingest_number_controls_use_defaults(self) -> None:
        request = build_ingest_request(
            "历史会议原文", None, "", "u", "p", "2026-07-21", "history", "", "",
            *(_optional_int(value) for value in (0, 0.0, None, "", 0, 0)),
        )
        self.assertEqual(WebIngestConfig(), request.config)
        self.assertEqual(2400, _optional_int("2400"))
        self.assertEqual(2400, _optional_int(2400.0))

    def test_ingest_config_explicit_then_environment_then_backend_default(self) -> None:
        previous = os.environ.get("SMW_TYPED_EXTRACTION_MAX_TOKENS")
        os.environ["SMW_TYPED_EXTRACTION_MAX_TOKENS"] = "4100"
        try:
            from_env = resolve_ingest_config(
                WebIngestConfig(), runtime_info={"llm_backend": "deepseek_ws"}
            )
            explicit = resolve_ingest_config(
                WebIngestConfig(typed_extraction_max_tokens=2700),
                runtime_info={"llm_backend": "deepseek_ws"},
            )
        finally:
            if previous is None:
                os.environ.pop("SMW_TYPED_EXTRACTION_MAX_TOKENS", None)
            else:
                os.environ["SMW_TYPED_EXTRACTION_MAX_TOKENS"] = previous
        self.assertEqual(4100, from_env["typed_extraction_max_tokens"])
        self.assertEqual(2700, explicit["typed_extraction_max_tokens"])
        self.assertEqual(1000, from_env["chunk_char_limit"])

    def test_ingest_request_has_no_summarize_or_run_type_fields(self) -> None:
        request = WebIngestRequest(
            history_sources=[{"label": "h", "text": "raw"}],
            user_id="u",
            project="p",
        )
        payload = request.model_dump()
        self.assertNotIn("run_type", payload)
        self.assertNotIn("summary_request", payload)

    def test_store_probe_v3_and_operation_readback_are_identity_scoped(self) -> None:
        memory = ProbeMemoryV3()
        probe = probe_memory_store(memory, user_id="u", project="demo")
        readback = verify_operation_readback(
            memory, ["mem-1"], user_id="u", project="demo"
        )
        self.assertEqual({"user_id": "u"}, memory.filters)
        self.assertEqual("ready", probe["status"])
        self.assertEqual(1, probe["matching_count"])
        self.assertEqual("verified", readback["status"])

    def test_store_probe_falls_back_to_legacy_entity_parameter(self) -> None:
        memory = ProbeMemoryLegacy()
        probe = probe_memory_store(memory, user_id="u", project="demo")
        self.assertEqual({"user_id": "u"}, memory.filters)
        self.assertEqual("ready", probe["status"])
        self.assertEqual(1, probe["matching_count"])

    def test_store_probe_does_not_mask_inventory_failures(self) -> None:
        probe = probe_memory_store(
            ProbeMemoryFailure(), user_id="u", project="demo"
        )
        self.assertEqual("error", probe["status"])
        self.assertIn("Qdrant inventory unavailable", probe["error"])


if __name__ == "__main__":
    unittest.main()
