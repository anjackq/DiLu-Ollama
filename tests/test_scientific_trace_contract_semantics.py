from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from dilu.runtime.scientific_trace import (
    ScientificTraceWriteError,
    ScientificTraceWriter,
)
from tests.scientific_trace_support import backend_schema_record
from tests.test_scientific_trace import _record


def _replace_final(payload: dict[str, Any], name: str, value: object) -> None:
    generation = payload["generation"]
    generation[name] = value
    generation["attempts"][-1][name] = value


class ScientificTraceContractSemanticTests(unittest.TestCase):
    def test_resume_rejects_prompt_contract_and_response_body_forgery(self) -> None:
        mutations = (
            lambda payload: _replace_final(
                payload,
                "thinking_output",
                "forged reasoning",
            ),
            lambda payload: _replace_final(payload, "raw_output", "forged"),
            lambda payload: _replace_final(payload, "response_body", "forged"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(index=index):
                self._assert_resume_rejects(_record().to_dict(), mutate)

    def test_resume_rejects_backend_schema_raw_contract_forgery(self) -> None:
        self._assert_resume_rejects(
            backend_schema_record().to_dict(),
            lambda payload: _replace_final(payload, "raw_output", "not-json"),
        )

    def _assert_resume_rejects(
        self,
        payload: dict[str, Any],
        mutate: Callable[[dict[str, Any]], None],
    ) -> None:
        mutate(payload)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "decision_traces.jsonl"
            path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ScientificTraceWriteError):
                ScientificTraceWriter(path, artifact_root=root, resume=True)


if __name__ == "__main__":
    unittest.main()
