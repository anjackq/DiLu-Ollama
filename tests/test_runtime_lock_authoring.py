from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

from dilu.runtime._minimal_factorial_manifest import RuntimeSnapshot
from dilu.runtime.minimal_factorial_schedule import (
    case_fingerprint,
    load_experiment_manifest,
)
from dilu.runtime.runtime_lock_authoring import S1AuthoringResult
from dilu.runtime.scientific_transport_types import canonical_action_text_schema

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "iclr2027" / "minimal_factorial.yaml"
REVISION = "4" * 40
DIGESTS = {
    "qwen3:0.6b": "sha256:" + "a" * 64,
    "llama3.2:1b": "sha256:" + "b" * 64,
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def fake_snapshot() -> RuntimeSnapshot:
    manifest = load_experiment_manifest(MANIFEST_PATH)
    cases = json.loads((ROOT / manifest.case_path).read_text(encoding="utf-8"))
    return RuntimeSnapshot.create(
        {
            "case_set_fingerprint": case_fingerprint(cases),
            "code_revision": REVISION,
            "source_sha256": {"frozen": "1" * 64},
            "environment_config": {"name": "fake"},
            "scoring_fingerprint": "2" * 64,
            "trace_schema_sha256": "3" * 64,
        }
    )


class FakeResponse:
    def __init__(
        self,
        model: str,
        *,
        content: str = "Response to user:#### 3",
        status: int = 200,
        history: tuple[object, ...] = (),
        malformed: bool = False,
        url: str = "http://localhost:11434/api/chat",
    ) -> None:
        self.status_code = status
        self.history = history
        self.url = url
        self._malformed = malformed
        self._payload: dict[str, object] = {
            "model": model,
            "message": {"role": "assistant", "content": content, "thinking": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 19,
            "eval_count": 7,
            "total_duration": 120_000_000,
            "load_duration": 10_000_000,
            "prompt_eval_duration": 40_000_000,
            "eval_duration": 60_000_000,
        }
        self.text = (
            "{malformed" if malformed else canonical_bytes(self._payload).decode()
        )

    def json(self) -> dict[str, object]:
        if self._malformed:
            raise ValueError("malformed response")
        return self._payload

    def raise_for_status(self) -> None:
        if not 200 <= self.status_code < 300:
            raise ValueError(f"status={self.status_code}")


class NativeFakes:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.post_calls: list[dict[str, object]] = []

    def get(self, _url: str, *, timeout: float, allow_redirects: bool) -> FakeResponse:
        del timeout
        if allow_redirects:
            raise AssertionError("identity redirects were enabled")
        get_count = sum(event.startswith("get:") for event in self.events)
        model = "qwen3:0.6b" if get_count < 2 else "llama3.2:1b"
        self.events.append(f"get:{model}")
        response = FakeResponse(model)
        response._payload = {
            "models": [
                {"name": tag, "model": tag, "digest": digest}
                for tag, digest in DIGESTS.items()
            ]
        }
        response.text = canonical_bytes(response._payload).decode()
        response.url = "http://localhost:11434/api/tags"
        return response

    def post(
        self,
        url: str,
        *,
        data: bytes,
        headers: dict[str, str],
        timeout: float,
        allow_redirects: bool,
    ) -> FakeResponse:
        if allow_redirects is not False:
            raise AssertionError("generation redirects were enabled")
        payload: dict[str, object] = json.loads(data)
        model = payload["model"]
        if not isinstance(model, str):
            raise AssertionError("model must be a string")
        self.events.append(f"post:{model}")
        self.post_calls.append(
            {
                "url": url,
                "data": data,
                "headers": headers,
                "timeout": timeout,
                "allow_redirects": allow_redirects,
                "payload": payload,
            }
        )
        content = "Response to user:#### 3"
        if "format" in payload:
            content = json.dumps(content)
        return FakeResponse(model, content=content, url=url)


def run_authoring(
    output_root: Path,
    fakes: NativeFakes,
    *,
    snapshots: list[RuntimeSnapshot] | None = None,
    publication_hook: Callable[[Path], None] | None = None,
    publication_boundary_hook: Callable[[int, Path], None] | None = None,
) -> S1AuthoringResult:
    from dilu.runtime.runtime_lock_authoring import author_verified_runtime_locks

    values = iter(snapshots or [fake_snapshot(), fake_snapshot()])

    def next_snapshot(*_args: object) -> RuntimeSnapshot:
        fakes.events.append("snapshot")
        return next(values)

    with (
        mock.patch(
            "dilu.runtime.runtime_lock_authoring.build_runtime_snapshot",
            side_effect=next_snapshot,
        ),
        mock.patch("requests.get", side_effect=AssertionError("real GET reached")),
        mock.patch("requests.post", side_effect=AssertionError("real POST reached")),
    ):
        optional_kwargs = (
            {}
            if publication_boundary_hook is None
            else {"publication_boundary_hook": publication_boundary_hook}
        )
        return author_verified_runtime_locks(
            ROOT,
            output_root=output_root,
            get=fakes.get,
            post=fakes.post,
            publication_hook=publication_hook,
            **optional_kwargs,
        )


class RuntimeLockAuthoringTests(unittest.TestCase):
    def test_probe_and_authored_locks_are_exact_and_verified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            fakes = NativeFakes()
            hook_observations: list[tuple[bool, bool, bool]] = []

            result = run_authoring(
                output,
                fakes,
                publication_hook=lambda root: hook_observations.append(
                    (
                        (root / "s1" / "model_preflight.json").is_file(),
                        (root / "smoke" / "campaign_manifest.json").is_file(),
                        (root / "llm_campaign" / "union_schedule.json").is_file(),
                    )
                ),
            )

            self.assertEqual(len(fakes.post_calls), 6)
            self.assertEqual(
                fakes.events,
                [
                    "snapshot",
                    "get:qwen3:0.6b",
                    *(["post:qwen3:0.6b"] * 3),
                    "get:qwen3:0.6b",
                    "get:llama3.2:1b",
                    *(["post:llama3.2:1b"] * 3),
                    "get:llama3.2:1b",
                    "snapshot",
                ],
            )
            for offset in (0, 3):
                first, repeat, schema = fakes.post_calls[offset : offset + 3]
                self.assertEqual(first["data"], repeat["data"])
                self.assertEqual(first["payload"], repeat["payload"])
                self.assertNotIn("format", first["payload"])
                self.assertEqual(
                    schema["payload"]["format"], canonical_action_text_schema()
                )
            preflight_bytes = (output / "s1" / "model_preflight.json").read_bytes()
            preflight = json.loads(preflight_bytes)
            self.assertEqual(len(preflight["records"]), 6)
            self.assertEqual(
                result.preflight_sha256,
                "sha256:" + hashlib.sha256(preflight_bytes).hexdigest(),
            )
            self.assertTrue(
                all(record["canonical_action"] == 3 for record in preflight["records"])
            )
            required_evidence = {
                "request",
                "payload",
                "payload_sha256",
                "request_body",
                "response_body",
                "raw_response",
                "canonical_action",
                "stop_reason",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "backend_timing",
                "identity_before",
                "identity_after",
            }
            self.assertTrue(
                all(required_evidence <= set(record) for record in preflight["records"])
            )
            self.assertEqual(len(result.lock_artifacts), 16)
            self.assertEqual(
                {
                    (item.model_slot, item.condition_id)
                    for item in result.lock_artifacts
                },
                {
                    (slot, f"c{condition:03b}")
                    for slot in ("qwen_06b", "llama_1b")
                    for condition in range(8)
                },
            )
            for item in result.lock_artifacts:
                runtime_bytes = item.runtime_lock_path.read_bytes()
                authorization = json.loads(item.authorization_path.read_bytes())
                self.assertEqual(
                    set(authorization),
                    {"artifact_type", "runtime_lock_sha256"},
                )
                self.assertEqual(
                    authorization,
                    {
                        "artifact_type": "runtime_lock_authorization_v1",
                        "runtime_lock_sha256": (
                            "sha256:" + hashlib.sha256(runtime_bytes).hexdigest()
                        ),
                    },
                )
                self.assertEqual(
                    item.verified_binding.to_dict(), json.loads(runtime_bytes)
                )
            self.assertEqual(hook_observations, [(True, True, True)])

    def test_idempotent_rerun_and_different_existing_artifact_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            first = run_authoring(output, NativeFakes())
            second = run_authoring(output, NativeFakes())
            self.assertEqual(first.preflight_sha256, second.preflight_sha256)
            lock = output / "s1" / "locks" / "qwen_06b" / "c000"
            lock.joinpath("RUNTIME_PROTOCOL_LOCK.json").write_bytes(b"{}")

            with self.assertRaisesRegex(
                ValueError,
                "already exists|bytes drifted",
            ):
                run_authoring(output, NativeFakes())


if __name__ == "__main__":
    unittest.main()
