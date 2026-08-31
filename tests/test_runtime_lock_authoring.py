from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest import mock

from dilu.runtime._grounded_decoding_lock_authoring import (
    GROUNDED_CAPABILITY_PREFLIGHT_ARTIFACT_TYPE,
    GROUNDED_PROBE_ACTION_IDS,
)
from dilu.runtime._minimal_factorial_manifest import RuntimeSnapshot
from dilu.runtime._runtime_lock_response_evidence import derive_response_evidence
from dilu.runtime._scientific_runtime_binding import load_verified_runtime_lock_binding
from dilu.runtime.action_resolution import FIXED_IDLE_ACTION_ID
from dilu.runtime.harness_config import OutputEnforcement
from dilu.runtime.minimal_factorial_schedule import (
    case_fingerprint,
    load_experiment_manifest,
)
from dilu.runtime.runtime_lock_authoring import GroundedDecodingS1Result, S1AuthoringResult
from dilu.runtime.scientific_transport_types import (
    SCHEMA_MECHANISM_GROUNDED,
    canonical_action_text_schema,
)
from tests.grounded_decoding_schedule_support import FROZEN_DIGESTS
from tests.scientific_transport_support import make_request, success_payload

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "iclr2027" / "minimal_factorial.yaml"
REVISION = "4" * 40
DIGESTS = {
    "qwen3:0.6b": "sha256:" + "a" * 64,
    "llama3.2:1b": "sha256:" + "b" * 64,
}
FIXED_IDLE_ACTION_TEXT = f"Response to user:#### {FIXED_IDLE_ACTION_ID}"
PROMPT_ONLY_INVALID_TEXT = f"{FIXED_IDLE_ACTION_TEXT} /no_think"

# V8 model roster (matches configs/iclr2027/grounded_decoding_v8.yaml). Every
# model's live digest is set to match its real frozen V5/V7 comparator
# digest except qwen_8b, whose live digest is deliberately wrong so both the
# match and mismatch branches of comparator_digest_match get exercised.
V8_MODEL_SLOTS = ("qwen_06b", "llama_1b", "llama_3b", "gemma_4b", "qwen_8b")
V8_LIVE_DIGESTS = {
    "qwen3:0.6b": FROZEN_DIGESTS["qwen_06b"],
    "llama3.2:1b": FROZEN_DIGESTS["llama_1b"],
    "llama3.2:3b": FROZEN_DIGESTS["llama_3b"],
    "gemma3:4b": FROZEN_DIGESTS["gemma_4b"],
    "qwen3:8b": "sha256:" + "9" * 64,
}
GROUNDED_ACTION_TEXT = f"Response to user:#### {GROUNDED_PROBE_ACTION_IDS[0]}"
GROUNDED_ACTION_TEXT_OUT_OF_ENUM = "Response to user:#### 2"


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
        content: str = FIXED_IDLE_ACTION_TEXT,
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
        content = PROMPT_ONLY_INVALID_TEXT
        if "format" in payload:
            content = json.dumps(FIXED_IDLE_ACTION_TEXT)
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
                self.assertEqual(
                    [first["timeout"], repeat["timeout"], schema["timeout"]],
                    [120.0, 30.0, 30.0],
                )
                self.assertEqual(first["data"], repeat["data"])
                self.assertEqual(first["payload"], repeat["payload"])
                self.assertNotIn("format", first["payload"])
                self.assertEqual(
                    first["payload"]["messages"],
                    [
                        {
                            "role": "system",
                            "content": (
                                "Return exactly this text and nothing else: "
                                f"{FIXED_IDLE_ACTION_TEXT}"
                            ),
                        },
                        {
                            "role": "user",
                            "content": "Perform the response-format capability check.",
                        },
                    ],
                )
                self.assertEqual(
                    schema["payload"]["format"], canonical_action_text_schema()
                )
            preflight_bytes = (output / "s1" / "model_preflight.json").read_bytes()
            preflight = json.loads(preflight_bytes)
            self.assertEqual(
                preflight["artifact_type"],
                "ollama_native_capability_preflight_v2",
            )
            self.assertEqual(len(preflight["records"]), 6)
            self.assertEqual(
                result.preflight_sha256,
                "sha256:" + hashlib.sha256(preflight_bytes).hexdigest(),
            )
            prompt_resolution = {
                "raw_response": PROMPT_ONLY_INVALID_TEXT,
                "syntax_status": "invalid",
                "strict_action": None,
                "recovered_action": None,
                "recovery_stage": "none",
                "violation": "syntax_invalid",
                "action_available": "not_applicable",
                "fallback_action": FIXED_IDLE_ACTION_ID,
                "final_resolved_action": FIXED_IDLE_ACTION_ID,
                "used_fallback": True,
            }
            schema_resolution = {
                "raw_response": FIXED_IDLE_ACTION_TEXT,
                "syntax_status": "strict_valid",
                "strict_action": FIXED_IDLE_ACTION_ID,
                "recovered_action": None,
                "recovery_stage": "none",
                "violation": None,
                "action_available": "available",
                "fallback_action": None,
                "final_resolved_action": FIXED_IDLE_ACTION_ID,
                "used_fallback": False,
            }
            for offset in (0, 3):
                first, repeat, schema = preflight["records"][offset : offset + 3]
                self.assertEqual(first["raw_response"], PROMPT_ONLY_INVALID_TEXT)
                self.assertEqual(first["contract_text"], PROMPT_ONLY_INVALID_TEXT)
                self.assertEqual(first["action_resolution"], prompt_resolution)
                self.assertEqual(repeat["action_resolution"], prompt_resolution)
                self.assertEqual(
                    schema["raw_response"], json.dumps(FIXED_IDLE_ACTION_TEXT)
                )
                self.assertEqual(schema["contract_text"], FIXED_IDLE_ACTION_TEXT)
                self.assertEqual(schema["action_resolution"], schema_resolution)
            self.assertEqual(
                [record["request"]["timeout_sec"] for record in preflight["records"]],
                [120.0, 30.0, 30.0, 120.0, 30.0, 30.0],
            )
            self.assertEqual(
                load_experiment_manifest(MANIFEST_PATH).transport.timeout_sec, 30.0
            )
            required_evidence = {
                "request",
                "payload",
                "payload_sha256",
                "request_body",
                "response_body",
                "raw_response",
                "contract_text",
                "action_resolution",
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

    def test_v5_lock_and_preflight_bytes_match_pre_task5_golden_hashes(self) -> None:
        # Golden hashes captured by running this exact fake-transport flow at
        # commit 9f21221 (the Task 5 baseline, before ConditionSpec.from_condition_id
        # replaced the int(condition_id[1:], 2) parse in build_lock_plans) and
        # verified byte-identical to a second capture taken after all Task 5
        # edits (via `git stash` / `git stash pop`, diffed with zero deltas).
        # A change to any of these hashes means V5/V7 lock authoring drifted.
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            result = run_authoring(output, NativeFakes())
            self.assertEqual(
                result.preflight_sha256,
                "sha256:11679cedb78f4487d0855a8f4b19b585a2faebac525ae713650837eae2e09811",
            )
            golden = {
                "s1/locks/qwen_06b/c000/RUNTIME_PROTOCOL_LOCK.json": (
                    "ee1e5a67cba326cfc75a9eba8ced95c7a501bc0cc959c5c026f7f16a577cbe09"
                ),
                "s1/locks/qwen_06b/c111/RUNTIME_PROTOCOL_LOCK.json": (
                    "8cda005be1d26b3392a2d10fa5f69ec929e4584f674e94700a96c51162cdce8a"
                ),
                "s1/locks/llama_1b/c000/RUNTIME_PROTOCOL_LOCK.json": (
                    "29a353113a16ed8732c08d68345207e0b8eb8e48f0a8d683de55272289e47056"
                ),
                "s1/locks/llama_1b/c111/RUNTIME_PROTOCOL_LOCK.json": (
                    "71af0fa09b6a241433b4e87b1323116044acbcfa1923d5ff67a879e6810868a1"
                ),
                "smoke/campaign_manifest.json": (
                    "34c6846244556112f6349313b00c867f1ac712c05012d9e141daac975af9c334"
                ),
                "llm_campaign/union_schedule.json": (
                    "58eb17e4e03555b3642b08246f57a25090ff29b82cb9b116622bff36d58a13ac"
                ),
            }
            for relative, expected in golden.items():
                actual = hashlib.sha256((output / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, expected, f"{relative} drifted from its golden hash")


class V8NativeFakes:
    """Fake transport for the 5-model, 4-probe-per-model V8 S1 flow.

    Every model's live digest matches its real frozen V5/V7 comparator
    digest except qwen3:8b, whose digest is deliberately wrong; this
    exercises both branches of ``comparator_digest_match`` against the
    real, committed frozen campaign artifacts (read-only).
    """

    def __init__(self, *, grounded_content: str = GROUNDED_ACTION_TEXT) -> None:
        self.events: list[str] = []
        self.post_calls: list[dict[str, object]] = []
        self._grounded_content = grounded_content

    def get(self, _url: str, *, timeout: float, allow_redirects: bool) -> FakeResponse:
        del timeout
        if allow_redirects:
            raise AssertionError("identity redirects were enabled")
        self.events.append("get")
        response = FakeResponse(next(iter(V8_LIVE_DIGESTS)))
        response._payload = {
            "models": [
                {"name": tag, "model": tag, "digest": digest}
                for tag, digest in V8_LIVE_DIGESTS.items()
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
        fmt = payload.get("format")
        enum = fmt.get("enum") if isinstance(fmt, dict) else None
        if isinstance(enum, list) and len(enum) == len(GROUNDED_PROBE_ACTION_IDS):
            content = json.dumps(self._grounded_content)
        elif "format" in payload:
            content = json.dumps(FIXED_IDLE_ACTION_TEXT)
        else:
            content = PROMPT_ONLY_INVALID_TEXT
        return FakeResponse(model, content=content, url=url)


def run_v8_authoring(
    output_root: Path,
    fakes: V8NativeFakes,
    *,
    snapshots: list[RuntimeSnapshot] | None = None,
    publication_hook: Callable[[Path], None] | None = None,
) -> GroundedDecodingS1Result:
    from dilu.runtime.runtime_lock_authoring import author_verified_grounded_decoding_locks

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
        return author_verified_grounded_decoding_locks(
            ROOT,
            output_root=output_root,
            get=fakes.get,
            post=fakes.post,
            publication_hook=publication_hook,
        )


class GroundedDecodingLockAuthoringTests(unittest.TestCase):
    """S1 probe and lock authoring extended to the grounded (O2) mechanism."""

    def test_v8_probes_four_calls_per_model_and_authors_ten_lock_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            fakes = V8NativeFakes()

            result = run_v8_authoring(output, fakes)

            # 5 models x 4 calls (prompt, prompt-repeat, schema, grounded-schema).
            self.assertEqual(len(fakes.post_calls), 20)
            self.assertEqual(
                sorted({call["payload"]["model"] for call in fakes.post_calls}),
                sorted(V8_LIVE_DIGESTS),
            )

            preflight_bytes = (output / "s1" / "model_preflight.json").read_bytes()
            preflight = json.loads(preflight_bytes)
            self.assertEqual(
                preflight["artifact_type"], GROUNDED_CAPABILITY_PREFLIGHT_ARTIFACT_TYPE
            )
            self.assertEqual(len(preflight["records"]), 20)
            self.assertEqual(
                result.preflight_sha256,
                "sha256:" + hashlib.sha256(preflight_bytes).hexdigest(),
            )

            # -- comparator_digest_match: recorded, both branches exercised --
            self.assertEqual(
                preflight["comparator_digest_match"],
                {
                    "qwen_06b": True,
                    "llama_1b": True,
                    "llama_3b": True,
                    "gemma_4b": True,
                    "qwen_8b": False,
                },
            )
            self.assertEqual(dict(result.comparator_digest_match), preflight["comparator_digest_match"])

            # -- capabilities record the grounded schema mechanism -----------
            self.assertEqual(len(result.capabilities), 5)
            for capabilities in result.capabilities.values():
                self.assertEqual(capabilities.schema_mechanism, SCHEMA_MECHANISM_GROUNDED)
                self.assertTrue(capabilities.schema_verified)

            # -- the grounded probe record itself -----------------------------
            grounded_records = [
                record
                for record in preflight["records"]
                if record["request"]["output_enforcement"] == "backend_schema_grounded"
            ]
            self.assertEqual(len(grounded_records), 5)
            for record in grounded_records:
                self.assertEqual(
                    record["payload"]["format"]["enum"],
                    [f"Response to user:#### {i}" for i in GROUNDED_PROBE_ACTION_IDS],
                )
                self.assertEqual(record["contract_text"], GROUNDED_ACTION_TEXT)
                self.assertEqual(record["action_resolution"]["strict_action"], 1)
                self.assertEqual(record["action_resolution"]["syntax_status"], "strict_valid")
                self.assertFalse(record["action_resolution"]["used_fallback"])

            # -- 10 lock/authorization pairs, all loading through the loader --
            self.assertEqual(len(result.lock_artifacts), 10)
            self.assertEqual(
                {(item.model_slot, item.condition_id) for item in result.lock_artifacts},
                {(slot, cid) for slot in V8_MODEL_SLOTS for cid in ("c120", "c121")},
            )
            for item in result.lock_artifacts:
                loaded = load_verified_runtime_lock_binding(
                    runtime_lock_path=item.runtime_lock_path,
                    authorization_path=item.authorization_path,
                )
                self.assertEqual(loaded.to_dict(), item.verified_binding.to_dict())
                self.assertEqual(loaded.condition_id, item.condition_id)

            # -- no smoke/llm_campaign files: Task 5 scopes S1 to preflight+locks
            self.assertFalse((output / "smoke").exists())
            self.assertFalse((output / "llm_campaign").exists())

    def test_v8_grounded_probe_rejects_action_outside_restricted_enum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            fakes = V8NativeFakes(grounded_content=GROUNDED_ACTION_TEXT_OUT_OF_ENUM)

            with self.assertRaisesRegex(
                ValueError,
                "schema rejection",
            ):
                run_v8_authoring(output, fakes)

    def test_grounded_probe_preserves_transport_drift_diagnostic(self) -> None:
        request = make_request(
            OutputEnforcement.BACKEND_SCHEMA_GROUNDED,
            available_action_ids=GROUNDED_PROBE_ACTION_IDS,
        )
        payload = success_payload(json.dumps(GROUNDED_ACTION_TEXT))
        payload["done_reason"] = " stop "

        with self.assertRaisesRegex(ValueError, "malformed native response"):
            derive_response_evidence(request, 200, payload, json.dumps(payload))

    def test_v8_lock_authoring_leaves_no_destination_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            fakes = V8NativeFakes(grounded_content=GROUNDED_ACTION_TEXT_OUT_OF_ENUM)
            with self.assertRaises(ValueError):
                run_v8_authoring(output, fakes)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
