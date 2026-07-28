from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_runtime_lock_authoring import (
    ROOT,
    FakeResponse,
    NativeFakes,
    fake_snapshot,
    run_authoring,
)


class ConfigurableFakes(NativeFakes):
    def __init__(
        self,
        *,
        post_failure: str | None = None,
        repeat_action: str | None = None,
        identity_drift: bool = False,
        identity_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.post_failure = post_failure
        self.repeat_action = repeat_action
        self.identity_drift = identity_drift
        self.identity_fallback = identity_fallback

    def get(self, url: str, *, timeout: float, allow_redirects: bool) -> FakeResponse:
        response = super().get(url, timeout=timeout, allow_redirects=allow_redirects)
        if self.identity_drift and len(self.post_calls) == 3:
            response._payload["models"][0]["digest"] = "sha256:" + "f" * 64
        if self.identity_fallback:
            response.url = "http://localhost:11434/v1/models"
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
        response = super().post(
            url,
            data=data,
            headers=headers,
            timeout=timeout,
            allow_redirects=allow_redirects,
        )
        call_index = len(self.post_calls)
        if self.repeat_action and call_index == 2:
            response._payload["message"]["content"] = self.repeat_action
            response.text = "{}"
        if self.post_failure == "schema" and call_index == 3:
            response._payload["message"]["content"] = "not-json"
        elif self.post_failure == "redirect" and call_index == 1:
            response.history = (object(),)
        elif self.post_failure == "fallback" and call_index == 1:
            response.url = "http://localhost:11434/v1/chat/completions"
        elif self.post_failure == "non2xx" and call_index == 1:
            response.status_code = 503
        elif self.post_failure == "malformed" and call_index == 1:
            response._malformed = True
            response.text = "{bad"
        return response


class RuntimeLockAuthoringFailureTests(unittest.TestCase):
    def test_probe_failures_block_all_publication(self) -> None:
        failures = {
            "schema": "schema",
            "redirect": "redirect",
            "fallback": "endpoint",
            "non2xx": "direct 2xx",
            "malformed": "malformed",
        }
        for failure, message in failures.items():
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "results"
                with self.assertRaisesRegex(ValueError, message):
                    run_authoring(
                        output,
                        ConfigurableFakes(post_failure=failure),
                    )
                self.assertFalse(output.exists())

    def test_repeat_action_mismatch_and_identity_drift_block(self) -> None:
        scenarios = (
            (
                ConfigurableFakes(repeat_action="Response to user:#### 4"),
                "repeat",
            ),
            (ConfigurableFakes(identity_drift=True), "identity drift"),
            (ConfigurableFakes(identity_fallback=True), "identity endpoint"),
        )
        for fakes, message in scenarios:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "results"
                with self.assertRaisesRegex(ValueError, message):
                    run_authoring(output, fakes)
                self.assertFalse(output.exists())

    def test_source_revision_and_dirty_snapshot_failures_precede_posts(self) -> None:
        failures = (
            ValueError("Git revision is missing."),
            ValueError("Git revision is not an exact commit SHA."),
            ValueError("Runtime snapshot requires a clean Git worktree."),
        )
        from dilu.runtime.runtime_lock_authoring import author_verified_runtime_locks

        for failure in failures:
            with (
                self.subTest(failure=str(failure)),
                tempfile.TemporaryDirectory() as tmp,
            ):
                fakes = NativeFakes()
                with (
                    mock.patch(
                        "dilu.runtime.runtime_lock_authoring.build_runtime_snapshot",
                        side_effect=failure,
                    ),
                    mock.patch(
                        "requests.get",
                        side_effect=AssertionError("real GET reached"),
                    ),
                    mock.patch(
                        "requests.post",
                        side_effect=AssertionError("real POST reached"),
                    ),
                    self.assertRaisesRegex(ValueError, str(failure)),
                ):
                    author_verified_runtime_locks(
                        ROOT,
                        output_root=Path(tmp) / "results",
                        get=fakes.get,
                        post=fakes.post,
                    )
                self.assertEqual(fakes.post_calls, [])

    def test_runtime_snapshot_or_schema_drift_after_probe_blocks_publication(
        self,
    ) -> None:
        drifted = fake_snapshot()
        object.__setattr__(drifted, "sha256", "f" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            with self.assertRaisesRegex(ValueError, "snapshot drift"):
                run_authoring(
                    output,
                    NativeFakes(),
                    snapshots=[fake_snapshot(), drifted],
                )
            self.assertFalse(output.exists())

        from dilu.runtime import runtime_lock_authoring

        original = runtime_lock_authoring.canonical_action_text_schema
        schemas = [original(), {"type": "string", "enum": ["drift"]}]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            with (
                mock.patch.object(
                    runtime_lock_authoring,
                    "canonical_action_text_schema",
                    side_effect=schemas,
                ),
                self.assertRaisesRegex(ValueError, "schema drift"),
            ):
                run_authoring(output, NativeFakes())
            self.assertFalse(output.exists())

        from dilu.runtime import scientific_transport_types

        drifted_schema = {"type": "string", "enum": ["drift"]}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "results"
            with (
                mock.patch.object(
                    scientific_transport_types,
                    "canonical_action_text_schema",
                    return_value=drifted_schema,
                ),
                self.assertRaisesRegex(ValueError, "schema drift"),
            ):
                run_authoring(output, NativeFakes())
            self.assertFalse(output.exists())

    def test_claim_locks_are_loaded_without_verified_live_constructor(self) -> None:
        from dilu.runtime._scientific_runtime_binding import VerifiedRuntimeLockBinding

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(
                    VerifiedRuntimeLockBinding,
                    "from_runtime",
                    side_effect=AssertionError("unverified live claim constructor"),
                ),
                mock.patch.object(
                    VerifiedRuntimeLockBinding,
                    "from_mapping",
                    side_effect=AssertionError("unverified mapping claim constructor"),
                ),
            ):
                result = run_authoring(Path(tmp) / "results", NativeFakes())
            self.assertEqual(len(result.lock_artifacts), 16)


if __name__ == "__main__":
    unittest.main()
