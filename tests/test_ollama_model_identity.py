import dataclasses
import subprocess
import unittest
from unittest.mock import patch

from dilu.runtime.ollama_transport import (
    OllamaModelIdentity,
    inspect_ollama_model_identity,
    ollama_tags_url,
    parse_ollama_model_identity,
)
from evaluate_models_ollama import _inspect_ollama_model


DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_A_RAW = "a" * 64


class _FakeTagsResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code != 200:
            raise RuntimeError(f"status={self.status_code}")

    def json(self) -> dict:
        return self.payload


class OllamaModelIdentityTests(unittest.TestCase):
    def test_native_bare_hex_digest_is_canonicalized(self) -> None:
        identity = parse_ollama_model_identity(
            {
                "models": [
                    {
                        "name": "qwen3:0.6b",
                        "model": "qwen3:0.6b",
                        "digest": DIGEST_A_RAW.upper(),
                    }
                ]
            },
            "qwen3:0.6b",
        )

        self.assertEqual(identity.model_digest, DIGEST_A)

    def test_internal_identity_rejects_bare_hex_digest(self) -> None:
        with self.assertRaises(ValueError):
            OllamaModelIdentity(
                model_tag="qwen3:0.6b",
                model_digest=DIGEST_A_RAW,
            )

    def test_exact_unique_tag_returns_frozen_full_digest_identity(self) -> None:
        identity = parse_ollama_model_identity(
            {
                "models": [
                    {"name": "qwen3:0.6b", "digest": DIGEST_A.upper()},
                    {"name": "qwen3:8b", "digest": DIGEST_B},
                ]
            },
            "qwen3:0.6b",
        )

        self.assertEqual(
            identity,
            OllamaModelIdentity(model_tag="qwen3:0.6b", model_digest=DIGEST_A),
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            identity.model_digest = DIGEST_B

    def test_alias_missing_duplicate_and_invalid_digest_fail_closed(self) -> None:
        invalid_payloads = (
            {"models": [{"name": "qwen3:latest", "digest": DIGEST_A}]},
            {"models": []},
            {
                "models": [
                    {"name": "qwen3:0.6b", "digest": DIGEST_A},
                    {"model": "qwen3:0.6b", "digest": DIGEST_A},
                ]
            },
            {"models": [{"name": "qwen3:0.6b", "digest": "sha256:abcd"}]},
            {"models": [{"name": "qwen3:0.6b", "digest": "a" * 63}]},
            {"models": [{"name": "qwen3:0.6b", "digest": "a" * 65}]},
            {"models": [{"name": "qwen3:0.6b", "digest": "g" * 64}]},
            {"models": [{"name": "qwen3:0.6b", "digest": "sha512:" + "a" * 64}]},
            {"models": [{"name": "qwen3:0.6b"}]},
            {
                "models": [
                    {
                        "name": "qwen3:0.6b",
                        "model": "qwen3:latest",
                        "digest": DIGEST_A,
                    }
                ]
            },
            {"models": "not-a-list"},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    parse_ollama_model_identity(payload, "qwen3:0.6b")

    def test_tags_url_normalizes_supported_ollama_bases(self) -> None:
        bases = (
            "http://127.0.0.1:11434",
            "http://127.0.0.1:11434/v1",
            "http://127.0.0.1:11434/api/chat",
            "http://127.0.0.1:11434/v1/chat/completions",
        )

        for base in bases:
            with self.subTest(base=base):
                self.assertEqual(
                    ollama_tags_url(base),
                    "http://127.0.0.1:11434/api/tags",
                )

    def test_inspector_uses_native_tags_endpoint_and_exact_identity(self) -> None:
        calls: list[tuple[str, float]] = []

        def fake_get(
            url: str, *, timeout: float, allow_redirects: bool
        ) -> _FakeTagsResponse:
            self.assertFalse(allow_redirects)
            calls.append((url, timeout))
            return _FakeTagsResponse(
                {"models": [{"name": "qwen3:0.6b", "digest": DIGEST_A}]}
            )

        identity = inspect_ollama_model_identity(
            "http://127.0.0.1:11434/v1",
            "qwen3:0.6b",
            get=fake_get,
            timeout_sec=7.5,
        )

        self.assertEqual(identity.model_digest, DIGEST_A)
        self.assertEqual(calls, [("http://127.0.0.1:11434/api/tags", 7.5)])

    def test_identity_lookup_rejects_direct_redirect_status(self) -> None:
        def redirected_get(
            url: str, *, timeout: float, allow_redirects: bool
        ) -> _FakeTagsResponse:
            del url, timeout
            self.assertFalse(allow_redirects)
            return _FakeTagsResponse(
                {"models": [{"name": "qwen3:0.6b", "digest": DIGEST_A}]},
                status_code=302,
            )

        with self.assertRaises(ValueError):
            inspect_ollama_model_identity(
                "http://127.0.0.1:11434",
                "qwen3:0.6b",
                get=redirected_get,
            )

    def test_evaluator_metadata_captures_verified_immutable_digest(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["ollama", "show", "qwen3:0.6b"],
            returncode=0,
            stdout="family: qwen3\nparameters: 600M\nquantization: Q4_K_M\n",
            stderr="",
        )
        with (
            patch(
                "evaluate_models_ollama.inspect_ollama_model_identity",
                return_value=OllamaModelIdentity("qwen3:0.6b", DIGEST_A),
            ),
            patch("evaluate_models_ollama.subprocess.run", return_value=completed),
        ):
            metadata = _inspect_ollama_model("qwen3:0.6b")

        self.assertEqual(metadata["model_digest"], DIGEST_A)
        self.assertTrue(metadata["model_identity_verified"])
        self.assertEqual(metadata["model_identity_source"], "ollama_native_api_tags")


if __name__ == "__main__":
    unittest.main()
