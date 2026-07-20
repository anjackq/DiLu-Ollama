from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import dilu.runtime as runtime_api
from dilu.runtime import _scientific_runtime_binding as binding_module
from dilu.runtime._scientific_runtime_binding import (
    RuntimeLockBinding,
    VerifiedRuntimeLockBinding,
)
from tests.runtime_factorization_support import identity
from tests.scientific_transport_support import make_capabilities
from tests.test_scientific_driver_action_resolution import _scientific_config


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _runtime_lock_mapping() -> dict[str, str]:
    return RuntimeLockBinding.from_runtime(
        harness_config=_scientific_config(),
        identity=identity(),
        capabilities=make_capabilities(),
    ).to_dict()


def _write_artifacts(root: Path) -> tuple[Path, Path, bytes, bytes]:
    runtime_bytes = json.dumps(
        _runtime_lock_mapping(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    authorization_bytes = json.dumps(
        {
            "artifact_type": "runtime_lock_authorization_v1",
            "runtime_lock_sha256": _sha256(runtime_bytes),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    runtime_path = root / "RUNTIME_PROTOCOL_LOCK.json"
    authorization_path = root / "PROTOCOL_FROZEN.json"
    runtime_path.write_bytes(runtime_bytes)
    authorization_path.write_bytes(authorization_bytes)
    return runtime_path, authorization_path, runtime_bytes, authorization_bytes


class RuntimeLockLoaderTests(unittest.TestCase):
    def test_loader_verifies_independent_authorization_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_artifacts(Path(tmp))

            loaded = binding_module.load_verified_runtime_lock_binding(
                runtime_lock_path=paths[0],
                authorization_path=paths[1],
            )

            self.assertIsInstance(loaded, VerifiedRuntimeLockBinding)
            self.assertEqual(loaded.to_dict(), _runtime_lock_mapping())
            self.assertEqual(loaded.source_artifact_sha256, _sha256(paths[2]))
            self.assertEqual(
                loaded.authorization_artifact_sha256,
                _sha256(paths[3]),
            )

    def test_loader_hashes_exact_runtime_lock_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_artifacts(Path(tmp))
            paths[0].write_bytes(paths[2] + b"\n")

            with self.assertRaisesRegex(
                ValueError,
                "source artifact hash did not verify",
            ):
                binding_module.load_verified_runtime_lock_binding(
                    runtime_lock_path=paths[0],
                    authorization_path=paths[1],
                )

    def test_loader_rejects_duplicate_authorization_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_artifacts(Path(tmp))
            digest = _sha256(paths[2])
            paths[1].write_text(
                '{"artifact_type":"runtime_lock_authorization_v1",'
                f'"runtime_lock_sha256":"{digest}",'
                f'"runtime_lock_sha256":"{digest}"}}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
                binding_module.load_verified_runtime_lock_binding(
                    runtime_lock_path=paths[0],
                    authorization_path=paths[1],
                )

    def test_loader_requires_distinct_artifact_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_artifacts(Path(tmp))

            with self.assertRaisesRegex(ValueError, "must be distinct"):
                binding_module.load_verified_runtime_lock_binding(
                    runtime_lock_path=paths[0],
                    authorization_path=paths[0],
                )

    def test_verified_binding_cannot_be_directly_constructed(self) -> None:
        mapping = _runtime_lock_mapping()

        with self.assertRaises(TypeError):
            VerifiedRuntimeLockBinding(
                **mapping,
                source_artifact_sha256="sha256:" + "a" * 64,
                authorization_artifact_sha256="sha256:" + "b" * 64,
                binding_sha256="sha256:" + "c" * 64,
            )

    def test_verified_binding_disables_live_and_mapping_constructors(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be loaded"):
            VerifiedRuntimeLockBinding.from_mapping(_runtime_lock_mapping())
        with self.assertRaisesRegex(TypeError, "must be loaded"):
            VerifiedRuntimeLockBinding.from_runtime(
                harness_config=_scientific_config(),
                identity=identity(),
                capabilities=make_capabilities(),
            )

    def test_loader_proof_is_bound_to_loaded_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_artifacts(Path(tmp))
            loaded = binding_module.load_verified_runtime_lock_binding(
                runtime_lock_path=paths[0],
                authorization_path=paths[1],
            )
            drifted_mapping = loaded.to_dict()
            drifted_mapping["model_digest"] = "sha256:" + "f" * 64

            with self.assertRaisesRegex(ValueError, "load proof drifted"):
                dataclasses.replace(
                    loaded,
                    model_digest=drifted_mapping["model_digest"],
                    binding_sha256=binding_module._mapping_sha256(drifted_mapping),
                )

    def test_loader_is_a_stable_public_runtime_api(self) -> None:
        self.assertIn("load_verified_runtime_lock_binding", runtime_api.__all__)
        self.assertIs(
            runtime_api.load_verified_runtime_lock_binding,
            binding_module.load_verified_runtime_lock_binding,
        )


if __name__ == "__main__":
    unittest.main()
