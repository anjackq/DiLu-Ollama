"""Regression coverage for the two hardcoded 8-cell condition-space sites.

dilu/runtime/_minimal_factorial_provenance.py:52 and
dilu/runtime/_runtime_lock_existing.py:152 both used to hardcode the V5/V7
binary condition grid (``{f"c{index:03b}" for index in range(8)}``), which
would reject the ICLR 2027 grounded-decoding V8 condition ids ``c120``/
``c121``. Both were extended -- not replaced -- so V5/V7 behavior is
provably unchanged while V8's new ids are now accepted.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from tests.grounded_decoding_schedule_support import (
    FROZEN_DIGESTS,
    MANIFEST_PATH,
    ROOT,
    V5_MANIFEST_PATH,
    V7_MANIFEST_PATH,
    fake_git,
    frozen_bindings,
)


class ProvenanceConditionIdExtensionTests(unittest.TestCase):
    def test_frozen_condition_ids_extend_without_dropping_the_binary_grid(self) -> None:
        from dilu.runtime._minimal_factorial_provenance import _FROZEN_CONDITION_IDS

        binary = {f"c{index:03b}" for index in range(8)}
        self.assertTrue(binary.issubset(_FROZEN_CONDITION_IDS))
        self.assertEqual(_FROZEN_CONDITION_IDS - binary, {"c120", "c121"})

    def test_v5_frozen_schedule_still_validates_unchanged(self) -> None:
        from dilu.runtime._minimal_factorial_manifest import validate_schedule
        from dilu.runtime.minimal_factorial_schedule import (
            build_runtime_snapshot,
            build_union_schedule,
            load_experiment_manifest,
        )

        manifest = load_experiment_manifest(V5_MANIFEST_PATH)
        cases = json.loads((ROOT / manifest.case_path).read_text())
        digests = {
            "qwen_06b": FROZEN_DIGESTS["qwen_06b"],
            "llama_1b": FROZEN_DIGESTS["llama_1b"],
        }
        with patch("dilu.runtime._minimal_factorial_manifest.subprocess.run", fake_git):
            snapshot = build_runtime_snapshot(manifest, cases)
        schedule = build_union_schedule(manifest, cases, digests, runtime_snapshot=snapshot)
        validate_schedule(manifest, snapshot, schedule, cases)  # must not raise

    def test_v8_c120_row_validates_through_the_same_shared_validator(self) -> None:
        from dilu.runtime._minimal_factorial_provenance import validate_episode
        from dilu.runtime.grounded_decoding_schedule import (
            build_runtime_snapshot,
            build_v8_schedule,
            load_grounded_decoding_manifest,
        )

        manifest = load_grounded_decoding_manifest(MANIFEST_PATH)
        cases = json.loads((ROOT / manifest.case_path).read_text())
        with patch("dilu.runtime._minimal_factorial_manifest.subprocess.run", fake_git):
            snapshot = build_runtime_snapshot(manifest, cases)
        schedule = build_v8_schedule(
            manifest, cases, frozen_bindings(), runtime_snapshot=snapshot
        )

        models = {model.slot: model.tag for model in manifest.models}
        case_index = {case["case_id"]: case for case in cases["cases"]}
        fingerprint = snapshot.payload["case_set_fingerprint"]
        revision = snapshot.payload["code_revision"]
        row = schedule.stage1[0]
        self.assertEqual(row.condition_id, "c120")
        validate_episode(row, manifest, models, fingerprint, revision, case_index)  # ok

    def test_valid_but_unregistered_condition_is_still_rejected(self) -> None:
        # c020 (P0 O2 E0) is a real, internally-consistent ConditionSpec but
        # is not part of V8's registered grid; the extension must still
        # reject it, proving the guard is a real allowlist, not "anything
        # that parses."
        from dilu.runtime._harness_config_support import (
            ConditionSpec,
            ExecutionMode,
            OutputEnforcement,
            PolicyContent,
        )
        from dilu.runtime._minimal_factorial_provenance import validate_episode
        from dilu.runtime.grounded_decoding_schedule import (
            build_runtime_snapshot,
            build_v8_schedule,
            load_grounded_decoding_manifest,
        )

        manifest = load_grounded_decoding_manifest(MANIFEST_PATH)
        cases = json.loads((ROOT / manifest.case_path).read_text())
        with patch("dilu.runtime._minimal_factorial_manifest.subprocess.run", fake_git):
            snapshot = build_runtime_snapshot(manifest, cases)
        schedule = build_v8_schedule(
            manifest, cases, frozen_bindings(), runtime_snapshot=snapshot
        )

        models = {model.slot: model.tag for model in manifest.models}
        case_index = {case["case_id"]: case for case in cases["cases"]}
        fingerprint = snapshot.payload["case_set_fingerprint"]
        revision = snapshot.payload["code_revision"]
        row = schedule.stage1[0]

        p0_o2 = ConditionSpec(
            PolicyContent.HISTORICAL_DILU_2024,
            OutputEnforcement.BACKEND_SCHEMA_GROUNDED,
            ExecutionMode.UNSHIELDED_OPERATIONAL,
        )
        self.assertEqual(p0_o2.condition_id(), "c020")
        bad = replace(row, condition=replace(row.condition, condition=p0_o2), condition_id="c020")
        with self.assertRaises(ValueError):
            validate_episode(bad, manifest, models, fingerprint, revision, case_index)


class RuntimeLockPathExtensionTests(unittest.TestCase):
    def test_lock_relative_paths_preserves_binary_layout_and_accepts_grounded_ids(
        self,
    ) -> None:
        from dilu.runtime._runtime_lock_existing import (
            _expected_relative_paths,
            _lock_relative_paths,
        )
        from dilu.runtime.minimal_factorial_schedule import load_experiment_manifest

        v7_manifest = load_experiment_manifest(V7_MANIFEST_PATH)
        original = _expected_relative_paths(v7_manifest)
        via_helper = _lock_relative_paths(
            v7_manifest, tuple(f"c{index:03b}" for index in range(8))
        )
        self.assertEqual(original, via_helper)  # zero behavior change for V5/V7

        grounded_paths = _lock_relative_paths(v7_manifest, ("c120", "c121"))
        grounded_dirs = {path.parts[3] for path in grounded_paths if len(path.parts) > 3}
        self.assertEqual(grounded_dirs, {"c120", "c121"})
        self.assertEqual(len(grounded_paths), 4 + len(v7_manifest.models) * 2 * 2)


if __name__ == "__main__":
    unittest.main()
