"""Focused coverage for every fail-closed branch in the V8 comparator-digest
comparison path: ``_grounded_decoding_lock_authoring._load_frozen_model_digests``
and ``resolve_comparator_digest_matches``.

Deliberately uses synthetic in-memory/tempdir JSON fixtures rather than the
real frozen V5/V7 ``campaign_manifest.json`` files -- the point is to
construct malformed inputs that the real, always-consistent frozen files
can never exercise. Each test targets exactly one ``raise ValueError(...)``
branch; the surrounding fixture is otherwise a fully valid frozen campaign
manifest so earlier guards do not fire first.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from dilu.runtime._grounded_decoding_lock_authoring import (
    _load_frozen_model_digests,
    resolve_comparator_digest_matches,
)
from dilu.runtime.ollama_transport import OllamaModelIdentity

V5_CAMPAIGN_ID = "synthetic-v5-campaign"
V7_CAMPAIGN_ID = "synthetic-v7-campaign"


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _valid_manifest_payload(
    campaign_id: str = V5_CAMPAIGN_ID,
    rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if rows is None:
        rows = [
            {"campaign_id": campaign_id, "model_tag": "modelA:1b", "model_digest": "sha256:" + "a" * 64},
            {"campaign_id": campaign_id, "model_tag": "modelB:2b", "model_digest": "sha256:" + "b" * 64},
        ]
    return {"manifest": {"campaign_id": campaign_id}, "schedule": rows}


class LoadFrozenModelDigestsBranchTests(unittest.TestCase):
    """Every raise in _load_frozen_model_digests, one synthetic fixture each."""

    def test_unreadable_or_malformed_json_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign_manifest.json"
            path.write_bytes(b"{not valid json")
            with self.assertRaisesRegex(
                ValueError, "could not be read as a frozen campaign manifest"
            ):
                _load_frozen_model_digests(path, expected_campaign_id=V5_CAMPAIGN_ID)

    def test_missing_required_top_level_keys_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign_manifest.json"
            # Valid JSON, but not a mapping with {"manifest", "schedule"}.
            _write(path, {"manifest": {"campaign_id": V5_CAMPAIGN_ID}})
            with self.assertRaisesRegex(
                ValueError, "is not a valid frozen campaign manifest"
            ):
                _load_frozen_model_digests(path, expected_campaign_id=V5_CAMPAIGN_ID)

    def test_decoded_json_not_a_mapping_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign_manifest.json"
            _write(path, [1, 2, 3])
            with self.assertRaisesRegex(
                ValueError, "is not a valid frozen campaign manifest"
            ):
                _load_frozen_model_digests(path, expected_campaign_id=V5_CAMPAIGN_ID)

    def test_manifest_section_campaign_id_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign_manifest.json"
            _write(path, _valid_manifest_payload(campaign_id="a-different-campaign"))
            with self.assertRaisesRegex(
                ValueError, "campaign_id does not match the trusted frozen manifest"
            ):
                _load_frozen_model_digests(path, expected_campaign_id=V5_CAMPAIGN_ID)

    def test_manifest_section_not_a_mapping_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign_manifest.json"
            payload = _valid_manifest_payload()
            payload["manifest"] = "not-a-mapping"
            _write(path, payload)
            with self.assertRaisesRegex(
                ValueError, "campaign_id does not match the trusted frozen manifest"
            ):
                _load_frozen_model_digests(path, expected_campaign_id=V5_CAMPAIGN_ID)

    def test_empty_schedule_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign_manifest.json"
            _write(path, _valid_manifest_payload(rows=[]))
            with self.assertRaisesRegex(
                ValueError, "has an empty or invalid schedule"
            ):
                _load_frozen_model_digests(path, expected_campaign_id=V5_CAMPAIGN_ID)

    def test_schedule_not_a_list_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign_manifest.json"
            payload = _valid_manifest_payload()
            payload["schedule"] = "not-a-list"
            _write(path, payload)
            with self.assertRaisesRegex(
                ValueError, "has an empty or invalid schedule"
            ):
                _load_frozen_model_digests(path, expected_campaign_id=V5_CAMPAIGN_ID)

    def test_row_campaign_id_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign_manifest.json"
            rows = [
                {"campaign_id": V5_CAMPAIGN_ID, "model_tag": "modelA:1b", "model_digest": "sha256:" + "a" * 64},
                {"campaign_id": "some-other-campaign", "model_tag": "modelB:2b", "model_digest": "sha256:" + "b" * 64},
            ]
            _write(path, _valid_manifest_payload(rows=rows))
            with self.assertRaisesRegex(
                ValueError, "contains a row with a mismatched campaign_id"
            ):
                _load_frozen_model_digests(path, expected_campaign_id=V5_CAMPAIGN_ID)

    def test_row_not_a_mapping_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign_manifest.json"
            rows = [
                {"campaign_id": V5_CAMPAIGN_ID, "model_tag": "modelA:1b", "model_digest": "sha256:" + "a" * 64},
                "not-a-mapping-row",
            ]
            _write(path, _valid_manifest_payload(rows=rows))
            with self.assertRaisesRegex(
                ValueError, "contains a row with a mismatched campaign_id"
            ):
                _load_frozen_model_digests(path, expected_campaign_id=V5_CAMPAIGN_ID)

    def test_row_missing_model_tag_or_digest_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign_manifest.json"
            rows = [
                {"campaign_id": V5_CAMPAIGN_ID, "model_tag": "modelA:1b"},  # no model_digest
            ]
            _write(path, _valid_manifest_payload(rows=rows))
            with self.assertRaisesRegex(
                ValueError, "row is missing model_tag/model_digest"
            ):
                _load_frozen_model_digests(path, expected_campaign_id=V5_CAMPAIGN_ID)

    def test_row_digest_wrong_type_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign_manifest.json"
            rows = [
                {"campaign_id": V5_CAMPAIGN_ID, "model_tag": "modelA:1b", "model_digest": 12345},
            ]
            _write(path, _valid_manifest_payload(rows=rows))
            with self.assertRaisesRegex(
                ValueError, "row is missing model_tag/model_digest"
            ):
                _load_frozen_model_digests(path, expected_campaign_id=V5_CAMPAIGN_ID)

    def test_inconsistent_digest_for_same_tag_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign_manifest.json"
            rows = [
                {"campaign_id": V5_CAMPAIGN_ID, "model_tag": "modelA:1b", "model_digest": "sha256:" + "a" * 64},
                {"campaign_id": V5_CAMPAIGN_ID, "model_tag": "modelA:1b", "model_digest": "sha256:" + "c" * 64},
            ]
            _write(path, _valid_manifest_payload(rows=rows))
            with self.assertRaisesRegex(
                ValueError, "records inconsistent digests for modelA:1b"
            ):
                _load_frozen_model_digests(path, expected_campaign_id=V5_CAMPAIGN_ID)

    def test_valid_manifest_loads_the_expected_digest_map(self) -> None:
        # Positive control: proves the fixture builder itself produces a
        # loadable manifest (so the negative tests above are failing on the
        # intended guard, not on an unrelated fixture mistake).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "campaign_manifest.json"
            _write(path, _valid_manifest_payload())
            digests = _load_frozen_model_digests(path, expected_campaign_id=V5_CAMPAIGN_ID)
            self.assertEqual(
                digests,
                {"modelA:1b": "sha256:" + "a" * 64, "modelB:2b": "sha256:" + "b" * 64},
            )


@dataclass(frozen=True)
class _FakeModelSpec:
    slot: str
    tag: str


@dataclass(frozen=True)
class _FakeComparatorPaths:
    v5_manifest: str
    v7_manifest: str


class _FakeGroundedDecodingManifest:
    """Minimal duck-typed stand-in for GroundedDecodingManifest.

    resolve_comparator_digest_matches only reads .repo_root(),
    .comparators.{v5_manifest,v7_manifest}, and .models -- everything else
    on the real dataclass is irrelevant to this code path, so a full
    frozen-manifest-shaped fixture would only add noise.
    """

    def __init__(self, root: Path, v5_manifest: str, v7_manifest: str, models) -> None:
        self._root = root
        self.comparators = _FakeComparatorPaths(v5_manifest, v7_manifest)
        self.models = models

    def repo_root(self) -> Path:
        return self._root


class ResolveComparatorDigestMatchesBranchTests(unittest.TestCase):
    """Every raise in resolve_comparator_digest_matches, one fixture each."""

    def _write_frozen(self, directory: Path, name: str, campaign_id: str, rows) -> str:
        relative = f"{name}.json"
        _write(directory / relative, {"manifest": {"campaign_id": campaign_id}, "schedule": rows})
        return relative

    def test_v5_v7_overlapping_model_tags_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared_tag = "shared:1b"
            v5_relative = self._write_frozen(
                root,
                "v5",
                V5_CAMPAIGN_ID,
                [{"campaign_id": V5_CAMPAIGN_ID, "model_tag": shared_tag, "model_digest": "sha256:" + "a" * 64}],
            )
            v7_relative = self._write_frozen(
                root,
                "v7",
                V7_CAMPAIGN_ID,
                [{"campaign_id": V7_CAMPAIGN_ID, "model_tag": shared_tag, "model_digest": "sha256:" + "b" * 64}],
            )
            manifest = _FakeGroundedDecodingManifest(
                root, v5_relative, v7_relative, [_FakeModelSpec("slot_a", shared_tag)]
            )
            bindings = {"slot_a": OllamaModelIdentity(shared_tag, "sha256:" + "a" * 64)}
            with self.assertRaisesRegex(
                ValueError, r"V5 and V7 comparator campaigns share model tags: \['shared:1b'\]"
            ):
                resolve_comparator_digest_matches(
                    manifest,
                    bindings,
                    SimpleNamespace(campaign_id=V5_CAMPAIGN_ID),
                    SimpleNamespace(campaign_id=V7_CAMPAIGN_ID),
                )

    def test_missing_live_binding_for_a_manifest_model_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v5_relative = self._write_frozen(
                root,
                "v5",
                V5_CAMPAIGN_ID,
                [{"campaign_id": V5_CAMPAIGN_ID, "model_tag": "modelA:1b", "model_digest": "sha256:" + "a" * 64}],
            )
            v7_relative = self._write_frozen(
                root,
                "v7",
                V7_CAMPAIGN_ID,
                [{"campaign_id": V7_CAMPAIGN_ID, "model_tag": "modelC:3b", "model_digest": "sha256:" + "d" * 64}],
            )
            manifest = _FakeGroundedDecodingManifest(
                root, v5_relative, v7_relative, [_FakeModelSpec("slot_a", "modelA:1b")]
            )
            with self.assertRaisesRegex(
                ValueError, "Missing or mismatched live binding for slot_a"
            ):
                resolve_comparator_digest_matches(
                    manifest,
                    {},  # no binding recorded for slot_a
                    SimpleNamespace(campaign_id=V5_CAMPAIGN_ID),
                    SimpleNamespace(campaign_id=V7_CAMPAIGN_ID),
                )

    def test_mismatched_live_binding_tag_for_a_manifest_model_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v5_relative = self._write_frozen(
                root,
                "v5",
                V5_CAMPAIGN_ID,
                [{"campaign_id": V5_CAMPAIGN_ID, "model_tag": "modelA:1b", "model_digest": "sha256:" + "a" * 64}],
            )
            v7_relative = self._write_frozen(
                root,
                "v7",
                V7_CAMPAIGN_ID,
                [{"campaign_id": V7_CAMPAIGN_ID, "model_tag": "modelC:3b", "model_digest": "sha256:" + "d" * 64}],
            )
            manifest = _FakeGroundedDecodingManifest(
                root, v5_relative, v7_relative, [_FakeModelSpec("slot_a", "modelA:1b")]
            )
            bindings = {
                "slot_a": OllamaModelIdentity("a-different-tag:1b", "sha256:" + "a" * 64)
            }
            with self.assertRaisesRegex(
                ValueError, "Missing or mismatched live binding for slot_a"
            ):
                resolve_comparator_digest_matches(
                    manifest,
                    bindings,
                    SimpleNamespace(campaign_id=V5_CAMPAIGN_ID),
                    SimpleNamespace(campaign_id=V7_CAMPAIGN_ID),
                )

    def test_no_frozen_digest_recorded_for_a_manifest_model_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Neither frozen file mentions "modelZ:9b" at all.
            v5_relative = self._write_frozen(
                root,
                "v5",
                V5_CAMPAIGN_ID,
                [{"campaign_id": V5_CAMPAIGN_ID, "model_tag": "modelA:1b", "model_digest": "sha256:" + "a" * 64}],
            )
            v7_relative = self._write_frozen(
                root,
                "v7",
                V7_CAMPAIGN_ID,
                [{"campaign_id": V7_CAMPAIGN_ID, "model_tag": "modelB:2b", "model_digest": "sha256:" + "b" * 64}],
            )
            manifest = _FakeGroundedDecodingManifest(
                root, v5_relative, v7_relative, [_FakeModelSpec("slot_z", "modelZ:9b")]
            )
            bindings = {"slot_z": OllamaModelIdentity("modelZ:9b", "sha256:" + "c" * 64)}
            with self.assertRaisesRegex(
                ValueError, "No frozen comparator digest recorded for modelZ:9b"
            ):
                resolve_comparator_digest_matches(
                    manifest,
                    bindings,
                    SimpleNamespace(campaign_id=V5_CAMPAIGN_ID),
                    SimpleNamespace(campaign_id=V7_CAMPAIGN_ID),
                )

    def test_valid_inputs_report_match_and_mismatch_correctly(self) -> None:
        # Positive control: proves the fixture plumbing itself is sound, and
        # that the function's *result* (not just its guards) behaves as
        # documented -- both branches of the match/mismatch comparison.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v5_relative = self._write_frozen(
                root,
                "v5",
                V5_CAMPAIGN_ID,
                [{"campaign_id": V5_CAMPAIGN_ID, "model_tag": "modelA:1b", "model_digest": "sha256:" + "a" * 64}],
            )
            v7_relative = self._write_frozen(
                root,
                "v7",
                V7_CAMPAIGN_ID,
                [{"campaign_id": V7_CAMPAIGN_ID, "model_tag": "modelB:2b", "model_digest": "sha256:" + "b" * 64}],
            )
            manifest = _FakeGroundedDecodingManifest(
                root,
                v5_relative,
                v7_relative,
                [_FakeModelSpec("slot_a", "modelA:1b"), _FakeModelSpec("slot_b", "modelB:2b")],
            )
            bindings = {
                "slot_a": OllamaModelIdentity("modelA:1b", "sha256:" + "a" * 64),  # matches
                "slot_b": OllamaModelIdentity("modelB:2b", "sha256:" + "9" * 64),  # mismatches
            }
            matches = resolve_comparator_digest_matches(
                manifest,
                bindings,
                SimpleNamespace(campaign_id=V5_CAMPAIGN_ID),
                SimpleNamespace(campaign_id=V7_CAMPAIGN_ID),
            )
            self.assertEqual(matches, {"slot_a": True, "slot_b": False})


if __name__ == "__main__":
    unittest.main()
