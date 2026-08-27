"""Shared fixtures for the grounded-decoding V8 test files.

Not a ``test_*.py`` module itself, so ``unittest discover`` does not collect
it directly; both ``test_grounded_decoding_schedule.py`` and
``test_grounded_decoding_condition_space.py`` import from here to avoid
duplicating the frozen digests, manifest paths, and the mocked-git snapshot
helper.
"""

from __future__ import annotations

import json
from pathlib import Path

from dilu.runtime.ollama_transport import OllamaModelIdentity

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs" / "iclr2027" / "grounded_decoding_v8.yaml"
V5_MANIFEST_PATH = ROOT / "configs" / "iclr2027" / "minimal_factorial.yaml"
V7_MANIFEST_PATH = ROOT / "configs" / "iclr2027" / "model_breadth_factorial_v7.yaml"
V5_EPISODES_PATH = (
    ROOT / "results" / "iclr2027_minimal_factorial_v5" / "llm_campaign" / "episodes.jsonl"
)
V7_EPISODES_PATH = (
    ROOT
    / "results"
    / "iclr2027_model_breadth_factorial_v7"
    / "llm_campaign"
    / "episodes.jsonl"
)
V5_CAMPAIGN_MANIFEST_PATH = (
    ROOT / "results" / "iclr2027_minimal_factorial_v5" / "llm_campaign" / "campaign_manifest.json"
)
V7_CAMPAIGN_MANIFEST_PATH = (
    ROOT
    / "results"
    / "iclr2027_model_breadth_factorial_v7"
    / "llm_campaign"
    / "campaign_manifest.json"
)

# Measured 2026-08-27 from campaign_manifest.json's runtime_snapshot.simulator_versions
# in both frozen campaigns (identical in V5 and V7).
FROZEN_SIMULATOR_VERSIONS = {
    "gymnasium": "0.29.1",
    "highway-env": "1.8.2",
    "numpy": "1.26.4",
}

# Measured 2026-08-27 from the real frozen artifacts; comparator anchors
# (one model digest per model per campaign, no intra-campaign drift).
FROZEN_DIGESTS = {
    "qwen_06b": "sha256:7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435",
    "llama_1b": "sha256:baf6a787fdffd633537aa2eb51cfd54cb93ff08e28040095462bb63daf552878",
    "llama_3b": "sha256:a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72",
    "gemma_4b": "sha256:a2af6cc3eb7fa8be8504abaf9b04e88f17a119ec3f04a3addf55f92841195f5a",
    "qwen_8b": "sha256:500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41",
}
MODEL_TAGS = {
    "qwen_06b": "qwen3:0.6b",
    "llama_1b": "llama3.2:1b",
    "llama_3b": "llama3.2:3b",
    "gemma_4b": "gemma3:4b",
    "qwen_8b": "qwen3:8b",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frozen_bindings() -> dict[str, OllamaModelIdentity]:
    return {
        slot: OllamaModelIdentity(model_tag=MODEL_TAGS[slot], model_digest=digest)
        for slot, digest in FROZEN_DIGESTS.items()
    }


def matching_snapshot(snapshot):
    """Return ``snapshot`` with simulator_versions forced to the frozen pin.

    The live conda env on this machine does not necessarily match the
    pinned ``gymnasium==0.29.1``/``highway-env==1.8.2``/``numpy==1.26.4``
    recorded in the frozen V5/V7 manifests, so tests that need the
    comparator contract's simulator-version gate to *pass* build a snapshot
    with that field overridden rather than asserting anything about what
    happens to be installed locally. Everything else in the payload (case
    set fingerprint, code revision, ...) is left exactly as built.
    """
    from dilu.runtime.minimal_factorial_schedule import RuntimeSnapshot

    return RuntimeSnapshot.create(
        {**snapshot.payload, "simulator_versions": dict(FROZEN_SIMULATOR_VERSIONS)}
    )


def write_fake_campaign_manifest(path: Path, *, campaign_id: str, simulator_versions: dict) -> None:
    """Write a minimal but shape-valid campaign_manifest.json test fixture."""
    payload = {
        "manifest": {"campaign_id": campaign_id},
        "runtime_snapshot": {"simulator_versions": simulator_versions},
        "runtime_snapshot_sha256": "sha256:" + "0" * 64,
        "schedule": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def fake_git(command, **_kwargs):
    """A subprocess.run stand-in: clean tree, tracked sources, fixed HEAD."""
    action = command[1]
    stdout = (
        "cb965349db000dbdcb5201914bfbc95ca35bb62e\n"
        if action == "rev-parse"
        else command[-1] + "\n"
        if action == "ls-files"
        else ""
    )
    return type("Completed", (), {"stdout": stdout, "returncode": 0})()


__all__ = [
    "FROZEN_DIGESTS",
    "FROZEN_SIMULATOR_VERSIONS",
    "MANIFEST_PATH",
    "MODEL_TAGS",
    "ROOT",
    "V5_CAMPAIGN_MANIFEST_PATH",
    "V5_EPISODES_PATH",
    "V5_MANIFEST_PATH",
    "V7_CAMPAIGN_MANIFEST_PATH",
    "V7_EPISODES_PATH",
    "V7_MANIFEST_PATH",
    "fake_git",
    "frozen_bindings",
    "matching_snapshot",
    "read_jsonl",
    "write_fake_campaign_manifest",
]
