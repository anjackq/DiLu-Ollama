import hashlib
import tempfile
import unittest
from pathlib import Path

import yaml

from dilu.driver_agent.driverAgent import DriverAgent
from dilu.driver_agent.prompt_modules import (
    ORIGINAL_DILU_PROMPT_BYTES,
    ORIGINAL_DILU_PROMPT_SHA256,
    PromptArtifact,
    build_policy_prompt,
    build_prompt_artifact,
    build_system_prompt,
    load_original_dilu_prompt,
)
from dilu.runtime.harness_config import (
    ConditionSpec,
    ExecutionMode,
    FallbackPolicy,
    HarnessConfig,
    OutputEnforcement,
    ParserMode,
    PolicyContent,
    ResolverMode,
    RetryPolicy,
    ShieldConfig,
    ThinkMode,
    TraceLevel,
    TransportConfig,
    TransportProfile,
)
from scripts.extract_original_dilu_prompt import (
    ExtractionSpec,
    extract_original_dilu_prompt,
    write_extracted_artifacts,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = REPO_ROOT / "dilu" / "driver_agent" / "prompts" / "original_dilu_2024.txt"
PROVENANCE_PATH = REPO_ROOT / "provenance" / "original_dilu_2024_prompt.yaml"
REVISION = "1eed4ed"
FULL_REVISION = "1eed4ed0bd2e483c2a604adc63f4a21c445dba06"
BLOB = "91888022745e4edbb9dff5e0528f5d6bf3498713"


def _harness_config(
    policy_content: PolicyContent = PolicyContent.HISTORICAL_DILU_2024,
    output_enforcement: OutputEnforcement = OutputEnforcement.PROMPT_ONLY,
) -> HarnessConfig:
    return HarnessConfig(
        condition=ConditionSpec(
            policy_content=policy_content,
            output_enforcement=output_enforcement,
            execution_mode=ExecutionMode.UNSHIELDED_OPERATIONAL,
        ),
        parser_mode=ParserMode.STRICT_ONLY,
        resolver_mode=ResolverMode.DISABLED,
        fallback_policy=FallbackPolicy.FIXED_IDLE,
        shield=ShieldConfig.implementation_defaults(),
        transport=TransportConfig(
            profile=TransportProfile.OLLAMA_NATIVE_CHAT,
            think_mode=ThinkMode.NO_THINK,
            temperature=0.0,
            context_tokens=4096,
            max_output_tokens=64,
            timeout_sec=60.0,
            generation_seed_master=20270713,
            allow_transport_fallback=False,
            adaptive_timeout=False,
        ),
        retry_policy=RetryPolicy(
            max_transport_unavailable_retries=1,
            retry_cooldown_sec=10.0,
            retry_on_timeout=False,
            retry_on_empty_output=False,
            retry_on_schema_rejection=False,
        ),
        trace_level=TraceLevel.MANDATORY_SCIENTIFIC,
    )


class PromptProvenanceTests(unittest.TestCase):
    def test_original_dilu_prompt_hash(self) -> None:
        content = PROMPT_PATH.read_bytes()

        self.assertEqual(len(content), ORIGINAL_DILU_PROMPT_BYTES)
        self.assertEqual(
            hashlib.sha256(content).hexdigest(), ORIGINAL_DILU_PROMPT_SHA256
        )
        self.assertEqual(
            load_original_dilu_prompt(PROMPT_PATH).encode("utf-8"), content
        )
        self.assertTrue(content.endswith(b"every step.\n"))

    def test_provenance_record_binds_revision_blob_scope_and_hash(self) -> None:
        provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))

        self.assertEqual(provenance["revision"], FULL_REVISION)
        self.assertEqual(provenance["git_blob"], BLOB)
        self.assertEqual(provenance["normalized_bytes"], ORIGINAL_DILU_PROMPT_BYTES)
        self.assertEqual(provenance["sha256"], ORIGINAL_DILU_PROMPT_SHA256)
        self.assertEqual(provenance["artifact_scope"], "system_message_only")
        self.assertFalse(provenance["claims_exact_historical_runtime"])

    def test_git_extractor_reproduces_artifact_and_fails_closed(self) -> None:
        spec = ExtractionSpec(
            revision=REVISION,
            source_path="dilu/driver_agent/driverAgent.py",
            expected_blob=BLOB,
            expected_bytes=ORIGINAL_DILU_PROMPT_BYTES,
            expected_sha256=ORIGINAL_DILU_PROMPT_SHA256,
        )
        extracted = extract_original_dilu_prompt(REPO_ROOT, spec)

        self.assertEqual(extracted.text.encode("utf-8"), PROMPT_PATH.read_bytes())
        self.assertEqual(extracted.revision, FULL_REVISION)
        self.assertEqual(extracted.git_blob, BLOB)
        for invalid_spec in (
            ExtractionSpec(**{**spec.__dict__, "expected_blob": "0" * 40}),
            ExtractionSpec(**{**spec.__dict__, "expected_sha256": "0" * 64}),
        ):
            with self.assertRaises(ValueError):
                extract_original_dilu_prompt(REPO_ROOT, invalid_spec)

    def test_extractor_writes_prompt_and_provenance_atomically(self) -> None:
        spec = ExtractionSpec(
            revision=REVISION,
            source_path="dilu/driver_agent/driverAgent.py",
            expected_blob=BLOB,
            expected_bytes=ORIGINAL_DILU_PROMPT_BYTES,
            expected_sha256=ORIGINAL_DILU_PROMPT_SHA256,
        )
        extracted = extract_original_dilu_prompt(REPO_ROOT, spec)

        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "prompt.txt"
            provenance_path = Path(directory) / "prompt.yaml"
            write_extracted_artifacts(extracted, prompt_path, provenance_path)
            self.assertEqual(prompt_path.read_bytes(), PROMPT_PATH.read_bytes())
            record = yaml.safe_load(provenance_path.read_text(encoding="utf-8"))
            self.assertEqual(record["sha256"], ORIGINAL_DILU_PROMPT_SHA256)


class PromptCompositionTests(unittest.TestCase):
    def test_prompt_modules_compose_deterministically(self) -> None:
        artifact = build_prompt_artifact(PolicyContent.MODULAR_HARNESS)

        self.assertIsInstance(artifact, PromptArtifact)
        self.assertEqual(
            artifact.component_names(),
            (
                "minimal_policy",
                "anti_passive",
                "lane_boundary",
                "flow_policy",
                "observation_action_domain",
                "strict_contract",
            ),
        )
        self.assertEqual(
            build_policy_prompt(PolicyContent.MODULAR_HARNESS),
            "\n".join(component.text for component in artifact.components[:4]),
        )
        self.assertEqual(
            build_system_prompt(PolicyContent.MODULAR_HARNESS), artifact.system_prompt()
        )
        self.assertEqual(
            build_prompt_artifact(PolicyContent.MODULAR_HARNESS).component_hashes(),
            artifact.component_hashes(),
        )
        self.assertEqual(
            artifact.component_hashes(),
            (
                (
                    "minimal_policy",
                    "cd720cf8b36adef0e6f09b057e77330cc1d000f4352aa5381b8fd054d4239090",
                ),
                (
                    "anti_passive",
                    "9b8ae5f9f1cbc8319cd846ff42b4dc1d8331cc1641228c019528acdde7dc0294",
                ),
                (
                    "lane_boundary",
                    "0cb2c8087dad1c37a2b4d2d377f772bc13f70c1680b55d502b2489dd670af67b",
                ),
                (
                    "flow_policy",
                    "ace8cade0aa08e143cf967ae70a1496fa4445a7b5a59e1cb538125b38015e606",
                ),
                (
                    "observation_action_domain",
                    "7ced94d1e3e9a791593dea0f98f3e0a5bc85818e88af6a7235f0cfe8e3606593",
                ),
                (
                    "strict_contract",
                    "581bfcabd4103bff9253aa9147033adfb385b5e09187b3d9a2d051befc58dd0c",
                ),
            ),
        )
        self.assertEqual(
            artifact.prompt_hash(),
            "64bfc9dda50ab80c2fe204f27b030832ee3be76be642612df0ba3871857bf6a4",
        )

    def test_policy_factor_changes_only_policy_components(self) -> None:
        historical = build_prompt_artifact(PolicyContent.HISTORICAL_DILU_2024)
        modular = build_prompt_artifact(PolicyContent.MODULAR_HARNESS)
        historical_components = dict(historical.component_texts())
        modular_components = dict(modular.component_texts())

        for shared_name in ("observation_action_domain", "strict_contract"):
            self.assertEqual(
                historical_components[shared_name], modular_components[shared_name]
            )
        self.assertNotEqual(
            build_policy_prompt(PolicyContent.HISTORICAL_DILU_2024),
            build_policy_prompt(PolicyContent.MODULAR_HARNESS),
        )
        self.assertEqual(
            build_policy_prompt(PolicyContent.HISTORICAL_DILU_2024),
            load_original_dilu_prompt(PROMPT_PATH).splitlines()[0],
        )
        self.assertEqual(
            historical.prompt_hash(),
            "7f7f9e7c0e9778eea61f3e215ce4773a00c4722ae8d6ac053c4c088157744c76",
        )
        self.assertEqual(historical.provenance_scope, "historical_policy_content")

    def test_output_enforcement_does_not_change_prompt_text_or_hash(self) -> None:
        prompt_only = build_prompt_artifact(
            PolicyContent.MODULAR_HARNESS,
            output_enforcement=OutputEnforcement.PROMPT_ONLY,
        )
        schema = build_prompt_artifact(
            PolicyContent.MODULAR_HARNESS,
            output_enforcement=OutputEnforcement.BACKEND_SCHEMA,
        )

        self.assertEqual(prompt_only.system_prompt(), schema.system_prompt())
        self.assertEqual(prompt_only.prompt_hash(), schema.prompt_hash())
        self.assertNotEqual(prompt_only.output_enforcement, schema.output_enforcement)

    def test_confirmatory_prompt_rejects_unhashed_few_shot_examples(self) -> None:
        for invalid_value in (1, 0.0, False):
            with self.subTest(value=invalid_value):
                with self.assertRaises(ValueError):
                    build_prompt_artifact(
                        PolicyContent.MODULAR_HARNESS,
                        few_shot_num=invalid_value,
                    )

    def test_driver_agent_scientific_branch_delegates_and_records_artifact(
        self,
    ) -> None:
        agent = DriverAgent.__new__(DriverAgent)
        agent.scientific_harness_config = _harness_config(
            PolicyContent.MODULAR_HARNESS,
            OutputEnforcement.BACKEND_SCHEMA,
        )

        prompt = agent._build_system_message(fallback_action_id=1)

        self.assertEqual(prompt, build_system_prompt(PolicyContent.MODULAR_HARNESS))
        self.assertIsInstance(agent.last_prompt_artifact, PromptArtifact)
        self.assertEqual(
            agent.last_prompt_artifact.output_enforcement,
            OutputEnforcement.BACKEND_SCHEMA,
        )

    def test_driver_agent_scientific_branch_rejects_few_shot_messages(self) -> None:
        agent = DriverAgent.__new__(DriverAgent)
        agent.scientific_harness_config = _harness_config()

        with self.assertRaises(ValueError):
            agent.few_shot_decision(
                fewshot_messages=["historical example"],
                fewshot_answers=["Response to user:#### 1"],
            )


if __name__ == "__main__":
    unittest.main()
