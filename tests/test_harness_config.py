import dataclasses
import math
import os
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from dilu.runtime.config_loader import load_runtime_config
from dilu.runtime import safety_shields
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
    diff_conditions,
    resolve_main_conditions,
)


def _condition() -> ConditionSpec:
    return ConditionSpec(
        policy_content=PolicyContent.HISTORICAL_DILU_2024,
        output_enforcement=OutputEnforcement.PROMPT_ONLY,
        execution_mode=ExecutionMode.UNSHIELDED_OPERATIONAL,
    )


def _scientific_config(**overrides: Any) -> HarnessConfig:
    values = {
        "condition": _condition(),
        "parser_mode": ParserMode.STRICT_ONLY,
        "resolver_mode": ResolverMode.DISABLED,
        "fallback_policy": FallbackPolicy.FIXED_IDLE,
        "shield": ShieldConfig.implementation_defaults(),
        "transport": TransportConfig(
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
        "retry_policy": RetryPolicy(
            max_transport_unavailable_retries=1,
            retry_cooldown_sec=10.0,
            retry_on_timeout=False,
            retry_on_empty_output=False,
            retry_on_schema_rejection=False,
        ),
        "trace_level": TraceLevel.MANDATORY_SCIENTIFIC,
    }
    values.update(overrides)
    return HarnessConfig(**values)


class HarnessConfigTests(unittest.TestCase):
    def test_config_and_nested_configs_are_frozen(self) -> None:
        config = _scientific_config()

        for target, field, value in (
            (config, "parser_mode", ParserMode.DETERMINISTIC_RECOVERY),
            (config.condition, "execution_mode", ExecutionMode.SHIELDED),
            (config.transport, "timeout_sec", 5.0),
            (config.shield, "front_critical_gap_m", 1.0),
            (config.retry_policy, "retry_on_timeout", True),
        ):
            with self.subTest(field=field):
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(target, field, value)

    def test_resolve_main_conditions_builds_fixed_factorial_ids(self) -> None:
        conditions = resolve_main_conditions(_scientific_config())

        self.assertEqual(len(conditions), 8)
        self.assertEqual(
            [item.condition_id() for item in conditions],
            ["c000", "c001", "c010", "c011", "c100", "c101", "c110", "c111"],
        )
        self.assertEqual(len({item.config_hash() for item in conditions}), 8)

    def test_harness_factors_resolve_independently(self) -> None:
        conditions = {
            (
                item.condition.policy_content,
                item.condition.output_enforcement,
                item.condition.execution_mode,
            ): item
            for item in resolve_main_conditions(_scientific_config())
        }
        factors = (
            (
                "condition.policy_content",
                (PolicyContent.HISTORICAL_DILU_2024, PolicyContent.MODULAR_HARNESS),
            ),
            (
                "condition.output_enforcement",
                (OutputEnforcement.PROMPT_ONLY, OutputEnforcement.BACKEND_SCHEMA),
            ),
            (
                "condition.execution_mode",
                (ExecutionMode.UNSHIELDED_OPERATIONAL, ExecutionMode.SHIELDED),
            ),
        )

        edge_count = 0
        keys = list(conditions)
        for left_index, left_key in enumerate(keys):
            for right_key in keys[left_index + 1 :]:
                changed_indexes = [
                    index for index in range(3) if left_key[index] != right_key[index]
                ]
                if len(changed_indexes) != 1:
                    continue
                edge_count += 1
                expected_path = factors[changed_indexes[0]][0]
                self.assertEqual(
                    set(diff_conditions(conditions[left_key], conditions[right_key])),
                    {expected_path},
                )
        self.assertEqual(edge_count, 12)

    def test_canonical_serialization_has_stable_golden_hash(self) -> None:
        config = _scientific_config()
        reordered = dict(reversed(list(config.to_canonical_dict().items())))
        reconstructed = HarnessConfig.from_mapping(reordered)

        self.assertEqual(reconstructed, config)
        self.assertEqual(reconstructed.to_canonical_dict(), config.to_canonical_dict())
        self.assertEqual(
            config.config_hash(),
            "eeacd106023bb40ec7a38f1cca59af06804d2b821ee0fc53f86d626f513fe05d",
        )
        self.assertRegex(config.config_hash(), r"^[0-9a-f]{64}$")

    def test_factor_and_runtime_identities_are_separate(self) -> None:
        config = _scientific_config()
        runtime_changed = dataclasses.replace(
            config,
            transport=dataclasses.replace(config.transport, timeout_sec=90.0),
        )
        factor_changed = dataclasses.replace(
            config,
            condition=dataclasses.replace(
                config.condition,
                policy_content=PolicyContent.MODULAR_HARNESS,
            ),
        )

        self.assertEqual(runtime_changed.condition_id(), config.condition_id())
        self.assertNotEqual(runtime_changed.config_hash(), config.config_hash())
        self.assertNotEqual(factor_changed.condition_id(), config.condition_id())
        self.assertNotEqual(factor_changed.config_hash(), config.config_hash())

    def test_claim_bearing_hash_rejects_invalid_scientific_config(self) -> None:
        invalid = dataclasses.replace(
            _scientific_config(),
            parser_mode=ParserMode.DETERMINISTIC_RECOVERY,
        )

        with self.assertRaises(ValueError):
            invalid.config_hash()

    def test_from_mapping_is_strict_and_environment_independent(self) -> None:
        mapping = _scientific_config().to_canonical_dict()
        invalid_mappings = []
        for path in ("top", "condition", "transport", "shield", "retry_policy"):
            candidate = _scientific_config().to_canonical_dict()
            target = candidate if path == "top" else candidate[path]
            target["misspelled_field"] = True
            invalid_mappings.append((path, candidate))
        missing = _scientific_config().to_canonical_dict()
        del missing["parser_mode"]
        invalid_mappings.append(("missing", missing))

        for reason, candidate in invalid_mappings:
            with self.subTest(reason=reason):
                with self.assertRaises(ValueError):
                    HarnessConfig.from_mapping(candidate)

        with mock.patch.dict(
            os.environ,
            {"DILU_PROMPT_PROFILE": "invalid", "OLLAMA_THINK_MODE": "auto"},
        ):
            self.assertEqual(HarnessConfig.from_mapping(mapping), _scientific_config())

    def test_direct_construction_rejects_unresolved_or_mistyped_values(self) -> None:
        invalid_configs = (
            dataclasses.replace(
                _scientific_config(),
                condition=ConditionSpec(
                    policy_content="historical_dilu_2024",
                    output_enforcement=OutputEnforcement.PROMPT_ONLY,
                    execution_mode=ExecutionMode.UNSHIELDED_OPERATIONAL,
                ),
            ),
            dataclasses.replace(
                _scientific_config(),
                transport=dataclasses.replace(
                    _scientific_config().transport,
                    context_tokens=True,
                ),
            ),
            dataclasses.replace(
                _scientific_config(),
                transport=dataclasses.replace(
                    _scientific_config().transport,
                    temperature=math.inf,
                ),
            ),
            dataclasses.replace(
                _scientific_config(),
                transport=dataclasses.replace(
                    _scientific_config().transport,
                    generation_seed_master=1 << 32,
                ),
            ),
        )

        for invalid in invalid_configs:
            with self.assertRaises(ValueError):
                invalid.validate_scientific()

    def test_scientific_validation_accepts_all_main_conditions(self) -> None:
        for condition in resolve_main_conditions(_scientific_config()):
            with self.subTest(condition=condition.condition_id()):
                condition.validate_scientific()

    def test_scientific_validation_rejects_confounded_controls(self) -> None:
        config = _scientific_config()
        invalid_configs = (
            dataclasses.replace(config, parser_mode=ParserMode.DETERMINISTIC_RECOVERY),
            dataclasses.replace(config, resolver_mode=ResolverMode.ASSISTED),
            dataclasses.replace(config, fallback_policy=FallbackPolicy.FIXED_SLOWER),
            dataclasses.replace(
                config,
                transport=dataclasses.replace(
                    config.transport,
                    profile=TransportProfile.OLLAMA_OPENAI_COMPATIBLE,
                ),
            ),
            dataclasses.replace(
                config,
                transport=dataclasses.replace(
                    config.transport, think_mode=ThinkMode.AUTO
                ),
            ),
            dataclasses.replace(
                config,
                transport=dataclasses.replace(
                    config.transport,
                    allow_transport_fallback=True,
                ),
            ),
            dataclasses.replace(
                config,
                transport=dataclasses.replace(config.transport, adaptive_timeout=True),
            ),
            dataclasses.replace(config, trace_level=TraceLevel.DISABLED),
            dataclasses.replace(
                config,
                retry_policy=dataclasses.replace(
                    config.retry_policy,
                    retry_on_timeout=True,
                ),
            ),
        )

        for invalid in invalid_configs:
            with self.assertRaises(ValueError):
                invalid.validate_scientific()

    def test_scientific_validation_rejects_invalid_shield_constants(self) -> None:
        shield = ShieldConfig.implementation_defaults()
        invalid_shields = (
            dataclasses.replace(shield, front_critical_gap_m=0.0),
            dataclasses.replace(
                shield,
                front_critical_gap_m=13.0,
                front_caution_gap_m=12.0,
            ),
            dataclasses.replace(
                shield,
                front_critical_ttc_sec=4.0,
                front_caution_ttc_sec=3.0,
            ),
        )

        for invalid_shield in invalid_shields:
            with self.assertRaises(ValueError):
                dataclasses.replace(
                    _scientific_config(), shield=invalid_shield
                ).validate_scientific()

    def test_scientific_validation_rejects_mistyped_nested_configs(self) -> None:
        config = _scientific_config()

        for field_name in ("shield", "transport", "retry_policy"):
            with self.subTest(field=field_name):
                with self.assertRaises(ValueError):
                    dataclasses.replace(
                        config, **{field_name: {}}
                    ).validate_scientific()

    def test_shield_defaults_match_executed_primitive_constants(self) -> None:
        shield = ShieldConfig.implementation_defaults()

        for field in dataclasses.fields(shield):
            if field.name == "stage_order":
                continue
            self.assertEqual(
                getattr(shield, field.name),
                getattr(safety_shields, field.name.upper()),
                msg=field.name,
            )

    def test_protocol_constants_freeze_pre_s1_fields_only(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "iclr2027"
            / "protocol_constants.yaml"
        )
        protocol = load_runtime_config(path)

        self.assertEqual(protocol["status"], "pre_runtime_lock_template")
        self.assertFalse(protocol["claim_bearing_config"])
        self.assertEqual(
            protocol["main_factors"],
            {
                "policy_content": ["historical_dilu_2024", "modular_harness"],
                "output_enforcement": ["prompt_only", "backend_schema"],
                "execution_mode": ["unshielded_operational", "shielded"],
            },
        )
        self.assertEqual(
            protocol["fixed_harness"]["shield"],
            _scientific_config().to_canonical_dict()["shield"],
        )
        self.assertEqual(
            protocol["fixed_harness"]["retry_policy"],
            _scientific_config().to_canonical_dict()["retry_policy"],
        )
        for field_name in protocol["runtime_bound_fields"]:
            self.assertEqual(
                protocol["transport_template"][field_name],
                "__runtime_bound_after_s1__",
            )


if __name__ == "__main__":
    unittest.main()
