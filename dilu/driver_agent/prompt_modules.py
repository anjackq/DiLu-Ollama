from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from dilu.runtime.harness_config import OutputEnforcement, PolicyContent


ORIGINAL_DILU_PROMPT_BYTES = 836
ORIGINAL_DILU_PROMPT_SHA256 = (
    "170ff62b29d558fea590f234f3994a4b72100efbacdff5ccd518c24629bf764a"
)
ORIGINAL_DILU_PROMPT_PATH = (
    Path(__file__).resolve().parent / "prompts" / "original_dilu_2024.txt"
)

HISTORICAL_POLICY = (
    "You are ChatGPT, a large language model trained by OpenAI. Now you act as a "
    "mature driving assistant, who can give accurate and correct advice for human "
    "driver in complex urban driving scenarios."
)
MINIMAL_POLICY = (
    "You are a local language-model driving decision module. Select one high-level "
    "action by prioritizing collision avoidance, legal feasibility, and safe progress."
)
ANTI_PASSIVE = (
    "Do not game safety by repeatedly braking, idling, or stopping when the observed "
    "road ahead permits safe progress; resume traffic-flow speed after a hazard clears."
)
LANE_BOUNDARY = (
    "Change lanes only when the requested direction is available and both target-lane "
    "front and rear gaps are clearly safe; otherwise select a longitudinal action."
)
FLOW_POLICY = (
    "Treat unnecessary low-speed blocking and lane-change refusal as policy failures, "
    "while never trading collision safety for speed or task completion."
)
OBSERVATION_ACTION_DOMAIN = (
    "You receive a structured driving-scenario description, previous decisions, and "
    "the available high-level actions. Action IDs are: 0 lane left, 1 keep lane/IDLE, "
    "2 lane right, 3 accelerate, and 4 decelerate. Judge semantic availability from "
    "the supplied state; do not invent observations or actions."
)
STRICT_CONTRACT = (
    "Return exactly one non-empty line in this form: Response to user:#### <action_id>. "
    "Replace <action_id> with one integer from 0 to 4. Output no reasoning, JSON, "
    "Markdown, action name, alternative, or additional text."
)


@dataclass(frozen=True)
class PromptComponent:
    name: str
    text: str

    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptArtifact:
    policy_content: PolicyContent
    output_enforcement: OutputEnforcement
    components: tuple[PromptComponent, ...]
    provenance_scope: str
    few_shot_num: int = 0

    def component_names(self) -> tuple[str, ...]:
        return tuple(component.name for component in self.components)

    def component_texts(self) -> tuple[tuple[str, str], ...]:
        return tuple((component.name, component.text) for component in self.components)

    def component_hashes(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (component.name, component.sha256()) for component in self.components
        )

    def system_prompt(self) -> str:
        component_map = dict(self.component_texts())
        policy_names = self.component_names()[:-2]
        policy_text = "\n".join(component_map[name] for name in policy_names)
        return "\n\n".join(
            (
                policy_text,
                component_map["observation_action_domain"],
                component_map["strict_contract"],
            )
        )

    def prompt_hash(self) -> str:
        return hashlib.sha256(self.system_prompt().encode("utf-8")).hexdigest()


def load_original_dilu_prompt(path: Path = ORIGINAL_DILU_PROMPT_PATH) -> str:
    content = path.read_bytes()
    if len(content) != ORIGINAL_DILU_PROMPT_BYTES:
        raise ValueError("Original DiLu prompt byte length does not match provenance.")
    if hashlib.sha256(content).hexdigest() != ORIGINAL_DILU_PROMPT_SHA256:
        raise ValueError("Original DiLu prompt hash does not match provenance.")
    return content.decode("utf-8")


def build_policy_prompt(policy_content: PolicyContent) -> str:
    return "\n".join(component.text for component in _policy_components(policy_content))


def build_prompt_artifact(
    policy_content: PolicyContent,
    *,
    output_enforcement: OutputEnforcement = OutputEnforcement.PROMPT_ONLY,
    few_shot_num: int = 0,
) -> PromptArtifact:
    if not isinstance(policy_content, PolicyContent):
        raise ValueError("Scientific policy content must be a resolved PolicyContent.")
    if not isinstance(output_enforcement, OutputEnforcement):
        raise ValueError(
            "Scientific output enforcement must be a resolved OutputEnforcement."
        )
    if (
        isinstance(few_shot_num, bool)
        or not isinstance(few_shot_num, int)
        or few_shot_num != 0
    ):
        raise ValueError("Confirmatory prompts require few_shot_num=0.")
    components = _policy_components(policy_content) + (
        PromptComponent("observation_action_domain", OBSERVATION_ACTION_DOMAIN),
        PromptComponent("strict_contract", STRICT_CONTRACT),
    )
    provenance_scope = (
        "historical_policy_content"
        if policy_content is PolicyContent.HISTORICAL_DILU_2024
        else "modular_harness_policy"
    )
    return PromptArtifact(
        policy_content=policy_content,
        output_enforcement=output_enforcement,
        components=components,
        provenance_scope=provenance_scope,
        few_shot_num=few_shot_num,
    )


def build_system_prompt(
    policy_content: PolicyContent,
    *,
    output_enforcement: OutputEnforcement = OutputEnforcement.PROMPT_ONLY,
    few_shot_num: int = 0,
) -> str:
    return build_prompt_artifact(
        policy_content,
        output_enforcement=output_enforcement,
        few_shot_num=few_shot_num,
    ).system_prompt()


def _policy_components(policy_content: PolicyContent) -> tuple[PromptComponent, ...]:
    if policy_content is PolicyContent.HISTORICAL_DILU_2024:
        return (PromptComponent("historical_policy", HISTORICAL_POLICY),)
    if policy_content is PolicyContent.MODULAR_HARNESS:
        return (
            PromptComponent("minimal_policy", MINIMAL_POLICY),
            PromptComponent("anti_passive", ANTI_PASSIVE),
            PromptComponent("lane_boundary", LANE_BOUNDARY),
            PromptComponent("flow_policy", FLOW_POLICY),
        )
    raise ValueError(f"Unsupported policy content: {policy_content!r}")


__all__ = [
    "ORIGINAL_DILU_PROMPT_BYTES",
    "ORIGINAL_DILU_PROMPT_SHA256",
    "PromptArtifact",
    "PromptComponent",
    "build_policy_prompt",
    "build_prompt_artifact",
    "build_system_prompt",
    "load_original_dilu_prompt",
]
