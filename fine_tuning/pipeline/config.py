import os
from typing import Dict


FAMILY_DEFAULTS: Dict[str, Dict[str, object]] = {
    "llama3": {
        "chat_template": "llama-3",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_seq_length": 2048,
    },
    "qwen": {
        "chat_template": "chatml",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_seq_length": 4096,
    },
    "mistral": {
        "chat_template": "chatml",
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "max_seq_length": 4096,
    },
}


def apply_windows_short_temp(short_temp: str = "C:\\ui") -> str:
    """Keep Triton/Inductor cache paths short on Windows."""
    if os.name != "nt":
        return short_temp
    if not os.path.exists(short_temp):
        try:
            os.makedirs(short_temp, exist_ok=True)
        except Exception:
            short_temp = os.path.join(os.environ.get("HOMEDRIVE", "C:"), "\\temp_ui")
            os.makedirs(short_temp, exist_ok=True)

    os.environ["TMPDIR"] = short_temp
    os.environ["TEMP"] = short_temp
    os.environ["TMP"] = short_temp
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = short_temp
    os.environ["TRITON_CACHE_DIR"] = short_temp
    return short_temp


def infer_model_family(model_name: str) -> str:
    name = (model_name or "").lower()
    if "llama" in name:
        return "llama3"
    if "qwen" in name:
        return "qwen"
    if "mistral" in name:
        return "mistral"
    return "llama3"


def resolve_family_config(model_family: str, max_seq_length: int = 0) -> Dict[str, object]:
    family = (model_family or "").lower().strip()
    if family not in FAMILY_DEFAULTS:
        raise ValueError(f"Unsupported model family: {model_family}. Expected one of {list(FAMILY_DEFAULTS.keys())}")
    resolved = dict(FAMILY_DEFAULTS[family])
    if max_seq_length and max_seq_length > 0:
        resolved["max_seq_length"] = int(max_seq_length)
    return resolved
