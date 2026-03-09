import os
import re
import json
import tempfile
from datetime import datetime
from typing import Dict, List, Optional


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ensure_parent_dir(file_path: str) -> str:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return file_path


def timestamped_results_path(prefix: str, ext: str = ".json", results_dir: str = "results") -> str:
    ensure_dir(results_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(results_dir, f"{prefix}_{ts}{ext}")


def current_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify_model_name(model_name: str) -> str:
    text = (model_name or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown_model"


def build_experiment_root(results_root: str, experiment_id: Optional[str] = None) -> str:
    root = results_root or os.path.join("results", "experiments")
    exp_id = (experiment_id or "").strip() or current_timestamp()
    return ensure_dir(os.path.join(root, exp_id))


def build_model_root(experiment_root: str, model_name: str) -> str:
    model_slug = slugify_model_name(model_name)
    return ensure_dir(os.path.join(experiment_root, "models", model_slug))


def build_model_run_dir(experiment_root: str, model_name: str, run_id: Optional[str] = None) -> str:
    rid = (run_id or "").strip() or f"run_{current_timestamp()}"
    model_root = build_model_root(experiment_root, model_name)
    return ensure_dir(os.path.join(model_root, "runs", rid))


def ensure_experiment_layout(experiment_root: str, model_names: List[str]) -> Dict[str, str]:
    ensure_dir(experiment_root)
    ensure_dir(os.path.join(experiment_root, "compare"))

    model_roots: Dict[str, str] = {}
    for model_name in model_names or []:
        model_root = build_model_root(experiment_root, model_name)
        ensure_dir(os.path.join(model_root, "runs"))
        ensure_dir(os.path.join(model_root, "eval"))
        ensure_dir(os.path.join(model_root, "plots"))
        model_roots[str(model_name)] = model_root

    return model_roots


def write_json_atomic(path: str, payload: Dict) -> str:
    ensure_parent_dir(path)
    folder = os.path.dirname(path) or "."
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=folder, suffix=".tmp") as tf:
        json.dump(payload, tf, indent=2)
        tmp_path = tf.name
    os.replace(tmp_path, path)
    return path


def read_json(path: str, default: Optional[Dict] = None) -> Dict:
    if not os.path.exists(path):
        return {} if default is None else dict(default)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
