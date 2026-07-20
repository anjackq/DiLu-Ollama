from __future__ import annotations

import argparse
import ast
import hashlib
import logging
import os
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml


DEFAULT_REVISION = "1eed4ed"
DEFAULT_SOURCE_PATH = "dilu/driver_agent/driverAgent.py"
DEFAULT_BLOB = "91888022745e4edbb9dff5e0528f5d6bf3498713"
DEFAULT_BYTES = 836
DEFAULT_SHA256 = "170ff62b29d558fea590f234f3994a4b72100efbacdff5ccd518c24629bf764a"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionSpec:
    revision: str
    source_path: str
    expected_blob: str
    expected_bytes: int
    expected_sha256: str


@dataclass(frozen=True)
class ExtractedPrompt:
    revision: str
    source_path: str
    git_blob: str
    text: str
    raw_chars: int
    normalized_bytes: int
    sha256: str

    def provenance_record(self) -> dict[str, object]:
        return {
            "artifact_id": "original_dilu_2024_system_prompt",
            "artifact_scope": "system_message_only",
            "scope": "DriverAgent.few_shot_decision.system_message",
            "runtime_equivalence": False,
            "claims_exact_historical_runtime": False,
            "revision": self.revision,
            "source": self.source_path,
            "git_blob": self.git_blob,
            "extraction": "direct AST assignment; textwrap.dedent JoinedStr",
            "normalization": (
                "substitute three bare delimiter fields with ####; textwrap.dedent; "
                "CRLF/CR to LF; UTF-8; no trim"
            ),
            "raw_chars": self.raw_chars,
            "normalized_bytes": self.normalized_bytes,
            "sha256": self.sha256,
        }


def extract_original_dilu_prompt(
    repo_root: Path, spec: ExtractionSpec
) -> ExtractedPrompt:
    root = repo_root.resolve()
    revision = _git_text(root, "rev-parse", f"{spec.revision}^{{commit}}")
    git_blob = _git_text(root, "rev-parse", f"{spec.revision}:{spec.source_path}")
    if git_blob != spec.expected_blob:
        raise ValueError(
            f"Historical prompt blob mismatch: expected {spec.expected_blob}, got {git_blob}."
        )
    source_bytes = _git_bytes(root, "cat-file", "blob", git_blob)
    try:
        source = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Historical prompt source is not strict UTF-8.") from exc
    raw = _extract_joined_string(source)
    normalized = textwrap.dedent(raw).replace("\r\n", "\n").replace("\r", "\n")
    normalized_bytes = normalized.encode("utf-8")
    digest = hashlib.sha256(normalized_bytes).hexdigest()
    if len(normalized_bytes) != spec.expected_bytes:
        raise ValueError(
            f"Historical prompt length mismatch: expected {spec.expected_bytes}, "
            f"got {len(normalized_bytes)}."
        )
    if digest != spec.expected_sha256:
        raise ValueError(
            f"Historical prompt hash mismatch: expected {spec.expected_sha256}, got {digest}."
        )
    return ExtractedPrompt(
        revision=revision,
        source_path=spec.source_path,
        git_blob=git_blob,
        text=normalized,
        raw_chars=len(raw),
        normalized_bytes=len(normalized_bytes),
        sha256=digest,
    )


def write_extracted_artifacts(
    extracted: ExtractedPrompt, prompt_path: Path, provenance_path: Path
) -> None:
    _atomic_write(prompt_path, extracted.text.encode("utf-8"))
    _atomic_write(provenance_path, _provenance_bytes(extracted))


def verify_extracted_artifacts(
    extracted: ExtractedPrompt, prompt_path: Path, provenance_path: Path
) -> None:
    expected = {
        prompt_path: extracted.text.encode("utf-8"),
        provenance_path: _provenance_bytes(extracted),
    }
    for path, expected_bytes in expected.items():
        if not path.is_file():
            raise ValueError(f"Missing checked-in provenance artifact: {path}")
        if path.read_bytes() != expected_bytes:
            raise ValueError(f"Checked-in provenance artifact differs: {path}")


def _extract_joined_string(source: str) -> str:
    tree = ast.parse(source)
    driver_classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "DriverAgent"
    ]
    if len(driver_classes) != 1:
        raise ValueError("Expected exactly one direct DriverAgent class.")
    methods = [
        node
        for node in driver_classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "few_shot_decision"
    ]
    if len(methods) != 1:
        raise ValueError("Expected exactly one direct few_shot_decision method.")
    assignments = [
        node
        for node in methods[0].body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "system_message"
            for target in node.targets
        )
    ]
    if len(assignments) != 1:
        raise ValueError("Expected exactly one direct system_message assignment.")
    joined = _dedent_joined_string(assignments[0].value)
    chunks = []
    delimiter_count = 0
    for value in joined.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            chunks.append(value.value)
            continue
        if _is_bare_delimiter(value):
            chunks.append("####")
            delimiter_count += 1
            continue
        raise ValueError(
            "Historical prompt contains an unsupported f-string expression."
        )
    if delimiter_count != 3:
        raise ValueError(
            f"Expected three delimiter substitutions, got {delimiter_count}."
        )
    return "".join(chunks)


def _dedent_joined_string(value: ast.expr) -> ast.JoinedStr:
    if not isinstance(value, ast.Call) or len(value.args) != 1 or value.keywords:
        raise ValueError("system_message must be one textwrap.dedent call.")
    function = value.func
    if not (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "textwrap"
        and function.attr == "dedent"
        and isinstance(value.args[0], ast.JoinedStr)
    ):
        raise ValueError("system_message must wrap one f-string in textwrap.dedent.")
    return value.args[0]


def _is_bare_delimiter(value: ast.AST) -> bool:
    return (
        isinstance(value, ast.FormattedValue)
        and isinstance(value.value, ast.Name)
        and value.value.id == "delimiter"
        and value.conversion == -1
        and value.format_spec is None
    )


def _git_text(repo_root: Path, *args: str) -> str:
    return _git_bytes(repo_root, *args).decode("utf-8").strip()


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    result = subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git provenance command failed: {message}")
    return result.stdout


def _provenance_bytes(extracted: ExtractedPrompt) -> bytes:
    content = yaml.safe_dump(
        extracted.provenance_record(),
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    )
    return content.encode("utf-8")


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract the provenance-locked DiLu prompt."
    )
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--prompt-output",
        type=Path,
        default=Path("dilu/driver_agent/prompts/original_dilu_2024.txt"),
    )
    parser.add_argument(
        "--provenance-output",
        type=Path,
        default=Path("provenance/original_dilu_2024_prompt.yaml"),
    )
    parser.add_argument("--verify", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    spec = ExtractionSpec(
        revision=args.revision,
        source_path=DEFAULT_SOURCE_PATH,
        expected_blob=DEFAULT_BLOB,
        expected_bytes=DEFAULT_BYTES,
        expected_sha256=DEFAULT_SHA256,
    )
    extracted = extract_original_dilu_prompt(root, spec)
    prompt_path = root / args.prompt_output
    provenance_path = root / args.provenance_output
    if args.verify:
        verify_extracted_artifacts(extracted, prompt_path, provenance_path)
        LOGGER.info("Historical DiLu prompt artifacts verified.")
    else:
        write_extracted_artifacts(extracted, prompt_path, provenance_path)
        LOGGER.info("Historical DiLu prompt artifacts written.")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
