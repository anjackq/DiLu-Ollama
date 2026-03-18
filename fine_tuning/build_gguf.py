import argparse
import json
import os
import re
import subprocess
import sys
import shutil
from typing import Optional


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

ENHANCED_SYSTEM_PROMPT = """You are an autonomous driving decision engine.
Choose exactly one Action_id from 0..4 for the given scenario.

Allowed actions:
0 = Turn-left
1 = IDLE
2 = Turn-right
3 = Acceleration
4 = Deceleration

Output requirements (strict):
1) Exactly two lines.
2) Line 1: Reasoning: <one short sentence, max 20 words>
3) Line 2: Response to user:#### <0|1|2|3|4>
4) No markdown, no bullet lists, no extra sections, no chain-of-thought.

Safety fallback:
If uncertain or constraints conflict, choose Deceleration (4)."""

SYSTEM_BLOCK_RE = re.compile(r'(?ims)^\s*SYSTEM\s+"""(.*?)"""[ \t]*\n?')


def _abs_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(ROOT_DIR, path))


def _run(cmd: list[str], cwd: Optional[str] = None) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd or ROOT_DIR)


def _is_llama_cpp_dir(path: str) -> bool:
    if not path:
        return False
    return os.path.exists(os.path.join(path, "convert_hf_to_gguf.py"))


def _discover_llama_cpp_dir() -> Optional[str]:
    # 1) explicit environment variable
    env_dir = str(os.getenv("LLAMA_CPP_DIR", "")).strip()
    if env_dir and _is_llama_cpp_dir(env_dir):
        return os.path.abspath(env_dir)

    # 2) common local checkout locations
    home = os.path.expanduser("~")
    candidates = [
        os.path.join(ROOT_DIR, "llama.cpp"),
        os.path.join(os.path.dirname(ROOT_DIR), "llama.cpp"),
        os.path.join(home, "llama.cpp"),
        r"C:\dev\llama.cpp",
    ]
    for path in candidates:
        if _is_llama_cpp_dir(path):
            return os.path.abspath(path)

    # 3) derive from binaries found in PATH (winget/choco/manual installs)
    bin_candidates = [
        shutil.which("llama-quantize"),
        shutil.which("quantize"),
        shutil.which("llama-cli"),
    ]
    for bin_path in [p for p in bin_candidates if p]:
        current = os.path.abspath(os.path.dirname(bin_path))
        # Try current dir and several parent levels.
        for _ in range(8):
            if _is_llama_cpp_dir(current):
                return current
            # Try common parent/sibling layout where binaries live in build/bin/...
            sibling = os.path.join(current, "..", "..", "..")
            sibling = os.path.abspath(sibling)
            if _is_llama_cpp_dir(sibling):
                return sibling
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

    return None


def _find_quantize_binary(llama_cpp_dir: str) -> Optional[str]:
    candidates = [
        os.path.join(llama_cpp_dir, "build", "bin", "Release", "llama-quantize.exe"),
        os.path.join(llama_cpp_dir, "build", "bin", "Release", "quantize.exe"),
        os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize.exe"),
        os.path.join(llama_cpp_dir, "build", "bin", "quantize.exe"),
        os.path.join(llama_cpp_dir, "llama-quantize.exe"),
        os.path.join(llama_cpp_dir, "quantize.exe"),
        os.path.join(llama_cpp_dir, "llama-quantize"),
        os.path.join(llama_cpp_dir, "quantize"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _slugify(name: str) -> str:
    out = []
    for ch in name.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in {":", "/", "\\", " ", ".", "-"}:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "model"


def _maybe_patch_phi_tokenizer_config(hf_model_dir: str) -> Optional[tuple[str, str]]:
    """Patch tokenizer_config for Phi3/Phi4-style exports lacking tokenizer.model.

    Some llama.cpp revisions expect Phi3 models to provide tokenizer.model unless
    tokenizer_class is GPT2Tokenizer. Unsloth merged exports can contain only
    tokenizer.json + tokenizer_config.json with tokenizer_class=TokenizersBackend.
    In that case we patch tokenizer_class temporarily for conversion and restore it.
    """
    config_path = os.path.join(hf_model_dir, "config.json")
    tokenizer_config_path = os.path.join(hf_model_dir, "tokenizer_config.json")
    tokenizer_json_path = os.path.join(hf_model_dir, "tokenizer.json")
    tokenizer_model_path = os.path.join(hf_model_dir, "tokenizer.model")

    if not os.path.isfile(config_path):
        return None
    if not os.path.isfile(tokenizer_config_path):
        return None
    if not os.path.isfile(tokenizer_json_path):
        return None
    if os.path.isfile(tokenizer_model_path):
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            model_config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    model_type = str(model_config.get("model_type", "")).lower()
    architectures = model_config.get("architectures", [])
    is_phi3_family = model_type == "phi3" or any(
        isinstance(a, str) and a.lower().startswith("phi3") for a in architectures
    )
    if not is_phi3_family:
        return None

    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
        original_text = f.read()
    try:
        tokenizer_config = json.loads(original_text)
    except json.JSONDecodeError:
        return None

    tokenizer_class = str(tokenizer_config.get("tokenizer_class", ""))
    if tokenizer_class == "GPT2Tokenizer":
        return None

    tokenizer_config["tokenizer_class"] = "GPT2Tokenizer"
    with open(tokenizer_config_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(
        "[INFO] Applied temporary tokenizer compatibility patch for llama.cpp: "
        f"{tokenizer_config_path} tokenizer_class={tokenizer_class!r} -> 'GPT2Tokenizer'"
    )
    return tokenizer_config_path, original_text


def _restore_patched_file(path: str, original_text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(original_text)


def _extract_first_system_prompt(text: str) -> Optional[str]:
    matches = SYSTEM_BLOCK_RE.findall(text or "")
    if not matches:
        return None
    candidate = matches[0].strip()
    return candidate or None


def _remove_system_blocks(text: str) -> str:
    return SYSTEM_BLOCK_RE.sub("", text or "")


def _resolve_system_prompt(explicit_prompt: str, template_prompt: Optional[str]) -> str:
    if explicit_prompt.strip():
        return explicit_prompt.strip()
    if template_prompt and template_prompt.strip():
        return template_prompt.strip()
    return ENHANCED_SYSTEM_PROMPT.strip()


def _has_system_prompt(text: str) -> bool:
    return bool(SYSTEM_BLOCK_RE.search(text or ""))


def _backfill_missing_system_prompts(output_dir: str, fallback_prompt: str) -> int:
    patched_count = 0
    fallback_prompt = fallback_prompt.strip()
    if not fallback_prompt:
        return patched_count

    for filename in sorted(os.listdir(output_dir)):
        if not filename.endswith(".gguf.Modelfile"):
            continue
        path = os.path.join(output_dir, filename)
        if not os.path.isfile(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if _has_system_prompt(content):
            continue

        updated = content.rstrip() + f'\n\nSYSTEM """{fallback_prompt}"""\n'
        with open(path, "w", encoding="utf-8") as f:
            f.write(updated)
        patched_count += 1

    return patched_count


def _find_auto_template_modelfile(base_name: str) -> str:
    """Locate a template Modelfile for GGUF generation.

    Preferred order:
    1) fine_tuning/template_modelfile/<base_name>.Modelfile
    2) fine_tuning/modelfiles/<base_name>.Modelfile (legacy location)
    3) fine_tuning/template_modelfile/dilu-reasoning.Modelfile or dilu-instruct.Modelfile
       selected by model-name heuristic
    """
    lowered = base_name.lower()
    looks_reasoning = any(token in lowered for token in ["-r1", "_r1", "reason", "qwq"])

    candidates = [
        os.path.join(ROOT_DIR, "fine_tuning", "template_modelfile", f"{base_name}.Modelfile"),
        os.path.join(ROOT_DIR, "fine_tuning", "modelfiles", f"{base_name}.Modelfile"),
        os.path.join(
            ROOT_DIR,
            "fine_tuning",
            "template_modelfile",
            "dilu-reasoning.Modelfile" if looks_reasoning else "dilu-instruct.Modelfile",
        ),
        os.path.join(ROOT_DIR, "fine_tuning", "template_modelfile", "dilu-instruct.Modelfile"),
        os.path.join(ROOT_DIR, "fine_tuning", "template_modelfile", "dilu-reasoning.Modelfile"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert merged HF model exports to GGUF, optionally quantize, and optionally create an Ollama model."
    )
    parser.add_argument(
        "--llama-cpp-dir",
        default="",
        help="Path to local llama.cpp checkout (must contain convert_hf_to_gguf.py). If omitted, auto-discovery is used.",
    )
    parser.add_argument(
        "--hf-model-dir",
        required=True,
        help="Path to merged HF model directory (e.g., fine_tuning/merged_models/...).",
    )
    parser.add_argument(
        "--output-dir",
        default="fine_tuning/gguf",
        help="Output directory for generated GGUF files and Modelfile.",
    )
    parser.add_argument(
        "--name",
        default="",
        help="Logical model name used for output file names and Ollama model tag.",
    )
    parser.add_argument(
        "--outtype",
        default="f16",
        choices=["f16", "bf16", "f32", "q8_0", "auto"],
        help="GGUF output type for convert_hf_to_gguf.py.",
    )
    parser.add_argument(
        "--quantize",
        default="",
        help="Optional post-quantization type (e.g. Q4_K_M, Q5_K_M, Q8_0).",
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Optional system prompt text to embed in generated Modelfile.",
    )
    parser.add_argument(
        "--template-modelfile",
        default="",
        help="Optional template Modelfile to inherit PARAMETER/SYSTEM sections from. "
             "If omitted, auto-uses fine_tuning/template_modelfile/<name>.Modelfile "
             "(fallback: fine_tuning/modelfiles/<name>.Modelfile).",
    )
    parser.add_argument(
        "--ollama-model",
        default="",
        help="Optional Ollama model tag to build (implies --create-ollama).",
    )
    parser.add_argument(
        "--create-ollama",
        action="store_true",
        help="If set, run `ollama create` using generated Modelfile.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to run llama.cpp conversion script.",
    )
    return parser.parse_args()


def _write_modelfile(
    modelfile_path: str,
    gguf_path: str,
    system_prompt: str,
    template_modelfile: str = "",
) -> None:
    rel_gguf_path = os.path.relpath(gguf_path, os.path.dirname(modelfile_path))
    rel_gguf_path = rel_gguf_path.replace("\\", "/")
    if not rel_gguf_path.startswith("."):
        rel_gguf_path = f"./{rel_gguf_path}"

    template_path = template_modelfile.strip()
    lines = []
    template_system_prompt = None
    if template_path:
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template Modelfile not found: {template_path}")
        with open(template_path, "r", encoding="utf-8") as f:
            template_text = f.read()

        template_system_prompt = _extract_first_system_prompt(template_text)
        template_lines = _remove_system_blocks(template_text).splitlines()

        from_replaced = False
        for line in template_lines:
            stripped = line.strip()
            upper = stripped.upper()
            if upper.startswith("ADAPTER "):
                # GGUF Modelfiles should load from a GGUF file directly.
                continue
            if upper.startswith("FROM "):
                lines.append(f"FROM {rel_gguf_path}")
                from_replaced = True
                continue
            lines.append(line)
        if not from_replaced:
            lines.insert(0, f"FROM {rel_gguf_path}")
    else:
        lines = [
            "# Auto-generated by fine_tuning/build_gguf.py",
            f"FROM {rel_gguf_path}",
            "",
            "PARAMETER temperature 0.1",
            "PARAMETER num_ctx 4096",
        ]
    resolved_system_prompt = _resolve_system_prompt(system_prompt, template_system_prompt)
    lines.extend(["", f'SYSTEM """{resolved_system_prompt}"""'])

    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()

    llama_cpp_dir = _abs_path(args.llama_cpp_dir) if args.llama_cpp_dir.strip() else _discover_llama_cpp_dir()
    hf_model_dir = _abs_path(args.hf_model_dir)
    output_dir = _abs_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if not llama_cpp_dir:
        raise FileNotFoundError(
            "Could not auto-detect llama.cpp directory. Set --llama-cpp-dir explicitly or set LLAMA_CPP_DIR."
        )
    llama_cpp_dir = os.path.abspath(llama_cpp_dir)

    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found: {convert_script}")
    if not os.path.isdir(hf_model_dir):
        raise FileNotFoundError(f"HF model directory not found: {hf_model_dir}")

    base_name = args.name.strip() or _slugify(os.path.basename(hf_model_dir))
    base_gguf = os.path.join(output_dir, f"{base_name}.{args.outtype}.gguf")
    patched = _maybe_patch_phi_tokenizer_config(hf_model_dir)
    try:
        _run(
            [
                args.python,
                convert_script,
                hf_model_dir,
                "--outfile",
                base_gguf,
                "--outtype",
                args.outtype,
            ],
            cwd=llama_cpp_dir,
        )
    finally:
        if patched:
            patched_path, original_text = patched
            _restore_patched_file(patched_path, original_text)
            print(f"[INFO] Restored original tokenizer config: {patched_path}")

    final_gguf = base_gguf
    quant = args.quantize.strip()
    if quant:
        quantize_bin = _find_quantize_binary(llama_cpp_dir)
        if not quantize_bin:
            raise FileNotFoundError(
                "Quantize binary not found. Build llama.cpp first (llama-quantize/quantize), "
                "or run without --quantize."
            )
        quant_gguf = os.path.join(output_dir, f"{base_name}.{quant}.gguf")
        _run([quantize_bin, base_gguf, quant_gguf, quant], cwd=llama_cpp_dir)
        final_gguf = quant_gguf

    modelfile_path = os.path.join(output_dir, f"{base_name}.gguf.Modelfile")
    template_modelfile = args.template_modelfile.strip()
    if template_modelfile:
        template_modelfile = _abs_path(template_modelfile)
    else:
        template_modelfile = _find_auto_template_modelfile(base_name)
    if template_modelfile:
        print(f"- Template Modelfile: {template_modelfile}")
    else:
        print("- Template Modelfile: <none> (using enhanced fallback SYSTEM prompt)")

    _write_modelfile(modelfile_path, final_gguf, args.system_prompt, template_modelfile=template_modelfile)
    patched_modelfiles = _backfill_missing_system_prompts(output_dir, ENHANCED_SYSTEM_PROMPT)

    print("\n[OK] GGUF build complete")
    print(f"- Base GGUF: {base_gguf}")
    if quant:
        print(f"- Quantized GGUF: {final_gguf}")
    print(f"- Modelfile: {modelfile_path}")
    if patched_modelfiles:
        print(f"- Backfilled SYSTEM prompt in {patched_modelfiles} existing GGUF Modelfile(s)")

    ollama_model = args.ollama_model.strip()
    if args.create_ollama or ollama_model:
        if not ollama_model:
            ollama_model = base_name
        _run(["ollama", "create", ollama_model, "-f", modelfile_path], cwd=ROOT_DIR)
        print(f"- Ollama model: {ollama_model}")


if __name__ == "__main__":
    main()
