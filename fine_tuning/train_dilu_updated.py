import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from fine_tuning.pipeline import (
    apply_windows_short_temp,
    build_training_metadata,
    default_lora_config,
    extract_action_id,
    infer_model_family,
    resolve_family_config,
    save_json,
    to_chat_texts,
)


DATA_FILE = "data/gold_standard_data_clean.jsonl"
OUTPUT_DIR = "fine_tuning/checkpoints"
MERGED_MODEL_DIR = "fine_tuning/merged_models/dilu-llama3_1-8b-v1"
MODEL_NAME = "unsloth/Llama-3.1-8B-Instruct"


def _resolve_data_file(path_str: str) -> str:
    if os.path.exists(path_str):
        return path_str
    sandbox_path = os.path.join("/mnt/data", os.path.basename(path_str))
    if os.path.exists(sandbox_path):
        return sandbox_path
    return path_str


def _dataset_action_distribution(dataset):
    counts = {}
    for row in dataset:
        action = extract_action_id(row.get("output", ""))
        key = "unknown" if action is None else str(action)
        counts[key] = counts.get(key, 0) + 1
    return counts


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DiLu local models and export merged weights for Ollama.")
    parser.add_argument("--data-file", default=DATA_FILE, help="Path to cleaned JSONL dataset.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Training checkpoint directory.")
    parser.add_argument("--merged-model-dir", default=MERGED_MODEL_DIR, help="Merged model output directory.")
    parser.add_argument("--model-name", default=MODEL_NAME, help="Base model to fine-tune.")
    parser.add_argument("--model-family", default="auto", choices=["auto", "llama3", "qwen", "mistral"], help="Model-family presets.")
    parser.add_argument("--chat-template", default="", help="Override tokenizer chat template.")
    parser.add_argument("--lora-target-modules", default="", help="Comma-separated override for LoRA target modules.")
    parser.add_argument("--max-seq-length", type=int, default=0, help="Override max sequence length (0 = family default).")
    parser.add_argument("--max-steps", type=int, default=60, help="Training max steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Training learning rate.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio (0 disables eval).")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size.")
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--save-adapter-only", action="store_true", help="Save only LoRA adapter instead of merged model.")
    parser.add_argument("--adapter-dir", default="fine_tuning/adapters/latest", help="Adapter output path when --save-adapter-only is used.")
    return parser.parse_args()


def train(args):
    import inspect
    import unsloth  # noqa: F401 - must be imported before trl/transformers for patching
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer

    apply_windows_short_temp()
    family = infer_model_family(args.model_name) if args.model_family == "auto" else args.model_family
    family_cfg = resolve_family_config(family, max_seq_length=args.max_seq_length)

    chat_template = args.chat_template or family_cfg["chat_template"]
    lora_targets = family_cfg["lora_targets"]
    if args.lora_target_modules.strip():
        lora_targets = [x.strip() for x in args.lora_target_modules.split(",") if x.strip()]

    print(f"Loading model: {args.model_name}")
    print(f"Family: {family} | Template: {chat_template} | MaxSeq: {family_cfg['max_seq_length']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=int(family_cfg["max_seq_length"]),
        dtype=None,
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template,
        mapping={"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    )

    lora_cfg = default_lora_config()
    model = FastLanguageModel.get_peft_model(
        model,
        target_modules=lora_targets,
        **lora_cfg,
    )

    data_file = _resolve_data_file(args.data_file)
    print(f"Loading dataset: {data_file}")
    dataset = load_dataset("json", data_files=data_file, split="train")
    eval_dataset = None
    if 0 < args.val_ratio < 1:
        split = dataset.train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset

    train_dist = _dataset_action_distribution(train_dataset)
    eval_dist = _dataset_action_distribution(eval_dataset) if eval_dataset is not None else None
    print(f"Train rows: {len(train_dataset)} | Eval rows: {len(eval_dataset) if eval_dataset is not None else 0}")
    print(f"Train action distribution: {train_dist}")
    if eval_dist is not None:
        print(f"Eval action distribution: {eval_dist}")

    eval_enabled = eval_dataset is not None and len(eval_dataset) > 0
    training_args_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        warmup_steps=5,
        learning_rate=args.learning_rate,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        report_to="none",
    )
    if eval_enabled:
        training_args_kwargs["eval_steps"] = 10
    eval_mode = "steps" if eval_enabled else "no"
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in ta_params:
        training_args_kwargs["evaluation_strategy"] = eval_mode
    elif "eval_strategy" in ta_params:
        training_args_kwargs["eval_strategy"] = eval_mode

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=int(family_cfg["max_seq_length"]),
        dataset_num_proc=1,
        formatting_func=lambda examples: to_chat_texts(tokenizer, examples),
        args=TrainingArguments(**training_args_kwargs),
    )

    print("Starting training...")
    trainer.train()

    os.makedirs(args.output_dir, exist_ok=True)
    metadata = build_training_metadata(
        base_model=args.model_name,
        model_family=family,
        chat_template=chat_template,
        lora_target_modules=lora_targets,
        data_file=data_file,
        output_dir=args.output_dir,
        merged_model_dir=args.merged_model_dir,
        adapter_dir=args.adapter_dir if args.save_adapter_only else None,
        max_seq_length=int(family_cfg["max_seq_length"]),
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        val_ratio=args.val_ratio,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        train_rows=len(train_dataset),
        eval_rows=len(eval_dataset) if eval_dataset is not None else 0,
        train_action_distribution=train_dist,
        eval_action_distribution=eval_dist,
    )
    save_json(os.path.join(args.output_dir, "run_metadata.json"), metadata)

    if args.save_adapter_only:
        print(f"Saving adapter only: {args.adapter_dir}")
        model.save_pretrained(args.adapter_dir)
        tokenizer.save_pretrained(args.adapter_dir)
    else:
        print(f"Saving merged model: {args.merged_model_dir}")
        model.save_pretrained_merged(
            args.merged_model_dir,
            tokenizer,
            save_method="merged_16bit",
        )


if __name__ == "__main__":
    train(parse_args())
