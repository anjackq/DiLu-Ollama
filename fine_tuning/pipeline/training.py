from typing import Dict, List


def to_chat_texts(tokenizer, examples) -> List[str]:
    texts: List[str] = []
    for instruction, input_text, output_text in zip(
        examples["instruction"],
        examples["input"],
        examples["output"],
    ):
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
        ]
        texts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )
    return texts


def default_lora_config() -> Dict[str, object]:
    return {
        "r": 16,
        "lora_alpha": 16,
        "lora_dropout": 0,
        "bias": "none",
        "use_gradient_checkpointing": "unsloth",
        "random_state": 3407,
    }

