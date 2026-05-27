import os
import json
import gc
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

import random
import numpy as np


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# =========================
# 1) Model configuration
# =========================

MODEL_DIRS = {
    "llama31": "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/Llama-3.1-8B-Instruct",
    "mistral": "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/Mistral-7B-Instruct-v0.3",
    "deepseek": "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/deepseek-llm-7b-chat",
    "gemma2": "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/gemma-2-9b-it",
    "glm4": "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/glm-4-9b-chat-hf",
    "qwen25": "/dss/dssmcmlfs01/pn25ju/pn25ju-dss-0000/models/Qwen2.5-7B-Instruct",
}

# =========================
# 2) Task configuration
# =========================

TASK_CONFIGS = {
    ("task1", "zh_en"): {
        "input_path": "data/idiom_base.json",
        "source_field": "source_idiom",
        "output_field": "translation_en",
        "output_dir": "outputs/task1",
    },
    ("task1", "en_zh"): {
        "input_path": "outputs/task1/task1_zh_en_{model}_{zh_en_prompt_version}_clean.jsonl",
        "source_field": "translation_en",
        "output_field": "translation_zh",
        "output_dir": "outputs/task1",
    },

    ("task2", "zh_en"): {
        "input_path": "data/idiom_base.json",
        "source_field": "sentence",
        "output_field": "translation_en",
        "output_dir": "outputs/task2",
    },
    ("task2", "en_zh"): {
        "input_path": "outputs/task2/task2_zh_en_{model}_{zh_en_prompt_version}_clean.jsonl",
        "source_field": "translation_en",
        "output_field": "translation_zh",
        "output_dir": "outputs/task2",
    },

    ("task3_3", "zh_en"): {
        "input_path": "data/idiom_polarity.json",
        "source_field": "sentence",
        "output_field": "translation_en",
        "output_dir": "outputs/task3_3",
    },
    ("task3_3", "en_zh"): {
        "input_path": "outputs/task3_3/task3_3_zh_en_{model}_{zh_en_prompt_version}_clean.jsonl",
        "source_field": "translation_en",
        "output_field": "translation_zh",
        "output_dir": "outputs/task3_3",
    },
}

# =========================
# 3) Prompt configuration
# =========================

PROMPT_CONFIGS = {
    ("task1", "zh_en", "v1"): {
        "prompt_path": "prompts/task1_zh_en_v1.txt",
        "prompt_fields": ["source_text"],
    },
    ("task1", "zh_en", "v2"): {
        "prompt_path": "prompts/task1_zh_en_v2.txt",
        "prompt_fields": ["source_text"],
    },
    ("task1", "en_zh", "v1"): {
        "prompt_path": "prompts/task1_en_zh_v1.txt",
        "prompt_fields": ["source_text"],
    },
    ("task1", "en_zh", "v2"): {
        "prompt_path": "prompts/task1_en_zh_v2.txt",
        "prompt_fields": ["source_text"],
    },
    ("task1", "en_zh", "v3"): {
        "prompt_path": "prompts/task1_en_zh_v3.txt",
        "prompt_fields": ["source_text"],
    },

    ("task2", "zh_en", "v1"): {
        "prompt_path": "prompts/task2_zh_en_v1.txt",
        "prompt_fields": ["source_text"],
    },
    ("task2", "zh_en", "v2"): {
        "prompt_path": "prompts/task2_zh_en_v2.txt",
        "prompt_fields": ["source_text"],
    },
    ("task2", "zh_en", "v3"): {
        "prompt_path": "prompts/task2_zh_en_v3.txt",
        "prompt_fields": ["source_text"],
    },
    ("task2", "en_zh", "v1"): {
        "prompt_path": "prompts/task2_en_zh_v1.txt",
        "prompt_fields": ["source_text"],
    },
    ("task2", "en_zh", "v2"): {
        "prompt_path": "prompts/task2_en_zh_v2.txt",
        "prompt_fields": ["source_text"],
    },
    ("task2", "en_zh", "v3"): {
        "prompt_path": "prompts/task2_en_zh_v3.txt",
        "prompt_fields": ["source_text", "source_idiom"],
    },
    ("task2", "en_zh", "v4"): {
        "prompt_path": "prompts/task2_en_zh_v4.txt",
        "prompt_fields": ["source_text", "source_idiom"],
    },
    ("task2", "en_zh", "v5"): {
        "prompt_path": "prompts/task2_en_zh_v5.txt",
        "prompt_fields": ["source_text", "source_idiom"],
    },
    ("task2", "en_zh", "v6"): {
        "prompt_path": "prompts/task2_en_zh_v6.txt",
        "prompt_fields": ["source_text"],
    },
    ("task2", "en_zh", "v7"): {
        "prompt_path": "prompts/task2_en_zh_v7.txt",
        "prompt_fields": ["source_text", "source_idiom"],
    },
    ("task2", "en_zh", "v8"): {
        "prompt_path": "prompts/task2_en_zh_v8.txt",
        "prompt_fields": ["source_text"],
    },
    ("task2", "en_zh", "v9"): {
        "prompt_path": "prompts/task2_en_zh_v9.txt",
        "prompt_fields": ["source_text"],
    },
    ("task2", "en_zh", "v10"): {
        "prompt_path": "prompts/task2_en_zh_v10.txt",
        "prompt_fields": ["source_text"],
    },

    ("task3_3", "zh_en", "v1"): {
        "prompt_path": "prompts/task3_3_zh_en_v1.txt",
        "prompt_fields": ["source_text"],
    },
    ("task3_3", "zh_en", "v2"): {
        "prompt_path": "prompts/task3_3_zh_en_v2.txt",
        "prompt_fields": ["source_text"],
    },
    ("task3_3", "en_zh", "v1"): {
        "prompt_path": "prompts/task3_3_en_zh_v1.txt",
        "prompt_fields": ["source_text"],
    },
    ("task3_3", "en_zh", "v2"): {
        "prompt_path": "prompts/task3_3_en_zh_v2.txt",
        "prompt_fields": ["source_text"],
    },
    ("task3_3", "en_zh", "v3"): {
        "prompt_path": "prompts/task3_3_en_zh_v3.txt",
        "prompt_fields": ["source_text", "source_idiom"],
    },
}

# =========================
# 4) Generation settings
# =========================

MAX_NEW_TOKENS = 128
DO_SAMPLE = False
TEMPERATURE = 0.0
TOP_P = 1.0

# =========================
# 5) Utilities
# =========================


def load_samples(path: str) -> List[Dict[str, Any]]:
    """Load samples from JSON or JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read().strip()

    if not txt:
        return []

    if txt.startswith("["):
        data = json.loads(txt)
        if not isinstance(data, list):
            raise ValueError(f"{path} looks like a JSON array but is not a list.")
        return data

    samples = []
    for line in txt.splitlines():
        line = line.strip()
        if line:
            samples.append(json.loads(line))
    return samples


def load_prompt_template(path: str) -> str:
    """Load prompt template from file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def ensure_dir(path: str) -> None:
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def validate_sample(sample: Dict[str, Any], source_field: str) -> None:
    """Ensure required fields exist in sample."""
    if "index" not in sample:
        raise KeyError("Each sample must contain an 'index' field.")
    if source_field not in sample:
        raise KeyError(
            f"Missing source field '{source_field}' in sample index={sample.get('index')}"
        )


def build_prompt_values(
    sample: Dict[str, Any],
    source_field: str,
    prompt_fields: List[str]
) -> Dict[str, Any]:
    """Build only the fields allowed to be exposed to the model."""
    values = {}

    for field in prompt_fields:
        if field == "source_text":
            validate_sample(sample, source_field)
            values["source_text"] = sample[source_field]
        else:
            if field not in sample:
                raise KeyError(
                    f"Missing required prompt field '{field}' in sample index={sample.get('index')}"
                )
            values[field] = sample[field]

    return values


def render_prompt(
    template: str,
    sample: Dict[str, Any],
    source_field: str,
    prompt_fields: List[str]
) -> str:
    """Render prompt text using template and allowed fields."""
    values = build_prompt_values(sample, source_field, prompt_fields)
    return template.format(**values)


def build_messages(prompt_text: str) -> List[Dict[str, str]]:
    """Wrap prompt text in a user message for chat models."""
    return [{"role": "user", "content": prompt_text}]


def maybe_print_prompt(prompt_text: str, index_value: Any, show_prompt: bool) -> None:
    """Print prompt for debugging."""
    if show_prompt:
        print("\n========== PROMPT ==========")
        print(f"index: {index_value}")
        print(prompt_text)
        print("======== END PROMPT ========\n")


def is_upstream_invalid_en(sample: Dict[str, Any]) -> bool:
    """Check whether upstream zh->en result is already invalid."""
    return sample.get("is_invalid_en") == "True"


# =========================
# 6) Translation
# =========================

@torch.no_grad()
def translate_with_model(
    model_dir: str,
    samples: List[Dict[str, Any]],
    source_field: str,
    output_field: str,
    prompt_template: str,
    prompt_fields: List[str],
    direction: str,
    show_prompt: bool = False,
) -> List[Dict[str, Any]]:
    """Run translation using a specified model."""
    print(f"\n=== Loading model: {model_dir} ===")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        use_fast=True,
        trust_remote_code=True,
    )

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    results = []
    model_name = os.path.basename(model_dir.rstrip("/"))

    for i, ex in enumerate(tqdm(samples, desc=model_name)):
        # -------------------------------------------------
        # Special handling for en_zh:
        # if upstream zh_en is invalid, skip generation
        # -------------------------------------------------
        if direction == "en_zh" and is_upstream_invalid_en(ex):
            result = dict(ex)
            result[output_field] = ""
            result["is_invalid_zh"] = "True"
            result["invalid_type"] = "upstream_invalid_en"
            results.append(result)
            continue

        prompt_text = render_prompt(
            template=prompt_template,
            sample=ex,
            source_field=source_field,
            prompt_fields=prompt_fields,
        )

        if show_prompt and i < 3:
            maybe_print_prompt(prompt_text, ex.get("index"), show_prompt=True)

        # Prepare input for chat or non-chat models
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                messages = build_messages(prompt_text)
                rendered = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if isinstance(rendered, torch.Tensor):
                    inputs = {"input_ids": rendered}
                else:
                    inputs = dict(rendered)
            except Exception:
                enc = tokenizer(prompt_text, return_tensors="pt")
                inputs = dict(enc)
        else:
            enc = tokenizer(prompt_text, return_tensors="pt")
            inputs = dict(enc)

        # Move tensors to model device
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(model.device)

        # Remove token_type_ids if present
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        # Generate output
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode output
        prompt_len = inputs["input_ids"].shape[-1]
        gen_ids = output_ids[0, prompt_len:]
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        generated_text = generated_text.replace("\u200b", "").strip()

        # Keep only the first paragraph / line
        generated_text = generated_text.split("\n\n")[0].strip()
        generated_text = generated_text.split("\n")[0].strip()

        result = dict(ex)
        result[output_field] = generated_text
        results.append(result)

    # Cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return results


# =========================
# 7) Main
# =========================

def main():
    """Main entry point for translation pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=["task1", "task2", "task3_3"])
    parser.add_argument("--direction", required=True, choices=["zh_en", "en_zh"])
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_DIRS.keys()) + ["all"],
        help="Choose one model or 'all'.",
    )
    parser.add_argument(
        "--prompt_version",
        required=True,
        help="Prompt version for zh_en or en_zh.",
    )
    parser.add_argument(
        "--zh_en_prompt_version",
        default=None,
        help="Required for en_zh: specifies which zh_en output version to use.",
    )
    parser.add_argument(
        "--show_prompt",
        action="store_true",
        help="Print first 3 rendered prompts for debugging.",
    )
    args = parser.parse_args()

    task_key = (args.task, args.direction)

    # Determine prompt version
    if args.direction == "zh_en":
        zh_en_prompt_version = args.prompt_version
        en_zh_prompt_version = None
        prompt_key = (args.task, args.direction, zh_en_prompt_version)
    else:
        if args.zh_en_prompt_version is None:
            raise ValueError("For en_zh, you must provide --zh_en_prompt_version.")
        zh_en_prompt_version = args.zh_en_prompt_version
        en_zh_prompt_version = args.prompt_version
        prompt_key = (args.task, args.direction, en_zh_prompt_version)

    if task_key not in TASK_CONFIGS:
        raise ValueError(f"No task config found for {task_key}")

    if prompt_key not in PROMPT_CONFIGS:
        raise ValueError(f"No prompt config found for {prompt_key}")

    task_config = TASK_CONFIGS[task_key]
    prompt_config = PROMPT_CONFIGS[prompt_key]

    prompt_template = load_prompt_template(prompt_config["prompt_path"])
    print(f"Loaded prompt template from: {prompt_config['prompt_path']}")

    model_names = list(MODEL_DIRS.keys()) if args.model == "all" else [args.model]

    for model_name in model_names:
        model_dir = MODEL_DIRS[model_name]

        input_path = task_config["input_path"].format(
            model=model_name,
            zh_en_prompt_version=zh_en_prompt_version,
        )
        samples = load_samples(input_path)
        print(f"Loaded {len(samples)} samples from {input_path}")

        ensure_dir(task_config["output_dir"])

        if args.direction == "zh_en":
            out_filename = (
                f"{args.task}_{args.direction}_{model_name}_{zh_en_prompt_version}.jsonl"
            )
        else:
            out_filename = (
                f"{args.task}_{args.direction}_{model_name}_"
                f"{zh_en_prompt_version}_{en_zh_prompt_version}.jsonl"
            )

        out_path = os.path.join(task_config["output_dir"], out_filename)

        results = translate_with_model(
            model_dir=model_dir,
            samples=samples,
            source_field=task_config["source_field"],
            output_field=task_config["output_field"],
            prompt_template=prompt_template,
            prompt_fields=prompt_config["prompt_fields"],
            direction=args.direction,
            show_prompt=args.show_prompt,
        )

        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()