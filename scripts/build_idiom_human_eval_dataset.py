import os
import json
import random
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation


# =========================
# Config
# =========================

SAMPLE_FILE = "data/idiom_eval_human.json"
OUTPUT_DIR = "human_eval/input"

RANDOM_SEED = 42

MODELS = {
    "llama31": "M01",
    "mistral": "M02",
    "deepseek": "M03",
    "gemma2": "M04",
    "glm4": "M05",
    "qwen25": "M06",
}


TASK_CONFIGS = [
    {
        "task": "task1",
        "eval_type": "zh_en",
        "version": "v1_v1",
        "input_pattern": "outputs/task1/task1_en_zh_{model}_v1_v1_clean.jsonl",
        "output_file": "task1_zh_en_human_eval.json",
        "eval_fields": [
            "code_switching",
            "grammar",
            "fluency",
            "adequacy",
            "english_rendering_strategy",
        ],
        "remove_fields": [
            "sentence",
            "translation_zh",
            "is_invalid_zh",
            "invalid_type_zh",
        ],
    },
    {
        "task": "task1",
        "eval_type": "en_zh",
        "version": "v1_v1",
        "input_pattern": "outputs/task1/task1_en_zh_{model}_v1_v1_clean.jsonl",
        "output_file": "task1_en_zh_human_eval.json",
        "eval_fields": [
            "code_switching",
            "grammar",
            "fluency",
            "adequacy",
            "idiom_form",
        ],
        "remove_fields": [
            "sentence",
            "source_idiom",
            "is_invalid_en",
            "invalid_type_en",
        ],
    },
    {
        "task": "task1",
        "eval_type": "zh_zh",
        "version": "v1_v1",
        "input_pattern": "outputs/task1/task1_en_zh_{model}_v1_v1_clean.jsonl",
        "output_file": "task1_zh_zh_human_eval.json",
        "eval_fields": [
            "code_switching",
            "grammar",
            "fluency",
            "adequacy",
            "idiom_form",
        ],
        "remove_fields": [
            "sentence",
            "translation_en",
            "is_invalid_en",
            "invalid_type_en",
        ],
    },
    {
        "task": "task2",
        "eval_type": "zh_en",
        "version": "v3_v1",
        "input_pattern": "outputs/task2/task2_en_zh_{model}_v3_v1_clean.jsonl",
        "output_file": "task2_zh_en_v3_v1_human_eval.json",
        "eval_fields": [
            "code_switching",
            "grammar",
            "fluency",
            "adequacy",
            "idiom_related_meaning_translation",
        ],
        "remove_fields": [
            "translation_zh",
            "is_invalid_zh",
            "invalid_type_zh",
        ],
    },
    {
        "task": "task2",
        "eval_type": "en_zh",
        "version": "v3_v1",
        "input_pattern": "outputs/task2/task2_en_zh_{model}_v3_v1_clean.jsonl",
        "output_file": "task2_en_zh_v3_v1_human_eval.json",
        "eval_fields": [
            "code_switching",
            "grammar",
            "fluency",
            "adequacy",
            "idiom_related_meaning_translation",
            "idiom_use",
        ],
        "remove_fields": [
            "source_idiom",
            "sentence",
            "is_invalid_en",
            "invalid_type_en",
        ],
    },
    {
        "task": "task2",
        "eval_type": "zh_zh",
        "version": "v3_v1",
        "input_pattern": "outputs/task2/task2_en_zh_{model}_v3_v1_clean.jsonl",
        "output_file": "task2_zh_zh_v3_v1_human_eval.json",
        "eval_fields": [
            "code_switching",
            "grammar",
            "fluency",
            "adequacy",
            "idiom_related_meaning_preservation",
            "idiom_use",
        ],
        "remove_fields": [
            "translation_en",
            "is_invalid_en",
            "invalid_type_en",
        ],
    },
    {
        "task": "task2",
        "eval_type": "en_zh",
        "version": "v3_v2",
        "input_pattern": "outputs/task2/task2_en_zh_{model}_v3_v2_clean.jsonl",
        "output_file": "task2_en_zh_v3_v2_human_eval.json",
        "eval_fields": [
            "code_switching",
            "grammar",
            "fluency",
            "adequacy",
            "idiom_reintroduction",
        ],
        "remove_fields": [
            "source_idiom",
            "sentence",
            "is_invalid_en",
            "invalid_type_en",
        ],
    },
    {
        "task": "task2",
        "eval_type": "zh_zh",
        "version": "v3_v2",
        "input_pattern": "outputs/task2/task2_en_zh_{model}_v3_v2_clean.jsonl",
        "output_file": "task2_zh_zh_v3_v2_human_eval.json",
        "eval_fields": [
            "code_switching",
            "grammar",
            "fluency",
            "adequacy",
            "idiom_reintroduction",
        ],
        "remove_fields": [
            "translation_en",
            "is_invalid_en",
            "invalid_type_en",
        ],
    },
    {
        "task": "task2",
        "eval_type": "en_zh",
        "version": "v3_v3",
        "input_pattern": "outputs/task2/task2_en_zh_{model}_v3_v3_clean.jsonl",
        "output_file": "task2_en_zh_v3_v3_human_eval.json",
        "target_idiom": True,
        "eval_fields": [
            "code_switching",
            "grammar",
            "fluency",
            "adequacy",
            "target_idiom_use",
        ],
        "remove_fields": [
            "sentence",
            "is_invalid_en",
            "invalid_type_en",
        ],
    },
    {
        "task": "task2",
        "eval_type": "zh_zh",
        "version": "v3_v3",
        "input_pattern": "outputs/task2/task2_en_zh_{model}_v3_v3_clean.jsonl",
        "output_file": "task2_zh_zh_v3_v3_human_eval.json",
        "target_idiom": True,
        "eval_fields": [
            "code_switching",
            "grammar",
            "fluency",
            "adequacy",
            "target_idiom_use",
        ],
        "remove_fields": [
            "translation_en",
            "is_invalid_en",
            "invalid_type_en",
        ],
    },
]


SCORE_FIELDS = {
    "code_switching",
    "grammar",
    "fluency",
    "adequacy",
    "english_rendering_strategy",
    "idiom_form",
    "idiom_related_meaning_translation",
    "idiom_related_meaning_preservation",
    "idiom_reintroduction",
    "target_idiom_use",
}

CATEGORICAL_FIELDS = {
    "idiom_use": ["I0", "I1", "NA"],
}


# =========================
# Helpers
# =========================

def load_json_or_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    if not text:
        return []

    if text.startswith("["):
        return json.loads(text)

    data = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            data.append(json.loads(line))

    return data


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def save_excel(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="annotation")

        worksheet = writer.sheets["annotation"]
        worksheet.freeze_panes = "A2"

        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(wrap_text=True, vertical="top")

        for column_cells in worksheet.columns:
            column_letter = column_cells[0].column_letter
            header = column_cells[0].value

            if header in [
                "sentence",
                "translation_en",
                "translation_zh",
            ]:
                worksheet.column_dimensions[column_letter].width = 45
            elif header in [
                "eval_id",
                "task",
                "eval_type",
                "version",
                "index",
                "source_idiom",
                "target_idiom",
                "model_id_anonymous",
            ]:
                worksheet.column_dimensions[column_letter].width = 18
            else:
                worksheet.column_dimensions[column_letter].width = 16

            for cell in column_cells:
                cell.alignment = Alignment(wrap_text=True, vertical="top")

        max_row = worksheet.max_row

        for cell in worksheet[1]:
            header = cell.value
            column_letter = cell.column_letter

            if header in SCORE_FIELDS:
                validation = DataValidation(
                    type="list",
                    formula1='"1,2,3,4,5,NA"',
                    allow_blank=True,
                )
                worksheet.add_data_validation(validation)
                validation.add(f"{column_letter}2:{column_letter}{max_row}")

            elif header in CATEGORICAL_FIELDS:
                values = ",".join(CATEGORICAL_FIELDS[header])
                validation = DataValidation(
                    type="list",
                    formula1=f'"{values}"',
                    allow_blank=True,
                )
                worksheet.add_data_validation(validation)
                validation.add(f"{column_letter}2:{column_letter}{max_row}")


def index_by_index(data):
    return {int(item["index"]): item for item in data}


def get_bool_value(item, key, default=None):
    value = item.get(key, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False

    return value


def should_mark_eval_as_na(config, result_item):
    eval_type = config["eval_type"]

    if eval_type == "zh_en":
        return get_bool_value(result_item, "is_invalid_en", False) is True

    if eval_type in ["en_zh", "zh_zh"]:
        return get_bool_value(result_item, "is_invalid_zh", False) is True

    return False


def build_eval_item(eval_id, config, sample_item, result_item, anonymous_model_id):
    item = {
        "eval_id": eval_id,
        "task": config["task"],
        "eval_type": config["eval_type"],
        "version": config["version"],
        "index": sample_item["index"],
        "source_idiom": sample_item["source_idiom"],
        "sentence": sample_item["sentence"],
        "translation_en": result_item.get("translation_en"),
        "is_invalid_en": get_bool_value(result_item, "is_invalid_en", None),
    }

    if "invalid_type_en" in result_item:
        item["invalid_type_en"] = result_item.get("invalid_type_en")

    item["translation_zh"] = result_item.get("translation_zh")
    item["is_invalid_zh"] = get_bool_value(result_item, "is_invalid_zh", None)

    if "invalid_type_zh" in result_item:
        item["invalid_type_zh"] = result_item.get("invalid_type_zh")

    item["model_id_anonymous"] = anonymous_model_id

    if config.get("target_idiom"):
        item["target_idiom"] = sample_item["source_idiom"]

    mark_as_na = should_mark_eval_as_na(config, result_item)

    for field in config["eval_fields"]:
        if mark_as_na:
            item[field] = "NA"
        else:
            item[field] = ""

    for field in config.get("remove_fields", []):
        item.pop(field, None)

    return item


# =========================
# Main
# =========================

def main():
    random.seed(RANDOM_SEED)

    samples = load_json_or_jsonl(SAMPLE_FILE)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for config in TASK_CONFIGS:
        all_eval_items = []
        counter = 1

        for sample_item in samples:
            source_index = int(sample_item["index"])

            for model_name, anonymous_model_id in MODELS.items():
                input_file = config["input_pattern"].format(model=model_name)

                if not os.path.exists(input_file):
                    raise FileNotFoundError(f"Input file not found: {input_file}")

                result_data = load_json_or_jsonl(input_file)
                result_by_index = index_by_index(result_data)

                if source_index not in result_by_index:
                    raise KeyError(
                        f"Index {source_index} not found in {input_file}"
                    )

                result_item = result_by_index[source_index]

                eval_id = (
                    f"{config['task']}_"
                    f"{config['eval_type']}_"
                    f"{config['version']}_"
                    f"{counter:04d}"
                )

                eval_item = build_eval_item(
                    eval_id=eval_id,
                    config=config,
                    sample_item=sample_item,
                    result_item=result_item,
                    anonymous_model_id=anonymous_model_id,
                )

                all_eval_items.append(eval_item)
                counter += 1

        json_output_path = os.path.join(OUTPUT_DIR, config["output_file"])
        save_json(all_eval_items, json_output_path)

        excel_output_file = config["output_file"].replace(".json", ".xlsx")
        excel_output_path = os.path.join(OUTPUT_DIR, excel_output_file)
        save_excel(all_eval_items, excel_output_path)

        print(f"Saved {len(all_eval_items)} items to {json_output_path}")
        print(f"Saved {len(all_eval_items)} items to {excel_output_path}")


if __name__ == "__main__":
    main()