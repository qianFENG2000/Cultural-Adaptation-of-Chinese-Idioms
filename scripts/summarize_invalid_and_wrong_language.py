import json
import csv
from pathlib import Path
from collections import Counter, defaultdict


# =========================================================
# Config
# =========================================================

ROOT_DIR = Path("outputs")
TASK_DIRS = ["task1", "task2"]

OUTPUT_DIR = Path("eval/merged")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INVALID_TYPES = [
    "meta_output",
    "upstream_invalid_en",
    "refusal",
    "wrong_input_claim",
    "explanatory_translation",
    "repetition",
    "encoding_error",
]

EVAL_TYPES = ["zh_en", "en_zh", "zh_zh"]

TASK2_CONDITION_MAP = {
    "v1": "C1",
    "v2": "C2",
    "v3": "C3",
}


# =========================================================
# Helpers
# =========================================================

def load_jsonl(path: Path):
    """Load a JSONL file."""
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if line:
                row = json.loads(line)
                row["_line_no"] = line_no
                rows.append(row)

    return rows


def safe_divide(a, b):
    """Safely divide two numbers."""
    return a / b if b else 0.0


def parse_filename(path: Path):
    """
    Parse filenames like:
      task1_en_zh_deepseek_v1_v1_clean.jsonl
      task2_en_zh_deepseek_v3_v1_clean.jsonl

    Returns:
      task
      file_direction
      model
      zh_en_prompt_version
      en_zh_prompt_version
      condition
      filename
    """
    stem = path.stem

    if stem.endswith("_clean"):
        stem = stem[:-6]

    parts = stem.split("_")

    if len(parts) < 6:
        raise ValueError(f"Unexpected filename format: {path.name}")

    task = parts[0]
    file_direction = "_".join(parts[1:3])
    model = parts[3]
    zh_en_prompt_version = parts[4]
    en_zh_prompt_version = parts[5]

    condition = ""

    if task == "task2":
        condition = TASK2_CONDITION_MAP.get(
            en_zh_prompt_version,
            en_zh_prompt_version,
        )

    return {
        "task": task,
        "file_direction": file_direction,
        "model": model,
        "zh_en_prompt_version": zh_en_prompt_version,
        "en_zh_prompt_version": en_zh_prompt_version,
        "condition": condition,
        "filename": path.name,
    }


def str_is_true(value):
    """Treat boolean True or string 'True' as True."""
    return str(value).strip() == "True"


def get_invalid_type(row, eval_type):
    """
    invalid_type:
      zh_en -> invalid_type_en
      en_zh -> invalid_type_zh
      zh_zh -> prefer invalid_type_en, else invalid_type_zh
    """
    invalid_type_en = str(row.get("invalid_type_en", "")).strip()
    invalid_type_zh = str(row.get("invalid_type_zh", "")).strip()

    if eval_type == "zh_en":
        return invalid_type_en if invalid_type_en else None

    if eval_type == "en_zh":
        return invalid_type_zh if invalid_type_zh else None

    if eval_type == "zh_zh":
        if invalid_type_en:
            return invalid_type_en
        if invalid_type_zh:
            return invalid_type_zh
        return None

    return None


def is_invalid_row(row, eval_type):
    """
    invalid:
      zh_en -> is_invalid_en == True
      en_zh -> is_invalid_zh == True
      zh_zh -> is_invalid_en == True OR is_invalid_zh == True

    Returns:
      (is_invalid: bool, invalid_type: str | None)
    """
    invalid_type = get_invalid_type(row, eval_type)

    invalid_en = str_is_true(row.get("is_invalid_en", "False"))
    invalid_zh = str_is_true(row.get("is_invalid_zh", "False"))

    if eval_type == "zh_en":
        if invalid_en:
            return True, invalid_type if invalid_type else "unspecified_invalid"

    elif eval_type == "en_zh":
        if invalid_zh:
            return True, invalid_type if invalid_type else "unspecified_invalid"

    elif eval_type == "zh_zh":
        if invalid_en or invalid_zh:
            return True, invalid_type if invalid_type else "unspecified_invalid"

    return False, None


def is_wrong_language_row(row, eval_type):
    """
    wrong_language:
      zh_en -> language_status_en == "wrong_language"
      en_zh -> language_status_zh == "wrong_language"
      zh_zh -> language_status_en == "wrong_language" OR language_status_zh == "wrong_language"
    """
    status_en = str(row.get("language_status_en", "")).strip()
    status_zh = str(row.get("language_status_zh", "")).strip()

    if eval_type == "zh_en":
        return status_en == "wrong_language"

    if eval_type == "en_zh":
        return status_zh == "wrong_language"

    if eval_type == "zh_zh":
        return (
            status_en == "wrong_language"
            or status_zh == "wrong_language"
        )

    return False


def gather_files():
    """Collect all cleaned JSONL files from outputs/task1 and outputs/task2."""
    files = []

    for task_dir in TASK_DIRS:
        full_dir = ROOT_DIR / task_dir

        if not full_dir.exists():
            continue

        for path in sorted(full_dir.glob("*_clean.jsonl")):
            files.append(path)

    return files


def compute_stats_for_file_and_eval_type(path: Path, eval_type: str):
    """
    Compute invalid, wrong-language, and invalid-type statistics
    for one cleaned output file and one evaluation type.
    """
    meta = parse_filename(path)
    rows = load_jsonl(path)

    total = len(rows)
    invalid_count = 0
    wrong_language_count = 0
    invalid_type_counter = Counter()

    for row in rows:
        invalid, invalid_type = is_invalid_row(row, eval_type)
        wrong_language = is_wrong_language_row(row, eval_type)

        if invalid:
            invalid_count += 1
            invalid_type_counter[invalid_type] += 1

        if wrong_language:
            wrong_language_count += 1

    evaluated = total - invalid_count

    record = {
        **meta,
        "eval_type": eval_type,
        "total": total,
        "invalid": invalid_count,
        "invalid_rate": safe_divide(invalid_count, total),
        "wrong_language": wrong_language_count,
        "wrong_language_rate": safe_divide(wrong_language_count, total),
        "evaluated": evaluated,
    }

    return record, invalid_type_counter


def write_csv(path: Path, rows, fieldnames):
    """Write rows to a CSV file."""
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# =========================================================
# Main
# =========================================================

def main():
    files = gather_files()

    if not files:
        raise FileNotFoundError(
            "No *_clean.jsonl files found under outputs/task1 or outputs/task2"
        )

    # -----------------------------------------------------
    # 1. Per-file-per-eval_type statistics
    #    Output:
    #    overall_distribution_by_file_eval_type.csv
    # -----------------------------------------------------

    per_file_rows = []

    # This is used to build the second requested output.
    agg_invalid_types = defaultdict(Counter)

    for path in files:
        for eval_type in EVAL_TYPES:
            record, invalid_type_counter = compute_stats_for_file_and_eval_type(
                path,
                eval_type,
            )

            per_file_rows.append(record)

            agg_key = (
                record["task"],
                record["eval_type"],
                record["condition"],
                record["model"],
            )

            for invalid_type, count in invalid_type_counter.items():
                agg_invalid_types[agg_key][invalid_type] += count

    # -----------------------------------------------------
    # 2. Aggregate invalid-type breakdown by:
    #    task + eval_type + condition + model
    #    Output:
    #    invalid_type_breakdown_by_task_eval_condition_model.csv
    # -----------------------------------------------------

    aggregate_invalid_type_rows = []

    for (task, eval_type, condition, model), counter in sorted(agg_invalid_types.items()):
        row = {
            "task": task,
            "eval_type": eval_type,
            "condition": condition,
            "model": model,
        }

        for invalid_type in INVALID_TYPES:
            row[invalid_type] = counter.get(invalid_type, 0)

        row["unspecified_invalid"] = counter.get("unspecified_invalid", 0)

        aggregate_invalid_type_rows.append(row)

    # -----------------------------------------------------
    # Write only the two requested CSV outputs
    # -----------------------------------------------------

    write_csv(
        OUTPUT_DIR / "overall_distribution_by_file_eval_type.csv",
        per_file_rows,
        [
            "filename",
            "task",
            "file_direction",
            "eval_type",
            "condition",
            "model",
            "zh_en_prompt_version",
            "en_zh_prompt_version",
            "total",
            "invalid",
            "invalid_rate",
            "wrong_language",
            "wrong_language_rate",
            "evaluated",
        ],
    )

    write_csv(
        OUTPUT_DIR / "invalid_type_breakdown_by_task_eval_condition_model.csv",
        aggregate_invalid_type_rows,
        [
            "task",
            "eval_type",
            "condition",
            "model",
        ] + INVALID_TYPES + ["unspecified_invalid"],
    )

    print(f"Done. Outputs saved to: {OUTPUT_DIR.resolve()}")
    print("Generated:")
    print("- overall_distribution_by_file_eval_type.csv")
    print("- invalid_type_breakdown_by_task_eval_condition_model.csv")


if __name__ == "__main__":
    main()