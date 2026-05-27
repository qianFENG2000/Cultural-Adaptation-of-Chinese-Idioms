import json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd


# =========================
# Config
# =========================

# Summary files are stored under eval/
INPUT_DIR = Path("eval")

# Merged table will be saved here
OUTPUT_DIR = Path("eval/merged")

# Only one output file
OUTPUT_CSV = OUTPUT_DIR / "summary_auto_eval.csv"


# =========================
# Helper functions
# =========================

def is_la_file(path: Path) -> bool:
    """Check whether a summary file is a language-aware filtered file."""
    return path.stem.endswith("_la")


def load_summary_file(path: Path) -> List[Dict[str, Any]]:
    """Load one summary JSON file."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, but got {type(data)}")

    rows = []

    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Expected dict items in {path}, but got {type(item)}")

        row = item.copy()
        row["source_file"] = path.name
        rows.append(row)

    return rows


def collect_summary_files(input_dir: Path) -> List[Path]:
    """
    Collect original summary files only.

    Files ending with '_la.json' are excluded.
    """
    all_files = sorted(input_dir.rglob("*summary*.json"))

    if not all_files:
        raise FileNotFoundError(f"No summary JSON files found in {input_dir}")

    original_files = [p for p in all_files if not is_la_file(p)]

    if not original_files:
        raise FileNotFoundError(
            f"No original summary JSON files found in {input_dir}. "
            "All summary files may be *_la.json files."
        )

    return original_files


def build_raw_dataframe(files: List[Path]) -> pd.DataFrame:
    """Load a list of summary files into one dataframe."""
    rows = []

    for file in files:
        rows.extend(load_summary_file(file))

    df = pd.DataFrame(rows)

    required_cols = [
        "task",
        "eval_type",
        "model",
        "zh_en_prompt_version",
        "en_zh_prompt_version",
        "num_total",
        "num_valid",
        "num_invalid",
        "bleu",
        "rouge1",
        "rouge2",
        "rougeL",
        "bertscore_f1",
        "sbert_cosine",
        "cometkiwi",
        "metricx_qe",
        "comet",
        "source_file",
    ]

    for col in required_cols:
        if col not in df.columns:
            df[col] = pd.NA

    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add valid_rate and invalid_rate."""
    df = df.copy()

    df["valid_rate"] = df["num_valid"] / df["num_total"]
    df["invalid_rate"] = df["num_invalid"] / df["num_total"]

    return df


def build_long_table(files: List[Path]) -> pd.DataFrame:
    """Build one long-format automatic evaluation summary table."""
    df = build_raw_dataframe(files)

    df = add_derived_columns(df)

    df["prompt_setting"] = (
        "zh" + df["zh_en_prompt_version"].astype(str)
        + "_enzh" + df["en_zh_prompt_version"].astype(str)
    )

    model_order = {
        "llama31": 1,
        "mistral": 2,
        "deepseek": 3,
        "gemma2": 4,
        "glm4": 5,
        "qwen25": 6,
    }

    df["model_order"] = df["model"].map(model_order).fillna(999)

    ordered_cols = [
        "task",
        "eval_type",
        "model",
        "model_order",
        "zh_en_prompt_version",
        "en_zh_prompt_version",
        "prompt_setting",

        "num_total",
        "num_valid",
        "num_invalid",
        "valid_rate",
        "invalid_rate",

        "bleu",
        "rouge1",
        "rouge2",
        "rougeL",
        "bertscore_f1",
        "sbert_cosine",
        "cometkiwi",
        "metricx_qe",
        "comet",

        "source_file",
    ]

    for col in ordered_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[ordered_cols]

    df = df.sort_values(
        by=[
            "task",
            "eval_type",
            "zh_en_prompt_version",
            "en_zh_prompt_version",
            "model_order",
        ]
    ).reset_index(drop=True)

    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_files = collect_summary_files(INPUT_DIR)

    print(f"Found {len(summary_files)} original summary files.")
    print("Files ending with '_la.json' were excluded.")

    df = build_long_table(summary_files)

    df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
        na_rep="NA"
    )

    print(f"Saved summary table to: {OUTPUT_CSV}")

    print("\nPreview:")
    print(df.head(10).to_string(index=False, na_rep="NA"))


if __name__ == "__main__":
    main()