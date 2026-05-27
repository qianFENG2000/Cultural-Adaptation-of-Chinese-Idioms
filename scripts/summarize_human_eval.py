import json
from pathlib import Path
import pandas as pd

# =========================================================
# Config
# =========================================================
INPUT_DIRS = [
    Path("human_eval/task1"),
    Path("human_eval/task2"),
]
ANN_FILE = Path("data/idiom_sample_ann.json")
OUTPUT_FILE = Path("human_eval/merged_human_eval.csv")

MODELS = {
    "llama31": "M01",
    "mistral": "M02",
    "deepseek": "M03",
    "gemma2": "M04",
    "glm4": "M05",
    "qwen25": "M06",
}

# 反向映射：M01 -> llama31
MODEL_ID_TO_NAME = {v: k for k, v in MODELS.items()}

FINAL_COLS = [
    "eval_id",
    "task",
    "eval_type",
    "version",
    "annotator",
    "index",
    "source_idiom",
    "model",
    "transparency",
    "polarity",
    "reference",
    "evaluation_item",
    "idiom_use",
    "code_switching",
    "grammar",
    "fluency",
    "adequacy",
    "task_specific_dim",
    "total_score",
]


# =========================================================
# Helpers
# =========================================================
def load_annotation_map(path: Path) -> dict:
    """
    Build a mapping:
      index -> {"source_idiom": ..., "transparency": ..., "polarity": ...}
    Supports either JSON array or JSONL.
    """
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Annotation file is empty: {path}")

    if text.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]

    ann_map = {}
    for row in rows:
        idx = row.get("index")
        if idx is None:
            continue
        ann_map[idx] = {
            "source_idiom": row.get("source_idiom"),
            "transparency": row.get("transparency"),
            "polarity": row.get("polarity"),
        }
    return ann_map


def parse_filename(filename: str) -> dict:
    """
    Examples:
      task1_zh_en_human_eval_a1.xlsx
      task2_en_zh_v3_v1_human_eval_a2.xlsx
    """
    stem = Path(filename).stem
    parts = stem.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected filename format: {filename}")

    task = parts[0]
    eval_type = "_".join(parts[1:3])

    annotator = parts[-1]
    if annotator not in {"a1", "a2"}:
        raise ValueError(f"Cannot parse annotator from filename: {filename}")

    if task == "task1":
        version = "v1_v1"
    elif task == "task2":
        if len(parts) < 6:
            raise ValueError(f"Task 2 filename missing prompt versions: {filename}")
        version = f"{parts[3]}_{parts[4]}"
    else:
        raise ValueError(f"Unsupported task in filename: {filename}")

    return {
        "task": task,
        "eval_type": eval_type,
        "version": version,
        "annotator": annotator,
    }


def get_reference_and_eval_cols(task: str, eval_type: str) -> tuple[str, str]:
    """
    task1:
      zh_en -> source_idiom -> reference, translation_en -> evaluation_item
      en_zh -> translation_en -> reference, translation_zh -> evaluation_item
      zh_zh -> source_idiom -> reference, translation_zh -> evaluation_item

    task2:
      zh_en -> sentence -> reference, translation_en -> evaluation_item
      en_zh -> translation_en -> reference, translation_zh -> evaluation_item
      zh_zh -> sentence -> reference, translation_zh -> evaluation_item
    """
    if task == "task1":
        if eval_type == "zh_en":
            return "source_idiom", "translation_en"
        elif eval_type == "en_zh":
            return "translation_en", "translation_zh"
        elif eval_type == "zh_zh":
            return "source_idiom", "translation_zh"

    elif task == "task2":
        if eval_type == "zh_en":
            return "sentence", "translation_en"
        elif eval_type == "en_zh":
            return "translation_en", "translation_zh"
        elif eval_type == "zh_zh":
            return "sentence", "translation_zh"

    raise ValueError(f"Unsupported combination: task={task}, eval_type={eval_type}")


def find_task_specific_dim_col(df: pd.DataFrame) -> str:
    """
    Take the first column after 'adequacy' and rename it to task_specific_dim,
    regardless of its original name.
    """
    cols = list(df.columns)
    if "adequacy" not in cols:
        raise ValueError("Column 'adequacy' not found.")

    adequacy_idx = cols.index("adequacy")
    if adequacy_idx + 1 >= len(cols):
        raise ValueError("No column found after 'adequacy'.")

    return cols[adequacy_idx + 1]


def map_model_id_to_name(value):
    """
    Convert anonymous model id (e.g., M01) back to real model name (e.g., llama31).
    If the value is already a real model name or unknown, keep it as-is.
    """
    if pd.isna(value):
        return pd.NA

    v = str(value).strip()
    return MODEL_ID_TO_NAME.get(v, v)


def process_one_file(path: Path, ann_map: dict) -> pd.DataFrame:
    meta = parse_filename(path.name)
    task = meta["task"]
    eval_type = meta["eval_type"]
    version = meta["version"]
    annotator = meta["annotator"]

    df = pd.read_excel(path)
    df = df.dropna(how="all").reset_index(drop=True)

    ref_col, eval_col = get_reference_and_eval_cols(task, eval_type)
    task_specific_col = find_task_specific_dim_col(df)

    required_base = [
        "eval_id",
        "index",
        "model_id_anonymous",
        "code_switching",
        "grammar",
        "fluency",
        "adequacy",
    ]

    for c in required_base + [ref_col, eval_col, task_specific_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in file {path.name}")

    out = pd.DataFrame(index=df.index)

    out["eval_id"] = df["eval_id"]
    out["task"] = task
    out["eval_type"] = eval_type
    out["version"] = version
    out["annotator"] = annotator
    out["index"] = df["index"]

    out["source_idiom"] = out["index"].map(
        lambda x: ann_map.get(x, {}).get("source_idiom")
    )

    # 取出 model_id_anonymous，并映射回真实模型名
    out["model"] = df["model_id_anonymous"].map(map_model_id_to_name)

    out["transparency"] = out["index"].map(
        lambda x: ann_map.get(x, {}).get("transparency")
    )
    out["polarity"] = out["index"].map(
        lambda x: ann_map.get(x, {}).get("polarity")
    )

    out["reference"] = df[ref_col]
    out["evaluation_item"] = df[eval_col]

    if "idiom_use" in df.columns:
        out["idiom_use"] = df["idiom_use"]
    else:
        out["idiom_use"] = pd.NA

    # Convert score columns to numeric before calculating total_score
    out["code_switching"] = pd.to_numeric(df["code_switching"], errors="coerce")
    out["grammar"] = pd.to_numeric(df["grammar"], errors="coerce")
    out["fluency"] = pd.to_numeric(df["fluency"], errors="coerce")
    out["adequacy"] = pd.to_numeric(df["adequacy"], errors="coerce")
    out["task_specific_dim"] = pd.to_numeric(df[task_specific_col], errors="coerce")

    score_cols = [
        "code_switching",
        "grammar",
        "fluency",
        "adequacy",
        "task_specific_dim",
    ]

    # 平均五个维度，得到总分
    out["total_score"] = out[score_cols].mean(axis=1)

    return out[FINAL_COLS]


def gather_xlsx_files(input_dirs: list[Path]) -> list[Path]:
    files = []
    for folder in input_dirs:
        if not folder.exists():
            print(f"Warning: input directory not found: {folder}")
            continue
        files.extend(sorted(folder.glob("*.xlsx")))
    return files


# =========================================================
# Main
# =========================================================
def main():
    ann_map = load_annotation_map(ANN_FILE)

    xlsx_files = gather_xlsx_files(INPUT_DIRS)
    if not xlsx_files:
        raise FileNotFoundError("No .xlsx files found in the specified input directories.")

    all_dfs = []
    for file in xlsx_files:
        try:
            one_df = process_one_file(file, ann_map)
            all_dfs.append(one_df)
            print(f"Processed: {file} ({len(one_df)} rows)")
        except Exception as e:
            print(f"Skipped {file}: {e}")

    if not all_dfs:
        raise ValueError("No files were successfully processed.")

    merged = pd.concat(all_dfs, ignore_index=True)
    merged.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"\nSaved merged csv to: {OUTPUT_FILE.resolve()}")
    print(f"Total rows: {len(merged)}")


if __name__ == "__main__":
    main()