import pandas as pd
from pathlib import Path


# =========================
# Config
# =========================

INPUT_FILE = Path("eval/merged/summary_auto_eval.csv")
OUTPUT_DIR = Path("eval/merged")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KEEP_COLS = [
    "task",
    "eval_type",
    "zh_en_prompt_version",
    "en_zh_prompt_version",
    "model",
    "num_valid",
    "bleu",
    "bertscore_f1",
    "sbert_cosine",
    "cometkiwi",
    "metricx_qe",
    "comet",
]

GROUP_KEYS = [
    "task",
    "eval_type",
    "zh_en_prompt_version",
    "en_zh_prompt_version",
]

METRIC_COLS = [
    "bleu",
    "bertscore_f1",
    "sbert_cosine",
    "cometkiwi",
    "metricx_qe",
    "comet",
]


# =========================
# Helper functions
# =========================

def clean_display_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep required columns and round metric columns to 2 decimals."""
    out = df.copy()

    cols = [c for c in KEEP_COLS if c in out.columns]
    out = out[cols]

    for col in METRIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    if "num_valid" in out.columns:
        out["num_valid"] = pd.to_numeric(out["num_valid"], errors="coerce")

    return out


def main():
    df = pd.read_csv(INPUT_FILE)

    # =========================
    # 1. Save full cleaned CSV
    # =========================

    full_df = clean_display_df(df)

    full_output_file = OUTPUT_DIR / "auto_eval_full_table.csv"
    full_df.to_csv(full_output_file, index=False, encoding="utf-8-sig")

    # =========================
    # 2. Save grouped CSV files
    # =========================

    grouped_rows = []

    for keys, group_df in df.groupby(GROUP_KEYS, sort=True):
        task, eval_type, zh_ver, en_ver = keys

        display_df = clean_display_df(group_df)

        output_filename = f"{task}_{eval_type}_{zh_ver}_{en_ver}.csv"
        output_path = OUTPUT_DIR / output_filename

        display_df.to_csv(output_path, index=False, encoding="utf-8-sig")

        grouped_rows.append(
            {
                "task": task,
                "eval_type": eval_type,
                "zh_en_prompt_version": zh_ver,
                "en_zh_prompt_version": en_ver,
                "rows": len(display_df),
                "csv_file": output_filename,
            }
        )

    # =========================
    # 3. Save grouped table index
    # =========================

    grouped_index_df = pd.DataFrame(grouped_rows)

    grouped_index_file = OUTPUT_DIR / "grouped_table_index.csv"
    grouped_index_df.to_csv(
        grouped_index_file,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Done. CSV files saved to: {OUTPUT_DIR.resolve()}")
    print("Generated:")
    print(f"- {full_output_file.name}")
    print("- grouped CSV files by task / eval_type / prompt versions")
    print(f"- {grouped_index_file.name}")


if __name__ == "__main__":
    main()