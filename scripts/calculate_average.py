import pandas as pd
from pathlib import Path


# =========================================================
# Config
# =========================================================

INPUT_FILE = Path("human_eval/merged_human_eval.csv")
OUTPUT_DIR = Path("human_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only one output file:
# average different annotators for the same eval_id
OUTPUT_AVG_BY_ITEM = OUTPUT_DIR / "merged_human_eval_avg_by_eval_id.csv"


SCORE_COLS = [
    "code_switching",
    "grammar",
    "fluency",
    "adequacy",
    "task_specific_dim",
    "total_score",
]

META_COLS = [
    "task",
    "eval_type",
    "version",
    "index",
    "source_idiom",
    "model",
    "transparency",
    "polarity",
    "reference",
    "evaluation_item",
]


# =========================================================
# Helpers
# =========================================================

def to_numeric_scores(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Convert score columns to numeric values.
    Non-numeric values are converted to NaN.
    """
    out = df.copy()

    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def first_non_null(series: pd.Series):
    """
    Return the first non-null value in a Series.
    If all values are null, return pd.NA.
    """
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) > 0 else pd.NA


# =========================================================
# Main
# =========================================================

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE.resolve()}")

    df = pd.read_csv(INPUT_FILE)
    df = to_numeric_scores(df, SCORE_COLS)

    # -----------------------------------------------------
    # Average scores from different annotators for the same eval_id
    # -----------------------------------------------------

    # Average score columns
    agg_dict = {
        col: "mean"
        for col in SCORE_COLS
        if col in df.columns
    }

    # Keep the first non-null value for metadata columns within each eval_id
    for col in META_COLS:
        if col in df.columns:
            agg_dict[col] = first_non_null

    # Keep idiom_use if it exists, but do not average it
    if "idiom_use" in df.columns:
        agg_dict["idiom_use"] = first_non_null

    avg_by_item = (
        df.groupby("eval_id", dropna=False, sort=True)
        .agg(agg_dict)
        .reset_index()
    )

    # Reorder columns in the item-level averaged file
    item_cols = [
        "eval_id",
        "task",
        "eval_type",
        "version",
        "index",
        "source_idiom",
        "model",
        "transparency",
        "polarity",
        "reference",
        "evaluation_item",
    ]

    if "idiom_use" in avg_by_item.columns:
        item_cols.append("idiom_use")

    item_cols += [
        col for col in SCORE_COLS
        if col in avg_by_item.columns
    ]

    item_cols = [
        col for col in item_cols
        if col in avg_by_item.columns
    ]

    avg_by_item = avg_by_item[item_cols]

    avg_by_item.to_csv(
        OUTPUT_AVG_BY_ITEM,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"Saved averaged item-level file to: {OUTPUT_AVG_BY_ITEM.resolve()}")


if __name__ == "__main__":
    main()