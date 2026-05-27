import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

# =========================================================
# Config
# =========================================================
INPUT_FILE = Path("human_eval/merged_human_eval.csv")
OUTPUT_DIR = Path("human_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OVERALL_OUT = OUTPUT_DIR / "overall_weighted_kappa.csv"
DETAIL_OUT = OUTPUT_DIR / "detailed_weighted_kappa.csv"

DIMENSIONS = [
    "code_switching",
    "grammar",
    "fluency",
    "adequacy",
    "task_specific_dim",
]

GROUP_COLS = [
    "task",
    "eval_type",
    "version",
]


# =========================================================
# Helpers
# =========================================================
def normalize_scores(df: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    """
    Convert rating columns to numeric where possible.
    Non-numeric values become NaN.
    """
    out = df.copy()
    for col in dimensions:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def pair_annotators(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """
    Pair a1 and a2 by eval_id for one dimension.
    Returns a dataframe with columns:
      eval_id, a1, a2
    Only keeps rows where both annotators have non-missing scores.
    """
    sub = df[["eval_id", "annotator", dimension]].copy()
    sub = sub[sub["annotator"].isin(["a1", "a2"])]
    sub = sub.dropna(subset=[dimension])

    wide = sub.pivot_table(
        index="eval_id",
        columns="annotator",
        values=dimension,
        aggfunc="first"
    ).reset_index()

    if "a1" not in wide.columns or "a2" not in wide.columns:
        return pd.DataFrame(columns=["eval_id", "a1", "a2"])

    wide = wide.dropna(subset=["a1", "a2"]).copy()
    return wide


def compute_quadratic_weighted_kappa(a1_scores, a2_scores):
    """
    Compute quadratic weighted Cohen's kappa.
    """
    return cohen_kappa_score(a1_scores, a2_scores, weights="quadratic")


# =========================================================
# Main
# =========================================================
def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE.resolve()}")

    df = pd.read_csv(INPUT_FILE)
    df = normalize_scores(df, DIMENSIONS)

    # -----------------------------------------------------
    # 1. Overall weighted kappa by dimension
    # -----------------------------------------------------
    overall_rows = []

    for dim in DIMENSIONS:
        if dim not in df.columns:
            continue

        paired = pair_annotators(df, dim)
        n_pairs = len(paired)

        if n_pairs == 0:
            kappa = pd.NA
        else:
            kappa = compute_quadratic_weighted_kappa(paired["a1"], paired["a2"])

        overall_rows.append({
            "dimension": dim,
            "n_pairs": n_pairs,
            "weighted_kappa": kappa,
        })

    overall_df = pd.DataFrame(overall_rows)
    overall_df.to_csv(OVERALL_OUT, index=False, encoding="utf-8-sig")

    # -----------------------------------------------------
    # 2. Detailed table by task + eval_type + version + dimension
    # -----------------------------------------------------
    detail_rows = []

    grouped = df.groupby(GROUP_COLS, dropna=False, sort=True)

    for group_keys, group_df in grouped:
        task, eval_type, version = group_keys

        for dim in DIMENSIONS:
            if dim not in group_df.columns:
                continue

            paired = pair_annotators(group_df, dim)
            n_pairs = len(paired)

            if n_pairs == 0:
                kappa = pd.NA
            else:
                kappa = compute_quadratic_weighted_kappa(paired["a1"], paired["a2"])

            detail_rows.append({
                "task": task,
                "eval_type": eval_type,
                "version": version,
                "dimension": dim,
                "n_pairs": n_pairs,
                "weighted_kappa": kappa,
            })

    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(DETAIL_OUT, index=False, encoding="utf-8-sig")

    print(f"Saved: {OVERALL_OUT.resolve()}")
    print(f"Saved: {DETAIL_OUT.resolve()}")


if __name__ == "__main__":
    main()