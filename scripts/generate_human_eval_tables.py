import pandas as pd
from pathlib import Path


# =========================================================
# Config
# =========================================================

INPUT_FILE = Path("human_eval/merged_human_eval_avg_by_eval_id.csv")
OUTPUT_DIR = Path("human_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCORE_COLS = [
    "code_switching",
    "grammar",
    "fluency",
    "adequacy",
    "task_specific_dim",
]

MODEL_ORDER = [
    "deepseek",
    "gemma2",
    "glm4",
    "llama31",
    "mistral",
    "qwen25",
]

DIRECTION_ORDER = [
    "zh_en",
    "en_zh",
    "zh_zh",
]

TRANSPARENCY_ORDER = [
    "high",
    "low",
]

VERSION_TO_CONDITION = {
    "v1_v1": "--",
    "v3_v1": "C1",
    "v3_v2": "C2",
    "v3_v3": "C3",
}


# =========================================================
# Helper functions
# =========================================================

def mean_scores(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate score columns by mean."""
    return (
        df.groupby(group_cols, dropna=False)[SCORE_COLS]
        .mean()
        .reset_index()
    )


def add_condition_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add readable condition labels from version."""
    df = df.copy()
    df["condition"] = df["version"].map(VERSION_TO_CONDITION).fillna(df["version"])
    return df


def model_sort_key(model: str) -> int:
    try:
        return MODEL_ORDER.index(model)
    except ValueError:
        return len(MODEL_ORDER)


def direction_sort_key(direction: str) -> int:
    try:
        return DIRECTION_ORDER.index(direction)
    except ValueError:
        return len(DIRECTION_ORDER)


def transparency_sort_key(transparency: str) -> int:
    try:
        return TRANSPARENCY_ORDER.index(transparency)
    except ValueError:
        return len(TRANSPARENCY_ORDER)


def task_specific_dimension_label(task: str, eval_type: str, condition: str) -> str:
    """Return readable task-specific dimension name."""
    if task == "task1" and eval_type == "zh_en":
        return "Rendering Strategy"
    if task == "task1" and eval_type in ["en_zh", "zh_zh"]:
        return "Idiom Form"

    if task == "task2" and eval_type == "zh_en":
        return "Idiom Meaning Translation"

    if task == "task2" and condition == "C1":
        return "Idiom Meaning Preservation"
    if task == "task2" and condition == "C2":
        return "Idiom Reintroduction"
    if task == "task2" and condition == "C3":
        return "Target Idiom Use"

    return "Task-Specific Dim."


def save_csv(df: pd.DataFrame, filename: str) -> None:
    """Save dataframe as CSV."""
    output_path = OUTPUT_DIR / filename
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {output_path}")


# =========================================================
# Table 1: Overall human evaluation scores
# =========================================================

def make_overall_table(df: pd.DataFrame) -> pd.DataFrame:
    d = add_condition_column(df)

    tab = mean_scores(d, ["task", "eval_type", "condition"])

    tab["task_order"] = tab["task"].map({"task1": 1, "task2": 2})
    tab["condition_order"] = tab["condition"].map({"--": 0, "C1": 1, "C2": 2, "C3": 3})
    tab["direction_order"] = tab["eval_type"].map({"zh_en": 1, "en_zh": 2, "zh_zh": 3})

    tab = tab.sort_values(
        ["task_order", "condition_order", "direction_order"]
    ).reset_index(drop=True)

    tab = tab.drop(columns=["task_order", "condition_order", "direction_order"])

    return tab


# =========================================================
# Tables 2--5: Single-direction by-model tables
# =========================================================

def make_model_table(
    df: pd.DataFrame,
    task: str,
    eval_type: str,
    version: str,
) -> pd.DataFrame:
    d = df[
        (df["task"] == task)
        & (df["eval_type"] == eval_type)
        & (df["version"] == version)
    ].copy()

    tab = mean_scores(d, ["model"])
    tab["model_order"] = tab["model"].map(model_sort_key)

    tab = tab.sort_values("model_order").reset_index(drop=True)
    tab = tab.drop(columns=["model_order"])

    return tab


# =========================================================
# Tables 6--8: Task 2 C1/C2/C3 with en_zh and zh_zh
# =========================================================

def make_task2_condition_table(
    df: pd.DataFrame,
    version: str,
) -> pd.DataFrame:
    d = df[
        (df["task"] == "task2")
        & (df["version"] == version)
        & (df["eval_type"].isin(["en_zh", "zh_zh"]))
    ].copy()

    tab = mean_scores(d, ["eval_type", "model"])

    tab["direction_order"] = tab["eval_type"].map(direction_sort_key)
    tab["model_order"] = tab["model"].map(model_sort_key)

    tab = tab.sort_values(
        ["direction_order", "model_order"]
    ).reset_index(drop=True)

    tab = tab.drop(columns=["direction_order", "model_order"])

    return tab


# =========================================================
# Table 9: Transparency overall table
# =========================================================

def make_transparency_overall_table(df: pd.DataFrame) -> pd.DataFrame:
    general_cols = [
        "code_switching",
        "grammar",
        "fluency",
        "adequacy",
    ]

    tab = (
        df.groupby(["transparency"], dropna=False)[general_cols]
        .mean()
        .reset_index()
    )

    tab["transparency_order"] = tab["transparency"].map(transparency_sort_key)

    tab = tab.sort_values("transparency_order").reset_index(drop=True)
    tab = tab.drop(columns=["transparency_order"])

    return tab


# =========================================================
# Table 10: Task-specific transparency table
# =========================================================

def make_transparency_task_specific_table(df: pd.DataFrame) -> pd.DataFrame:
    d = add_condition_column(df)

    tab = (
        d.groupby(
            ["task", "eval_type", "condition", "transparency"],
            dropna=False,
        )["task_specific_dim"]
        .mean()
        .reset_index()
    )

    wide = (
        tab.pivot_table(
            index=["task", "eval_type", "condition"],
            columns="transparency",
            values="task_specific_dim",
            aggfunc="mean",
        )
        .reset_index()
    )

    if "high" not in wide.columns:
        wide["high"] = pd.NA
    if "low" not in wide.columns:
        wide["low"] = pd.NA

    wide["task_specific_dimension"] = wide.apply(
        lambda row: task_specific_dimension_label(
            row["task"],
            row["eval_type"],
            row["condition"],
        ),
        axis=1,
    )

    wide["task_order"] = wide["task"].map({"task1": 1, "task2": 2})
    wide["condition_order"] = wide["condition"].map({"--": 0, "C1": 1, "C2": 2, "C3": 3})
    wide["direction_order"] = wide["eval_type"].map({"zh_en": 1, "en_zh": 2, "zh_zh": 3})

    wide = wide.sort_values(
        ["task_order", "condition_order", "direction_order"]
    ).reset_index(drop=True)

    wide = wide.drop(
        columns=["task_order", "condition_order", "direction_order"]
    )

    wide = wide[
        [
            "task",
            "eval_type",
            "condition",
            "task_specific_dimension",
            "high",
            "low",
        ]
    ]

    return wide


# =========================================================
# Table 11: Diagnostic analysis for Task 2 C2/C3
# =========================================================

def make_task2_idiom_diagnostic_table(df: pd.DataFrame) -> pd.DataFrame:
    d = add_condition_column(df)

    d = d[
        (d["task"] == "task2")
        & (d["condition"].isin(["C2", "C3"]))
        & (d["eval_type"].isin(["en_zh", "zh_zh"]))
    ].copy()

    rows = []

    for (condition, eval_type, model), g in d.groupby(
        ["condition", "eval_type", "model"],
        dropna=False,
    ):
        n = len(g)

        if n == 0:
            score1_rate = pd.NA
            n_above_1 = 0
            mean_given_above_1 = pd.NA
        else:
            score1_rate = (g["task_specific_dim"] == 1).mean()
            above_1 = g[g["task_specific_dim"] > 1]
            n_above_1 = len(above_1)
            mean_given_above_1 = (
                above_1["task_specific_dim"].mean()
                if n_above_1 > 0
                else pd.NA
            )

        rows.append(
            {
                "condition": condition,
                "eval_type": eval_type,
                "model": model,
                "n": n,
                "score1_rate": score1_rate,
                "n_above_1": n_above_1,
                "mean_given_above_1": mean_given_above_1,
            }
        )

    tab = pd.DataFrame(rows)

    tab["condition_order"] = tab["condition"].map({"C2": 1, "C3": 2})
    tab["direction_order"] = tab["eval_type"].map(direction_sort_key)
    tab["model_order"] = tab["model"].map(model_sort_key)

    tab = tab.sort_values(
        ["condition_order", "direction_order", "model_order"]
    ).reset_index(drop=True)

    tab = tab.drop(
        columns=["condition_order", "direction_order", "model_order"]
    )

    return tab


# =========================================================
# Main
# =========================================================

def main():
    df = pd.read_csv(INPUT_FILE)

    required_cols = {
        "task",
        "eval_type",
        "version",
        "model",
        "transparency",
        *SCORE_COLS,
    }

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    tables = {
        "human_overall_scores.csv": make_overall_table(df),

        "human_task1_zh_en.csv": make_model_table(
            df=df,
            task="task1",
            eval_type="zh_en",
            version="v1_v1",
        ),

        "human_task1_en_zh.csv": make_model_table(
            df=df,
            task="task1",
            eval_type="en_zh",
            version="v1_v1",
        ),

        "human_task1_zh_zh.csv": make_model_table(
            df=df,
            task="task1",
            eval_type="zh_zh",
            version="v1_v1",
        ),

        "human_task2_zh_en.csv": make_model_table(
            df=df,
            task="task2",
            eval_type="zh_en",
            version="v3_v1",
        ),

        "human_task2_c1.csv": make_task2_condition_table(
            df=df,
            version="v3_v1",
        ),

        "human_task2_c2.csv": make_task2_condition_table(
            df=df,
            version="v3_v2",
        ),

        "human_task2_c3.csv": make_task2_condition_table(
            df=df,
            version="v3_v3",
        ),

        "human_transparency_overall.csv": make_transparency_overall_table(df),

        "human_transparency_task_specific.csv": make_transparency_task_specific_table(df),

        "human_task2_idiom_diagnostic.csv": make_task2_idiom_diagnostic_table(df),
    }

    for filename, table in tables.items():
        save_csv(table, filename)

    print(f"\nDone. Wrote {len(tables)} CSV files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()