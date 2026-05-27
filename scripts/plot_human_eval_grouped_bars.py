import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# 1. Output directory
# =========================
output_dir = "human_eval/human_pic"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 2. Model order and hatch patterns
# =========================
model_order = ["deepseek", "gemma2", "glm4", "llama31", "mistral", "qwen25"]

model_name_map = {
    "deepseek": "DeepSeek",
    "gemma2": "Gemma-2",
    "glm4": "GLM-4",
    "llama31": "Llama-3.1",
    "mistral": "Mistral",
    "qwen25": "Qwen2.5",
}

hatch_map = {
    "deepseek": "",
    "gemma2": "//",
    "glm4": "\\\\",
    "llama31": "xx",
    "mistral": "..",
    "qwen25": "++",
}

# =========================
# 3. Helper functions
# =========================
def add_bar_labels(ax, bars, fontsize=5.5, rotation=0):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
        )


def plot_model_grouped_bar(
    data,
    score_cols,
    score_display_labels,
    out_filename,
    figsize=(12, 5.2),
):
    """
    For one-direction model-by-dimension tables.

    data columns:
    Model, score1, score2, ...
    """

    columns = ["Model"] + score_cols
    df = pd.DataFrame(data, columns=columns)

    x = np.arange(len(score_cols))
    n_models = len(model_order)
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=figsize)

    for i, model in enumerate(model_order):
        row = df[df["Model"] == model]
        if row.empty:
            continue

        values = row[score_cols].iloc[0].values.astype(float)
        offset = (i - (n_models - 1) / 2) * bar_width

        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            color="white",
            edgecolor="black",
            linewidth=0.7,
            hatch=hatch_map[model],
            label=model_name_map[model],
        )

        add_bar_labels(ax, bars, fontsize=5.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(score_display_labels, rotation=25, ha="right")
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 5.75)
    ax.grid(True, axis="y", alpha=0.3)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0, 0.84, 1])

    out_path = os.path.join(output_dir, out_filename)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


def plot_task2_faceted_grouped_bar(
    data,
    score_cols,
    score_display_labels,
    out_filename,
    figsize=(12, 9),
):
    """
    For Task 2.1, Task 2.2, Task 2.3.
    Facet by direction:
    en→zh on top, zh→zh on bottom.

    data columns:
    Direction, Model, score1, score2, ...
    """

    columns = ["Direction", "Model"] + score_cols
    df = pd.DataFrame(data, columns=columns)

    directions = ["en→zh", "zh→zh"]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharey=True)

    x = np.arange(len(score_cols))
    n_models = len(model_order)
    bar_width = 0.15

    for ax, direction in zip(axes, directions):
        sub = df[df["Direction"] == direction]

        for i, model in enumerate(model_order):
            row = sub[sub["Model"] == model]
            if row.empty:
                continue

            values = row[score_cols].iloc[0].values.astype(float)
            offset = (i - (n_models - 1) / 2) * bar_width

            bars = ax.bar(
                x + offset,
                values,
                width=bar_width,
                color="white",
                edgecolor="black",
                linewidth=0.7,
                hatch=hatch_map[model],
                label=model_name_map[model],
            )

            add_bar_labels(ax, bars, fontsize=5.5, rotation=0)

        ax.set_xticks(x)
        ax.set_xticklabels(score_display_labels, rotation=25, ha="right")
        ax.set_ylabel("Mean score")
        ax.set_ylim(0, 5.75)
        ax.set_xlabel(direction)
        ax.grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.98, 0.5),
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0, 0.88, 1])

    out_path = os.path.join(output_dir, out_filename)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


def plot_transparency_grouped_bar(
    data,
    score_cols,
    score_display_labels,
    out_filename,
    figsize=(8.5, 4.8),
):
    """
    For Table human_transparency_overall.
    Each dimension has two bars:
    High transparency and Low transparency.
    """

    df = pd.DataFrame(data, columns=["Transparency"] + score_cols)

    x = np.arange(len(score_cols))
    bar_width = 0.32

    transparency_order = ["High transparency", "Low transparency"]
    transparency_hatches = {
        "High transparency": "",
        "Low transparency": "//",
    }

    fig, ax = plt.subplots(figsize=figsize)

    for i, trans in enumerate(transparency_order):
        row = df[df["Transparency"] == trans]
        values = row[score_cols].iloc[0].values.astype(float)
        offset = (i - 0.5) * bar_width

        bars = ax.bar(
            x + offset,
            values,
            width=bar_width,
            color="white",
            edgecolor="black",
            linewidth=0.7,
            hatch=transparency_hatches[trans],
            label=trans,
        )

        add_bar_labels(ax, bars, fontsize=6, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(score_display_labels, rotation=25, ha="right")
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 5.75)
    ax.grid(True, axis="y", alpha=0.3)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0, 0.82, 1])

    out_path = os.path.join(output_dir, out_filename)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


def plot_transparency_task_specific_bar(
    data,
    out_filename,
    figsize=(15, 5.8),
):
    """
    For Table human_transparency_task_specific.
    Each task/condition has two bars:
    High transparency and Low transparency.
    """

    df = pd.DataFrame(
        data,
        columns=[
            "Setting",
            "High Transparency",
            "Low Transparency",
        ],
    )

    x = np.arange(len(df))
    bar_width = 0.32

    fig, ax = plt.subplots(figsize=figsize)

    bars_high = ax.bar(
        x - bar_width / 2,
        df["High Transparency"],
        width=bar_width,
        color="white",
        edgecolor="black",
        linewidth=0.7,
        hatch="",
        label="High transparency",
    )

    bars_low = ax.bar(
        x + bar_width / 2,
        df["Low Transparency"],
        width=bar_width,
        color="white",
        edgecolor="black",
        linewidth=0.7,
        hatch="//",
        label="Low transparency",
    )

    add_bar_labels(ax, bars_high, fontsize=5.5, rotation=0)
    add_bar_labels(ax, bars_low, fontsize=5.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(df["Setting"], rotation=35, ha="right")
    ax.set_ylabel("Mean score")
    ax.set_ylim(0, 5.75)
    ax.grid(True, axis="y", alpha=0.3)

    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0, 0.84, 1])

    out_path = os.path.join(output_dir, out_filename)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


# =========================================================
# 4. plot_human_task1_zh_en_bar.py
# =========================================================
score_cols_task1_zh_en = [
    "Code-Switching",
    "Grammar",
    "Fluency",
    "Adequacy",
    "Rendering Strategy",
]

score_labels_task1_zh_en = [
    "Code-Switching",
    "Grammar",
    "Fluency",
    "Adequacy",
    "Rendering\nStrategy",
]

data_task1_zh_en = [
    ["deepseek", 4.92, 4.96, 4.54, 2.75, 2.88],
    ["gemma2",   5.00, 4.79, 4.83, 3.00, 3.42],
    ["glm4",     4.92, 4.92, 4.75, 3.25, 3.25],
    ["llama31",  5.00, 5.00, 4.83, 3.58, 3.46],
    ["mistral",  5.00, 4.58, 4.42, 1.71, 1.92],
    ["qwen25",   4.92, 4.58, 4.58, 3.71, 3.38],
]

plot_model_grouped_bar(
    data=data_task1_zh_en,
    score_cols=score_cols_task1_zh_en,
    score_display_labels=score_labels_task1_zh_en,
    out_filename="fig_human_task1_zh_en_bar.pdf",
)


# =========================================================
# 5. plot_human_task1_en_zh_bar.py
# =========================================================
score_cols_task1_en_zh = [
    "Code-Switching",
    "Grammar",
    "Fluency",
    "Adequacy",
    "Idiom Form",
]

score_labels_task1_en_zh = [
    "Code-Switching",
    "Grammar",
    "Fluency",
    "Adequacy",
    "Idiom\nForm",
]

data_task1_en_zh = [
    ["deepseek", 5.00, 4.54, 4.46, 3.62, 3.50],
    ["gemma2",   5.00, 4.83, 4.75, 4.12, 3.83],
    ["glm4",     5.00, 4.83, 4.79, 3.67, 4.50],
    ["llama31",  5.00, 4.21, 3.71, 3.38, 3.08],
    ["mistral",  5.00, 3.04, 2.67, 2.54, 2.67],
    ["qwen25",   5.00, 5.00, 5.00, 4.00, 4.67],
]

plot_model_grouped_bar(
    data=data_task1_en_zh,
    score_cols=score_cols_task1_en_zh,
    score_display_labels=score_labels_task1_en_zh,
    out_filename="fig_human_task1_en_zh_bar.pdf",
)


# =========================================================
# 6. plot_human_task1_zh_zh_bar.py
# =========================================================
data_task1_zh_zh = [
    ["deepseek", 5.00, 4.54, 4.46, 2.96, 3.50],
    ["gemma2",   5.00, 4.83, 4.75, 3.21, 3.83],
    ["glm4",     5.00, 4.83, 4.79, 3.08, 4.50],
    ["llama31",  5.00, 4.21, 3.71, 2.58, 3.08],
    ["mistral",  5.00, 3.04, 2.67, 1.71, 2.67],
    ["qwen25",   5.00, 5.00, 5.00, 3.58, 4.67],
]

plot_model_grouped_bar(
    data=data_task1_zh_zh,
    score_cols=score_cols_task1_en_zh,
    score_display_labels=score_labels_task1_en_zh,
    out_filename="fig_human_task1_zh_zh_bar.pdf",
)


# =========================================================
# 7. plot_human_task2_zh_en_bar.py
# =========================================================
score_cols_task2_zh_en = [
    "Code-Switching",
    "Grammar",
    "Fluency",
    "Adequacy",
    "Idiom Meaning Translation",
]

score_labels_task2_zh_en = [
    "Code-Switching",
    "Grammar",
    "Fluency",
    "Adequacy",
    "Idiom Meaning\nTranslation",
]

data_task2_zh_en = [
    ["deepseek", 5.00, 4.38, 3.62, 3.54, 3.04],
    ["gemma2",   5.00, 4.92, 4.08, 3.62, 3.38],
    ["glm4",     5.00, 4.46, 3.75, 3.21, 3.12],
    ["llama31",  5.00, 4.58, 3.50, 3.08, 2.92],
    ["mistral",  5.00, 4.46, 3.17, 2.75, 2.50],
    ["qwen25",   5.00, 4.54, 3.54, 3.46, 3.21],
]

plot_model_grouped_bar(
    data=data_task2_zh_en,
    score_cols=score_cols_task2_zh_en,
    score_display_labels=score_labels_task2_zh_en,
    out_filename="fig_human_task2_zh_en_bar.pdf",
)


# =========================================================
# 8. Task 2.1 faceted bar: en→zh on top, zh→zh below
# =========================================================
score_cols_task2_faceted = [
    "Code-Switching",
    "Grammar",
    "Fluency",
    "Adequacy",
    "Task-Specific",
]

score_labels_c1 = [
    "Code-Switching",
    "Grammar",
    "Fluency",
    "Adequacy",
    "Idiom Meaning\nPreservation",
]

data_c1 = [
    ["en→zh", "deepseek", 4.83, 3.25, 3.58, 4.33, 4.42],
    ["en→zh", "gemma2",   5.00, 4.25, 4.33, 4.54, 4.58],
    ["en→zh", "glm4",     5.00, 3.38, 3.42, 4.50, 4.21],
    ["en→zh", "llama31",  5.00, 3.46, 3.50, 4.29, 4.33],
    ["en→zh", "mistral",  5.00, 2.29, 2.33, 3.50, 3.50],
    ["en→zh", "qwen25",   5.00, 3.50, 3.50, 4.33, 4.46],
    ["zh→zh", "deepseek", 4.83, 3.25, 3.58, 3.42, 3.04],
    ["zh→zh", "gemma2",   5.00, 4.25, 4.33, 3.67, 3.08],
    ["zh→zh", "glm4",     5.00, 3.38, 3.42, 3.29, 2.92],
    ["zh→zh", "llama31",  5.00, 3.46, 3.50, 2.88, 2.42],
    ["zh→zh", "mistral",  5.00, 2.29, 2.33, 2.62, 2.25],
    ["zh→zh", "qwen25",   5.00, 3.50, 3.50, 3.46, 3.33],
]

plot_task2_faceted_grouped_bar(
    data=data_c1,
    score_cols=score_cols_task2_faceted,
    score_display_labels=score_labels_c1,
    out_filename="fig_human_task2_c1_faceted_bar.pdf",
)


# =========================================================
# 9. Task 2.2 faceted bar: en→zh on top, zh→zh below
# =========================================================
score_labels_c2 = [
    "Code-Switching",
    "Grammar",
    "Fluency",
    "Adequacy",
    "Idiom\nReintroduction",
]

data_c2 = [
    ["en→zh", "deepseek", 5.00, 4.21, 4.00, 4.33, 2.75],
    ["en→zh", "gemma2",   5.00, 3.96, 4.00, 3.92, 2.75],
    ["en→zh", "glm4",     5.00, 4.25, 4.04, 4.29, 2.83],
    ["en→zh", "llama31",  5.00, 4.12, 4.04, 3.75, 2.42],
    ["en→zh", "mistral",  5.00, 2.75, 2.62, 2.88, 1.21],
    ["en→zh", "qwen25",   5.00, 3.67, 3.38, 3.50, 2.96],
    ["zh→zh", "deepseek", 5.00, 4.21, 4.00, 3.75, 2.33],
    ["zh→zh", "gemma2",   5.00, 3.96, 4.00, 3.29, 2.62],
    ["zh→zh", "glm4",     5.00, 4.25, 4.04, 3.54, 2.33],
    ["zh→zh", "llama31",  5.00, 4.12, 4.04, 3.08, 1.71],
    ["zh→zh", "mistral",  5.00, 2.75, 2.62, 1.92, 1.17],
    ["zh→zh", "qwen25",   5.00, 3.67, 3.38, 2.46, 2.62],
]

plot_task2_faceted_grouped_bar(
    data=data_c2,
    score_cols=score_cols_task2_faceted,
    score_display_labels=score_labels_c2,
    out_filename="fig_human_task2_c2_faceted_bar.pdf",
)


# =========================================================
# 10. Task 2.3 faceted bar: en→zh on top, zh→zh below
# =========================================================
score_labels_c3 = [
    "Code-Switching",
    "Grammar",
    "Fluency",
    "Adequacy",
    "Target Idiom\nUse",
]

data_c3 = [
    ["en→zh", "deepseek", 5.00, 4.45, 4.55, 4.09, 1.32],
    ["en→zh", "gemma2",   5.00, 4.79, 4.71, 4.04, 4.21],
    ["en→zh", "glm4",     5.00, 4.29, 4.00, 4.00, 3.04],
    ["en→zh", "llama31",  5.00, 4.12, 3.62, 3.62, 2.96],
    ["en→zh", "mistral",  5.00, 3.04, 2.83, 2.75, 1.62],
    ["en→zh", "qwen25",   4.83, 4.29, 4.33, 3.42, 3.62],
    ["zh→zh", "deepseek", 5.00, 4.45, 4.55, 3.73, 1.36],
    ["zh→zh", "gemma2",   5.00, 4.79, 4.71, 4.25, 4.12],
    ["zh→zh", "glm4",     5.00, 4.29, 4.00, 3.92, 3.17],
    ["zh→zh", "llama31",  5.00, 4.12, 3.62, 3.12, 2.75],
    ["zh→zh", "mistral",  5.00, 3.04, 2.83, 2.25, 1.58],
    ["zh→zh", "qwen25",   4.83, 4.29, 4.33, 3.50, 3.54],
]

plot_task2_faceted_grouped_bar(
    data=data_c3,
    score_cols=score_cols_task2_faceted,
    score_display_labels=score_labels_c3,
    out_filename="fig_human_task2_c3_faceted_bar.pdf",
)


# =========================================================
# 11. plot_human_transparency_overall_bar.py
# =========================================================
score_cols_transparency_overall = [
    "Code-Switching",
    "Grammar",
    "Fluency",
    "Adequacy",
]

score_labels_transparency_overall = [
    "Code-Switching",
    "Grammar",
    "Fluency",
    "Adequacy",
]

data_transparency_overall = [
    ["High transparency", 5.00, 4.17, 3.93, 3.66],
    ["Low transparency",  4.97, 4.00, 3.87, 3.15],
]

plot_transparency_grouped_bar(
    data=data_transparency_overall,
    score_cols=score_cols_transparency_overall,
    score_display_labels=score_labels_transparency_overall,
    out_filename="fig_human_transparency_overall_bar.pdf",
)


# =========================================================
# 12. plot_human_transparency_task_specific_bar.py
# =========================================================
data_transparency_task_specific = [
    ["Task 1 zh→en\nRendering Strategy",              3.44, 2.65],
    ["Task 1 en→zh\nIdiom Form",                      4.08, 3.33],
    ["Task 1 zh→zh\nIdiom Form",                      4.08, 3.33],
    ["Task 2 zh→en\nIdiom Meaning Translation",       3.39, 2.67],
    ["Task 2.1 en→zh\nIdiom Meaning Preservation",    4.62, 3.88],
    ["Task 2.1 zh→zh\nIdiom Meaning Preservation",    3.10, 2.58],
    ["Task 2.2 en→zh\nIdiom Reintroduction",          2.65, 2.32],
    ["Task 2.2 zh→zh\nIdiom Reintroduction",          2.36, 1.90],
    ["Task 2.3 en→zh\nTarget Idiom Use",              3.33, 2.32],
    ["Task 2.3 zh→zh\nTarget Idiom Use",              3.04, 2.51],
]

plot_transparency_task_specific_bar(
    data=data_transparency_task_specific,
    out_filename="fig_human_transparency_task_specific_bar.pdf",
)

print("All figures saved in:", output_dir)