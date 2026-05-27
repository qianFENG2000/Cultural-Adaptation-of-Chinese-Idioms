import math
import os
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. Read data
# =========================
csv_path = "eval/merged/auto_eval_full_table.csv"
df = pd.read_csv(csv_path)

# =========================
# 2. Output directory
# =========================
output_dir = "eval/merged/auto_pic"
os.makedirs(output_dir, exist_ok=True)

# Optional: prettier model names
model_name_map = {
    "llama31": "Llama-3.1",
    "mistral": "Mistral",
    "deepseek": "DeepSeek",
    "gemma2": "Gemma-2",
    "glm4": "GLM-4",
    "qwen25": "Qwen2.5",
}

df["model_display"] = df["model"].map(model_name_map).fillna(df["model"])

# Fixed model order
model_order = ["llama31", "mistral", "deepseek", "gemma2", "glm4", "qwen25"]
model_labels = [model_name_map[m] for m in model_order]

# =========================
# 3. Helper functions
# =========================
metric_label_map = {
    "bleu": "BLEU",
    "bertscore_f1": "BERTScore",
    "sbert_cosine": "SBERT",
    "cometkiwi": "COMETKiwi",
    "metricx_qe": "MetricX-QE",
    "comet": "COMET",
}

def make_line_figure(
    df,
    eval_type,
    condition_specs,
    metrics,
    out_path,
    model_order,
    model_labels,
    figsize=(12, 8),
):
    n_metrics = len(metrics)
    ncols = 2
    nrows = math.ceil(n_metrics / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    x = list(range(len(condition_specs)))
    x_labels = [c[3] for c in condition_specs]

    all_handles = None
    all_labels = None

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]

        for m, m_label in zip(model_order, model_labels):
            y = []
            for task, zh_ver, en_ver, _ in condition_specs:
                sub = df[
                    (df["task"] == task)
                    & (df["eval_type"] == eval_type)
                    & (df["zh_en_prompt_version"] == zh_ver)
                    & (df["en_zh_prompt_version"] == en_ver)
                    & (df["model"] == m)
                ]

                if sub.empty:
                    y.append(float("nan"))
                else:
                    y.append(sub.iloc[0][metric])

            ax.plot(
                x,
                y,
                marker="o",
                markersize=3,
                linewidth=0.8,
                label=m_label
            )

        ylabel = metric_label_map.get(metric, metric)
        if metric == "metricx_qe":
            ylabel = "MetricX-QE\n(lower better)"

        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=0)
        ax.grid(True, alpha=0.3)

        if all_handles is None:
            all_handles, all_labels = ax.get_legend_handles_labels()

    # Hide unused axes
    for j in range(n_metrics, len(axes)):
        axes[j].axis("off")

    # Legend closer to the plot
    fig.legend(
        all_handles,
        all_labels,
        loc="center left",
        bbox_to_anchor=(0.94, 0.5),
        frameon=False
    )

    fig.tight_layout(rect=[0, 0, 0.90, 1])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

# =========================
# 4. Figure 1: zh->en
#    BLEU removed
# =========================
fig1_conditions = [
    ("task1", "v1", "v1", "Task 1"),
    ("task2", "v3", "v1", "Task 2"),
]

fig1_metrics = [
    "bertscore_f1",
    "sbert_cosine",
    "cometkiwi",
    "metricx_qe",
]

make_line_figure(
    df=df,
    eval_type="zh_en",
    condition_specs=fig1_conditions,
    metrics=fig1_metrics,
    out_path=os.path.join(output_dir, "fig1_auto_zh_en.pdf"),
    model_order=model_order,
    model_labels=model_labels,
    figsize=(12, 8),
)

# =========================
# 5. Figure 2: en->zh
#    BLEU removed
# =========================
fig2_conditions = [
    ("task1", "v1", "v1", "Task 1"),
    ("task2", "v3", "v1", "Task 2.1"),
    ("task2", "v3", "v2", "Task 2.2"),
    ("task2", "v3", "v3", "Task 2.3"),
]

fig2_metrics = [
    "bertscore_f1",
    "sbert_cosine",
    "cometkiwi",
    "metricx_qe",
]

make_line_figure(
    df=df,
    eval_type="en_zh",
    condition_specs=fig2_conditions,
    metrics=fig2_metrics,
    out_path=os.path.join(output_dir, "fig2_auto_en_zh.pdf"),
    model_order=model_order,
    model_labels=model_labels,
    figsize=(12, 8),
)

# =========================
# 6. Figure 3: zh->zh
#    BLEU retained
# =========================
fig3_conditions = [
    ("task1", "v1", "v1", "Task 1"),
    ("task2", "v3", "v1", "Task 2.1"),
    ("task2", "v3", "v2", "Task 2.2"),
    ("task2", "v3", "v3", "Task 2.3"),
]

fig3_metrics = [
    "bleu",
    "bertscore_f1",
    "sbert_cosine",
    "comet",
]

make_line_figure(
    df=df,
    eval_type="zh_zh",
    condition_specs=fig3_conditions,
    metrics=fig3_metrics,
    out_path=os.path.join(output_dir, "fig3_auto_zh_zh.pdf"),
    model_order=model_order,
    model_labels=model_labels,
    figsize=(12, 8),
)

print("Saved:")
print(" - eval/merged/auto_pic/fig1_auto_zh_en.pdf")
print(" - eval/merged/auto_pic/fig2_auto_en_zh.pdf")
print(" - eval/merged/auto_pic/fig3_auto_zh_zh.pdf")