# ─────────────────────────────────────────────────────────────
# Plot Results Script
# ─────────────────────────────────────────────────────────────

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

RESULTS_FILE = os.path.join(BASE_DIR, "all_results.json")
PCA_FILE     = os.path.join(BASE_DIR, "pca_results.json")
SAVE_DIR     = os.path.join(BASE_DIR, "plots")

os.makedirs(SAVE_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="talk")

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.titlesize": 16,
    "axes.labelsize": 13,
})


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_results(path):
    with open(path, "r") as f:
        return pd.DataFrame(json.load(f))


df = load_results(RESULTS_FILE)

pca_df = None
if os.path.exists(PCA_FILE):
    pca_df = load_results(PCA_FILE)


# ─────────────────────────────────────────────
# PLOT 1 — MODEL PERFORMANCE
# ─────────────────────────────────────────────
def plot_model_comparison(df):
    plt.figure()

    sns.barplot(
        data=df,
        x="model",
        y="accuracy",
        hue="feature_space",
        palette="Set2"
    )

    plt.xticks(rotation=30)
    plt.title("Model Performance Across Feature Spaces")
    plt.ylabel("Accuracy")

    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/model_comparison.png")
    plt.close()


# ─────────────────────────────────────────────
# PLOT 2 — FEATURE SPACE COMPARISON
# ─────────────────────────────────────────────
def plot_feature_comparison(df):
    plt.figure()

    sns.barplot(
        data=df,
        x="feature_space",
        y="accuracy",
        palette="muted"
    )

    plt.title("Accuracy by Feature Representation")
    plt.ylabel("Accuracy")

    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/feature_comparison.png")
    plt.close()


# ─────────────────────────────────────────────
# PLOT 3 — PCA / SVD CURVES
# ─────────────────────────────────────────────
def plot_pca_curves(pca_df):
    if pca_df is None:
        return

    # extract dimension
    pca_df["dim"] = pca_df["feature_space"].str.extract(r"\((\d+)\)").astype(int)
    pca_df["type"] = pca_df["feature_space"].str.extract(r"(TF-IDF|BERT)")

    plt.figure()

    sns.lineplot(
        data=pca_df,
        x="dim",
        y="accuracy",
        hue="type",
        marker="o"
    )

    plt.title("Accuracy vs Dimensionality Reduction")
    plt.xlabel("Number of Components")

    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/pca_curves.png")
    plt.close()


# ─────────────────────────────────────────────
# PLOT 4 — ACCURACY VS TIME
# ─────────────────────────────────────────────
def plot_time_vs_accuracy(df):
    plt.figure()

    sns.scatterplot(
        data=df,
        x="train_time_s",
        y="accuracy",
        hue="feature_space",
        style="model",
        s=100
    )

    plt.title("Accuracy vs Training Time")

    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/time_vs_accuracy.png")
    plt.close()


# ─────────────────────────────────────────────
# PLOT 5 — TOP MODELS
# ─────────────────────────────────────────────
def plot_top_models(df):
    top_df = df.sort_values("accuracy", ascending=False).head(10)

    plt.figure()

    sns.barplot(
        data=top_df,
        x="accuracy",
        y="model",
        hue="feature_space"
    )

    plt.title("Top Performing Models")

    sns.despine()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/top_models.png")
    plt.close()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    plot_model_comparison(df)
    plot_feature_comparison(df)
    plot_time_vs_accuracy(df)
    plot_top_models(df)

    if pca_df is not None:
        plot_pca_curves(pca_df)

    print(f"All plots saved in: {SAVE_DIR}/")


if __name__ == "__main__":
    main()