"""
Step 5 - Charts for the report (all saved to outputs/charts/ at 150 dpi).

fig1_class_distribution.png     - ham vs spam counts
fig2_length_histogram.png       - token-length distribution by class
fig3_top_tokens.png             - most frequent tokens in spam vs ham
fig4_training_curves.png        - loss + F1 across epochs
fig5_confusion_matrix.png       - LSTM confusion matrix on the test set
fig6_roc_curves.png             - ROC: LSTM vs Naive Bayes baseline

Run:  python visualise.py
"""

import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "outputs")
CHART_DIR = os.path.join(OUT_DIR, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
BLUE, RED, GREEN = "#4C72B0", "#C44E52", "#55A868"


def main():
    stats = json.load(open(os.path.join(OUT_DIR, "data_stats.json")))
    history = json.load(open(os.path.join(OUT_DIR, "history.json")))
    metrics = json.load(open(os.path.join(OUT_DIR, "metrics.json")))
    train_df = pd.read_csv(os.path.join(OUT_DIR, "train.csv"))
    train_df["clean"] = train_df.clean.fillna("").astype(str)

    # ---- fig 1: class distribution ---------------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(["Ham (legitimate)", "Spam"],
                  [stats["n_ham"], stats["n_spam"]], color=[BLUE, RED], width=0.55)
    ax.bar_label(bars, fmt="%d", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of messages")
    ax.set_title("Class distribution of the SMS Spam Collection", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "fig1_class_distribution.png"), dpi=150)
    plt.close(fig)

    # ---- fig 2: message length histogram ----------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bins = np.arange(0, 61, 2)
    ham = train_df[train_df.label == "ham"].n_tokens.clip(upper=60)
    spam = train_df[train_df.label == "spam"].n_tokens.clip(upper=60)
    ax.hist(ham, bins=bins, alpha=0.75, label="Ham", color=BLUE, density=True)
    ax.hist(spam, bins=bins, alpha=0.75, label="Spam", color=RED, density=True)
    ax.axvline(50, color="black", ls="--", lw=1)
    ax.annotate("MAX_LEN = 50\n(99th percentile)", xy=(50, ax.get_ylim()[1] * 0.92),
                fontsize=8, ha="left")
    ax.set_xlabel("Message length (tokens)")
    ax.set_ylabel("Density")
    ax.set_title("Message length distribution (train split)", fontsize=11)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "fig2_length_histogram.png"), dpi=150)
    plt.close(fig)

    # ---- fig 3: top tokens in spam vs ham ---------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6))
    for ax, label, color in [(axes[0], "spam", RED), (axes[1], "ham", BLUE)]:
        tokens = Counter()
        for t in train_df[train_df.label == label].clean:
            tokens.update(t.split())
        top = tokens.most_common(12)[::-1]
        ax.barh([w for w, _ in top], [c for _, c in top], color=color)
        ax.set_title(f"Top tokens - {label}", fontsize=10)
    fig.suptitle("Most frequent tokens per class (train split)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(CHART_DIR, "fig3_top_tokens.png"), dpi=150)
    plt.close(fig)

    # ---- fig 4: training curves -------------------------------------------
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4))
    axes[0].plot(ep, [h["train_loss"] for h in history], "-o", ms=3, label="train", color=BLUE)
    axes[0].plot(ep, [h["val_loss"] for h in history], "-o", ms=3, label="validation", color=RED)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE loss")
    axes[0].set_title("Loss", fontsize=10); axes[0].legend(fontsize=8)
    axes[1].plot(ep, [h["train_acc"] for h in history], "-o", ms=3, label="train acc", color=BLUE)
    axes[1].plot(ep, [h["val_acc"] for h in history], "-o", ms=3, label="val acc", color=GREEN)
    axes[1].plot(ep, [h["val_f1"] for h in history], "-s", ms=3, label="val F1 (spam)", color=RED)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
    axes[1].set_title("Accuracy / F1", fontsize=10); axes[1].legend(fontsize=8)
    fig.suptitle("LSTM training progress", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(os.path.join(CHART_DIR, "fig4_training_curves.png"), dpi=150)
    plt.close(fig)

    # ---- fig 5: confusion matrix ------------------------------------------
    cm = metrics["lstm"]["confusion"]
    mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    sns.heatmap(mat, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                annot_kws={"size": 13, "weight": "bold"},
                xticklabels=["Pred ham", "Pred spam"],
                yticklabels=["Actual ham", "Actual spam"])
    ax.set_title("LSTM confusion matrix (test set)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "fig5_confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ---- fig 6: ROC curves -------------------------------------------------
    fpr, tpr = np.load(os.path.join(OUT_DIR, "roc_points.npy"))
    nb_fpr, nb_tpr = np.load(os.path.join(OUT_DIR, "roc_points_nb.npy"))
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    ax.plot(fpr, tpr, color=RED, lw=2,
            label=f"LSTM (AUC = {metrics['lstm']['roc_auc']:.3f})")
    ax.plot(nb_fpr, nb_tpr, color=BLUE, lw=2, ls="--",
            label=f"Naive Bayes (AUC = {metrics['naive_bayes']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls=":", label="Random (AUC = 0.5)")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the test set", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "fig6_roc_curves.png"), dpi=150)
    plt.close(fig)

    print("Saved 6 charts to outputs/charts/:")
    for name in sorted(os.listdir(CHART_DIR)):
        print("  -", name)


if __name__ == "__main__":
    main()
