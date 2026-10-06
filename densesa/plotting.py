"""Figures. Everything is saved at 600 dpi unless the canvas would exceed
18000 px on its longest side, in which case the dpi is reduced to fit."""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import cm as mpl_cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

COLORS = {"DenseSANet": "#2ECC71", "VGG16": "#3498DB", "CustomCNN": "#E74C3C"}
LABELS = {"DenseSANet": "DenseSA-Net", "VGG16": "VGG16", "CustomCNN": "CustomCNN"}


def save(fig, path, dpi=600, max_side=18000):
    w, h = fig.get_size_inches()
    dpi = min(dpi, int(max_side / max(w, h)))
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def overlay(rgb, heat, alpha=0.45, cmap="jet"):
    col = matplotlib.colormaps[cmap](heat)[..., :3]
    return np.clip(rgb * (1 - alpha) + col * alpha, 0, 1)


def colorbar(fig, ax, label, cmap="jet"):
    sm = mpl_cm.ScalarMappable(cmap=cmap)
    sm.set_array([0, 1])
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.06)
    cb = fig.colorbar(sm, cax=cax)
    cb.ax.tick_params(labelsize=7)
    cb.set_label(label, fontsize=7)


def confusion_grid(cms, accs, classes, title, path, dpi=600):
    names = list(cms)
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 6.2))
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        sns.heatmap(cms[name], annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    annot_kws={"size": 9}, linewidths=0.3, linecolor="white")
        ax.set_title(f"{LABELS.get(name, name)}\n(Acc = {accs[name]:.4f})",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, path, dpi)


def per_class_bars(df, title, path, dpi=600):
    metrics = [("Precision", "#2ECC71"), ("Sensitivity", "#3498DB"),
               ("Specificity", "#E67E22"), ("F1-Score", "#E74C3C")]
    fig, ax = plt.subplots(figsize=(max(10, 1.7 * len(df)), 6))
    x = np.arange(len(df))
    w = 0.2
    for k, (col, color) in enumerate(metrics):
        off = (k - 1.5) * w
        ax.bar(x + off, df[col], w, label=col, color=color, edgecolor="white")
        for i, v in enumerate(df[col]):
            ax.text(i + off, v + 0.002, f"{v:.3f}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Class"], rotation=25, ha="right")
    low = df[[c for c, _ in metrics]].values.min()
    ax.set_ylim(max(0.0, low - 0.03), 1.02)
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title, fontweight="bold")
    fig.tight_layout()
    save(fig, path, dpi)


def cv_boxplot(scores, title, path, dpi=600):
    names = list(scores)
    fig, ax = plt.subplots(figsize=(9, 6))
    bp = ax.boxplot([scores[n] for n in names], patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 2.2})
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(COLORS.get(n, "#999999"))
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([LABELS.get(n, n) for n in names])
    ax.set_ylabel("Fold accuracy")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", alpha=0.35)
    low = min(min(v) for v in scores.values())
    for i, n in enumerate(names, 1):
        ax.text(i, low - 0.004, f"mean={np.mean(scores[n]):.4f}", ha="center",
                fontsize=9, color=COLORS.get(n, "black"), fontweight="bold")
    fig.tight_layout()
    save(fig, path, dpi)


def factorial_bars(df, title, path, dpi=600):
    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(df))
    w = 0.26
    for off, col, color in ((-w, "Accuracy", "#2ECC71"), (0, "F1-Score", "#3498DB"),
                            (w, "AUC", "#E74C3C")):
        ax.bar(x + off, df[col], w, label=col, color=color, edgecolor="white")
        for i, v in enumerate(df[col]):
            ax.text(i + off, v + 0.0012, f"{v:.4f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{b}" for a, b in zip(df["Variant"], df["Label"])],
                       rotation=25, ha="right", fontsize=8.5)
    ax.set_ylim(df[["Accuracy", "F1-Score", "AUC"]].values.min() - 0.01, 1.008)
    ax.set_ylabel("Score")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, path, dpi)


def effects_plot(eff, title, path, dpi=600):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, metric in zip(axes, ("Accuracy", "F1-Score", "AUC")):
        sub = eff[eff["Metric"] == metric].sort_values("Effect")
        colors = ["#2ECC71" if v >= 0 else "#E74C3C" for v in sub["Effect"]]
        ax.barh(sub["Term"], sub["Effect"], color=colors, edgecolor="white")
        ax.axvline(0, color="black", lw=1)
        for i, v in enumerate(sub["Effect"]):
            ax.text(v, i, f" {v:+.4f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.set_title(metric, fontweight="bold")
        ax.set_xlabel("Effect")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, path, dpi)


def saliency_rows(rows, title, path, dpi=600, title_color=None):
    """rows: list of (rgb, gradcam, attention, caption)."""
    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.4 * n))
    axes = np.atleast_2d(axes)
    for r, (rgb, heat, att, caption) in enumerate(rows):
        axes[r, 0].imshow(rgb)
        axes[r, 0].set_title(caption, fontsize=9, color=title_color or "black",
                             fontweight="bold" if title_color else "normal")
        axes[r, 1].imshow(overlay(rgb, heat))
        axes[r, 1].set_title("Grad-CAM", fontsize=9)
        colorbar(fig, axes[r, 1], "Normalised activation")
        axes[r, 2].imshow(overlay(rgb, att))
        axes[r, 2].set_title("Spatial attention map", fontsize=9)
        colorbar(fig, axes[r, 2], "Attention weight")
        for a in axes[r]:
            a.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.999)
    fig.tight_layout()
    save(fig, path, dpi)


def alignment_vs_chance(summary, path, dpi=600, topk=0.20):
    specs = [("PointingGame", "PointingGame_chance", "Pointing game"),
             ("EnergyInMask", "EnergyInMask_chance", "CAM energy in nucleus"),
             ("IoU_topk", "IoU_chance", f"IoU (top-{int(topk * 100)}%)")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    x = np.arange(len(summary))
    w = 0.35
    for ax, (obs, ch, title) in zip(axes, specs):
        ax.bar(x - w / 2, summary[obs], w, label="Observed", color="#2ECC71", edgecolor="white")
        ax.bar(x + w / 2, summary[ch], w, label="Chance", color="#95A5A6", edgecolor="white")
        for i, (o, c) in enumerate(zip(summary[obs], summary[ch])):
            ax.text(i - w / 2, o + 0.006, f"{o:.3f}", ha="center", fontsize=8)
            ax.text(i + w / 2, c + 0.006, f"{c:.3f}", ha="center", fontsize=8)
            lift = o / c if c > 0 else 0
            ax.text(i, max(o, c) + 0.055, f"{lift:.1f}x", ha="center", fontsize=10,
                    fontweight="bold", color="#1A7A1A" if lift >= 1.5 else "#B91C1C")
        ax.set_xticks(x)
        ax.set_xticklabels(summary["Dataset"])
        ax.set_ylim(0, max(summary[[obs, ch]].values.max() * 1.35, 0.25))
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Attention-morphology alignment relative to chance",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    save(fig, path, dpi)
