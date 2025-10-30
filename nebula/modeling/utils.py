import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

from nebula.commons import Logger
from nebula.data.dataset import CLASSES

logger = Logger()
sns.set_theme(style="white")
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.linewidth": 1.2,
    }
)


class EarlyStopping:
    def __init__(self, patience: int, metric_name: str = "f1", mode: str = "max"):
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric_value: float) -> bool:
        score = metric_value

        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score
        else:
            improved = score < self.best_score

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f" EarlyStopping counter: {self.counter}/{self.patience} "
                f"(best {self.metric_name}: {self.best_score:.4f})"
            )
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(
                    f"Early stopping triggered! No improvement in {self.metric_name} "
                    f"for {self.patience} epochs."
                )
                logger.info(f"Best {self.metric_name}: {self.best_score:.4f}")
                return True

        return False


def plot_training_history(history: dict, save_path=None):
    """Plot training loss, CE loss, DA loss, and accuracies."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Losses
    ax1.plot(
        epochs, history["train_loss"], label="Total Loss", color="blue", linewidth=2
    )
    ax1.plot(epochs, history["ce_loss"], label="CE Loss", color="green", linewidth=2)
    ax1.plot(epochs, history["da_loss"], label="DA Loss", color="red", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(False)

    # Accuracy on twin axis
    ax2 = ax1.twinx()
    ax2.plot(
        epochs,
        history["source_acc"],
        label="Source Acc",
        color="orange",
        linestyle="-",
        linewidth=2,
    )
    ax2.plot(
        epochs,
        history["target_acc"],
        label="Target Acc",
        color="purple",
        linestyle="-",
        linewidth=2,
    )
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        bbox_to_anchor=(0.5, -0.15),
        loc="upper center",
        ncol=3,
    )

    plt.title("Training History")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)

    return fig


def plot_diag_history(diag_history: dict, save_path=None):
    """Plot selected diagnostic metrics over epochs."""
    if not diag_history:
        logger.info("No diagnostic history available to plot.")
        return

    epochs = range(1, max(len(v) for v in diag_history.values()) + 1)
    metrics_to_plot = ["mmd2", "sinkhorn_div", "domain_auc", "domain_acc"]

    fig = plt.figure(figsize=(8, 4.5))
    for metric in metrics_to_plot:
        if metric in diag_history:
            plt.plot(epochs, diag_history[metric], label=metric, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Metric Value")
    plt.title("Diagnostic Metrics Over Epochs")
    plt.legend()
    ax = plt.gca()
    ax.grid(False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)

    return fig


def plot_confusion_matrix(cm, normalize=True, save_path=None):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, None]

    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)

    return fig


def plot_embedding_scatter(embed_data: dict, title: str = "", save_path=None):
    """Plot 2D embedding colored by domain and by class in two subplots.

    Expects embed_data with keys: 'xy' (Nx2), 'domain' (N,), 'label' (N,).
    """
    import numpy as np

    xy = embed_data["xy"]
    domain = embed_data["domain"]
    label = embed_data["label"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Domain subplot
    for dom, color, marker, name in [
        (0, "#1f77b4", "o", "Source"),
        (1, "#d62728", "s", "Target"),
    ]:
        mask = domain == dom
        axes[0].scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=color,
            marker=marker,
            label=name,
            alpha=0.6,
            s=18,
            linewidth=0.0,
        )
    axes[0].set_title("Domain separation")
    axes[0].legend(frameon=False)

    # Class subplot
    palette = sns.color_palette("Set2", n_colors=len(CLASSES))
    for i, cls_name in enumerate(CLASSES):
        mask = label == i
        axes[1].scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=[palette[i]],
            label=cls_name,
            alpha=0.6,
            s=18,
            linewidth=0.0,
        )
    axes[1].set_title("Class separation")
    axes[1].legend(frameon=False, ncol=2)

    if title:
        fig.suptitle(title)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return fig


def plot_outlier_distribution(
    scores, threshold: float | None = None, title: str = "", save_path=None
):
    """Plot histogram of outlier scores and optional threshold line."""
    import numpy as np

    scores = scores if scores is not None else []
    scores = np.asarray(scores)
    scores = scores[np.isfinite(scores)] if scores.size else scores

    fig = plt.figure(figsize=(7, 4))
    if scores.size:
        plt.hist(scores, bins=40, color="#4e79a7", alpha=0.85, edgecolor="white")
        if threshold is not None and np.isfinite(threshold):
            plt.axvline(
                threshold,
                color="#e15759",
                linestyle="--",
                linewidth=2,
                label=f"q-thr = {threshold:.3f}",
            )
            plt.legend(frameon=False)
    plt.xlabel("Outlier score")
    plt.ylabel("Count")
    plt.title(title or "Target outlier scores")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return fig


def plot_per_class_bars(
    values_by_class: dict,
    ylabel: str,
    title: str = "",
    save_path=None,
    ylim: tuple | None = None,
):
    """Generic per-class bar plot.

    values_by_class: mapping from class name -> float value
    """
    cls_names = list(values_by_class.keys())
    vals = [values_by_class[c] for c in cls_names]

    fig = plt.figure(figsize=(8, 4))
    ax = sns.barplot(x=cls_names, y=vals, hue=cls_names, palette="Set2", legend=False)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if title:
        ax.set_title(title)
    for p, v in zip(ax.patches, vals):
        try:
            ax.annotate(
                f"{v:.2f}",
                (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=10,
            )
        except Exception:
            pass

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return fig


def plot_ot_mass_heatmap(
    ot_mass: torch.Tensor | None, title: str = "OT class-to-class mass", save_path=None
):
    """Plot OT transport mass matrix as heatmap.

    ot_mass: (C, C) array-like with rows source classes, columns target predicted classes.
    """
    import numpy as np

    if ot_mass is None:
        return None
    M = (
        ot_mass.detach().cpu().numpy()
        if hasattr(ot_mass, "detach")
        else np.asarray(ot_mass)
    )

    fig = plt.figure(figsize=(6.5, 5.2))
    ax = sns.heatmap(
        M,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=CLASSES,
        yticklabels=CLASSES,
        cbar_kws={"label": "Mass"},
    )
    ax.set_xlabel("Target (pred)")
    ax.set_ylabel("Source (true)")
    ax.set_title(title)

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    return fig


def add_equation_box(ax, text: str, loc: str = "upper right"):
    """Render a small LaTeX/text box on the given Axes.

    loc: one of {'upper left','upper right','lower left','lower right'}
    """

    loc_to_coords = {
        "upper left": (0.02, 0.98, "top", "left"),
        "upper right": (0.98, 0.98, "top", "right"),
        "lower left": (0.02, 0.02, "bottom", "left"),
        "lower right": (0.98, 0.02, "bottom", "right"),
    }
    x, y, va, ha = loc_to_coords.get(loc, loc_to_coords["upper right"])

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=12,
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#f8f9fb",
            "edgecolor": "#c9d3dd",
            "linewidth": 1.0,
        },
    )
