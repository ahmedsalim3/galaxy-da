from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nebula.commons import Logger
from nebula.visualizations.utils import (COLORS, add_equation_box, add_legend,
                                         add_warmup_region, get_loss_eq,
                                         prepare_data, set_plot_style)

logger = Logger()
set_plot_style()


def plot_training_history(
    history,
    trainer,
    save_path,
) -> plt.Figure:
    """
    Legacy interface - calls plot_training_summary
    """
    if "epoch_warmup" in history:
        warmup_epochs = int(sum(history.get("epoch_warmup", [])))
    else:
        warmup_epochs = int(
            getattr(getattr(trainer, "config", object()), "warmup_epochs", 0) or 0
        )
    logger.info(f"Warmup epochs: {warmup_epochs}")
    return plot_training_summary(history, warmup_epochs, trainer, save_path)


def plot_training_summary(
    history,
    warmup_epochs,
    trainer=None,
    save_path=None,
):
    """Plot comprehensive training summary with loss and accuracy subplots.

    Args:
        history: Training history dict
        warmup_epochs: Number of warmup epochs
        trainer: Trainer instance (for equation display)
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, (ax_loss, ax_acc) = plt.subplots(
        1, 2, figsize=(20, 6), gridspec_kw={"wspace": 0.25}
    )

    # ========== Loss Subplot ==========
    total_loss = prepare_data(history, "train_loss", epochs)
    ce_loss = prepare_data(history, "ce_loss", epochs)
    da_loss = prepare_data(history, "da_loss", epochs)
    eta1 = prepare_data(history, "eta_1", epochs)
    eta2 = prepare_data(history, "eta_2", epochs)
    sigma = prepare_data(history, "sigma", epochs)

    has_eta = (eta1.size > 0 and not np.all(np.isnan(eta1))) or (
        eta2.size > 0 and not np.all(np.isnan(eta2))
    )
    has_sigma = sigma.size > 0 and not np.all(np.isnan(sigma))

    add_warmup_region(ax_loss, warmup_epochs, epochs)

    # Plot loss components
    if total_loss.size > 0:
        ax_loss.plot(
            epochs,
            total_loss,
            label="Total Loss",
            color=COLORS["total_loss"],
            linewidth=2.5,
            marker="o",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )
    if ce_loss.size > 0:
        ax_loss.plot(
            epochs,
            ce_loss,
            label="Classification",
            color=COLORS["ce_loss"],
            linewidth=2.0,
            marker="s",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )
    if da_loss.size > 0 and not np.all(np.isnan(da_loss)) and not np.all(da_loss == 0):
        ax_loss.plot(
            epochs,
            da_loss,
            label="Domain Adapt.",
            color=COLORS["da_loss"],
            linewidth=2.0,
            marker="^",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )

    ax_loss.set_xlabel("Epoch", fontweight="bold")
    ax_loss.set_ylabel("Loss", fontweight="bold")
    ax_loss.set_title("Loss Components", pad=15)
    ax_loss.set_ylim(bottom=0)

    # Add eta/sigma curves on twin axis if present
    ax_twin = None
    if has_eta or has_sigma:
        ax_twin = ax_loss.twinx()
        ax_twin.spines["right"].set_visible(True)
        ax_twin.spines["right"].set_edgecolor(COLORS["total_loss"])
        ax_twin.spines["right"].set_linewidth(1.2)

        if has_eta:
            if eta1.size > 0 and not np.all(np.isnan(eta1)):
                ax_twin.plot(
                    epochs,
                    eta1,
                    label=r"$\eta_1$",
                    color=COLORS["eta1"],
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.9,
                )
            if eta2.size > 0 and not np.all(np.isnan(eta2)):
                ax_twin.plot(
                    epochs,
                    eta2,
                    label=r"$\eta_2$",
                    color=COLORS["eta2"],
                    linewidth=1.5,
                    linestyle="--",
                    alpha=0.9,
                )
        if has_sigma:
            ax_twin.plot(
                epochs,
                sigma,
                label=r"$\sigma$",
                color=COLORS["sigma"],
                linewidth=1.5,
                linestyle=":",
                alpha=0.9,
            )

        ax_twin.set_ylabel(r"$\eta$ / $\sigma$ Value", fontweight="bold")
        ax_twin.grid(False)

    add_legend(ax_loss, ax_twin, loc="upper right")
    add_equation_box(ax_loss, get_loss_eq(trainer), position="top_left")

    # ========== Accuracy Subplot ==========
    source_acc = prepare_data(history, "source_acc", epochs)
    target_acc = prepare_data(history, "target_acc", epochs)

    add_warmup_region(ax_acc, warmup_epochs, epochs)

    if source_acc.size > 0:
        ax_acc.plot(
            epochs,
            source_acc,
            label="Source",
            color=COLORS["source_acc"],
            linewidth=2.5,
            marker="o",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )
    if (
        target_acc.size > 0
        and not np.all(np.isnan(target_acc))
        and not np.all(target_acc == 0)
    ):
        ax_acc.plot(
            epochs,
            target_acc,
            label="Target",
            color=COLORS["target_acc"],
            linewidth=2.5,
            marker="s",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )

    ax_acc.set_xlabel("Epoch", fontweight="bold")
    ax_acc.set_ylabel("Accuracy (%)", fontweight="bold")
    ax_acc.set_title("Classification Accuracy", pad=15)
    ax_acc.set_ylim([0, 105])

    add_legend(ax_acc, loc="lower right")

    # ========== Save Figures ==========
    plt.tight_layout()

    if save_path:
        base_path = Path(save_path)
        base_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(base_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved training summary plot to {save_path}")

        # Save individual components
        loss_path = base_path.parent / f"{base_path.stem}_loss.png"
        acc_path = base_path.parent / f"{base_path.stem}_accuracy.png"

        fig_loss_only = plot_loss_components(history, warmup_epochs, trainer, loss_path)
        plt.close(fig_loss_only)

        fig_acc_only = plot_accuracy_curves(history, warmup_epochs, acc_path)
        plt.close(fig_acc_only)

    return fig


def plot_loss_components(
    history,
    warmup_epochs: int = 0,
    trainer=None,
    save_path=None,
):
    """Plot training loss components with optional eta/sigma curves.

    Args:
        history: Training history dict
        warmup_epochs: Number of warmup epochs
        trainer: Trainer instance (for equation box)
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        logger.warning("No training history to plot (loss)")
        return plt.figure()

    # Prepare data
    total_loss = prepare_data(history, "train_loss", epochs)
    ce_loss = prepare_data(history, "ce_loss", epochs)
    da_loss = prepare_data(history, "da_loss", epochs)
    eta1 = prepare_data(history, "eta_1", epochs)
    eta2 = prepare_data(history, "eta_2", epochs)
    sigma = prepare_data(history, "sigma", epochs)

    has_eta = (eta1.size > 0 and not np.all(np.isnan(eta1))) or (
        eta2.size > 0 and not np.all(np.isnan(eta2))
    )
    has_sigma = sigma.size > 0 and not np.all(np.isnan(sigma))

    # Create figure
    fig, ax = plt.subplots(figsize=(9, 6))

    add_warmup_region(ax, warmup_epochs, epochs)

    # Plot loss components
    if total_loss.size > 0:
        ax.plot(
            epochs,
            total_loss,
            label="Total Loss",
            color=COLORS["total_loss"],
            linewidth=2.5,
            marker="o",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )
    if ce_loss.size > 0:
        ax.plot(
            epochs,
            ce_loss,
            label="Classification Loss",
            color=COLORS["ce_loss"],
            linewidth=2.0,
            marker="s",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )
    if da_loss.size > 0 and not np.all(np.isnan(da_loss)) and not np.all(da_loss == 0):
        ax.plot(
            epochs,
            da_loss,
            label="Domain Adaptation Loss",
            color=COLORS["da_loss"],
            linewidth=2.0,
            marker="^",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )

    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")
    ax.set_title("Training Loss Components", pad=15)
    ax.set_ylim(bottom=0)

    # Add eta/sigma curves on twin axis if present
    ax_twin = None
    if has_eta or has_sigma:
        ax_twin = ax.twinx()
        ax_twin.spines["right"].set_visible(True)
        ax_twin.spines["right"].set_edgecolor(COLORS["total_loss"])
        ax_twin.spines["right"].set_linewidth(1.2)

        if has_eta:
            if eta1.size > 0 and not np.all(np.isnan(eta1)):
                ax_twin.plot(
                    epochs,
                    eta1,
                    label=r"$\eta_1$ (Weight)",
                    color=COLORS["eta1"],
                    linewidth=1.8,
                    linestyle="--",
                    alpha=0.9,
                )
            if eta2.size > 0 and not np.all(np.isnan(eta2)):
                ax_twin.plot(
                    epochs,
                    eta2,
                    label=r"$\eta_2$ (Weight)",
                    color=COLORS["eta2"],
                    linewidth=1.8,
                    linestyle="--",
                    alpha=0.9,
                )
            ax_twin.set_ylabel(r"$\eta$ (Uncertainty Weight)", fontweight="bold")

        if has_sigma:
            ax_twin.plot(
                epochs,
                sigma,
                label=r"$\sigma$ (Blur)",
                color=COLORS["sigma"],
                linewidth=1.8,
                linestyle=":",
                alpha=0.9,
            )
            if not has_eta:
                ax_twin.set_ylabel(r"$\sigma$ (Sinkhorn Blur)", fontweight="bold")
            else:
                ax_twin.set_ylabel(r"$\eta$ / $\sigma$ Value", fontweight="bold")

        ax_twin.grid(False)

    add_legend(ax, ax_twin, loc="upper right")
    add_equation_box(ax, get_loss_eq(trainer), position="top_left")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved loss plot to {save_path}")

    return fig


def plot_accuracy_curves(
    history,
    warmup_epochs,
    save_path=None,
):
    """Plot source and target accuracy over training.

    Args:
        history: Training history dict
        warmup_epochs: Number of warmup epochs
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    # Try to determine epochs from train_loss or source_acc
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        epochs = list(range(1, len(history.get("source_acc", [])) + 1))

    if not epochs:
        logger.warning("No training history to plot (accuracy)")
        return plt.figure()

    fig, ax = plt.subplots(figsize=(9, 6))

    # Prepare data
    source_acc = prepare_data(history, "source_acc", epochs)
    target_acc = prepare_data(history, "target_acc", epochs)

    add_warmup_region(ax, warmup_epochs, epochs)

    # Plot accuracies
    if source_acc.size > 0:
        ax.plot(
            epochs,
            source_acc,
            label="Source Domain",
            color=COLORS["source_acc"],
            linewidth=2.5,
            marker="o",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )

    if (
        target_acc.size > 0
        and not np.all(np.isnan(target_acc))
        and not np.all(target_acc == 0)
    ):
        ax.plot(
            epochs,
            target_acc,
            label="Target Domain",
            color=COLORS["target_acc"],
            linewidth=2.5,
            marker="s",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )

    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title("Training Accuracy", pad=15)
    ax.set_ylim([0, 105])

    add_legend(ax, loc="lower right")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved accuracy plot to {save_path}")

    return fig
