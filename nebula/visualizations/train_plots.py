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
    """Plot comprehensive training summary with 2x2 subplot layout.

    Subplots:
        - Top-left: Main losses (total, CE, DA)
        - Top-right: Auxiliary losses (alignment, entropy)
        - Bottom-left: Classification accuracy
        - Bottom-right: Trainable parameters (eta_1, eta_2, sigma, lambda_grl)

    Args:
        history: Training history dict
        warmup_epochs: Number of warmup epochs
        trainer: Trainer instance (for equation display)
        save_path: Path to save figure

    Returns:
        Matplotlib figure
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)
    
    ax_main_loss = fig.add_subplot(gs[0, 0])
    ax_aux_loss = fig.add_subplot(gs[0, 1])
    ax_acc = fig.add_subplot(gs[1, 0])
    ax_params = fig.add_subplot(gs[1, 1])

    # Prepare all data
    total_loss = prepare_data(history, "train_loss", epochs)
    ce_loss = prepare_data(history, "ce_loss", epochs)
    da_loss = prepare_data(history, "da_loss", epochs)
    align_loss = prepare_data(history, "align_loss", epochs)
    entropy_loss = prepare_data(history, "entropy_loss", epochs)
    eta1 = prepare_data(history, "eta_1", epochs)
    eta2 = prepare_data(history, "eta_2", epochs)
    sigma = prepare_data(history, "sigma", epochs)
    lambda_grl = prepare_data(history, "lambda_grl", epochs)
    source_acc = prepare_data(history, "source_acc", epochs)
    target_acc = prepare_data(history, "target_acc", epochs)
    target_f1 = prepare_data(history, "target_f1", epochs)

    # Check what we have
    has_align = align_loss.size > 0 and not np.all(np.isnan(align_loss)) and not np.all(align_loss == 0)
    has_entropy = entropy_loss.size > 0 and not np.all(np.isnan(entropy_loss)) and not np.all(entropy_loss == 0)
    has_eta1 = eta1.size > 0 and not np.all(np.isnan(eta1))
    has_eta2 = eta2.size > 0 and not np.all(np.isnan(eta2))
    has_sigma = sigma.size > 0 and not np.all(np.isnan(sigma))
    has_lambda_grl = lambda_grl.size > 0 and not np.all(np.isnan(lambda_grl))
    has_da = da_loss.size > 0 and not np.all(np.isnan(da_loss)) and not np.all(da_loss == 0)

    # ========== SUBPLOT 1: Main Losses ==========
    add_warmup_region(ax_main_loss, warmup_epochs, epochs)
    
    # Plot loss components
    if total_loss.size > 0:
        ax_main_loss.plot(
            epochs, total_loss,
            label=r"$L_{\text{total}}$",
            color=COLORS["total_loss"],
            linewidth=2.5,
            marker="o",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )
    if ce_loss.size > 0:
        ax_main_loss.plot(
            epochs, ce_loss,
            label=r"$L_{\text{CE}}$",
            color=COLORS["ce_loss"],
            linewidth=2.0,
            marker="s",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )
    if has_da:
        ax_main_loss.plot(
            epochs, da_loss,
            label=r"$L_{\text{DA}}$",
            color=COLORS["da_loss"],
            linewidth=2.0,
            marker="^",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )

    ax_main_loss.set_xlabel("Epoch", fontweight="bold")
    ax_main_loss.set_ylabel("Loss", fontweight="bold")
    ax_main_loss.set_title("Main Loss Components", pad=15, fontsize=14, fontweight="bold")
    if total_loss.size > 0 or ce_loss.size > 0 or (has_da and da_loss.size > 0):
        all_losses = []
        if total_loss.size > 0:
            all_losses.extend(total_loss[~np.isnan(total_loss)])
        if ce_loss.size > 0:
            all_losses.extend(ce_loss[~np.isnan(ce_loss)])
        if has_da and da_loss.size > 0:
            all_losses.extend(da_loss[~np.isnan(da_loss)])
        if all_losses:
            y_min = min(all_losses)
            y_max = max(all_losses)
            margin = (y_max - y_min) * 0.1 if y_max > y_min else abs(y_min) * 0.1
            ax_main_loss.set_ylim(bottom=y_min - margin if y_min < 0 else 0, top=y_max + margin)
    add_legend(ax_main_loss, loc="upper right")

    # ========== SUBPLOT 2: Auxiliary Losses ==========
    if has_align or has_entropy:
        add_warmup_region(ax_aux_loss, warmup_epochs, epochs)
        
        if has_align:
            ax_aux_loss.plot(
                epochs, align_loss,
                label=r"$L_{\text{align}}$",
                color=COLORS["align_loss"],
                linewidth=2.0,
                marker="d",
                markersize=4,
                markevery=max(1, len(epochs) // 15),
            )
        if has_entropy:
            ax_aux_loss.plot(
                epochs, entropy_loss,
                label=r"$L_{\text{entropy}}$",
                color=COLORS["entropy_loss"],
                linewidth=2.0,
                marker="v",
                markersize=4,
                markevery=max(1, len(epochs) // 15),
            )

        ax_aux_loss.set_xlabel("Epoch", fontweight="bold")
        ax_aux_loss.set_ylabel("Loss", fontweight="bold")
        ax_aux_loss.set_title("Auxiliary Loss Components", pad=15, fontsize=14, fontweight="bold")
        all_aux_losses = []
        if has_align and align_loss.size > 0:
            all_aux_losses.extend(align_loss[~np.isnan(align_loss)])
        if has_entropy and entropy_loss.size > 0:
            all_aux_losses.extend(entropy_loss[~np.isnan(entropy_loss)])
        if all_aux_losses:
            y_min = min(all_aux_losses)
            y_max = max(all_aux_losses)
            margin = (y_max - y_min) * 0.1 if y_max > y_min else abs(y_min) * 0.1
            ax_aux_loss.set_ylim(bottom=y_min - margin if y_min < 0 else 0, top=y_max + margin)
        add_legend(ax_aux_loss, loc="upper right")
    else:
        ax_aux_loss.text(0.5, 0.5, "No Auxiliary Losses", 
                        ha='center', va='center', fontsize=14, color='gray')
        ax_aux_loss.set_title("Auxiliary Loss Components", pad=15, fontsize=14, fontweight="bold")
        ax_aux_loss.axis('off')

    # ========== SUBPLOT 3: Accuracy ==========
    add_warmup_region(ax_acc, warmup_epochs, epochs)

    if source_acc.size > 0:
        ax_acc.plot(
            epochs, source_acc,
            label=r"$\text{Acc}_S$",
            color=COLORS["source_acc"],
            linewidth=2.5,
            marker="o",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )
    if target_acc.size > 0 and not np.all(np.isnan(target_acc)) and not np.all(target_acc == 0):
        ax_acc.plot(
            epochs, target_acc,
            label=r"$\text{Acc}_T$",
            color=COLORS["target_acc"],
            linewidth=2.5,
            marker="s",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )

    ax_acc.set_xlabel("Epoch", fontweight="bold")
    ax_acc.set_ylabel("Accuracy (%)", fontweight="bold")
    ax_acc.set_title("Classification Accuracy", pad=15, fontsize=14, fontweight="bold")
    ax_acc.set_ylim([0, 105])
    
    # Add F1 score on twin axis if available
    has_f1 = target_f1.size > 0 and not np.all(np.isnan(target_f1)) and not np.all(target_f1 == 0)
    if has_f1:
        ax_f1 = ax_acc.twinx()
        ax_f1.spines["right"].set_visible(True)
        ax_f1.spines["right"].set_edgecolor("#AAAAAA")
        ax_f1.spines["right"].set_linewidth(1.2)
        ax_f1.plot(
            epochs, target_f1,
            label=r"$\text{F1}_T$",
            color=COLORS["target_f1"],
            linewidth=2.5,
            marker="^",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
            linestyle="--",
        )
        ax_f1.set_ylabel("F1 Score", fontweight="bold")
        ax_f1.set_ylim([0, 1.05])
        add_legend(ax_acc, ax_f1, loc="lower right")
    else:
        add_legend(ax_acc, loc="lower right")

    # ========== SUBPLOT 4: Trainable Parameters ==========
    if has_eta1 or has_eta2 or has_sigma or has_lambda_grl:
        add_warmup_region(ax_params, warmup_epochs, epochs)
        
        if has_eta1:
            ax_params.plot(
                epochs, eta1,
                label=r"$\eta_1$",
                color=COLORS["eta1"],
                linewidth=2.0,
                marker="o",
                markersize=3,
                markevery=max(1, len(epochs) // 15),
            )
        if has_eta2:
            ax_params.plot(
                epochs, eta2,
                label=r"$\eta_2$",
                color=COLORS["eta2"],
                linewidth=2.0,
                marker="s",
                markersize=3,
                markevery=max(1, len(epochs) // 15),
            )
        if has_sigma:
            ax_params.plot(
                epochs, sigma,
                label=r"$\sigma$",
                color=COLORS["sigma"],
                linewidth=2.0,
                marker="^",
                markersize=3,
                markevery=max(1, len(epochs) // 15),
            )
        if has_lambda_grl:
            ax_params.plot(
                epochs, lambda_grl,
                label=r"$\lambda_{GRL}$",
                color=COLORS.get("lambda_grl", "#8B4513"),
                linewidth=2.0,
                marker="d",
                markersize=3,
                markevery=max(1, len(epochs) // 15),
            )

        ax_params.set_xlabel("Epoch", fontweight="bold")
        ax_params.set_ylabel("Parameter Value", fontweight="bold")
        ax_params.set_title("Trainable Parameters", pad=15, fontsize=14, fontweight="bold")
        ax_params.set_ylim(bottom=0)
        add_legend(ax_params, loc="upper right")
    else:
        ax_params.text(0.5, 0.5, "No Trainable Parameters", 
                      ha='center', va='center', fontsize=14, color='gray')
        ax_params.set_title("Trainable Parameters", pad=15, fontsize=14, fontweight="bold")
        ax_params.axis('off')

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
    align_loss = prepare_data(history, "align_loss", epochs)
    entropy_loss = prepare_data(history, "entropy_loss", epochs)
    eta1 = prepare_data(history, "eta_1", epochs)
    eta2 = prepare_data(history, "eta_2", epochs)
    sigma = prepare_data(history, "sigma", epochs)

    has_eta = (eta1.size > 0 and not np.all(np.isnan(eta1))) or (
        eta2.size > 0 and not np.all(np.isnan(eta2))
    )
    has_sigma = sigma.size > 0 and not np.all(np.isnan(sigma))

    fig, ax = plt.subplots(figsize=(14, 9))
    fig.subplots_adjust(top=0.90, bottom=0.30, left=0.22, right=0.92)

    add_warmup_region(ax, warmup_epochs, epochs)

    # Plot loss components
    if total_loss.size > 0:
        ax.plot(
            epochs,
            total_loss,
            label=r"$L_{\text{total}}$",
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
            label=r"$L_{\text{CE}}$",
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
            label=r"$L_{\text{DA}}$",
            color=COLORS["da_loss"],
            linewidth=2.0,
            marker="^",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )
    if (
        align_loss.size > 0
        and not np.all(np.isnan(align_loss))
        and not np.all(align_loss == 0)
    ):
        ax.plot(
            epochs,
            align_loss,
            label=r"$L_{\text{align}}$",
            color=COLORS["align_loss"],
            linewidth=2.0,
            marker="d",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )
    if (
        entropy_loss.size > 0
        and not np.all(np.isnan(entropy_loss))
        and not np.all(entropy_loss == 0)
    ):
        ax.plot(
            epochs,
            entropy_loss,
            label=r"$L_{\text{entropy}}$",
            color=COLORS["entropy_loss"],
            linewidth=2.0,
            marker="v",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
        )

    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Loss", fontweight="bold")
    ax.set_title("Training Loss Components", pad=15)
    all_losses = []
    if total_loss.size > 0:
        all_losses.extend(total_loss[~np.isnan(total_loss)])
    if ce_loss.size > 0:
        all_losses.extend(ce_loss[~np.isnan(ce_loss)])
    if da_loss.size > 0 and not np.all(np.isnan(da_loss)):
        all_losses.extend(da_loss[~np.isnan(da_loss)])
    if align_loss.size > 0 and not np.all(np.isnan(align_loss)):
        all_losses.extend(align_loss[~np.isnan(align_loss)])
    if entropy_loss.size > 0 and not np.all(np.isnan(entropy_loss)):
        all_losses.extend(entropy_loss[~np.isnan(entropy_loss)])
    if all_losses:
        y_min = min(all_losses)
        y_max = max(all_losses)
        margin = (y_max - y_min) * 0.1 if y_max > y_min else max(abs(y_min), abs(y_max)) * 0.1
        bottom = y_min - margin if y_min < 0 else 0
        top = y_max + margin
        ax.set_ylim(bottom=bottom, top=top)

    # Add eta/sigma curves on twin axis if present
    ax_twin = None
    if has_eta or has_sigma:
        ax_twin = ax.twinx()
        ax_twin.spines["right"].set_visible(True)
        ax_twin.spines["right"].set_edgecolor("#AAAAAA")
        ax_twin.spines["right"].set_linewidth(1.2)

        if has_eta:
            if eta1.size > 0 and not np.all(np.isnan(eta1)):
                ax_twin.plot(
                    epochs,
                    eta1,
                    label=r"$\eta_1$",
                    color=COLORS["eta1"],
                    linewidth=1.8,
                    linestyle="--",
                    alpha=0.9,
                )
            if eta2.size > 0 and not np.all(np.isnan(eta2)):
                ax_twin.plot(
                    epochs,
                    eta2,
                    label=r"$\eta_2$",
                    color=COLORS["eta2"],
                    linewidth=1.8,
                    linestyle="--",
                    alpha=0.9,
                )
            ax_twin.set_ylabel(r"$\eta$", fontweight="bold")

        if has_sigma:
            ax_twin.plot(
                epochs,
                sigma,
                label=r"$\sigma$",
                color=COLORS["sigma"],
                linewidth=1.8,
                linestyle=":",
                alpha=0.9,
            )
            if not has_eta:
                ax_twin.set_ylabel(r"$\sigma$", fontweight="bold")
            else:
                ax_twin.set_ylabel(r"$\eta$ / $\sigma$", fontweight="bold")

        ax_twin.grid(False)
        
        twin_values = []
        if has_eta:
            if eta1.size > 0 and not np.all(np.isnan(eta1)):
                twin_values.extend(eta1[~np.isnan(eta1)])
            if eta2.size > 0 and not np.all(np.isnan(eta2)):
                twin_values.extend(eta2[~np.isnan(eta2)])
        if has_sigma:
            if sigma.size > 0 and not np.all(np.isnan(sigma)):
                twin_values.extend(sigma[~np.isnan(sigma)])
        if twin_values:
            twin_min = min(twin_values)
            twin_max = max(twin_values)
            twin_margin = (twin_max - twin_min) * 0.1 if twin_max > twin_min else max(abs(twin_min), abs(twin_max)) * 0.1
            twin_bottom = twin_min - twin_margin if twin_min < 0 else 0
            ax_twin.set_ylim(bottom=twin_bottom, top=twin_max + twin_margin)

    add_legend(ax, ax_twin, loc="upper left", outside=True, transparent=True)

    equation = get_loss_eq(trainer)
    if equation:
        add_equation_box(ax, equation, position="bottom_center")

    plt.tight_layout(pad=1.5, rect=[0.20, 0.28, 0.95, 0.93])

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

    fig, ax = plt.subplots(figsize=(8, 6))

    # Prepare data
    source_acc = prepare_data(history, "source_acc", epochs)
    target_acc = prepare_data(history, "target_acc", epochs)
    target_f1 = prepare_data(history, "target_f1", epochs)

    add_warmup_region(ax, warmup_epochs, epochs)

    # Plot accuracies
    if source_acc.size > 0:
        ax.plot(
            epochs,
            source_acc,
            label=r"$\text{Acc}_S$",
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
            label=r"$\text{Acc}_T$",
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
    
    # Add F1 score on twin axis if available
    has_f1 = target_f1.size > 0 and not np.all(np.isnan(target_f1)) and not np.all(target_f1 == 0)
    if has_f1:
        ax_f1 = ax.twinx()
        ax_f1.spines["right"].set_visible(True)
        ax_f1.spines["right"].set_edgecolor("#AAAAAA")
        ax_f1.spines["right"].set_linewidth(1.2)
        ax_f1.plot(
            epochs, target_f1,
            label=r"$\text{F1}_T$",
            color=COLORS["target_f1"],
            linewidth=2.5,
            marker="^",
            markersize=4,
            markevery=max(1, len(epochs) // 15),
            linestyle="--",
        )
        ax_f1.set_ylabel("F1 Score", fontweight="bold")
        ax_f1.set_ylim([0, 1.05])
        add_legend(ax, ax_f1, loc="lower right")
    else:
        add_legend(ax, loc="lower right")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved accuracy plot to {save_path}")

    return fig
