from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nebula.commons import Logger
from nebula.visualizations.utils import (COLORS, add_legend, add_warmup_region,
                                         prepare_data, set_plot_style)

logger = Logger()
set_plot_style()


def plot_diag_history(
    diag_history,
    warmup_epochs=0,
    save_path=None,
):
    """Plot diagnostic history from domain adaptation training.

    Args:
        diag_history: Diagnostic history dict from trainer
        warmup_epochs: Number of warmup epochs
        save_path: Base path to save figures (will create multiple files)

    Returns:
        Dictionary of matplotlib figures
    """
    if not diag_history or not any(diag_history.get(k, []) for k in diag_history):
        logger.warning("No diagnostic history to plot")
        return {}

    # Determine epochs from any available metric
    epochs = None
    for key in ["target_acc", "mmd2", "sinkhorn_div", "domain_auc"]:
        data = diag_history.get(key, [])
        if data and len(data) > 0:
            epochs = list(range(1, len(data) + 1))
            break

    if epochs is None:
        logger.warning("Could not determine epochs from diagnostic history")
        return {}

    figures = {}

    # Plot 1: Domain Alignment Metrics
    fig_alignment = _plot_domain_alignment(diag_history, epochs, warmup_epochs)
    if fig_alignment:
        figures["alignment"] = fig_alignment
        if save_path:
            path = Path(save_path).parent / f"{Path(save_path).stem}_alignment.png"
            fig_alignment.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved domain alignment plot to {path}")

    # Plot 2: Target Performance Metrics
    fig_performance = _plot_target_performance(diag_history, epochs, warmup_epochs)
    if fig_performance:
        figures["performance"] = fig_performance
        if save_path:
            path = Path(save_path).parent / f"{Path(save_path).stem}_performance.png"
            fig_performance.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved target performance plot to {path}")

    # Plot 3: Class-wise Metrics
    fig_classwise = _plot_classwise_metrics(diag_history, epochs, warmup_epochs)
    if fig_classwise:
        figures["classwise"] = fig_classwise
        if save_path:
            path = Path(save_path).parent / f"{Path(save_path).stem}_classwise.png"
            fig_classwise.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved class-wise metrics plot to {path}")

    # Plot 4: OT Transport Matrix (if available)
    fig_ot = _plot_ot_transport(diag_history, epochs, warmup_epochs)
    if fig_ot:
        figures["ot_transport"] = fig_ot
        if save_path:
            path = Path(save_path).parent / f"{Path(save_path).stem}_ot_transport.png"
            fig_ot.savefig(path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved OT transport plot to {path}")

    return figures


def _plot_domain_alignment(diag_history, epochs, warmup_epochs=0):
    """Plot domain alignment metrics: MMD², Sinkhorn divergence, Domain AUC, Domain Accuracy."""
    mmd2 = prepare_data(diag_history, "mmd2", epochs)
    sinkhorn = prepare_data(diag_history, "sinkhorn_div", epochs)
    domain_auc = prepare_data(diag_history, "domain_auc", epochs)
    domain_acc = prepare_data(diag_history, "domain_acc", epochs)
    proxy_a = prepare_data(diag_history, "proxy_a_distance", epochs)

    # Check if we have any data
    has_data = any(
        arr.size > 0 and not np.all(np.isnan(arr))
        for arr in [mmd2, sinkhorn, domain_auc, domain_acc, proxy_a]
    )

    if not has_data:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"wspace": 0.25})

    # ========== Left: Distance Metrics ==========
    add_warmup_region(ax1, warmup_epochs, epochs)

    if mmd2.size > 0 and not np.all(np.isnan(mmd2)):
        ax1.plot(
            epochs,
            mmd2,
            label="MMD²",
            color=COLORS["da_loss"],
            linewidth=2.5,
            marker="o",
            markersize=4,
            markevery=max(1, len(epochs) // 10),
        )

    if sinkhorn.size > 0 and not np.all(np.isnan(sinkhorn)):
        ax1.plot(
            epochs,
            sinkhorn,
            label="Sinkhorn Divergence",
            color=COLORS["ce_loss"],
            linewidth=2.5,
            marker="s",
            markersize=4,
            markevery=max(1, len(epochs) // 10),
        )

    if proxy_a.size > 0 and not np.all(np.isnan(proxy_a)):
        ax1.plot(
            epochs,
            proxy_a,
            label="Proxy A-Distance",
            color=COLORS["sigma"],
            linewidth=2.5,
            marker="^",
            markersize=4,
            markevery=max(1, len(epochs) // 10),
        )

    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Distance", fontweight="bold")
    ax1.set_title("Domain Distance Metrics", pad=15)
    ax1.set_ylim(bottom=0)
    add_legend(ax1, loc="upper right")

    # ========== Right: Domain Discrimination ==========
    add_warmup_region(ax2, warmup_epochs, epochs)

    if domain_auc.size > 0 and not np.all(np.isnan(domain_auc)):
        ax2.plot(
            epochs,
            domain_auc,
            label="Domain AUC",
            color=COLORS["total_loss"],
            linewidth=2.5,
            marker="o",
            markersize=4,
            markevery=max(1, len(epochs) // 10),
        )

    if domain_acc.size > 0 and not np.all(np.isnan(domain_acc)):
        ax2.plot(
            epochs,
            domain_acc,
            label="Domain Accuracy",
            color=COLORS["da_loss"],
            linewidth=2.5,
            marker="s",
            markersize=4,
            markevery=max(1, len(epochs) // 10),
        )

    # Add reference line at 0.5 (random discrimination)
    ax2.axhline(
        y=0.5,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="Random (0.5)",
    )

    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Score", fontweight="bold")
    ax2.set_title("Domain Discrimination", pad=15)
    ax2.set_ylim([0, 1.05])
    add_legend(ax2, loc="upper right")

    plt.tight_layout()
    return fig


def _plot_target_performance(diag_history, epochs, warmup_epochs=0):
    """Plot target domain performance metrics."""
    target_acc = prepare_data(diag_history, "target_acc", epochs)
    target_f1 = prepare_data(diag_history, "target_macro_f1", epochs)
    roc_auc_macro = prepare_data(diag_history, "target_roc_auc_macro", epochs)
    roc_auc_weighted = prepare_data(diag_history, "target_roc_auc_weighted", epochs)
    auprc_macro = prepare_data(diag_history, "target_auprc_macro", epochs)
    auprc_weighted = prepare_data(diag_history, "target_auprc_weighted", epochs)

    # Check if we have any data
    has_data = any(
        arr.size > 0 and not np.all(np.isnan(arr))
        for arr in [
            target_acc,
            target_f1,
            roc_auc_macro,
            roc_auc_weighted,
            auprc_macro,
            auprc_weighted,
        ]
    )

    if not has_data:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"wspace": 0.25})

    # ========== Left: Accuracy & F1 ==========
    add_warmup_region(ax1, warmup_epochs, epochs)

    if target_acc.size > 0 and not np.all(np.isnan(target_acc)):
        ax1.plot(
            epochs,
            target_acc,
            label="Target Accuracy",
            color=COLORS["target_acc"],
            linewidth=2.5,
            marker="o",
            markersize=4,
            markevery=max(1, len(epochs) // 10),
        )

    if target_f1.size > 0 and not np.all(np.isnan(target_f1)):
        ax1.plot(
            epochs,
            target_f1,
            label="Target Macro F1",
            color=COLORS["ce_loss"],
            linewidth=2.5,
            marker="s",
            markersize=4,
            markevery=max(1, len(epochs) // 10),
        )

    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Score (%)", fontweight="bold")
    ax1.set_title("Target Domain: Accuracy & F1", pad=15)
    ax1.set_ylim([0, 105])
    add_legend(ax1, loc="lower right")

    # ========== Right: ROC-AUC & AUPRC ==========
    add_warmup_region(ax2, warmup_epochs, epochs)

    if roc_auc_macro.size > 0 and not np.all(np.isnan(roc_auc_macro)):
        ax2.plot(
            epochs,
            roc_auc_macro,
            label="ROC-AUC (Macro)",
            color=COLORS["eta1"],
            linewidth=2.5,
            marker="o",
            markersize=4,
            markevery=max(1, len(epochs) // 10),
        )

    if roc_auc_weighted.size > 0 and not np.all(np.isnan(roc_auc_weighted)):
        ax2.plot(
            epochs,
            roc_auc_weighted,
            label="ROC-AUC (Weighted)",
            color=COLORS["eta2"],
            linewidth=2.0,
            linestyle="--",
            marker="s",
            markersize=4,
            markevery=max(1, len(epochs) // 10),
        )

    if auprc_macro.size > 0 and not np.all(np.isnan(auprc_macro)):
        ax2.plot(
            epochs,
            auprc_macro,
            label="AUPRC (Macro)",
            color=COLORS["sigma"],
            linewidth=2.5,
            marker="^",
            markersize=4,
            markevery=max(1, len(epochs) // 10),
        )

    if auprc_weighted.size > 0 and not np.all(np.isnan(auprc_weighted)):
        ax2.plot(
            epochs,
            auprc_weighted,
            label="AUPRC (Weighted)",
            color=COLORS["da_loss"],
            linewidth=2.0,
            linestyle="--",
            marker="v",
            markersize=4,
            markevery=max(1, len(epochs) // 10),
        )

    ax2.set_xlabel("Epoch", fontweight="bold")
    ax2.set_ylabel("Score", fontweight="bold")
    ax2.set_title("Target Domain: ROC-AUC & AUPRC", pad=15)
    ax2.set_ylim([0, 1.05])
    add_legend(ax2, loc="lower right")

    plt.tight_layout()
    return fig


def _plot_classwise_metrics(diag_history, epochs, warmup_epochs=0):
    """Plot class-wise metrics: Recall, CMMD, Domain AUC."""
    recall_ell = prepare_data(diag_history, "recall_elliptical", epochs)
    recall_irr = prepare_data(diag_history, "recall_irregular", epochs)
    recall_spi = prepare_data(diag_history, "recall_spiral", epochs)

    cmmd_ell = prepare_data(diag_history, "cmmd_elliptical", epochs)
    cmmd_irr = prepare_data(diag_history, "cmmd_irregular", epochs)
    cmmd_spi = prepare_data(diag_history, "cmmd_spiral", epochs)

    dauc_ell = prepare_data(diag_history, "domain_auc_elliptical", epochs)
    dauc_irr = prepare_data(diag_history, "domain_auc_irregular", epochs)
    dauc_spi = prepare_data(diag_history, "domain_auc_spiral", epochs)

    # Check if we have any data
    has_recall = any(
        arr.size > 0 and not np.all(np.isnan(arr))
        for arr in [recall_ell, recall_irr, recall_spi]
    )
    has_cmmd = any(
        arr.size > 0 and not np.all(np.isnan(arr))
        for arr in [cmmd_ell, cmmd_irr, cmmd_spi]
    )
    has_dauc = any(
        arr.size > 0 and not np.all(np.isnan(arr))
        for arr in [dauc_ell, dauc_irr, dauc_spi]
    )

    if not (has_recall or has_cmmd or has_dauc):
        return None

    # Determine number of subplots needed
    n_plots = sum([has_recall, has_cmmd, has_dauc])
    if n_plots == 0:
        return None

    fig, axes = plt.subplots(
        1, n_plots, figsize=(6 * n_plots, 6), gridspec_kw={"wspace": 0.25}
    )
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Class colors
    class_colors = {
        "elliptical": COLORS["ce_loss"],
        "irregular": COLORS["da_loss"],
        "spiral": COLORS["eta1"],
    }

    # ========== Recall per Class ==========
    if has_recall:
        ax = axes[plot_idx]
        add_warmup_region(ax, warmup_epochs, epochs)

        if recall_ell.size > 0 and not np.all(np.isnan(recall_ell)):
            ax.plot(
                epochs,
                recall_ell,
                label="Elliptical",
                color=class_colors["elliptical"],
                linewidth=2.5,
                marker="o",
                markersize=4,
                markevery=max(1, len(epochs) // 10),
            )

        if recall_irr.size > 0 and not np.all(np.isnan(recall_irr)):
            ax.plot(
                epochs,
                recall_irr,
                label="Irregular",
                color=class_colors["irregular"],
                linewidth=2.5,
                marker="s",
                markersize=4,
                markevery=max(1, len(epochs) // 10),
            )

        if recall_spi.size > 0 and not np.all(np.isnan(recall_spi)):
            ax.plot(
                epochs,
                recall_spi,
                label="Spiral",
                color=class_colors["spiral"],
                linewidth=2.5,
                marker="^",
                markersize=4,
                markevery=max(1, len(epochs) // 10),
            )

        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Recall (%)", fontweight="bold")
        ax.set_title("Class-wise Recall", pad=15)
        ax.set_ylim([0, 105])
        add_legend(ax, loc="lower right")
        plot_idx += 1

    # ========== CMMD per Class ==========
    if has_cmmd:
        ax = axes[plot_idx]
        add_warmup_region(ax, warmup_epochs, epochs)

        if cmmd_ell.size > 0 and not np.all(np.isnan(cmmd_ell)):
            ax.plot(
                epochs,
                cmmd_ell,
                label="Elliptical",
                color=class_colors["elliptical"],
                linewidth=2.5,
                marker="o",
                markersize=4,
                markevery=max(1, len(epochs) // 10),
            )

        if cmmd_irr.size > 0 and not np.all(np.isnan(cmmd_irr)):
            ax.plot(
                epochs,
                cmmd_irr,
                label="Irregular",
                color=class_colors["irregular"],
                linewidth=2.5,
                marker="s",
                markersize=4,
                markevery=max(1, len(epochs) // 10),
            )

        if cmmd_spi.size > 0 and not np.all(np.isnan(cmmd_spi)):
            ax.plot(
                epochs,
                cmmd_spi,
                label="Spiral",
                color=class_colors["spiral"],
                linewidth=2.5,
                marker="^",
                markersize=4,
                markevery=max(1, len(epochs) // 10),
            )

        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("CMMD", fontweight="bold")
        ax.set_title("Conditional MMD (per class)", pad=15)
        ax.set_ylim(bottom=0)
        add_legend(ax, loc="upper right")
        plot_idx += 1

    # ========== Domain AUC per Class ==========
    if has_dauc:
        ax = axes[plot_idx]
        add_warmup_region(ax, warmup_epochs, epochs)

        if dauc_ell.size > 0 and not np.all(np.isnan(dauc_ell)):
            ax.plot(
                epochs,
                dauc_ell,
                label="Elliptical",
                color=class_colors["elliptical"],
                linewidth=2.5,
                marker="o",
                markersize=4,
                markevery=max(1, len(epochs) // 10),
            )

        if dauc_irr.size > 0 and not np.all(np.isnan(dauc_irr)):
            ax.plot(
                epochs,
                dauc_irr,
                label="Irregular",
                color=class_colors["irregular"],
                linewidth=2.5,
                marker="s",
                markersize=4,
                markevery=max(1, len(epochs) // 10),
            )

        if dauc_spi.size > 0 and not np.all(np.isnan(dauc_spi)):
            ax.plot(
                epochs,
                dauc_spi,
                label="Spiral",
                color=class_colors["spiral"],
                linewidth=2.5,
                marker="^",
                markersize=4,
                markevery=max(1, len(epochs) // 10),
            )

        # Add reference line
        ax.axhline(
            y=0.5,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
            label="Random (0.5)",
        )

        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Domain AUC", fontweight="bold")
        ax.set_title("Domain AUC (per class)", pad=15)
        ax.set_ylim([0, 1.05])
        add_legend(ax, loc="upper right")

    plt.tight_layout()
    return fig


def _plot_ot_transport(diag_history, epochs, warmup_epochs=0):
    """Plot OT transport matrix showing mass flow between classes."""
    # OT mass keys
    ot_keys = [
        "ot_mass_elliptical_to_elliptical",
        "ot_mass_elliptical_to_irregular",
        "ot_mass_elliptical_to_spiral",
        "ot_mass_irregular_to_elliptical",
        "ot_mass_irregular_to_irregular",
        "ot_mass_irregular_to_spiral",
        "ot_mass_spiral_to_elliptical",
        "ot_mass_spiral_to_irregular",
        "ot_mass_spiral_to_spiral",
    ]

    # Check if we have OT data
    ot_data = {k: prepare_data(diag_history, k, epochs) for k in ot_keys}
    has_ot = any(arr.size > 0 and not np.all(np.isnan(arr)) for arr in ot_data.values())

    ot_on_diag = prepare_data(diag_history, "ot_on_diag", epochs)

    if not has_ot and (ot_on_diag.size == 0 or np.all(np.isnan(ot_on_diag))):
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={"wspace": 0.3})

    # ========== Left: OT On-Diagonal Mass ==========
    ax1 = axes[0]
    add_warmup_region(ax1, warmup_epochs, epochs)

    if ot_on_diag.size > 0 and not np.all(np.isnan(ot_on_diag)):
        ax1.plot(
            epochs,
            ot_on_diag,
            label="On-Diagonal Mass",
            color=COLORS["ce_loss"],
            linewidth=2.5,
            marker="o",
            markersize=4,
            markevery=max(1, len(epochs) // 10),
        )

    ax1.axhline(
        y=1 / 3,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="Random (1/3)",
    )

    ax1.set_xlabel("Epoch", fontweight="bold")
    ax1.set_ylabel("Mass Fraction", fontweight="bold")
    ax1.set_title("OT On-Diagonal Mass", pad=15)
    ax1.set_ylim([0, 1.05])
    add_legend(ax1, loc="lower right")

    # ========== Right: Transport Matrix Heatmap (last epoch) ==========
    ax2 = axes[1]

    if has_ot:
        # Get last epoch values
        transport_matrix = np.zeros((3, 3))
        classes = ["Elliptical", "Irregular", "Spiral"]

        for i, source in enumerate(["elliptical", "irregular", "spiral"]):
            for j, target in enumerate(["elliptical", "irregular", "spiral"]):
                key = f"ot_mass_{source}_to_{target}"
                data = ot_data[key]
                if data.size > 0 and not np.isnan(data[-1]):
                    transport_matrix[i, j] = data[-1]

        # Create heatmap
        im = ax2.imshow(transport_matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar.set_label("Transport Mass", fontweight="bold")

        # Set ticks
        ax2.set_xticks(np.arange(3))
        ax2.set_yticks(np.arange(3))
        ax2.set_xticklabels(classes)
        ax2.set_yticklabels(classes)

        # Add text annotations
        for i in range(3):
            for j in range(3):
                text = ax2.text(
                    j,
                    i,
                    f"{transport_matrix[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if transport_matrix[i, j] > 0.5 else "black",
                    fontsize=12,
                    fontweight="bold",
                )

        ax2.set_xlabel("Target Class", fontweight="bold")
        ax2.set_ylabel("Source Class", fontweight="bold")
        ax2.set_title(f"OT Transport Matrix (Epoch {epochs[-1]})", pad=15)
    else:
        ax2.text(
            0.5,
            0.5,
            "No OT transport data available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=14,
            color="gray",
        )
        ax2.set_xticks([])
        ax2.set_yticks([])

    plt.tight_layout()
    return fig
