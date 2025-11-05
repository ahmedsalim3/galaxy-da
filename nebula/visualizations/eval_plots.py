from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap.umap_ as umap
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_curve)
from sklearn.preprocessing import label_binarize

from nebula.commons import Logger
from nebula.data.dataset import CLASSES
from nebula.visualizations.utils import COLORS, add_legend, set_plot_style

logger = Logger()
set_plot_style()
CMAP_SOURCE = plt.cm.Blues
CMAP_TARGET = plt.cm.Reds
COLORS_DOMAIN = {"source": "#2e3440", "target": "#bf616a"}
COLORS_CLASS = ["#9b59b6", "#f39c12", "#1abc9c"]  # elliptical, irregular, spiral


def plot_confusion_matrices(
    src_cm,
    tgt_cm,
    src_acc,
    tgt_acc,
    titles=["Source Test Set", "Target Test Set"],
    save_path=None,
) -> plt.Figure:
    """Plot side-by-side confusion matrices for source and target."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    accs = [src_acc, tgt_acc]

    for ax, cm, acc, title_base, cmap in zip(
        axes, [src_cm, tgt_cm], accs, titles, [CMAP_SOURCE, CMAP_TARGET]
    ):
        title = f"{title_base} (Acc: {acc * 100:.2f}%)"
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            ax=ax,
            xticklabels=CLASSES,
            yticklabels=CLASSES,
            cbar_kws={"label": "Count"},
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_xlabel("Predicted Label", fontsize=12)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {save_path.name}")
    return fig


def plot_domain_separation(
    src_z,
    tgt_z,
    method="tsne",
    save_path=None,
    mesh_step=0.15,
    point_size=20,
):
    """
    Plot source and target latent space with a binary LR domain boundary.
    """
    src_z = np.asarray(src_z)
    tgt_z = np.asarray(tgt_z)
    n_src = len(src_z)

    # embed once
    reducer = _get_reducer(method)
    embedded = reducer.fit_transform(np.vstack([src_z, tgt_z]))
    src_emb = embedded[:n_src]
    tgt_emb = embedded[n_src:]

    # domain labels: 0=source, 1=target
    domain_labels = np.hstack(
        [np.zeros(n_src, dtype=int), np.ones(len(tgt_z), dtype=int)]
    )

    # mesh
    pad = 0.8
    x_min, x_max = embedded[:, 0].min() - pad, embedded[:, 0].max() + pad
    y_min, y_max = embedded[:, 1].min() - pad, embedded[:, 1].max() + pad
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_step),
        np.arange(y_min, y_max, mesh_step),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # binary LR on domain
    domain_clf = LogisticRegression(max_iter=1000, solver="liblinear")
    domain_clf.fit(embedded, domain_labels)
    Z_prob = domain_clf.predict_proba(grid)[:, 1].reshape(xx.shape)  # P(target)

    # figure
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 6.5))

    # soft background (source vs target regions)
    ax.contourf(
        xx,
        yy,
        (Z_prob > 0.5).astype(int),
        alpha=0.12,
        cmap=ListedColormap([COLORS_DOMAIN["source"], COLORS_DOMAIN["target"]]),
        levels=[-0.5, 0.5, 1.5],
    )
    # decision contour at 0.5
    ax.contour(
        xx, yy, Z_prob, levels=[0.5], colors="black", linestyles="--", linewidths=1.6
    )

    # scatter
    ax.scatter(
        src_emb[:, 0],
        src_emb[:, 1],
        c=COLORS_DOMAIN["source"],
        s=point_size,
        alpha=0.7,
        marker="o",
        edgecolors="k",
        linewidths=0.35,
        label="Source",
    )
    ax.scatter(
        tgt_emb[:, 0],
        tgt_emb[:, 1],
        c=COLORS_DOMAIN["target"],
        s=point_size,
        alpha=0.7,
        marker="^",
        edgecolors="k",
        linewidths=0.35,
        label="Target",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    ax.set_title("Domain Separation with Boundary (Source vs Target)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_latent_space_two_panel(
    src_z,
    tgt_z,
    src_y,
    tgt_y,
    method="tsne",
    save_path=None,
    use_predictions=False,
    mesh_step=0.15,
):
    """
    Plot side by side source and target latent space with a binary LR domain boundary and class boundaries.
    Two-panel latent space visualization.

    Panel L: Source vs Target by class (clean, no boundaries)
    Panel R: Same scatter + combined boundaries:
             - Domain boundary (binary LR trained on all points)
             - Class boundaries (multinomial LR trained on all points)

    Both panels overlay Source (marker='o') and Target (marker='^').
    """

    src_z = np.asarray(src_z)
    tgt_z = np.asarray(tgt_z)
    src_y = np.asarray(src_y)
    tgt_y = np.asarray(tgt_y)
    combined_z = np.vstack([src_z, tgt_z])
    combined_y = np.hstack([src_y, tgt_y])
    n_src = len(src_z)

    # Domain labels: 0=source, 1=target
    domain_labels = np.hstack(
        [np.zeros(n_src, dtype=int), np.ones(len(tgt_z), dtype=int)]
    )

    reducer = _get_reducer(method)

    embedded = reducer.fit_transform(combined_z)
    src_emb = embedded[:n_src]
    tgt_emb = embedded[n_src:]

    x_min, x_max = embedded[:, 0].min() - 1, embedded[:, 0].max() + 1
    y_min, y_max = embedded[:, 1].min() - 1, embedded[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, mesh_step),
        np.arange(y_min, y_max, mesh_step),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Domain boundary (binary)
    domain_clf = LogisticRegression(max_iter=1000, solver="liblinear")
    domain_clf.fit(embedded, domain_labels)
    Z_domain = domain_clf.predict_proba(grid)[:, 1].reshape(xx.shape)  # prob target

    # Class boundaries (multinomial)
    class_clf = LogisticRegression(
        max_iter=1000, solver="lbfgs", multi_class="multinomial"
    )
    class_clf.fit(embedded, combined_y)
    Z_class = class_clf.predict(grid).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    def _scatter_overlay(ax):
        # source
        for c_idx, (name, color) in enumerate(zip(CLASSES, COLORS_CLASS)):
            m = src_y == c_idx
            if np.any(m):
                ax.scatter(
                    src_emb[m, 0],
                    src_emb[m, 1],
                    c=color,
                    s=18,
                    alpha=0.65,
                    marker="o",
                    edgecolors="k",
                    linewidths=0.3,
                    label=f"{name} (src)",
                )
        # target
        for c_idx, (name, color) in enumerate(zip(CLASSES, COLORS_CLASS)):
            m = tgt_y == c_idx
            if np.any(m):
                ax.scatter(
                    tgt_emb[m, 0],
                    tgt_emb[m, 1],
                    c=color,
                    s=18,
                    alpha=0.65,
                    marker="^",
                    edgecolors="k",
                    linewidths=0.3,
                    label=f"{name} (tgt)",
                )
        ax.grid(True, alpha=0.25)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Embedding 1")
        ax.set_ylabel("Embedding 2")

    # Left panel: (source vs target by class)
    ax = axes[0]
    _scatter_overlay(ax)
    ax.set_title("Source vs Target (by class)")
    # compact legend
    ax.legend(loc="best", frameon=True, fontsize=8, ncol=2)

    # Right panel: same + combined boundaries
    ax = axes[1]

    # Domain boundary (light background; 0=source,1=target)
    # Use a two-color ListedColormap for soft fill; emphasize the 0.5 contour
    ax.contourf(
        xx,
        yy,
        (Z_domain > 0.5).astype(int),
        alpha=0.12,
        cmap=ListedColormap([COLORS_DOMAIN["source"], COLORS_DOMAIN["target"]]),
        levels=[-0.5, 0.5, 1.5],
    )
    ax.contour(
        xx, yy, Z_domain, levels=[0.5], colors="black", linestyles="--", linewidths=1.6
    )

    # Class boundaries (discrete regions)
    ax.contourf(xx, yy, Z_class, alpha=0.10, cmap=ListedColormap(COLORS_CLASS))
    # optional hard class borders (thinner so they don't overpower points)
    try:
        # draw borders between adjacent class labels present
        unique_classes = np.unique(combined_y)
        # levels must be monotonically increasing; build from present classes
        levels = [k + 0.5 for k in unique_classes[:-1]]
        ax.contour(
            xx,
            yy,
            Z_class,
            levels=levels,
            colors="k",
            linestyles=":",
            linewidths=1.0,
            alpha=0.6,
        )
    except Exception:
        pass

    _scatter_overlay(ax)
    ax.set_title("Source vs Target (domain + class)")

    method_name = method.upper()
    title_suffix = "Predictions" if use_predictions else "True Labels"
    fig.suptitle(
        f"Latent Space ({method_name}) — {title_suffix}", fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def _get_reducer(method: str, random_state: int = 42, **kwargs):
    """
    Returns a dimensionality reducer object for 'tsne', 'pca', or 'umap'.
    If UMAP is not available, falls back to t-SNE.
    """
    method = method.lower()
    if method == "tsne":
        return TSNE(
            n_components=2,
            random_state=random_state,
            perplexity=kwargs.get("perplexity", 30),
        )
    if method == "pca":
        return PCA(n_components=2, random_state=random_state)
    if method == "umap":
        try:
            return umap.UMAP(
                n_components=2,
                random_state=random_state,
                n_neighbors=kwargs.get("n_neighbors", 15),
                min_dist=kwargs.get("min_dist", 0.1),
            )
        except Exception as e:
            logger.warning(
                f"UMAP requested but not available ({e}); falling back to t-SNE."
            )
            return TSNE(
                n_components=2,
                random_state=random_state,
                perplexity=kwargs.get("perplexity", 30),
            )
    raise ValueError(f"Unknown method: {method}")


def compute_reliability(y_true, y_proba, n_bins):
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)

    # bins and stats
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(confidences, bins) - 1
    ece = 0.0
    xs, ys, cs = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            bc = confidences[mask].mean()
            ba = accuracies[mask].mean()
            ece += mask.sum() / len(y_true) * abs(ba - bc)
            xs.append(bc)
            ys.append(ba)
            cs.append(mask.sum())
        else:
            xs.append(np.nan)
            ys.append(np.nan)
            cs.append(0)
    return xs, ys, cs, ece


def plot_reliability_diagrams_side_by_side(
    y_true_src,
    y_probas_src,
    y_true_tgt,
    y_probas_tgt,
    n_bins=15,
    titles=("Source Reliability (ECE)", "Target Reliability (ECE)"),
    save_path=None,
):
    """
    Plot source and target reliability diagrams side by side in one figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    for ax, (y_true, y_proba, title), color in zip(
        axes,
        zip([y_true_src, y_true_tgt], [y_probas_src, y_probas_tgt], titles),
        [COLORS["source_acc"], COLORS["target_acc"]],
    ):
        xs, ys, cs, ece = compute_reliability(y_true, y_proba, n_bins)
        ax.plot([0, 1], [0, 1], ls="--", c="#777777", linewidth=1.6, label="Perfect")
        ax.plot(
            xs,
            ys,
            marker="o",
            color=color,
            linewidth=2.2,
            markersize=4,
            label="Empirical",
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence", fontweight="bold")
        ax.set_ylabel("Accuracy", fontweight="bold")
        ax.set_title(f"{title} (ECE={ece:.3f})", pad=10)
        add_legend(ax, loc="lower right")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_roc_pr_curves(
    y_true, y_proba, class_names, title_prefix="Target", save_path=None
):
    """
    Plot macro-averaged ROC and PR curves in a 1x2 panel.
    """
    if y_true.size == 0 or y_proba.size == 0:
        logger.warning("Empty inputs to plot_roc_pr_curves; skipping.")
        return None

    classes = list(range(len(class_names)))
    y_bin = label_binarize(y_true, classes=classes)

    # ROC (macro)
    fprs, tprs, aucs = [], [], []
    for c in classes:
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_proba[:, c])
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auc(fpr, tpr))
    macro_auc = float(np.nanmean(aucs))

    # PR (macro)
    prs, recs, aps = [], [], []
    for c in classes:
        prec, rec, _ = precision_recall_curve(y_bin[:, c], y_proba[:, c])
        prs.append(prec)
        recs.append(rec)
        aps.append(average_precision_score(y_bin[:, c], y_proba[:, c]))
    macro_ap = float(np.nanmean(aps))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ROC plot
    ax = axes[0]
    for c, name in enumerate(class_names):
        fpr, tpr = fprs[c], tprs[c]
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={aucs[c]:.2f})",
            linewidth=2.2,
            color=COLORS_CLASS[c % len(COLORS_CLASS)],
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1.4)
    ax.set_xlabel("FPR", fontweight="bold")
    ax.set_ylabel("TPR", fontweight="bold")
    ax.set_title(f"{title_prefix} ROC (macro AUC={macro_auc:.2f})", pad=12)
    add_legend(ax, loc="lower right")

    ax = axes[1]
    for c, name in enumerate(class_names):
        ax.plot(
            recs[c],
            prs[c],
            label=f"{name} (AP={aps[c]:.2f})",
            linewidth=2.2,
            color=COLORS_CLASS[c % len(COLORS_CLASS)],
        )
    ax.set_xlabel("Recall", fontweight="bold")
    ax.set_ylabel("Precision", fontweight="bold")
    ax.set_title(f"{title_prefix} PR (macro AP={macro_ap:.2f})", pad=12)
    add_legend(ax, loc="lower left")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_per_class_bars(
    src_vals, tgt_vals, class_names, metric_name="Recall", save_path=None
):
    n = len(class_names)
    xs = np.arange(n)
    w = 0.4
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(
        xs - w / 2,
        src_vals,
        w,
        label="Source",
        color=COLORS["source_acc"],
        alpha=0.9,
    )
    ax.bar(
        xs + w / 2,
        tgt_vals,
        w,
        label="Target",
        color=COLORS["target_acc"],
        alpha=0.9,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1)
    ax.set_ylabel(metric_name, fontweight="bold")
    ax.set_title(f"Per-class {metric_name}: Source vs Target", pad=12)
    add_legend(ax, loc="upper right")
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_class_distribution(src_y, tgt_y, class_names, normalize=True, save_path=None):
    n = len(class_names)
    src_counts = np.array([(src_y == i).sum() for i in range(n)], dtype=float)
    tgt_counts = np.array([(tgt_y == i).sum() for i in range(n)], dtype=float)
    if normalize:
        src_vals = src_counts / max(src_counts.sum(), 1.0)
        tgt_vals = tgt_counts / max(tgt_counts.sum(), 1.0)
        ylabel = "Proportion"
    else:
        src_vals, tgt_vals = src_counts, tgt_counts
        ylabel = "Count"
    xs = np.arange(n)
    w = 0.4
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(
        xs - w / 2,
        src_vals,
        w,
        label="Source",
        color=COLORS["source_acc"],
        alpha=0.9,
    )
    ax.bar(
        xs + w / 2,
        tgt_vals,
        w,
        label="Target",
        color=COLORS["target_acc"],
        alpha=0.9,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(class_names)
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title("Class Distribution (Test Sets)", pad=12)
    add_legend(ax, loc="upper right")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig
