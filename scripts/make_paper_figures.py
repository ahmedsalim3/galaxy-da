import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

from nebula.commons import Logger
from nebula.data.dataset import CLASSES

os.environ["LOG_LEVEL"] = str(10)
logger = Logger()

METHOD_CONFIG = {
    "baseline": ("Baseline", "#7A7A7A"),
    "adversarial": ("DANN", "#E24A33"),
    "euclidean_fixed_lambda": (r"Euclidean (Fixed $\lambda$)", "#348ABD"),
    "euclidean_trainable_weights": (r"Euclidean ($\eta_1,\eta_2$)", "#988ED5"),
    "euclidean_trainable_weights_sigma_scheduler": (
        r"Euclidean (scheduler $\sigma$ + $\eta_1 \eta_2$)",
        "#FBC15E",
    ),
}

COLORS_DOMAIN = {
    "source": "#2e3440",
    "target": "#bf616a",
}

COLORS_CLASS = ["#9b59b6", "#f39c12", "#1abc9c"]

COLORS_METRICS = {
    "domain_auc": "#E24A33",
    "target_acc": "#348ABD",
    "target_f1": "#988ED5",
}

FONT_SIZE = 18
FONT_SIZE_LABEL = 20
FONT_SIZE_LEGEND = 18
FONT_SIZE_TICK = 16
FONT_SIZE_TITLE = 20
FONT_SIZE_ANNOT = 14

LINE_WIDTH = 3.5
LINE_WIDTH_BOUNDARY = 1.2
EDGE_WIDTH = 1.0

ALPHA_GRID = 0.4
ALPHA_SCATTER = 0.8
ALPHA_CONTOUR = 0.10

MARKER_SIZE = 10
SCATTER_SIZE = 20

LEGEND_FRAME_EDGE = "#888888"
LEGEND_FRAME_WIDTH = 1.2
LEGEND_FRAME_ALPHA = 0.9


def set_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 2.0,
            "lines.linewidth": LINE_WIDTH,
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE_LABEL,
            "legend.fontsize": FONT_SIZE_LEGEND,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "axes.grid": True,
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "grid.alpha": ALPHA_GRID,
            "xtick.major.width": 1.6,
            "ytick.major.width": 1.6,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
        }
    )
    plt.style.use("default")


def _load_data(exp_dir, full_z=True):
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        logger.error(f"Directory '{exp_dir}' not found.")
        sys.exit(1)
    logger.info(f"Loading {len(METHOD_CONFIG)} experiments from {exp_path}")
    experiments: dict[str, dict] = {}
    for folder, (display_name, color) in METHOD_CONFIG.items():
        method_path = exp_path / folder
        if not method_path.exists():
            logger.warning(f"  Method folder not found: {method_path}")
            continue
        logs_dir = method_path / "logs"
        eval_dir = method_path / "eval"
        data_dir = eval_dir / "data"
        exp_data: dict = {
            "key": folder,
            "display_name": display_name,
            "color": color,
            "history": None,
            "src_metrics": {},
            "tgt_metrics": {},
            "domain_metrics": {},
        }
        csv_files = list(logs_dir.glob("*_history.csv"))
        if not csv_files:
            logger.warning(f"  No history csv found for {method_path}")
        else:
            csv_path = csv_files[0]
            try:
                df = pd.read_csv(csv_path)
                if (
                    "epoch_warmup" in df.columns
                    and df["epoch_warmup"].dtype == "object"
                ):
                    df["epoch_warmup"] = df["epoch_warmup"].map(
                        {"True": True, "False": False}
                    )
                exp_data["history"] = df
                logger.debug(
                    f"  [Loaded] {display_name}: {len(df)} epochs from {csv_path.name}"
                )
            except Exception as e:
                logger.error(f"  reading history for {folder}: {e}")
        for metric_name in ["src_metrics", "tgt_metrics", "domain_metrics"]:
            metric_path = eval_dir / f"{metric_name}.json"
            if metric_path.exists():
                try:
                    with open(metric_path, "r") as f:
                        exp_data[metric_name] = json.load(f)
                except Exception as e:
                    logger.error(f"  reading {metric_name} for {folder}: {e}")
            else:
                logger.warning(f"  {metric_name}.json not found for {method_path}")
        if data_dir.exists():
            latent_keys = [
                "src_full_z",
                "tgt_full_z",
                "src_full_y",
                "tgt_full_y",
                "src_test_z",
                "tgt_test_z",
                "src_test_y",
                "tgt_test_y",
            ]
            for key in latent_keys:
                if (not full_z) and key.startswith(("src_full", "tgt_full")):
                    continue
                path = data_dir / f"{key}.npy"
                if path.exists():
                    try:
                        exp_data[key] = np.load(path)
                    except Exception as e:
                        logger.error(f"  loading {key} for {folder}: {e}")
                else:
                    if "test" in key:
                        logger.warning(f"  {key}.npy not found for {method_path}")
        experiments[display_name] = exp_data
    return experiments


def _plot_figure1(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "figure1_examples.pdf"
    root = Path(__file__).resolve().parent.parent
    src_img_dir = root / "data" / "source" / "galaxy_images_rgb"
    src_csv = root / "data" / "source" / "source_galaxy_labels.csv"
    tgt_img_dir = root / "data" / "target" / "gz2_images"
    tgt_csv = root / "data" / "target" / "gz2_galaxy_labels.csv"
    try:
        src_df = pd.read_csv(src_csv)
        tgt_df = pd.read_csv(tgt_csv)
    except Exception as e:
        logger.error(f"Failed to load label CSVs for figure 1: {e}")
        return
    src_df["classification"] = src_df["classification"].str.strip().str.lower()
    tgt_df["classification"] = tgt_df["classification"].str.strip().str.lower()
    titles = {
        "elliptical": "Elliptical",
        "irregular": "Irregular",
        "spiral": "Spiral",
    }
    n_classes = len(CLASSES)
    fig, axes = plt.subplots(2, n_classes, figsize=(4 * n_classes, 6))
    for j, cls in enumerate(CLASSES):
        ax_src = axes[0, j]
        row_src = src_df[src_df["classification"] == cls].head(1)
        if not row_src.empty:
            img_name = row_src.iloc[0]["image_path"]
            img_path = src_img_dir / img_name
            if img_path.exists():
                img = plt.imread(img_path)
                ax_src.imshow(img)
        ax_src.set_xticks([])
        ax_src.set_yticks([])
        if j == 0:
            ax_src.set_ylabel("Source", fontsize=FONT_SIZE_LABEL)
        ax_src.set_title(titles[cls], fontsize=FONT_SIZE_TITLE)
        ax_tgt = axes[1, j]
        row_tgt = tgt_df[tgt_df["classification"] == cls].head(1)
        if not row_tgt.empty:
            img_name = row_tgt.iloc[0]["subhalo_id"]
            img_path = tgt_img_dir / img_name
            if img_path.exists():
                img = plt.imread(img_path)
                ax_tgt.imshow(img)
        ax_tgt.set_xticks([])
        ax_tgt.set_yticks([])
        if j == 0:
            ax_tgt.set_ylabel("Target", fontsize=FONT_SIZE_LABEL)
    plt.tight_layout()
    logger.info(f"Saving figure 1 (examples) to {fig_path}")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_figure2(experiments: dict, output_path: str):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / "figure2_training.pdf"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.25)
    ax_metrics, ax_loss = axes
    for display_name, exp in experiments.items():
        df = exp["history"]
        if df is None:
            continue
        if "epoch_warmup" in df.columns:
            plot_df = df[df["epoch_warmup"] == False]
        else:
            plot_df = df
        if plot_df.empty:
            continue
        color = exp["color"]
        (line_acc,) = ax_metrics.plot(
            plot_df["epoch"],
            plot_df["target_acc"],
            color=color,
            linestyle="-",
            linewidth=LINE_WIDTH,
            alpha=0.9,
            label=display_name,
        )
        f1_scaled = plot_df["target_f1"] * 100
        ax_metrics.plot(
            plot_df["epoch"],
            f1_scaled,
            color=color,
            linestyle="--",
            linewidth=LINE_WIDTH,
            alpha=0.9,
            label="_nolegend_",
        )
        ax_loss.plot(
            plot_df["epoch"],
            plot_df["train_loss"],
            color=color,
            linestyle="-",
            linewidth=LINE_WIDTH,
            alpha=0.9,
            label=display_name,
        )
    ax_metrics.set_xlabel("Epoch", fontsize=FONT_SIZE_LABEL)
    ax_metrics.set_ylabel("Score (%)", fontsize=FONT_SIZE_LABEL)
    ax_metrics.yaxis.label.set_fontweight("medium")
    ax_metrics.grid(True, linestyle="--", alpha=ALPHA_GRID)
    style_lines = [
        Line2D(
            [0],
            [0],
            color="#2c3e50",
            linewidth=LINE_WIDTH,
            linestyle="-",
            label="Accuracy",
        ),
        Line2D(
            [0],
            [0],
            color="#2c3e50",
            linewidth=LINE_WIDTH,
            linestyle="--",
            label="F1 Score",
        ),
    ]
    style_legend = ax_metrics.legend(
        handles=style_lines,
        loc="lower right",
        title="Metric",
        frameon=True,
        fontsize=FONT_SIZE_LEGEND,
        title_fontsize=FONT_SIZE_TITLE,
    )
    ax_metrics.add_artist(style_legend)
    ax_loss.set_xlabel("Epoch", fontsize=FONT_SIZE_LABEL)
    ax_loss.set_ylabel(r"Total Loss ($\mathcal{L}$)", fontsize=FONT_SIZE_LABEL)
    ax_loss.yaxis.label.set_fontweight("medium")
    ax_loss.grid(True, linestyle="--", alpha=ALPHA_GRID)
    ax_loss.legend(
        loc="upper right",
        frameon=True,
        fontsize=FONT_SIZE_LEGEND,
        title_fontsize=FONT_SIZE_TITLE,
    )
    for leg in (
        fig.legends + list(ax_metrics.get_children()) + list(ax_loss.get_children())
    ):
        if isinstance(leg, Legend):
            leg.get_frame().set_edgecolor(LEGEND_FRAME_EDGE)
            leg.get_frame().set_linewidth(LEGEND_FRAME_WIDTH)
            leg.get_frame().set_alpha(LEGEND_FRAME_ALPHA)
    fig.set_constrained_layout(True)
    logger.info(f"Saving figure 2 (training) to {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _select_best_methods(
    experiments: dict, top_k: int = 2, by_domain_auc: bool = False
):
    baseline = experiments["Baseline"]
    others = []
    for name, exp in experiments.items():
        if name == "Baseline":
            continue
        tgt = exp["tgt_metrics"]
        dom_metrics = exp["domain_metrics"]
        if by_domain_auc:
            dom_test = dom_metrics["test"]["domain_auc"]
            dom_full = dom_metrics["full"]["domain_auc"]
            dom_avg = (dom_test + dom_full) / 2.0
            acc = tgt["tgt_acc"]
            others.append((name, dom_avg, acc))
        else:
            acc = tgt["tgt_acc"]
            dom = dom_metrics["test"]["domain_auc"]
            others.append((name, acc, dom))
    if by_domain_auc:
        others.sort(key=lambda t: (t[1], -t[2]))
    else:
        others.sort(key=lambda t: (-t[1], t[2]))
    best = [baseline] if baseline is not None else []
    best_names = ["Baseline"] if baseline is not None else []
    for name, _, _ in others[:top_k]:
        best.append(experiments[name])
        best_names.append(name)
    return best_names, best


def _plot_figure3(experiments: dict, output_path: str):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "figure3_latent_space.pdf"
    method_names, method_exps = _select_best_methods(
        experiments, top_k=2, by_domain_auc=True
    )
    if not method_exps:
        logger.warning("No experiments with latent features available for figure 3.")
        return
    n_cols = len(method_exps)
    legend_handles = []
    legend_labels = []
    for c_idx, (cls_name, color) in enumerate(zip(CLASSES, COLORS_CLASS)):
        h_src = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=color,
            markersize=MARKER_SIZE,
            markeredgecolor="k",
            markeredgewidth=0.5,
            linestyle="None",
        )
        legend_handles.append(h_src)
        legend_labels.append(f"{cls_name.capitalize()} (src)")
        h_tgt = plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor=color,
            markersize=MARKER_SIZE,
            markeredgecolor="k",
            markeredgewidth=0.5,
            linestyle="None",
        )
        legend_handles.append(h_tgt)
        legend_labels.append(f"{cls_name.capitalize()} (tgt)")

    def _plot_panel(ax, src_z, tgt_z, src_y, tgt_y, title=None, domain_auc=None):
        if src_z is None or tgt_z is None or src_y is None or tgt_y is None:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", fontsize=FONT_SIZE_LABEL
            )
            ax.set_xticks([])
            ax.set_yticks([])
            title_str = title
            if domain_auc is not None:
                if title_str is not None:
                    title_str += f"\nDomain AUC: {domain_auc:.3f}"
                else:
                    title_str = f"Domain AUC: {domain_auc:.3f}"
            ax.set_title(title_str, fontsize=FONT_SIZE_TITLE)
            return
        max_points = 4000
        src_idx = np.arange(len(src_z))
        tgt_idx = np.arange(len(tgt_z))
        rng = np.random.default_rng(42)
        if len(src_idx) > max_points // 2:
            src_idx = rng.choice(src_idx, size=max_points // 2, replace=False)
        if len(tgt_idx) > max_points // 2:
            tgt_idx = rng.choice(tgt_idx, size=max_points // 2, replace=False)
        src_z_sub = src_z[src_idx]
        tgt_z_sub = tgt_z[tgt_idx]
        src_y_sub = src_y[src_idx]
        tgt_y_sub = tgt_y[tgt_idx]
        combined_z = np.vstack([src_z_sub, tgt_z_sub])
        try:
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            embedded = reducer.fit_transform(combined_z)
        except Exception as e:
            logger.error(f"TSNE failed in figure 3: {e}")
            ax.text(
                0.5,
                0.5,
                "TSNE failed",
                ha="center",
                va="center",
                fontsize=FONT_SIZE_LABEL,
            )
            return
        
        n_src = len(src_z_sub)
        src_emb = embedded[:n_src]
        tgt_emb = embedded[n_src:]
        try:
            domain_labels = np.hstack(
                [np.zeros(len(src_emb), dtype=int), np.ones(len(tgt_emb), dtype=int)]
            )
            clf = LogisticRegression(max_iter=1000, solver="liblinear")
            clf.fit(embedded, domain_labels)
            x_min, x_max = embedded[:, 0].min() - 1, embedded[:, 0].max() + 1
            y_min, y_max = embedded[:, 1].min() - 1, embedded[:, 1].max() + 1
            xx, yy = np.meshgrid(
                np.arange(x_min, x_max, 0.25),
                np.arange(y_min, y_max, 0.25),
            )
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
            ax.contourf(
                xx,
                yy,
                (Z > 0.5).astype(int),
                alpha=ALPHA_CONTOUR,
                cmap=ListedColormap([COLORS_DOMAIN["source"], COLORS_DOMAIN["target"]]),
                levels=[-0.5, 0.5, 1.5],
            )
            ax.contour(
                xx,
                yy,
                Z,
                levels=[0.5],
                colors="black",
                linestyles="--",
                linewidths=LINE_WIDTH_BOUNDARY,
            )
        except Exception:
            pass
        for c_idx, (cls_name, color) in enumerate(zip(CLASSES, COLORS_CLASS)):
            mask_src = src_y_sub == c_idx
            mask_tgt = tgt_y_sub == c_idx
            if np.any(mask_src):
                ax.scatter(
                    src_emb[mask_src, 0],
                    src_emb[mask_src, 1],
                    c=color,
                    s=SCATTER_SIZE,
                    alpha=ALPHA_SCATTER,
                    marker="o",
                    edgecolors="k",
                    linewidths=0.3,
                )
            if np.any(mask_tgt):
                ax.scatter(
                    tgt_emb[mask_tgt, 0],
                    tgt_emb[mask_tgt, 1],
                    c=color,
                    s=SCATTER_SIZE,
                    alpha=ALPHA_SCATTER,
                    marker="^",
                    edgecolors="k",
                    linewidths=0.3,
                )
        ax.set_xticks([])
        ax.set_yticks([])
        title_str = title
        if domain_auc is not None:
            if title_str is not None:
                title_str += f"\nDomain AUC: {domain_auc:.3f}"
            else:
                title_str = f"Domain AUC: {domain_auc:.3f}"
        ax.set_title(title_str, fontsize=FONT_SIZE_TITLE)
        ax.grid(True, alpha=ALPHA_GRID)

    fig, axes = plt.subplots(2, n_cols, figsize=(5.0 * n_cols, 7.0))
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for col_idx, exp in enumerate(method_exps):
        name = exp["display_name"]
        domain_metrics = exp["domain_metrics"]
        domain_auc_full = domain_metrics["full"]["domain_auc"]
        ax_full = axes[0, col_idx]
        _plot_panel(
            ax_full,
            exp["src_full_z"],
            exp["tgt_full_z"],
            exp["src_full_y"],
            exp["tgt_full_y"],
            title=name,
            domain_auc=domain_auc_full,
        )
        domain_auc_test = domain_metrics["test"]["domain_auc"]
        ax_test = axes[1, col_idx]
        _plot_panel(
            ax_test,
            exp["src_test_z"],
            exp["tgt_test_z"],
            exp["src_test_y"],
            exp["tgt_test_y"],
            title=None,
            domain_auc=domain_auc_test,
        )
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        frameon=True,
        fontsize=FONT_SIZE_LEGEND,
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, left=0.08)
    for row_idx in range(2):
        bbox = axes[row_idx, 0].get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        x_center = bbox.x0 - 0.02
        label = "Full Data" if row_idx == 0 else "Test Data"
        t = fig.text(
            x_center,
            y_center,
            label,
            fontsize=FONT_SIZE_LABEL,
            rotation=90,
            ha="center",
            va="center",
            transform=fig.transFigure,
            zorder=100,
        )
        logger.info(f"Added label '{label}' at ({x_center:.3f}, {y_center:.3f})")
    logger.info(f"Saving figure 3 (latent space) to {fig_path}")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    plt.savefig(
        fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.1
    )
    plt.close(fig)
    fig_full, axes_full = plt.subplots(1, n_cols, figsize=(5.0 * n_cols, 4.0))
    if n_cols == 1:
        axes_full = np.array([axes_full])
    for col_idx, exp in enumerate(method_exps):
        name = exp["display_name"]
        domain_metrics = exp["domain_metrics"]
        domain_auc_full = domain_metrics["full"]["domain_auc"]
        ax = axes_full[col_idx]
        _plot_panel(
            ax,
            exp["src_full_z"],
            exp["tgt_full_z"],
            exp["src_full_y"],
            exp["tgt_full_y"],
            title=name,
            domain_auc=domain_auc_full,
        )
    fig_full.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        frameon=True,
        fontsize=FONT_SIZE_LEGEND,
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    fig_path_full = output_path / "figure3_latent_space_full.pdf"
    plt.savefig(fig_path_full, dpi=300, bbox_inches="tight")
    plt.savefig(fig_path_full.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig_full)
    logger.info(f"Saving figure 3 (full only) to {fig_path_full}")
    fig_test, axes_test = plt.subplots(1, n_cols, figsize=(5.0 * n_cols, 4.0))
    if n_cols == 1:
        axes_test = np.array([axes_test])
    for col_idx, exp in enumerate(method_exps):
        name = exp["display_name"]
        domain_metrics = exp["domain_metrics"]
        domain_auc_test = domain_metrics["test"]["domain_auc"]
        ax = axes_test[col_idx]
        _plot_panel(
            ax,
            exp["src_test_z"],
            exp["tgt_test_z"],
            exp["src_test_y"],
            exp["tgt_test_y"],
            title=name,
            domain_auc=domain_auc_test,
        )
    fig_test.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        frameon=True,
        fontsize=FONT_SIZE_LEGEND,
        ncol=3,
        bbox_to_anchor=(0.5, 0.0),
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    fig_path_test = output_path / "figure3_latent_space_test.pdf"
    plt.savefig(fig_path_test, dpi=300, bbox_inches="tight")
    plt.savefig(fig_path_test.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig_test)
    logger.info(f"Saving figure 3 (test only) to {fig_path_test}")


def _plot_figure4(experiments: dict, output_path: str):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    cm_metrics_fig_path = output_path / "figure4_cm_and_metrics.pdf"
    cm_fig_path = output_path / "figure4_cm.pdf"
    baseline = experiments["Baseline"]
    if baseline is None:
        logger.warning("Baseline experiment not found for figure 4.")
        return
    _, best_list = _select_best_methods(experiments, top_k=1)
    best = best_list[1] if len(best_list) > 1 else None
    if best is None:
        logger.warning("Best DA experiment not found for figure 4.")
        return

    methods, domain_aucs, tgt_accs, tgt_f1s = [], [], [], []
    for name, exp in experiments.items():
        tgt = exp["tgt_metrics"]
        dom = exp["domain_metrics"]["test"]
        if not tgt:
            continue
        methods.append(name)
        domain_aucs.append(dom["domain_auc"])
        tgt_accs.append(tgt["tgt_acc"])
        tgt_f1s.append(tgt["tgt_macro_f1"])

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 0.12, 1], hspace=0.12, wspace=0.25)
    ax_cm_base = fig.add_subplot(gs[0, 0])
    ax_cm_best = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[2, :])

    def _plot_cm(ax, exp, title, cmap):
        cm = np.array(exp["tgt_metrics"]["tgt_cm"])
        acc = exp["tgt_metrics"]["tgt_acc"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            ax=ax,
            xticklabels=[c.capitalize() for c in CLASSES],
            yticklabels=[c.capitalize() for c in CLASSES],
            # cbar_kws={"label": "Count"},
            annot_kws={"fontsize": FONT_SIZE_ANNOT},
            cbar=False,
        )
        ax.set_title(f"{title} (Acc: {acc * 100:.1f}%)", fontsize=FONT_SIZE_TITLE)
        ax.tick_params(labelsize=FONT_SIZE_ANNOT)

    _plot_cm(ax_cm_base, baseline, "Before DA", plt.cm.Blues)
    _plot_cm(ax_cm_best, best, "After DA", plt.cm.Reds)

    fig_cm_only = plt.figure(figsize=(12, 5))
    gs_cm = fig_cm_only.add_gridspec(1, 2, wspace=0.25)
    ax_cm_base_only = fig_cm_only.add_subplot(gs_cm[0, 0])
    ax_cm_best_only = fig_cm_only.add_subplot(gs_cm[0, 1])
    _plot_cm(ax_cm_base_only, baseline, "Before DA", plt.cm.Blues)
    _plot_cm(ax_cm_best_only, best, "After DA", plt.cm.Reds)
    plt.savefig(cm_fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(cm_fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig_cm_only)

    short = []
    for n in methods:
        if n == "Baseline":
            short.append("Baseline")
        elif n == "DANN":
            short.append("DANN")
        elif "Fixed" in n:
            short.append("Euclidean\n(Fixed λ)")
        elif "$η_1,η_2$" in n or "$\\eta_1,\\eta_2$" in n:
            short.append("Euclidean\n(η₁, η₂)")
        elif "scheduler" in n:
            short.append("Euclidean\n(Scheduler σ + η₁η₂)")
        else:
            short.append(n)

    x = np.arange(len(methods))
    w = 0.27
    bars1 = ax_bar.bar(
        x - w,
        domain_aucs,
        w,
        color=COLORS_METRICS["domain_auc"],
        edgecolor="black",
        linewidth=EDGE_WIDTH,
    )
    bars2 = ax_bar.bar(
        x,
        tgt_accs,
        w,
        color=COLORS_METRICS["target_acc"],
        edgecolor="black",
        linewidth=EDGE_WIDTH,
    )
    bars3 = ax_bar.bar(
        x + w,
        tgt_f1s,
        w,
        color=COLORS_METRICS["target_f1"],
        edgecolor="black",
        linewidth=EDGE_WIDTH,
    )

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(short, fontsize=FONT_SIZE_TICK - 4)
    ax_bar.set_ylabel("Score", fontsize=FONT_SIZE_LABEL - 4)
    ax_bar.set_ylim(0, 1)
    ax_bar.grid(True, axis="y", alpha=ALPHA_GRID)

    for bars in (bars1, bars2, bars3):
        for b in bars:
            h = b.get_height()
            ax_bar.text(
                b.get_x() + b.get_width() / 2,
                h + 0.01,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_ANNOT,
            )

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLORS_METRICS["domain_auc"],
            edgecolor="black",
            linewidth=EDGE_WIDTH,
            label="Domain AUC",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLORS_METRICS["target_acc"],
            edgecolor="black",
            linewidth=EDGE_WIDTH,
            label="Target Acc",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLORS_METRICS["target_f1"],
            edgecolor="black",
            linewidth=EDGE_WIDTH,
            label="Target Macro F1",
        ),
    ]

    bar_top = ax_bar.get_position().y1

    legend = fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        fontsize=FONT_SIZE_LEGEND - 4,
        frameon=True,
        bbox_to_anchor=(0.5, bar_top - 0.01),
        bbox_transform=fig.transFigure,
    )
    legend.get_frame().set_edgecolor(LEGEND_FRAME_EDGE)
    legend.get_frame().set_linewidth(LEGEND_FRAME_WIDTH)
    legend.get_frame().set_alpha(LEGEND_FRAME_ALPHA)

    plt.savefig(cm_metrics_fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(cm_metrics_fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    metrics_fig_path = output_path / "figure4_metrics.pdf"
    fig_metrics_only = plt.figure(figsize=(12, 5))
    ax_metrics_only = fig_metrics_only.add_subplot(1, 1, 1)
    bars1_metrics = ax_metrics_only.bar(
        x - w,
        domain_aucs,
        w,
        color=COLORS_METRICS["domain_auc"],
        edgecolor="black",
        linewidth=EDGE_WIDTH,
    )
    bars2_metrics = ax_metrics_only.bar(
        x,
        tgt_accs,
        w,
        color=COLORS_METRICS["target_acc"],
        edgecolor="black",
        linewidth=EDGE_WIDTH,
    )
    bars3_metrics = ax_metrics_only.bar(
        x + w,
        tgt_f1s,
        w,
        color=COLORS_METRICS["target_f1"],
        edgecolor="black",
        linewidth=EDGE_WIDTH,
    )
    ax_metrics_only.set_xticks(x)
    ax_metrics_only.set_xticklabels(short, fontsize=FONT_SIZE_TICK - 4)
    ax_metrics_only.set_ylabel("Score", fontsize=FONT_SIZE_LABEL - 4)
    ax_metrics_only.set_ylim(0, 1)
    ax_metrics_only.grid(True, axis="y", alpha=ALPHA_GRID)
    for bars in (bars1_metrics, bars2_metrics, bars3_metrics):
        for b in bars:
            h = b.get_height()
            ax_metrics_only.text(
                b.get_x() + b.get_width() / 2,
                h + 0.01,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=FONT_SIZE_ANNOT,
            )
    legend_metrics = fig_metrics_only.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        fontsize=FONT_SIZE_LEGEND - 4,
        frameon=True,
        bbox_to_anchor=(0.5, 1.0),
        bbox_transform=fig_metrics_only.transFigure,
    )
    legend_metrics.get_frame().set_edgecolor(LEGEND_FRAME_EDGE)
    legend_metrics.get_frame().set_linewidth(LEGEND_FRAME_WIDTH)
    legend_metrics.get_frame().set_alpha(LEGEND_FRAME_ALPHA)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(metrics_fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(metrics_fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig_metrics_only)


def _plot_figure5(experiments: dict, output_path: str):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "figure5_metrics_only.pdf"
    methods = []
    domain_aucs = []
    a_distances = []
    tgt_accs = []
    tgt_f1s = []
    for name, exp in experiments.items():
        tgt = exp["tgt_metrics"]
        dom = exp["domain_metrics"]["test"]
        if not tgt:
            continue
        methods.append(name)
        domain_aucs.append(dom["domain_auc"])
        a_distances.append(dom["a_distance"])
        tgt_accs.append(tgt["tgt_acc"])
        tgt_f1s.append(tgt["tgt_macro_f1"])
    if not methods:
        logger.warning("No metrics available for figure 5.")
        return
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    x = np.arange(len(methods))
    width = 0.18
    ax.bar(
        x - 1.5 * width,
        domain_aucs,
        width,
        label="Domain AUC",
        color=COLORS_METRICS["domain_auc"],
        edgecolor="black",
        linewidth=EDGE_WIDTH,
    )
    ax.bar(
        x - 0.5 * width,
        a_distances,
        width,
        label="A-distance",
        color="#7A7A7A",
        edgecolor="black",
        linewidth=EDGE_WIDTH,
    )
    ax.bar(
        x + 0.5 * width,
        tgt_accs,
        width,
        label="Target Acc",
        color=COLORS_METRICS["target_acc"],
        edgecolor="black",
        linewidth=EDGE_WIDTH,
    )
    ax.bar(
        x + 1.5 * width,
        tgt_f1s,
        width,
        label="Target Macro F1",
        color=COLORS_METRICS["target_f1"],
        edgecolor="black",
        linewidth=EDGE_WIDTH,
    )
    short = []
    for n in methods:
        if n == "Baseline":
            short.append("Baseline")
        elif n == "DANN":
            short.append("DANN")
        elif "Fixed" in n:
            short.append("Euclidean\n(Fixed λ)")
        elif "$η_1,η_2$" in n or "$\\eta_1,\\eta_2$" in n:
            short.append("Euclidean\n(η₁, η₂)")
        elif "scheduler" in n:
            short.append("Euclidean\n(Scheduler σ + η₁η₂)")
        else:
            short.append(n)

    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=FONT_SIZE_TICK - 4)
    ax.set_ylabel("Score", fontsize=FONT_SIZE_LABEL - 4)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=ALPHA_GRID)
    ax.tick_params(labelsize=FONT_SIZE_TICK - 4)

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLORS_METRICS["domain_auc"],
            edgecolor="black",
            linewidth=EDGE_WIDTH,
            label="Domain AUC",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#7A7A7A",
            edgecolor="black",
            linewidth=EDGE_WIDTH,
            label="A-distance",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLORS_METRICS["target_acc"],
            edgecolor="black",
            linewidth=EDGE_WIDTH,
            label="Target Acc",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=COLORS_METRICS["target_f1"],
            edgecolor="black",
            linewidth=EDGE_WIDTH,
            label="Target Macro F1",
        ),
    ]

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    legend = fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        fontsize=FONT_SIZE_LEGEND - 6,
        frameon=True,
        bbox_to_anchor=(0.5, 0.98),
        bbox_transform=fig.transFigure,
    )
    legend.get_frame().set_edgecolor(LEGEND_FRAME_EDGE)
    legend.get_frame().set_linewidth(LEGEND_FRAME_WIDTH)
    legend.get_frame().set_alpha(LEGEND_FRAME_ALPHA)
    logger.info(f"Saving figure 5 (metrics only) to {fig_path}")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_figure6(experiments: dict, output_path: str):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "figure6_ot_loss.pdf"

    best_name = None
    best_dom = None
    best_exp = None

    for name, exp in experiments.items():
        df = exp["history"]
        if df is None or "ot_loss" not in df.columns:
            continue
        dom_metrics = exp["domain_metrics"]
        dom_test = dom_metrics["test"]["domain_auc"]
        if best_dom is None or dom_test < best_dom:
            best_dom = dom_test
            best_name = name
            best_exp = exp

    if best_exp is None:
        logger.warning("No experiment with ot_loss found for figure 6.")
        return

    df = best_exp["history"].copy()
    if "epoch_warmup" in df.columns:
        df = df[df["epoch_warmup"] == False]
    if df.empty:
        logger.warning("Best OT experiment has empty history after warmup.")
        return

    epochs = df["epoch"].to_numpy()
    ce = df["ce_loss"].to_numpy() if "ce_loss" in df.columns else None
    da = df["da_loss"].to_numpy() if "da_loss" in df.columns else None
    ot_total = df["ot_loss"].to_numpy() if "ot_loss" in df.columns else None
    ot_topk = df["ot_topk_loss"].to_numpy() if "ot_topk_loss" in df.columns else None
    ot_match = df["ot_match_loss"].to_numpy() if "ot_match_loss" in df.columns else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_left, ax_right = axes

    if ce is not None:
        ax_left.plot(
            epochs,
            ce,
            color=METHOD_CONFIG["baseline"][1],
            linestyle="-",
            linewidth=LINE_WIDTH,
            label=r"$\mathcal{L}_{\mathrm{sup}}$",
        )
    if da is not None:
        ax_left.plot(
            epochs,
            da,
            color=METHOD_CONFIG["adversarial"][1],
            linestyle="-",
            linewidth=LINE_WIDTH,
            label=r"$\mathcal{L}_{D}$",
        )
    if ot_total is not None:
        ax_left.plot(
            epochs,
            ot_total,
            color=METHOD_CONFIG["euclidean_fixed_lambda"][1],
            linestyle="-",
            linewidth=LINE_WIDTH,
            label=r"$\mathcal{L}_{\mathrm{OT}}$",
        )

    ax_left.set_xlabel("Epoch", fontsize=FONT_SIZE_LABEL)
    ax_left.set_ylabel("Loss value", fontsize=FONT_SIZE_LABEL)
    ax_left.yaxis.label.set_fontweight("medium")
    ax_left.grid(True, linestyle="--", alpha=ALPHA_GRID)
    leg_left = ax_left.legend(
        loc="upper right", frameon=True, fontsize=FONT_SIZE_LEGEND
    )
    if leg_left:
        leg_left.get_frame().set_edgecolor(LEGEND_FRAME_EDGE)
        leg_left.get_frame().set_linewidth(LEGEND_FRAME_WIDTH)
        leg_left.get_frame().set_alpha(LEGEND_FRAME_ALPHA)
    ax_left.set_title(f"Supervised vs. Alignment", fontsize=FONT_SIZE_TITLE)

    if ot_total is not None:
        ax_right.plot(
            epochs,
            ot_total,
            color=METHOD_CONFIG["euclidean_fixed_lambda"][1],
            linestyle="-",
            linewidth=LINE_WIDTH,
            label=r"$\mathcal{L}_{\mathrm{OT}}$",
        )
    if ot_match is not None:
        ax_right.plot(
            epochs,
            ot_match,
            color=METHOD_CONFIG["euclidean_trainable_weights"][1],
            linestyle="-",
            linewidth=LINE_WIDTH,
            label=r"$\mathcal{L}_{\mathrm{match}}$",
        )
    if ot_topk is not None:
        ax_right.plot(
            epochs,
            ot_topk,
            color=METHOD_CONFIG["euclidean_trainable_weights_sigma_scheduler"][1],
            linestyle="--",
            linewidth=LINE_WIDTH,
            label=r"$\mathcal{L}_{\mathrm{top}\text{-}k}$",
        )

    ax_right.set_xlabel("Epoch", fontsize=FONT_SIZE_LABEL)
    ax_right.set_ylabel("OT loss components", fontsize=FONT_SIZE_LABEL)
    ax_right.yaxis.label.set_fontweight("medium")
    ax_right.grid(True, linestyle="--", alpha=ALPHA_GRID)
    leg_right = ax_right.legend(
        loc="upper right", frameon=True, fontsize=FONT_SIZE_LEGEND
    )
    if leg_right:
        leg_right.get_frame().set_edgecolor(LEGEND_FRAME_EDGE)
        leg_right.get_frame().set_linewidth(LEGEND_FRAME_WIDTH)
        leg_right.get_frame().set_alpha(LEGEND_FRAME_ALPHA)
    ax_right.set_title(f"OT Decomposition", fontsize=FONT_SIZE_TITLE)

    fig.tight_layout()
    logger.info(f"Saving figure 6 (OT losses) to {fig_path}")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(fig_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_summary(exp_dir: str, experiments: dict):
    exp_path = Path(exp_dir)
    readme_path = exp_path / "README.md"
    lines: list[str] = []
    lines.append("# Results")
    lines.append("")
    lines.append(
        "Domain adaptation experiments for galaxy morphology classification "
        "(source: TNG50 simulation, target: SDSS)."
    )
    lines.append("")
    
    def fmt_pct(x):
        return f"{x * 100:.1f}%" if x is not None and np.isfinite(x) else "–"

    def fmt_dec(x, d=3):
        return f"{x:.{d}f}" if x is not None and np.isfinite(x) else "–"
    
    def _fmt_best(m):
        return f"**{m}**"
    
    def is_close(a, b, rtol=1e-5, atol=1e-8):
        if a is None or b is None:
            return False
        return abs(a - b) <= (atol + rtol * abs(b))
    
    def calc_precision(cm, class_idx):
        if cm is None or class_idx >= len(cm):
            return None
        tp = cm[class_idx][class_idx]
        col_sum = sum(cm[i][class_idx] for i in range(len(cm)))
        return tp / col_sum if col_sum > 0 else 0.0

    lines.append("## Training Information")
    lines.append("")
    header = "| Method | Total Epochs | Best Epoch | Final Train Loss | Best Train Loss |"
    sep = "|" + "|".join(["---"] * 5) + "|"
    lines.append(header)
    lines.append(sep)
    
    train_data = {}
    for name, exp in experiments.items():
        df = exp["history"]
        if df is None or df.empty:
            train_data[name] = {"epochs": None, "best_epoch": None, "final_loss": None, "best_loss": None}
        else:
            epochs = len(df)
            best_idx = df["train_loss"].idxmin()
            best_epoch = int(df.loc[best_idx, "epoch"]) if "epoch" in df.columns else best_idx + 1
            final_loss = df["train_loss"].iloc[-1]
            best_loss = df["train_loss"].min()
            train_data[name] = {"epochs": epochs, "best_epoch": best_epoch, "final_loss": final_loss, "best_loss": best_loss}
    
    # Find best values (lower is better for losses)
    best_final_loss = min((v["final_loss"] for v in train_data.values() if v["final_loss"] is not None), default=None)
    best_best_loss = min((v["best_loss"] for v in train_data.values() if v["best_loss"] is not None), default=None)
    
    for name, data in train_data.items():
        if data["epochs"] is None:
            epochs = "–"
            best_epoch = "–"
            final_loss = "–"
            best_loss = "–"
        else:
            epochs = str(data["epochs"])
            best_epoch = str(data["best_epoch"])
            final_loss = fmt_dec(data["final_loss"], 4)
            if is_close(data["final_loss"], best_final_loss):
                final_loss = _fmt_best(final_loss)
            best_loss = fmt_dec(data["best_loss"], 4)
            if is_close(data["best_loss"], best_best_loss):
                best_loss = _fmt_best(best_loss)
        row = f"| {name} | {epochs} | {best_epoch} | {final_loss} | {best_loss} |"
        lines.append(row)
    lines.append("")
    
    lines.append("## Source-domain performance")
    lines.append("")
    header = "| Method | Source Acc | Source Macro F1 | Source ROC-AUC | Source AUPRC | Source ECE |"
    sep = "|" + "|".join(["---"] * 6) + "|"
    lines.append(header)
    lines.append(sep)
    
    src_data = {}
    for name, exp in experiments.items():
        src = exp["src_metrics"]
        if not src:
            src_data[name] = {"acc": None, "f1": None, "roc": None, "auprc": None, "ece": None}
        else:
            src_data[name] = {
                "acc": src.get("src_acc"),
                "f1": src.get("src_macro_f1"),
                "roc": src.get("src_roc_auc_macro"),
                "auprc": src.get("src_auprc_macro"),
                "ece": src.get("src_ece"),
            }
    
    # Find best values (higher is better for acc, f1, roc, auprc; lower is better for ece)
    best_acc = max((v["acc"] for v in src_data.values() if v["acc"] is not None), default=None)
    best_f1 = max((v["f1"] for v in src_data.values() if v["f1"] is not None), default=None)
    best_roc = max((v["roc"] for v in src_data.values() if v["roc"] is not None), default=None)
    best_auprc = max((v["auprc"] for v in src_data.values() if v["auprc"] is not None), default=None)
    best_ece = min((v["ece"] for v in src_data.values() if v["ece"] is not None), default=None)
    
    for name, data in src_data.items():
        if data["acc"] is None:
            row = f"| {name} | – | – | – | – | – |"
        else:
            acc = fmt_pct(data["acc"])
            if is_close(data["acc"], best_acc):
                acc = _fmt_best(acc)
            f1 = fmt_dec(data["f1"])
            if is_close(data["f1"], best_f1):
                f1 = _fmt_best(f1)
            roc = fmt_dec(data["roc"])
            if is_close(data["roc"], best_roc):
                roc = _fmt_best(roc)
            auprc = fmt_dec(data["auprc"])
            if is_close(data["auprc"], best_auprc):
                auprc = _fmt_best(auprc)
            ece = fmt_dec(data["ece"], 4)
            if is_close(data["ece"], best_ece):
                ece = _fmt_best(ece)
            row = f"| {name} | {acc} | {f1} | {roc} | {auprc} | {ece} |"
        lines.append(row)
    lines.append("")
    
    lines.append("## Target-domain performance")
    lines.append("")
    header = (
        "| Method | Target Acc | Target Macro F1 | Target ROC-AUC | "
        "Target AUPRC | Target ECE | Domain AUC (test) | A-distance (test) | Domain AUC (full) | A-distance (full) |"
    )
    sep = "|" + "|".join(["---"] * 10) + "|"
    lines.append(header)
    lines.append(sep)

    # Collect target data
    tgt_data = {}
    for name, exp in experiments.items():
        tgt = exp["tgt_metrics"]
        dom = exp["domain_metrics"]["test"]
        full_dom = exp["domain_metrics"]["full"]
        tgt_data[name] = {
            "acc": tgt["tgt_acc"],
            "f1": tgt["tgt_macro_f1"],
            "roc": tgt["tgt_roc_auc_macro"],
            "auprc": tgt["tgt_auprc_macro"],
            "ece": tgt["tgt_ece"],
            "dauc": dom["domain_auc"],
            "adist": dom["a_distance"],
            "full_dauc": full_dom["domain_auc"],
            "full_adist": full_dom["a_distance"],
        }
    
    # Find best values
    best_acc = max((v["acc"] for v in tgt_data.values()), default=None)
    best_f1 = max((v["f1"] for v in tgt_data.values()), default=None)
    best_roc = max((v["roc"] for v in tgt_data.values()), default=None)
    best_auprc = max((v["auprc"] for v in tgt_data.values()), default=None)
    best_ece = min((v["ece"] for v in tgt_data.values()), default=None)
    # Domain AUC: closest to 0.5 is best (perfect domain mixing)
    best_dauc_dist = min((abs(v["dauc"] - 0.5) for v in tgt_data.values()), default=None)
    best_adist = min((v["adist"] for v in tgt_data.values()), default=None)
    best_full_dauc_dist = min((abs(v["full_dauc"] - 0.5) for v in tgt_data.values()), default=None)
    best_full_adist = min((v["full_adist"] for v in tgt_data.values()), default=None)
    
    for name, data in tgt_data.items():
        acc = fmt_pct(data["acc"])
        if is_close(data["acc"], best_acc):
            acc = _fmt_best(acc)
        f1 = fmt_dec(data["f1"])
        if is_close(data["f1"], best_f1):
            f1 = _fmt_best(f1)
        roc = fmt_dec(data["roc"])
        if is_close(data["roc"], best_roc):
            roc = _fmt_best(roc)
        auprc = fmt_dec(data["auprc"])
        if is_close(data["auprc"], best_auprc):
            auprc = _fmt_best(auprc)
        ece = fmt_dec(data["ece"], 4)
        if is_close(data["ece"], best_ece):
            ece = _fmt_best(ece)
        dauc = fmt_dec(data["dauc"], 4)
        if is_close(abs(data["dauc"] - 0.5), best_dauc_dist):
            dauc = _fmt_best(dauc)
        adist = fmt_dec(data["adist"], 4)
        if is_close(data["adist"], best_adist):
            adist = _fmt_best(adist)
        full_dauc = fmt_dec(data["full_dauc"], 4)
        if is_close(abs(data["full_dauc"] - 0.5), best_full_dauc_dist):
            full_dauc = _fmt_best(full_dauc)
        full_adist = fmt_dec(data["full_adist"], 4)
        if is_close(data["full_adist"], best_full_adist):
            full_adist = _fmt_best(full_adist)
        row = (
            f"| {name} | {acc} | {f1} | {roc} | {auprc} | {ece} | "
            f"{dauc} | {adist} | {full_dauc} | {full_adist} |"
        )
        lines.append(row)
    lines.append("")
    lines.append("## Target per-class recall")
    lines.append("")
    header = "| Method | " + " | ".join(cls.capitalize() for cls in CLASSES) + " |"
    sep = "|" + "|".join(["---"] * (len(CLASSES) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    
    recall_data = {}
    for name, exp in experiments.items():
        tgt = exp["tgt_metrics"]
        recalls = tgt["tgt_recalls"]
        recall_data[name] = recalls
    
    best_recalls = []
    for i in range(len(CLASSES)):
        best_recall = max((recalls[i] if i < len(recalls) else -1 for recalls in recall_data.values()), default=None)
        best_recalls.append(best_recall)
    
    for name, recalls in recall_data.items():
        vals = []
        for i in range(len(CLASSES)):
            if i < len(recalls):
                val = fmt_pct(recalls[i])
                if is_close(recalls[i], best_recalls[i]):
                    val = _fmt_best(val)
            else:
                val = "–"
            vals.append(val)
        row = f"| {name} | " + " | ".join(vals) + " |"
        lines.append(row)
    lines.append("")
    
    lines.append("## Target per-class precision")
    lines.append("")
    header = "| Method | " + " | ".join(cls.capitalize() for cls in CLASSES) + " |"
    sep = "|" + "|".join(["---"] * (len(CLASSES) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    
    precision_data = {}
    for name, exp in experiments.items():
        tgt = exp["tgt_metrics"]
        cm = np.array(tgt.get("tgt_cm", []))
        precisions = [calc_precision(cm, i) for i in range(len(CLASSES))]
        precision_data[name] = precisions
    
    best_precisions = []
    for i in range(len(CLASSES)):
        best_prec = max((prec[i] if prec[i] is not None else -1 for prec in precision_data.values()), default=None)
        best_precisions.append(best_prec)
    
    for name, precisions in precision_data.items():
        vals = []
        for i in range(len(CLASSES)):
            if precisions[i] is not None:
                val = fmt_pct(precisions[i])
                if is_close(precisions[i], best_precisions[i]):
                    val = _fmt_best(val)
            else:
                val = "–"
            vals.append(val)
        row = f"| {name} | " + " | ".join(vals) + " |"
        lines.append(row)
    
    readme_path.write_text("\n".join(lines))
    logger.info(f"Wrote final Results to {readme_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot combined metrics for paper.")
    p.add_argument(
        "exp_dir", type=str, help="Path to experiment folder (e.g., experiments/)"
    )
    p.add_argument(
        "--output", type=str, default="paper-figures", help="Output filename"
    )
    p.add_argument(
        "--fig",
        type=int,
        default=None,
        help="Figure number to plot, if None, plot all figures",
    )
    args = p.parse_args()
    set_style()
    experiments = _load_data(args.exp_dir, full_z=True)
    output_dir = args.output
    try:
        if args.fig is None or args.fig == 1:
            _plot_figure1(output_dir)
    except Exception as e:
        logger.error(f"Error plotting figure 1: {e}")
    try:
        if args.fig is None or args.fig == 2:
            _plot_figure2(experiments, output_dir)
    except Exception as e:
        logger.error(f"Error plotting figure 2: {e}")
    
    try:
        if args.fig is None or args.fig == 3:
            _plot_figure3(experiments, output_dir)
    except Exception as e:
        logger.error(f"Error plotting figure 3: {e}")
    try:
        if args.fig is None or args.fig == 4:
            _plot_figure4(experiments, output_dir)
    except Exception as e:
        logger.error(f"Error plotting figure 4: {e}")
    try:
        if args.fig is None or args.fig == 5:
            _plot_figure5(experiments, output_dir)
    except Exception as e:
        logger.error(f"Error plotting figure 5: {e}")
    try:
        if args.fig is None or args.fig == 6:
            _plot_figure6(experiments, output_dir)
    except Exception as e:
        logger.error(f"Error plotting figure 6: {e}")
    try:
        _write_summary(args.exp_dir, experiments)
    except Exception as e:
        logger.error(f"Error writing summary: {e}")
