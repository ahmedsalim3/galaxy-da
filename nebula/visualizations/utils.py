import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nebula.commons import Logger

logger = Logger()


def set_plot_style():
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.2,
            "axes.titleweight": "bold",
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "grid.linestyle": "--",
            "grid.alpha": 0.6,
        }
    )


COLORS = {
    "total_loss": "#2e3440",
    "ce_loss": "#084d02",
    "da_loss": "#bf616a",
    "align_loss": "#5e81ac",
    "entropy_loss": "#b48ead",
    "source_acc": "#2e3440",
    "target_acc": "#bf616a",
    "target_f1": "#5e81ac",
    "eta1": "#88c0d0",
    "eta2": "#a64d79",
    "sigma": "#d08770",
    "lambda_grl": "#8B4513",
    "warmup_bg": "#5b94a4",
}


def prepare_data(history, key, epochs):
    data = history.get(key, [])
    if not data:
        return np.array([])
    assert len(data) == len(
        epochs
    ), f"Data for '{key}' and epochs must have the same length"
    data = data[: len(epochs)]
    return np.array([float(v) if v is not None else np.nan for v in data])


def add_warmup_region(ax, warmup_epochs, epochs):
    if warmup_epochs > 0 and len(epochs) > 0:
        ax.axvspan(
            epochs[0],
            min(warmup_epochs, epochs[-1]),
            alpha=0.12,
            color=COLORS["warmup_bg"],
            zorder=0,
            label="Warmup",
        )


def add_legend(ax, ax_twin=None, loc="upper right", outside=False, transparent=False):
    handles, labels = ax.get_legend_handles_labels()

    if ax_twin is not None:
        h2, l2 = ax_twin.get_legend_handles_labels()
        handles.extend(h2)
        labels.extend(l2)

    # remove duplicates while preserving order
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle

    # we can pass the loss equation below:
    # pass get_loss_eq(trainer) as a loss argument
    # import matplotlib.lines as mlines
    # if loss is not None:
    #     # Create a "fake" handle that's invisible
    #     empty_handle = mlines.Line2D([], [], color="none")
    #     handles.append(empty_handle)
    #     labels.append(loss)

    framealpha = 0.3 if transparent else 0.95
    
    if outside and loc == "upper left":
        ax.legend(
            unique_labels.values(),
            unique_labels.keys(),
            loc="center left",
            bbox_to_anchor=(-0.28, 0.5),
            framealpha=framealpha,
            bbox_transform=ax.transAxes,
            fontsize=11,
            frameon=True,
        )
    else:
        ax.legend(
            unique_labels.values(),
            unique_labels.keys(),
            loc=loc,
            framealpha=framealpha,
            fontsize=11,
        )


def add_equation_box(ax, equation, position="bottom_right"):
    """
    Add equation text box to axis.

    Args:
        ax: Matplotlib axis
        equation: LaTeX equation string, use get_loss_eq(trainer)
        position: One of 'bottom_left', 'bottom_right', 'top_left', 'top_right'
    """
    if not equation:
        return

    if position == "bottom_center":
        ax.text(
            0.5,
            -0.42,
            equation,
            ha="center",
            va="top",
            transform=ax.transAxes,
            fontsize=18,
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "white",
                "edgecolor": "#d8dee9",
                "linewidth": 1.2,
                "alpha": 0.92,
            },
        )
        return

    positions = {
        "bottom_left": (0.02, 0.05, "left", "bottom"),
        "bottom_right": (0.98, 0.05, "right", "bottom"),
        "top_left": (0.01, 0.98, "left", "top"),
        "top_right": (0.98, 0.98, "right", "top"),
    }

    x, y, ha, va = positions.get(position, positions["top_right"])

    ax.text(
        x,
        y,
        equation,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=15,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "white",
            "edgecolor": "#d8dee9",
            "linewidth": 1.2,
            "alpha": 0.92,
        },
    )


def get_loss_eq(trainer):
    """Return a LaTeX-formatted loss equation string for a given trainer.

    Supports:
        - NoDATrainer
        - DAFixedLambdaTrainer
        - DATrainableWeightsTrainer
        - DATrainableWeightsSigmaTrainer
        - DAAdversarialTrainer
    """

    def _format_num(num, precision=2):
        if abs(num) >= 1e-2 and abs(num) < 1e3:
            return f"{num:.{precision}f}".rstrip("0").rstrip(".")
        else:
            base, exp = f"{num:.{precision}e}".split("e")
            exp = int(exp)
            return rf"{base}\times10^{{{exp}}}"

    if trainer is None:
        return ""

    name = trainer.__class__.__name__

    # --- No domain adaptation ---
    if name == "NoDATrainer":
        return r"$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{CE}}$"

    # --- Fixed lambda DA ---
    if name == "DAFixedLambdaTrainer":
        λ = getattr(trainer.config, "lambda_da", None)
        method = getattr(trainer.config, "method", None)
        method_tag = rf"^{{\mathrm{{{method}}}}}" if isinstance(method, str) else ""

        # Check for optional losses
        λ_entropy = getattr(trainer.config, "lambda_entropy", 0.0)
        λ_ot = getattr(trainer.config, "lambda_ot", getattr(trainer, "lambda_ot", 0.0))
        has_entropy = λ_entropy > 0.0
        has_align = λ_ot > 0.0

        λ_str = _format_num(λ) if λ is not None else r"\lambda_{\mathrm{DA}}"

        eq_parts = [
            rf"$\mathcal{{L}}_{{\mathrm{{total}}}} = ",
            rf"\mathcal{{L}}_{{\mathrm{{CE}}}} + {λ_str}\,\mathcal{{L}}_{{\mathrm{{DA}}}}{method_tag}",
        ]

        if has_entropy:
            λ_ent_str = _format_num(λ_entropy)
            eq_parts.append(rf" + {λ_ent_str}\,\mathcal{{L}}_{{\mathrm{{entropy}}}}")

        if has_align:
            λ_ot_str = _format_num(λ_ot)
            eq_parts.append(rf" + {λ_ot_str}\,\mathcal{{L}}_{{\mathrm{{align}}}}")

        eq_parts.append("$")
        return "".join(eq_parts)

    # --- Trainable weights (η terms) ---
    if name in ["DATrainableWeightsTrainer", "DATrainableWeightsSigmaTrainer"]:
        method = getattr(trainer.config, "method", None)
        method_tag = rf"^{{\mathrm{{{method}}}}}" if isinstance(method, str) else ""
        sigma_tag = r"(\sigma_t)" if name == "DATrainableWeightsSigmaTrainer" else ""

        η1 = getattr(trainer.config, "eta1", None)
        η2 = getattr(trainer.config, "eta2", None)
        η1_str = _format_num(η1) if η1 is not None else r"\eta_1"
        η2_str = _format_num(η2) if η2 is not None else r"\eta_2"

        # For log term: use multiplication for numeric values, concatenation for symbolic
        if η1 is not None and η2 is not None:
            log_term = rf"\log({η1_str} \cdot {η2_str})"
        else:
            log_term = rf"\log({η1_str}{η2_str})"

        # Check for optional losses
        λ_entropy = getattr(trainer.config, "lambda_entropy", 0.0)
        λ_ot = getattr(trainer.config, "lambda_ot", getattr(trainer, "lambda_ot", 0.0))
        has_entropy = λ_entropy > 0.0
        has_align = λ_ot > 0.0

        eq_parts = [
            rf"$\mathcal{{L}}_{{\mathrm{{total}}}} = ",
            rf"\frac{{1}}{{2 \cdot {η1_str}^2}}\,\mathcal{{L}}_{{\mathrm{{CE}}}} + ",
            rf"\frac{{1}}{{2 \cdot {η2_str}^2}}\,\mathcal{{L}}_{{\mathrm{{DA}}}}{method_tag}{sigma_tag} ",
            rf"+ {log_term}",
        ]

        if has_entropy:
            λ_ent_str = _format_num(λ_entropy)
            eq_parts.append(rf" + {λ_ent_str}\,\mathcal{{L}}_{{\mathrm{{entropy}}}}")

        if has_align:
            λ_ot_str = _format_num(λ_ot)
            eq_parts.append(rf" + {λ_ot_str}\,\mathcal{{L}}_{{\mathrm{{align}}}}")

        eq_parts.append("$")
        return "".join(eq_parts)

    # --- Adversarial DA ---
    if name == "DAAdversarialTrainer":
        λ_grl = getattr(trainer.config, "lambda_grl", None)
        λ_grl_str = (
            _format_num(λ_grl) if λ_grl is not None else r"\lambda_{\mathrm{grl}}"
        )

        λ_entropy = getattr(trainer.config, "lambda_entropy", 0.0)
        λ_ot = getattr(trainer.config, "lambda_ot", getattr(trainer, "lambda_ot", 0.0))
        has_entropy = λ_entropy > 0.0
        has_align = λ_ot > 0.0

        # Note: lambda_grl controls gradient reversal strength, not a direct multiplier
        # But we show it in the equation for clarity
        eq_parts = [
            rf"$\mathcal{{L}}_{{\mathrm{{total}}}} = ",
            rf"\mathcal{{L}}_{{\mathrm{{CE}}}} + \lambda_{{\mathrm{{grl}}}}\,\mathcal{{L}}_{{\mathrm{{domain}}}}",
        ]
        if λ_grl is not None:
            eq_parts[1] = (
                rf"\mathcal{{L}}_{{\mathrm{{CE}}}} + {λ_grl_str}\,\mathcal{{L}}_{{\mathrm{{domain}}}}"
            )

        if has_entropy:
            λ_ent_str = _format_num(λ_entropy)
            eq_parts.append(rf" + {λ_ent_str}\,\mathcal{{L}}_{{\mathrm{{entropy}}}}")

        if has_align:
            λ_ot_str = _format_num(λ_ot)
            eq_parts.append(rf" + {λ_ot_str}\,\mathcal{{L}}_{{\mathrm{{align}}}}")

        eq_parts.append("$")
        return "".join(eq_parts)

    return ""
