import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn import metrics as skm

from nebula.commons import Logger
from nebula.visualizations import plot_training_history
from nebula.visualizations.eval_plots import (
    CLASSES, plot_class_distribution, plot_confusion_matrices,
    plot_domain_separation, plot_latent_space_two_panel, plot_per_class_bars,
    plot_reliability_diagrams_side_by_side, plot_roc_pr_curves)

logger = Logger()


def load_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)


def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def npy(eval_dir: Path, stem: str):
    # helper: eval_dir/data/<stem>.npy
    p = eval_dir / "data" / f"{stem}.npy"
    return np.load(p) if p.exists() else None


def find_experiments(root_dir: Path):
    exp_dirs = []
    root_path = Path(root_dir)
    if not root_path.exists():
        return exp_dirs

    for item in root_path.iterdir():
        if item.is_dir() and (item / "eval").exists():
            exp_dirs.append(item)

    return sorted(exp_dirs)


def load_history_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(how="all")
    for col in df.columns:
        if col in [
            "eta_1",
            "eta_2",
            "sigma",
            "train_loss",
            "ce_loss",
            "da_loss",
            "source_acc",
            "target_acc",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.to_dict(orient="list")


def _get_trainer(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config")

    config_class = config.__class__.__name__
    config_to_trainer = {
        "NoDAConfig": "NoDATrainer",
        "DAFixedLambdaConfig": "DAFixedLambdaTrainer",
        "DATrainableWeightsConfig": "DATrainableWeightsTrainer",
        "DATrainableWeightsSigmaConfig": "DATrainableWeightsSigmaTrainer",
        "DAAdversarialConfig": "DAAdversarialTrainer",
    }

    trainer_class_name = config_to_trainer.get(config_class)
    if trainer_class_name is None:
        return None

    mock_trainer = type(trainer_class_name, (), {})()
    mock_trainer.config = config

    return mock_trainer


def load_trainer(ckpt_path):
    if ckpt_path and Path(ckpt_path).exists():
        return _get_trainer(ckpt_path)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "paths",
        nargs="+",
        help="Path(s) to experiment directory or folder containing experiments",
    )
    ap.add_argument(
        "--many",
        action="store_true",
        help="Treat each path as an experiment directory (don't auto-discover)",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: [experiment_dir]/figures)",
    )
    ap.add_argument("--embed-method", default="tsne", choices=["tsne", "umap", "pca"])
    args = ap.parse_args()

    if args.many:
        exp_dirs = [p for p in Path(args.paths[0]).iterdir() if p.is_dir()]
    else:
        if len(args.paths) > 1:
            ap.error(
                "Without --many, only one path should be provided for auto-discovery"
            )
        root_dir = Path(args.paths[0])
        if (root_dir / "eval").exists():
            exp_dirs = [root_dir]
        else:
            exp_dirs = find_experiments(root_dir)
            if not exp_dirs:
                logger.warning(
                    f"Warning: No experiments found in {root_dir} (looking for subdirectories with eval/ folder)"
                )
                return

    logger.info(f"Found {len(exp_dirs)} experiment(s):")
    for ed in exp_dirs:
        logger.info(f"  - {ed}")

    df_rows = []

    for exp_dir in exp_dirs:
        eval_dir = exp_dir / "eval"
        logs_dir = exp_dir / "logs"
        ckpt_dir = exp_dir / "ckpts"
        exp_name = exp_dir.name
        logger.info(f"Processing {logs_dir}")
        if not eval_dir.exists():
            logger.warning(
                f"Warning: {exp_dir} does not have an eval/ subdirectory, skipping"
            )
            continue

        if args.output_dir:
            out_dir = Path(args.output_dir)
        else:
            out_dir = exp_dir / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)

        history_csv_files = list(logs_dir.glob("*_history.csv"))
        if history_csv_files:
            history_csv = history_csv_files[0]
        else:
            # Fallback to expected naming convention
            history_csv = logs_dir / f"{exp_name}_history.csv"
        if not history_csv.exists():
            raise FileNotFoundError(
                f"History CSV not found at {history_csv}. "
                f"Make sure experiment directory exists at {exp_dir}/"
            )

        # Find checkpoint for trainer config (look for any .pt file in ckpts directory)
        ckpt_files = list(ckpt_dir.glob("*.pt"))
        if ckpt_files:
            ckpt_path = ckpt_files[0]
        else:
            ckpt_path = ckpt_dir / f"{exp_name}.pt"
        if not ckpt_path.exists():
            logger.warning(
                f"Warning: Checkpoint not found at {ckpt_path}. Plotting without equation display."
            )
            ckpt_path = None

        history = load_history_from_csv(history_csv)
        trainer = load_trainer(ckpt_path) if ckpt_path else None
        plot_training_history(
            history,
            trainer=trainer,
            save_path=out_dir / (history_csv.with_suffix(".png")).name,
        )

        # --- Load metrics JSONs from eval/ subdirectory ---
        src_metrics = (
            load_json(eval_dir / "src_metrics.json")
            if (eval_dir / "src_metrics.json").exists()
            else {}
        )
        tgt_metrics = (
            load_json(eval_dir / "tgt_metrics.json")
            if (eval_dir / "tgt_metrics.json").exists()
            else {}
        )
        domain_metrics = (
            load_json(eval_dir / "domain_metrics.json")
            if (eval_dir / "domain_metrics.json").exists()
            else {}
        )

        # --- Load features for TEST sets from eval/data/ ---
        src_test_z = npy(eval_dir, "src_test_z")
        tgt_test_z = npy(eval_dir, "tgt_test_z")
        src_test_y = npy(eval_dir, "src_test_y")
        tgt_test_y = npy(eval_dir, "tgt_test_y")
        src_test_logits = npy(eval_dir, "src_test_logits")
        tgt_test_logits = npy(eval_dir, "tgt_test_logits")
        src_test_preds = npy(eval_dir, "src_test_preds")
        tgt_test_preds = npy(eval_dir, "tgt_test_preds")

        # --- Load features for FULL sets from eval/data/ ---
        src_full_z = npy(eval_dir, "src_full_z")
        tgt_full_z = npy(eval_dir, "tgt_full_z")
        src_full_y = npy(eval_dir, "src_full_y")
        tgt_full_y = npy(eval_dir, "tgt_full_y")
        src_full_preds = npy(eval_dir, "src_full_preds")
        tgt_full_preds = npy(eval_dir, "tgt_full_preds")

        n_classes = len(CLASSES)
        class_labels = list(range(n_classes))

        # --- Confusion Matrices for TEST sets ---
        if "src_cm" in src_metrics and "tgt_cm" in tgt_metrics:
            plot_confusion_matrices(
                np.array(src_metrics["src_cm"]),
                np.array(tgt_metrics["tgt_cm"]),
                src_metrics.get("src_acc", 0.0),
                tgt_metrics.get("tgt_acc", 0.0),
                titles=["Source Test Set", "Target Test Set"],
                save_path=out_dir / "confusion_matrices_test.png",
            )

        # --- Confusion Matrices for FULL sets ---
        if (
            src_full_y is not None
            and src_full_preds is not None
            and tgt_full_y is not None
            and tgt_full_preds is not None
        ):
            full_src_cm = skm.confusion_matrix(
                src_full_y, src_full_preds, labels=class_labels
            )
            full_tgt_cm = skm.confusion_matrix(
                tgt_full_y, tgt_full_preds, labels=class_labels
            )
            full_src_acc = skm.accuracy_score(src_full_y, src_full_preds)
            full_tgt_acc = skm.accuracy_score(tgt_full_y, tgt_full_preds)
            plot_confusion_matrices(
                full_src_cm,
                full_tgt_cm,
                full_src_acc,
                full_tgt_acc,
                titles=["Source Full Set", "Target Full Set"],
                save_path=out_dir / "confusion_matrices_full.png",
            )

        # --- Compute probabilities from logits ---
        if src_test_logits is not None:
            src_test_probas = F.softmax(
                torch.from_numpy(src_test_logits), dim=1
            ).numpy()
        else:
            src_test_probas = None

        if tgt_test_logits is not None:
            tgt_test_probas = F.softmax(
                torch.from_numpy(tgt_test_logits), dim=1
            ).numpy()
        else:
            tgt_test_probas = None

        # --- Reliability diagrams side by side ---
        if (
            src_test_y is not None
            and src_test_probas is not None
            and tgt_test_y is not None
            and tgt_test_probas is not None
        ):
            plot_reliability_diagrams_side_by_side(
                src_test_y,
                src_test_probas,
                tgt_test_y,
                tgt_test_probas,
                save_path=out_dir / "calibrations.png",
            )

        # --- ROC/PR curves (target) ---
        if tgt_test_y is not None and tgt_test_probas is not None:
            plot_roc_pr_curves(
                tgt_test_y,
                tgt_test_probas,
                class_names=CLASSES,
                title_prefix="Target",
                save_path=out_dir / "roc_pr_target.png",
            )

        # --- Per-class recall bars ---
        if "src_recalls" in src_metrics and "tgt_recalls" in tgt_metrics:
            plot_per_class_bars(
                src_metrics["src_recalls"],
                tgt_metrics["tgt_recalls"],
                class_names=CLASSES,
                metric_name="Recall",
                save_path=out_dir / "per_class_recall_bars.png",
            )

        # --- Class distribution on test sets ---
        if src_test_y is not None and tgt_test_y is not None:
            plot_class_distribution(
                src_test_y,
                tgt_test_y,
                class_names=CLASSES,
                normalize=True,
                save_path=out_dir / "class_distribution_test.png",
            )

        # --- Latent space plots (using FULL data) ---
        if (
            src_full_z is not None
            and tgt_full_z is not None
            and src_full_y is not None
            and tgt_full_y is not None
        ):
            # Latent space with true labels
            try:
                plot_latent_space_two_panel(
                    src_full_z,
                    tgt_full_z,
                    src_full_y,
                    tgt_full_y,
                    method=args.embed_method,
                    save_path=out_dir
                    / f"latent_two_panel_{args.embed_method}_true.png",
                    use_predictions=False,
                )
            except Exception as e:
                logger.warning(
                    f"Latent space plot (true labels) failed for {exp_dir}: {e}"
                )

            # Latent space with predictions
            if src_full_preds is not None and tgt_full_preds is not None:
                try:
                    plot_latent_space_two_panel(
                        src_full_z,
                        tgt_full_z,
                        src_full_preds,
                        tgt_full_preds,
                        method=args.embed_method,
                        save_path=out_dir
                        / f"latent_two_panel_{args.embed_method}_pred.png",
                        use_predictions=True,
                    )
                except Exception as e:
                    logger.warning(
                        f"Latent space plot (predictions) failed for {exp_dir}: {e}"
                    )

        # --- Domain separation plot ---
        if src_full_z is not None and tgt_full_z is not None:
            try:
                plot_domain_separation(
                    src_full_z,
                    tgt_full_z,
                    method=args.embed_method,
                    save_path=out_dir / f"latent_domain_only_{args.embed_method}.png",
                )
            except Exception as e:
                logger.warning(f"Domain separation plot failed for {exp_dir}: {e}")

        # for cross-run summary
        # TODO: add more metrics
        df_rows.append(
            {
                "run": exp_dir.name,
                "tgt_acc": tgt_metrics.get("tgt_acc", np.nan),
                "mmd2": safe_get(domain_metrics, "test", "mmd2", default=np.nan),
                "sinkhorn_div": safe_get(
                    domain_metrics, "test", "sinkhorn_div", default=np.nan
                ),
                "domain_auc": safe_get(
                    domain_metrics, "test", "domain_auc", default=np.nan
                ),
                "a_distance": safe_get(
                    domain_metrics, "test", "a_distance", default=np.nan
                ),
            }
        )

    # --- Across-run summary ---
    if len(df_rows) > 1:
        df = pd.DataFrame(df_rows)
        summary_dir = (
            exp_dirs[0].parent
            if all(ed.parent == exp_dirs[0].parent for ed in exp_dirs)
            else Path.cwd()
        )
        df.to_csv(summary_dir / "summary_metrics_across_runs.csv", index=False)


if __name__ == "__main__":
    main()
