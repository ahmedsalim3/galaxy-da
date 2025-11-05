import argparse
import json
from pathlib import Path

import numpy as np
import run_train as rt
import torch
import torch.nn.functional as F
from sklearn import metrics as skm
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from nebula.analysis.alignment import (domain_probe_auc,
                                       domain_probe_auc_per_class,
                                       mmd2_unbiased_gaussian,
                                       sinkhorn_divergence)
from nebula.analysis.features import extract_features
from nebula.commons import Logger, set_all_seeds
from nebula.data.dataset import CLASSES
from nebula.visualizations.eval_plots import (
    COLORS_CLASS, COLORS_DOMAIN, plot_class_distribution,
    plot_confusion_matrices, plot_domain_separation,
    plot_latent_space_two_panel, plot_per_class_bars,
    plot_reliability_diagrams_side_by_side, plot_roc_pr_curves)

logger = Logger()


def load_ckpt(ckpt_path, device):
    # load full config
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = ckpt["full_config"]

    # build model
    model = rt.build_model(
        config["model"]["type"],
        config["data"]["image_size"],
        model_config=config.get("model", {}),
    )

    # Load state dict
    try:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    except RuntimeError:
        # use strict=False for ESCNN
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model.to(device)
    model.eval()
    logger.info(f"Eval Model Loaded from {ckpt_path}")
    return model, config


def build_all_dataloaders(config):
    """
    Return all four dataloaders:
    src_full_loader, tgt_full_loader, src_test_loader, tgt_test_loader.
    """
    data_module = rt.build_data_module(config)
    source_dataset = data_module.src_dataset
    target_dataset = data_module.tgt_dataset

    # Ensure baseline method can load target data for eval
    if config["training"]["method"] == "baseline":
        try:
            # Attempt to get target test loader
            tgt_test_loader = data_module.target_test_loader
            if len(tgt_test_loader.dataset) == 0:
                raise AttributeError("Target test loader is empty")
        except (AttributeError, FileNotFoundError):
            # Force getting target data if not present
            logger.warning("Baseline config: manually loading target data for eval.")
            config["data"]["target_img_dir"] = config["data"].get(
                "target_img_dir", "data/target/gz2_images"
            )
            config["data"]["target_labels"] = config["data"].get(
                "target_labels", "data/target/gz2_galaxy_labels.csv"
            )
            data_module = rt.build_data_module(config)
            target_dataset = data_module.tgt_dataset

    src_full_loader = DataLoader(
        source_dataset,
        batch_size=data_module.batch_size,
        shuffle=False,
        num_workers=data_module.num_workers,
    )
    tgt_full_loader = DataLoader(
        target_dataset,
        batch_size=data_module.batch_size,
        shuffle=False,
        num_workers=data_module.num_workers,
    )

    src_test_loader = data_module.source_test_loader
    tgt_test_loader = data_module.target_test_loader

    return src_full_loader, tgt_full_loader, src_test_loader, tgt_test_loader


def compute_domain_metrics(
    src_z,
    tgt_z,
    src_y,
    tgt_y,
):
    """Compute comprehensive domain adaptation metrics."""
    n_classes = len(CLASSES)
    metrics = {}

    # Overall metrics
    metrics["domain_auc"], metrics["domain_acc"] = domain_probe_auc(src_z, tgt_z)
    metrics["a_distance"] = 2.0 * (1.0 - metrics["domain_acc"])
    # metrics["mmd2"] = mmd2_unbiased_gaussian(src_z, tgt_z)
    metrics["sinkhorn_div"] = sinkhorn_divergence(src_z, tgt_z, blur=0.05, p=2)

    # Per-class MMD
    # cmmd = {}
    # for i, cname in enumerate(CLASSES):
    #     Xs = src_z[src_y == i]
    #     Xt = tgt_z[tgt_y == i]
    #     cmmd[cname] = (
    #         mmd2_unbiased_gaussian(Xs, Xt) if (len(Xs) > 1 and len(Xt) > 1) else np.nan
    #     )

    # for i, val in cmmd.items():
    #     metrics[f"cmmd_{i}"] = val

    # Classwise domain AUC
    dom_auc_cls = domain_probe_auc_per_class(
        src_z, src_y, tgt_z, tgt_y, classes=range(n_classes)
    )
    for i, val in dom_auc_cls.items():
        metrics[f"domain_auc_{CLASSES[i]}"] = val

    # Per-class centroid distances (mean pairwise distance)
    centroid_dists = []
    for c, cname in enumerate(CLASSES):
        src_c = src_z[src_y == c]
        tgt_c = tgt_z[tgt_y == c]
        if len(src_c) > 0 and len(tgt_c) > 0:
            dist = np.linalg.norm(src_c.mean(0) - tgt_c.mean(0))
        else:
            dist = np.nan
        centroid_dists.append(dist)
        metrics[f"centroid_dist_{cname}"] = dist

    metrics["avg_centroid_distance"] = np.nanmean(centroid_dists)

    # Per-class variances
    for c, cname in enumerate(CLASSES):
        src_var = src_z[src_y == c].var() if (src_y == c).sum() > 0 else np.nan
        tgt_var = tgt_z[tgt_y == c].var() if (tgt_y == c).sum() > 0 else np.nan
        metrics[f"src_var_{cname}"] = src_var
        metrics[f"tgt_var_{cname}"] = tgt_var

    # Class distribution
    for c, cname in enumerate(CLASSES):
        src_pct = (src_y == c).sum() / len(src_y) * 100 if len(src_y) > 0 else 0
        tgt_pct = (tgt_y == c).sum() / len(tgt_y) * 100 if len(tgt_y) > 0 else 0
        metrics[f"src_class_pct_{cname}"] = src_pct
        metrics[f"tgt_class_pct_{cname}"] = tgt_pct

    # Silhouette score (domain separation)
    try:
        combined = np.vstack([src_z, tgt_z])
        labels = np.hstack([np.zeros(len(src_z)), np.ones(len(tgt_z))])
        if len(combined) > 5000:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(combined), 5000, replace=False)
            combined, labels = combined[idx], labels[idx]

        metrics["silhouette_score"] = skm.silhouette_score(combined, labels)
    except Exception:
        metrics["silhouette_score"] = np.nan

    return metrics


def compute_ece(y_true, y_proba, n_bins=10):
    if y_true.size == 0 or y_proba.size == 0:
        return np.nan

    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    accuracies = predictions == y_true

    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins[:-1]) - 1

    ece = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = accuracies[mask].mean()
            ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)

    return float(ece)


def save_features(feats, save_dir, prefix):
    save_dir.mkdir(parents=True, exist_ok=True)
    for key, arr in feats.items():
        if isinstance(arr, np.ndarray):
            np.save(save_dir / f"{prefix}_{key}.npy", arr)
    logger.info(f"Saved {prefix} features to {save_dir}")


def save_metrics(metrics, save_path):

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float_)):
            val = float(obj)
            return val if np.isfinite(val) else None
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.void, type(None))):
            return None
        if torch.is_tensor(obj):
            return obj.detach().cpu().numpy().tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_serializable(item) for item in obj]
        if isinstance(obj, float):
            return obj if np.isfinite(obj) else None
        return obj

    serializable = {k: to_serializable(v) for k, v in metrics.items()}
    with open(save_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved metrics to {save_path}")


def main():
    p = argparse.ArgumentParser(epilog="Domain Adaptation Evaluation")
    p.add_argument("ckpt_path", type=str, help="Path to model checkpoint (.pt file)")
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: [experiment_dir]/eval)",
    )
    p.add_argument(
        "--embed-method",
        type=str,
        default="tsne",
        choices=["tsne", "umap", "pca"],
        help="Embedding method.",
    )
    args = p.parse_args()

    ckpt_path = Path(args.ckpt_path)
    device = torch.device(args.device)

    # --- 1. Setup Output Directories ---
    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = ckpt_path.parent.parent / f"eval"

    plot_dir = output_root / "plots"
    data_dir = output_root / "data"
    output_root.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    global logger
    logger = Logger(output_root / "eval_logs.log")

    # --- Load model, config, and data
    model, config = load_ckpt(str(ckpt_path), device)
    seed = config.get("seed", 42)
    set_all_seeds(seed)

    logger.info(f"=" * 70)
    logger.info(f"Using device: {device}")
    logger.info(f"Experiment: {config['experiment_name']}")
    logger.info(f"Config: {config['config_file_path']}")
    logger.info(f"Output Root: {output_root}")
    logger.info(f"Data Dir: {data_dir}")
    logger.info(f"Plot Dir: {plot_dir}")
    logger.info(f"=" * 70)

    src_full_loader, tgt_full_loader, src_test_loader, tgt_test_loader = (
        build_all_dataloaders(config)
    )
    assert src_full_loader and tgt_full_loader, "Failed to load full data"
    assert src_test_loader and tgt_test_loader, "Failed to load test data"

    n_classes = len(CLASSES)
    class_labels = list(range(n_classes))

    # --- Extract features, and save features
    # extract_features function returns a dict with keys "z", "y", "logits", "preds"

    # extract features from full data
    logger.info("Extracting features from FULL source dataset...")
    src_full_feats = extract_features(
        model, src_full_loader, device, max_batches=None, feature_normalize=True
    )
    logger.info("Extracting features from FULL target dataset...")
    tgt_full_feats = extract_features(
        model, tgt_full_loader, device, max_batches=None, feature_normalize=True
    )

    # save features to npy files
    save_features(src_full_feats, data_dir, "src_full")
    save_features(tgt_full_feats, data_dir, "tgt_full")

    # extract features from test data
    logger.info("Extracting features from TEST source dataset...")
    src_test_feats = extract_features(
        model, src_test_loader, device, max_batches=None, feature_normalize=True
    )
    logger.info("Extracting features from TEST target dataset...")
    tgt_test_feats = extract_features(
        model, tgt_test_loader, device, max_batches=None, feature_normalize=True
    )
    save_features(src_test_feats, data_dir, "src_test")
    save_features(tgt_test_feats, data_dir, "tgt_test")
    logger.info(
        f"Full Data: {len(src_full_feats['z'])} src, {len(tgt_full_feats['z'])} tgt"
    )
    logger.info(
        f"Test Data: {len(src_test_feats['z'])} src, {len(tgt_test_feats['z'])} tgt"
    )

    # --- Compute performance metrics ( on TEST data)
    logger.info("Computing performance metrics on TEST sets...")
    src_metrics = {}
    tgt_metrics = {}
    domain_metrics = {}

    # ------------- SOURCE TEST METRICS -------------
    z_src_test = src_test_feats["z"]
    y_true_src = src_test_feats["y"]
    y_logits_src = src_test_feats["logits"]
    y_pred_src = src_test_feats["preds"]
    y_probas_src = F.softmax(torch.from_numpy(y_logits_src), dim=1).numpy()

    # calculate metrics:
    src_metrics["src_acc"] = skm.accuracy_score(y_true_src, y_pred_src)
    src_metrics["src_macro_f1"] = skm.f1_score(
        y_true_src, y_pred_src, average="macro", zero_division=0
    )
    src_metrics["src_recalls"] = skm.recall_score(
        y_true_src,
        y_pred_src,
        average=None,
        labels=class_labels,
        zero_division=0,
    )
    src_metrics["src_cm"] = skm.confusion_matrix(
        y_true_src, y_pred_src, labels=class_labels
    )
    src_metrics["src_ece"] = compute_ece(y_true_src, y_probas_src)
    # Binarize labels for AUC/AUPRC calculations
    y_true_src_bin = label_binarize(y_true_src, classes=class_labels)
    src_metrics["src_roc_auc_macro"] = skm.roc_auc_score(
        y_true_src_bin, y_probas_src, average="macro", multi_class="ovr"
    )
    src_metrics["src_roc_auc_weighted"] = skm.roc_auc_score(
        y_true_src_bin, y_probas_src, average="weighted", multi_class="ovr"
    )

    # --- AUPRC (Average Precision) ---
    src_metrics["src_auprc_macro"] = skm.average_precision_score(
        y_true_src_bin, y_probas_src, average="macro"
    )
    src_metrics["src_auprc_weighted"] = skm.average_precision_score(
        y_true_src_bin, y_probas_src, average="weighted"
    )

    # ------------- TARGET TEST METRICS -------------

    z_tgt_test = tgt_test_feats["z"]
    y_true_tgt = tgt_test_feats["y"]
    y_logits_tgt = tgt_test_feats["logits"]
    y_pred_tgt = tgt_test_feats["preds"]
    y_probas_tgt = F.softmax(torch.from_numpy(y_logits_tgt), dim=1).numpy()

    # calculate metrics:
    tgt_metrics["tgt_acc"] = skm.accuracy_score(y_true_tgt, y_pred_tgt)
    tgt_metrics["tgt_macro_f1"] = skm.f1_score(
        y_true_tgt, y_pred_tgt, average="macro", zero_division=0
    )
    tgt_metrics["tgt_recalls"] = skm.recall_score(
        y_true_tgt,
        y_pred_tgt,
        average=None,
        labels=class_labels,
        zero_division=0,
    )
    tgt_metrics["tgt_cm"] = skm.confusion_matrix(
        y_true_tgt, y_pred_tgt, labels=class_labels
    )
    tgt_metrics["tgt_ece"] = compute_ece(y_true_tgt, y_probas_tgt)
    # Binarize labels for AUC/AUPRC calculations
    y_true_tgt_bin = label_binarize(y_true_tgt, classes=class_labels)
    tgt_metrics["tgt_roc_auc_macro"] = skm.roc_auc_score(
        y_true_tgt_bin, y_probas_tgt, average="macro", multi_class="ovr"
    )
    tgt_metrics["tgt_roc_auc_weighted"] = skm.roc_auc_score(
        y_true_tgt_bin, y_probas_tgt, average="weighted", multi_class="ovr"
    )

    # --- AUPRC (Average Precision) ---
    tgt_metrics["tgt_auprc_macro"] = skm.average_precision_score(
        y_true_tgt_bin, y_probas_tgt, average="macro"
    )
    tgt_metrics["tgt_auprc_weighted"] = skm.average_precision_score(
        y_true_tgt_bin, y_probas_tgt, average="weighted"
    )

    # ------------- DOMAIN METRICS -------------
    logger.info("Computing domain alignment metrics on TEST dataset...")
    domain_metrics["test"] = compute_domain_metrics(
        src_z=src_test_feats["z"],
        tgt_z=tgt_test_feats["z"],
        src_y=src_test_feats["y"],
        tgt_y=tgt_test_feats["y"],
    )

    logger.info("Computing domain alignment metrics on FULL dataset...")
    domain_metrics["full"] = compute_domain_metrics(
        src_z=src_full_feats["z"],
        tgt_z=tgt_full_feats["z"],
        src_y=src_full_feats["y"],
        tgt_y=tgt_full_feats["y"],
    )

    logger.info("=" * 70)
    logger.info("--- Performance Metrics (Test Set) ---")
    logger.info(f"Source Accuracy: {src_metrics['src_acc'] * 100:.2f}%")
    logger.info(f"Target Accuracy: {tgt_metrics['tgt_acc'] * 100:.2f}%")
    logger.info(f"Source Macro F1: {src_metrics['src_macro_f1']:.4f}")
    logger.info(f"Target Macro F1: {tgt_metrics['tgt_macro_f1']:.4f}")
    logger.info(f"Source ECE: {src_metrics['src_ece']:.4f}")
    logger.info(f"Target ECE: {tgt_metrics['tgt_ece']:.4f}")
    # logger.info("--- Domain Alignment (Full Set) ---")
    # logger.info(f"MMD²: {domain_metrics['full']['mmd2']:.4f}")
    # logger.info(f"Sinkhorn Divergence: {domain_metrics['full']['sinkhorn_div']:.4f}")
    # logger.info(f"Domain AUC: {domain_metrics['full']['domain_auc']:.4f}")
    # logger.info(f"A-Distance: {domain_metrics['full']['a_distance']:.4f}")
    # logger.info(f"Silhouette Score: {domain_metrics['full']['silhouette_score']:.4f}")
    # logger.info(f"Avg Centroid Dist: {domain_metrics['full']['avg_centroid_distance']:.4f}")
    logger.info("--- Domain Alignment (Test Set) ---")
    # logger.info(f"MMD²: {domain_metrics['test']['mmd2']:.4f}")
    logger.info(f"Sinkhorn Divergence: {domain_metrics['test']['sinkhorn_div']:.4f}")
    logger.info(f"Domain AUC: {domain_metrics['test']['domain_auc']:.4f}")
    logger.info(f"A-Distance: {domain_metrics['test']['a_distance']:.4f}")
    logger.info(f"Silhouette Score: {domain_metrics['test']['silhouette_score']:.4f}")
    logger.info(
        f"Avg Centroid Dist: {domain_metrics['test']['avg_centroid_distance']:.4f}"
    )

    # ---- save all metrics
    save_metrics(src_metrics, output_root / "src_metrics.json")
    save_metrics(tgt_metrics, output_root / "tgt_metrics.json")
    save_metrics(domain_metrics, output_root / "domain_metrics.json")
    logger.info(f"See metrics at {output_root}")
    logger.info("=" * 70)
    # generate plots

    plot_confusion_matrices(
        src_metrics["src_cm"],
        tgt_metrics["tgt_cm"],
        src_metrics["src_acc"],
        tgt_metrics["tgt_acc"],
        titles=["Source Test Set", "Target Test Set"],
        save_path=plot_dir / "confusion_matrices_test.png",
    )

    # calculate full data cm
    full_src_cm = skm.confusion_matrix(
        src_full_feats["y"], src_full_feats["preds"], labels=class_labels
    )
    full_tgt_cm = skm.confusion_matrix(
        tgt_full_feats["y"], tgt_full_feats["preds"], labels=class_labels
    )
    full_src_acc = skm.accuracy_score(src_full_feats["y"], src_full_feats["preds"])
    full_tgt_acc = skm.accuracy_score(tgt_full_feats["y"], tgt_full_feats["preds"])

    plot_confusion_matrices(
        full_src_cm,
        full_tgt_cm,
        full_src_acc,
        full_tgt_acc,
        titles=["Source Full Set", "Target Full Set"],
        save_path=plot_dir / "confusion_matrices_full.png",
    )

    # Reliability diagrams
    plot_reliability_diagrams_side_by_side(
        y_true_src,
        y_probas_src,
        y_true_tgt,
        y_probas_tgt,
        save_path=plot_dir / "calibrations.png",
    )

    # ROC/PR (target)
    plot_roc_pr_curves(
        y_true_tgt,
        y_probas_tgt,
        class_names=CLASSES,
        title_prefix="Target",
        save_path=plot_dir / "roc_pr_target.png",
    )

    # Per-class recall bars
    plot_per_class_bars(
        src_metrics["src_recalls"],
        tgt_metrics["tgt_recalls"],
        class_names=CLASSES,
        metric_name="Recall",
        save_path=plot_dir / "per_class_recall_bars.png",
    )

    # Class distribution on test sets
    plot_class_distribution(
        src_test_feats["y"],
        tgt_test_feats["y"],
        class_names=CLASSES,
        normalize=True,
        save_path=plot_dir / "class_distribution_test.png",
    )

    # Latent space (true labels)
    # one clean 2-panel figure (true labels)
    plot_latent_space_two_panel(
        src_full_feats["z"],
        tgt_full_feats["z"],
        src_full_feats["y"],
        tgt_full_feats["y"],
        method=args.embed_method,
        save_path=plot_dir / f"latent_two_panel_{args.embed_method}_true.png",
        use_predictions=False,
    )

    # if you also want a predictions view, swap tgt_y to preds:
    plot_latent_space_two_panel(
        src_full_feats["z"],
        tgt_full_feats["z"],
        src_full_feats["preds"],
        tgt_full_feats["preds"],
        method=args.embed_method,
        save_path=plot_dir / f"latent_two_panel_{args.embed_method}_pred.png",
        use_predictions=True,
    )
    plot_domain_separation(
        src_full_feats["z"],
        tgt_full_feats["z"],
        method=args.embed_method,
        save_path=plot_dir / f"latent_domain_only_{args.embed_method}.png",
    )


if __name__ == "__main__":
    main()
