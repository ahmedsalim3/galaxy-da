from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics as skm
from sklearn.preprocessing import label_binarize

from nebula.analysis.alignment import (domain_probe_auc,
                                       domain_probe_auc_per_class,
                                       mmd2_unbiased_gaussian,
                                       sinkhorn_divergence,
                                       sinkhorn_plan_class_mass)
from nebula.analysis.embed import compute_embedding
from nebula.analysis.features import extract_features
from nebula.analysis.outliers import (class_conditional_mahalanobis_scores,
                                      knn_distance_scores,
                                      threshold_by_quantile)
from nebula.data.dataset import CLASSES


def compute_epoch_metrics(
    model: torch.nn.Module,
    eval_loader_source: torch.utils.data.DataLoader,
    eval_loader_target: torch.utils.data.DataLoader,
    class_names: Tuple[str, ...] = CLASSES,
    device: str = "cuda",
    max_batches: Optional[int] = None,
    blur: float = 0.05,
    ot_reg: float = 0.05,
    use_true_target_labels: bool = True,
    outlier_scoring: str | None = None,  # "cc-maha" | "knn" | None
    outlier_quantile: float = 0.95,
) -> Dict:
    """
    Compute comprehensive metrics for one epoch.
    This will report a post-hoc Gaussian MMD² with bandwidth set by the median heuristic;
    this diagnostic should not be backpropagated.
    """
    # Extract features
    src = extract_features(
        model,
        eval_loader_source,
        device,
        max_batches=max_batches,
        feature_normalize=True,
    )
    tgt = extract_features(
        model,
        eval_loader_target,
        device,
        max_batches=max_batches,
        feature_normalize=True,
    )

    # n_classes = len(class_names)
    # class_labels = list(range(n_classes))

    # Target performance
    y_true_t = tgt["y"]
    y_pred_t = tgt["preds"]
    y_logits_t = tgt["logits"]
    y_probas_t = (
        F.softmax(torch.from_numpy(y_logits_t), dim=1).numpy()
        if y_logits_t.size
        else np.array([])
    )

    acc = skm.accuracy_score(y_true_t, y_pred_t) if y_true_t.size else np.nan
    macro_f1 = (
        skm.f1_score(y_true_t, y_pred_t, average="macro", zero_division=0)
        if y_true_t.size
        else np.nan
    )
    recalls = (
        skm.recall_score(
            y_true_t,
            y_pred_t,
            average=None,
            labels=list(range(len(class_names))),
            zero_division=0,
        )
        if y_true_t.size
        else np.array([np.nan] * len(class_names))
    )
    cm = (
        skm.confusion_matrix(y_true_t, y_pred_t, labels=list(range(len(class_names))))
        if y_true_t.size
        else np.zeros((len(class_names), len(class_names)), int)
    )

    # Binarize labels for AUC/AUPRC calculations
    y_true_t_bin = (
        label_binarize(y_true_t, classes=list(range(len(class_names))))
        if y_true_t.size > 0
        else np.array([])
    )

    # --- ROC-AUC ---
    try:
        roc_auc_macro = (
            skm.roc_auc_score(
                y_true_t_bin, y_probas_t, average="macro", multi_class="ovr"
            )
            if y_true_t.size
            else np.nan
        )
        roc_auc_weighted = (
            skm.roc_auc_score(
                y_true_t_bin, y_probas_t, average="weighted", multi_class="ovr"
            )
            if y_true_t.size
            else np.nan
        )
    except ValueError:
        roc_auc_macro = np.nan
        roc_auc_weighted = np.nan

    # --- AUPRC (Average Precision) ---
    try:
        auprc_macro = (
            skm.average_precision_score(y_true_t_bin, y_probas_t, average="macro")
            if y_true_t.size
            else np.nan
        )
        auprc_weighted = (
            skm.average_precision_score(y_true_t_bin, y_probas_t, average="weighted")
            if y_true_t.size
            else np.nan
        )
    except ValueError:
        auprc_macro = np.nan
        auprc_weighted = np.nan

    # Alignment metrics
    # src["z"] and tgt["z"] are the L2-normalized embeddings
    mmd2 = mmd2_unbiased_gaussian(src["z"], tgt["z"])
    sink = sinkhorn_divergence(src["z"], tgt["z"], blur=blur, p=2)
    dom_auc, dom_acc = domain_probe_auc(src["z"], tgt["z"])
    proxy_a_dist = 2.0 * (1.0 - dom_acc) if dom_acc is not None else np.nan

    y_t_for_cc = y_true_t if (use_true_target_labels and y_true_t.size) else y_pred_t

    # Class-conditional MMD
    cmmd = {}
    for i, cname in enumerate(class_names):
        Xs = src["z"][src["y"] == i]
        Xt = tgt["z"][y_t_for_cc == i]
        cmmd[cname] = (
            mmd2_unbiased_gaussian(Xs, Xt) if (len(Xs) > 1 and len(Xt) > 1) else np.nan
        )

    # Classwise domain AUC
    dom_auc_cls = domain_probe_auc_per_class(
        src["z"], src["y"], tgt["z"], y_t_for_cc, classes=range(len(class_names))
    )

    # OT class mass
    ot_mass, ot_on_diag = sinkhorn_plan_class_mass(
        src["z"], tgt["z"], src["y"], y_t_for_cc, reg=ot_reg, n_classes=len(class_names)
    )

    # Outlier scoring on target features (optional)
    out_scores = None
    out_mask = None
    out_thr = None
    acc_non_outlier = np.nan

    if outlier_scoring is not None:
        if outlier_scoring == "cc-maha" and src["z"].size and src["y"].size:
            out_scores = class_conditional_mahalanobis_scores(
                src["z"], src["y"], tgt["z"]
            )
        elif outlier_scoring == "knn" and src["z"].size:
            out_scores = knn_distance_scores(src["z"], tgt["z"], k=5)

        if out_scores is not None and out_scores.size > 0:
            out_mask, out_thr = threshold_by_quantile(out_scores, q=outlier_quantile)

            if y_true_t.size and out_mask is not None:
                non_outlier_idx = ~out_mask
                if np.any(non_outlier_idx):
                    acc_non_outlier = skm.accuracy_score(
                        y_true_t[non_outlier_idx], y_pred_t[non_outlier_idx]
                    )

    # Embedding
    Z = np.vstack([src["z"], tgt["z"]])
    dom = np.hstack([np.zeros(len(src["z"]), int), np.ones(len(tgt["z"]), int)])
    lab = np.hstack([src["y"], y_t_for_cc])
    embed = compute_embedding(Z, dom, lab)

    metrics = {
        # --- Target Performance ---
        "target_acc": acc * 100 if acc == acc else np.nan,
        "target_macro_f1": macro_f1,
        "target_roc_auc_macro": roc_auc_macro,
        "target_roc_auc_weighted": roc_auc_weighted,
        "target_auprc_macro": auprc_macro,
        "target_auprc_weighted": auprc_weighted,
        "confusion_matrix": cm,
        # --- Domain Alignment ---
        "mmd2": mmd2,
        "sinkhorn_div": sink,
        "domain_auc": dom_auc,
        "domain_acc": dom_acc,
        "proxy_a_distance": proxy_a_dist,
        "ot_on_diag": ot_on_diag,
        # --- Outlier Analysis ---
        "outlier_scores": out_scores,
        "outlier_mask": out_mask,
        "outlier_threshold": out_thr,
        "target_acc_non_outlier": (
            acc_non_outlier * 100 if acc_non_outlier == acc_non_outlier else np.nan
        ),
        # --- Data for Plotting ---
        "embed": embed,
        "src_z": src["z"],
        "tgt_z": tgt["z"],
        "src_y": src["y"],
        "tgt_y_true": tgt["y"],
        "tgt_y_pred": tgt["preds"],
        "tgt_logits": tgt["logits"],
        "tgt_probas": y_probas_t,
    }

    # Add per-class recalls and domain AUCs as separate keys
    for i, r in zip(class_names, recalls):
        metrics[f"recall_{i}"] = r

    for i, val in dom_auc_cls.items():
        metrics[f"domain_auc_{class_names[i]}"] = val

    for i, val in cmmd.items():
        metrics[f"cmmd_{i}"] = val

    if ot_mass is not None and isinstance(ot_mass, np.ndarray):
        for i, src_name in enumerate(class_names):
            for j, tgt_name in enumerate(class_names):
                metrics[f"ot_mass_{src_name}_to_{tgt_name}"] = ot_mass[i, j]

    return metrics


@torch.no_grad()
def eval_accuracy(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: str
) -> float:
    """Quick accuracy evaluation."""
    model.eval()
    tot = 0
    correct = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        tot += y.size(0)
    return 100.0 * correct / max(1, tot)


@torch.no_grad()
def eval_f1_score(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> float:
    """Quick F1 score evaluation for imbalanced data."""
    model.eval()
    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model(x)
        pred = logits.argmax(1)
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    if len(all_preds) == 0:
        return 0.0

    return skm.f1_score(all_labels, all_preds, average="macro", zero_division=0)
