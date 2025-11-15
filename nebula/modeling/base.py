from __future__ import annotations

import copy
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from nebula.analysis.evaluate import (compute_epoch_metrics, eval_accuracy,
                                      eval_f1_score)
from nebula.commons import Logger
from nebula.data.class_weights import compute_class_weights
from nebula.modeling.configs import BaseTrainerConfig
from nebula.modeling.early_stoppings import EarlyStopping
from nebula.modeling.focal_loss import FocalLoss

logger = Logger()


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: BaseTrainerConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.config = config
        # ----------------
        self.device = device

        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.best_loss = float("inf")
        self.best_model = None
        self.best_metric_model = None
        self.early_stopping = None
        self.history = {
            "train_loss": [],
            "ce_loss": [],
            "da_loss": [],
            "source_acc": [],
            "target_acc": [],
            "target_f1": [],
        }
        self.diag_history = {
            "mmd2": [],
            "sinkhorn_div": [],
            "domain_auc": [],
            "domain_acc": [],
            "proxy_a_distance": [],
            "target_acc": [],
            "target_macro_f1": [],
            "target_roc_auc_macro": [],
            "target_roc_auc_weighted": [],
            "target_auprc_macro": [],
            "target_auprc_weighted": [],
            "recall_elliptical": [],
            "recall_irregular": [],
            "recall_spiral": [],
            "confusion_matrix": [],
            "cmmd_elliptical": [],
            "cmmd_irregular": [],
            "cmmd_spiral": [],
            "domain_auc_elliptical": [],
            "domain_auc_irregular": [],
            "domain_auc_spiral": [],
            # OT transport mass entries (source→target)
            "ot_mass_elliptical_to_elliptical": [],
            "ot_mass_elliptical_to_irregular": [],
            "ot_mass_elliptical_to_spiral": [],
            "ot_mass_irregular_to_elliptical": [],
            "ot_mass_irregular_to_irregular": [],
            "ot_mass_irregular_to_spiral": [],
            "ot_mass_spiral_to_elliptical": [],
            "ot_mass_spiral_to_irregular": [],
            "ot_mass_spiral_to_spiral": [],
            # --- summary ---
            "ot_on_diag": [],
            "outlier_scores": [],
            "outlier_mask": [],
            "outlier_threshold": [],
            "target_acc_non_outlier": [],
            # data
            "embed": [],
            "src_z": [],
            "tgt_z": [],
            "src_y": [],
            "tgt_y_true": [],
            "tgt_y_pred": [],
            "tgt_logits": [],
            "tgt_probas": [],
        }

    def _build_criterion(self, class_weights=None) -> nn.Module:
        crit_name = self.config.criterion
        if crit_name == "cross_entropy":
            if class_weights is not None:
                logger.info(
                    f"Building CrossEntropyLoss with class weights: {class_weights}"
                )
            else:
                logger.info("Building CrossEntropyLoss without class weights")
            return nn.CrossEntropyLoss(weight=class_weights)

        elif crit_name == "focal":
            gamma = self.config.focal_gamma
            alpha = self.config.focal_alpha
            reduction = self.config.focal_reduction

            logger.info(
                f"Building FocalLoss with gamma={gamma}, reduction={self.config.focal_reduction}"
            )

            # Case 1: Use class_weights as alpha (recommended for imbalanced data)
            if isinstance(alpha, str) and alpha == "class_weights":
                if class_weights is not None:
                    logger.info(
                        f"Using computed class_weights as alpha: {class_weights}"
                    )
                    return FocalLoss(
                        gamma=gamma,
                        alpha=class_weights,
                        reduction=reduction,
                    )
                else:
                    logger.warning(
                        "focal_alpha='class_weights' but no class_weights computed! Using no alpha."
                    )
                    return FocalLoss(
                        gamma=gamma,
                        alpha=None,
                        reduction=reduction,
                    )

            # Case 2: Use custom alpha weights (no class_weights as weight)
            elif alpha is not None:
                alpha_param = (
                    torch.tensor(alpha, device=self.device)
                    if isinstance(alpha, (list, float))
                    else None
                )
                logger.info(f"Using custom alpha: {alpha}")
                if class_weights is not None:
                    logger.warning(
                        f"Ignoring class_weights ({class_weights}) when custom alpha is set to avoid double-weighting"
                    )
                return FocalLoss(
                    gamma=gamma,
                    alpha=alpha_param,
                    reduction=reduction,
                )

            # Case 3: No alpha, but use class_weights as weight (CE-style weighting in focal loss)
            else:
                if class_weights is not None:
                    logger.info(
                        f"Using class_weights as weight parameter (CE-style): {class_weights}"
                    )
                    return FocalLoss(
                        gamma=gamma,
                        weight=class_weights,
                        reduction=reduction,
                    )
                else:
                    logger.info("No alpha or class_weights - using standard focal loss")
                    return FocalLoss(
                        gamma=gamma,
                        reduction=reduction,
                    )

        else:
            raise ValueError(f"Unknown criterion: {crit_name}")

    def _compute_class_weights(self, source_loader) -> Optional[torch.Tensor]:
        if not self.config.use_class_weights:
            return None
        logger.info("Computing class weights from source training data...")

        all_labels = None
        try:
            ds = getattr(source_loader, "dataset", None)
            if ds is not None and hasattr(ds, "dataset") and hasattr(ds, "indices"):
                base_ds = ds.dataset
                if hasattr(base_ds, "df"):
                    base_labels_np = base_ds.df["label"].values
                    idxs = np.asarray(ds.indices)
                    if getattr(base_ds, "include_rotations", False):
                        base_idxs = (idxs // 8).astype(int)
                    else:
                        base_idxs = idxs.astype(int)
                    selected = base_labels_np[base_idxs]
                    all_labels = torch.from_numpy(selected).to(self.device)
        except Exception as e:
            logger.warning(f"Could not compute class weights from source loader: {e}")

        if all_labels is None:
            labels_list = []
            for _, labels in source_loader:
                labels_list.append(labels.to(self.device))
            all_labels = torch.cat(labels_list, dim=0)

        # Compute weights
        class_weights = compute_class_weights(
            all_labels,
            method=self.config.class_weight_method,
            beta=self.config.class_weight_beta,
            normalize=True,
            device=self.device,
        )
        # -------------------------------------

        num_classes = len(class_weights)
        class_counts = torch.zeros(num_classes, dtype=torch.float32)
        for c in range(num_classes):
            class_counts[c] = (all_labels == c).sum()

        logger.debug("Class distribution and weights:")
        for c in range(num_classes):
            logger.debug(
                f"  Class {c}: {int(class_counts[c])} samples → weight {class_weights[c]:.4f}"
            )

        # ---------------- DEBUG: confirm class index order ----------------
        logger.debug("Class index order used for weights and focal alpha:")
        try:
            from nebula.data.dataset import get_label_mappings

            _, idx2label = get_label_mappings()
            logger.debug("Class order and weighting summary:")
            for i, name in idx2label.items():
                w_val = (
                    float(class_weights[i])
                    if class_weights is not None and len(class_weights) > i
                    else None
                )
                alpha_val = None
                if hasattr(self.config, "focal_alpha") and isinstance(
                    self.config.focal_alpha, (list, tuple)
                ):
                    if len(self.config.focal_alpha) > i:
                        alpha_val = float(self.config.focal_alpha[i])
                logger.debug(
                    f"  idx {i}: {name:<12} → weight={w_val}, alpha={alpha_val}"
                )
        except Exception as e:
            logger.warning(f"Could not print class order/weights/alpha mapping: {e}")
        logger.debug("------------------------------------------------------------")

        return class_weights

    def _compute_class_scales(self, source_loader) -> Optional[torch.Tensor]:
        """
        Compute data-driven class scales to boost minority class predictions.

        Unlike class weights (used in loss), these scales are applied to logits
        to boost predictions for underrepresented classes.

        Formula: scale = sqrt(max_count / count) normalized so majority class = 1.0
        This is less aggressive than full inverse frequency but still helps minorities.

        Returns:
            torch.Tensor of shape [num_classes] with scales for each class.
        """
        logger.info("Computing class scales from source training data...")

        all_labels = None
        try:
            ds = getattr(source_loader, "dataset", None)
            if ds is not None and hasattr(ds, "dataset") and hasattr(ds, "indices"):
                base_ds = ds.dataset
                if hasattr(base_ds, "df"):
                    base_labels_np = base_ds.df["label"].values
                    idxs = np.asarray(ds.indices)
                    if getattr(base_ds, "include_rotations", False):
                        base_idxs = (idxs // 8).astype(int)
                    else:
                        base_idxs = idxs.astype(int)
                    selected = base_labels_np[base_idxs]
                    all_labels = torch.from_numpy(selected).to(self.device)
        except Exception as e:
            logger.warning(f"Could not extract labels efficiently: {e}")

        if all_labels is None:
            try:
                labels_list = []
                for _, labels in source_loader:
                    labels_list.append(labels.to(self.device))
                all_labels = torch.cat(labels_list, dim=0)
            except Exception as e:
                logger.warning(
                    f"Could not compute class scales from source loader: {e}"
                )
                return None

        # Compute class counts
        num_classes = int(all_labels.max().item()) + 1
        class_counts = torch.zeros(num_classes, dtype=torch.float32, device=self.device)
        for c in range(num_classes):
            class_counts[c] = (all_labels == c).sum()

        # Compute scales: sqrt(max_count / count)
        max_count = class_counts.max()
        scales = torch.sqrt(
            max_count / (class_counts + 1e-8)
        )  # add epsilon to avoid div by zero

        # Normalize so the majority class has scale 1.0
        scales = scales / scales.max()

        # Apply a moderate boost: scale = 1.0 + 0.5 * (scale - 1.0)
        # This makes the effect less aggressive (halfway between no boost and full sqrt boost)
        scales = 1.0 + 0.5 * (scales - 1.0)

        logger.debug("Class distribution and scales:")
        try:
            from nebula.data.dataset import get_label_mappings

            _, idx2label = get_label_mappings()
            for i in range(num_classes):
                name = idx2label.get(i, f"class_{i}")
                count = int(class_counts[i].item())
                pct = 100.0 * count / len(all_labels)
                scale = scales[i].item()
                logger.debug(
                    f"  {name:<12}: {count:>5} samples ({pct:>5.1f}%) → scale {scale:.3f}"
                )
        except Exception as e:
            logger.debug(f"Could not get class names: {e}")
            for i in range(num_classes):
                count = int(class_counts[i].item())
                pct = 100.0 * count / len(all_labels)
                scale = scales[i].item()
                logger.debug(
                    f"  Class {i}: {count:>5} samples ({pct:>5.1f}%) → scale {scale:.3f}"
                )

        return scales

    def _build_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        assert self.optimizer is not None, "Optimizer must be built before scheduler"

        scheduler_type = self.config.lr_scheduler
        if scheduler_type is None:
            return None

        scheduler_type = scheduler_type.lower()
        num_epochs = self.config.num_epochs
        logger.info(f"Building LR scheduler: {scheduler_type}")

        if scheduler_type == "cosine":
            # Cosine annealing:
            # lr decays following a cosine curve
            min_lr = getattr(self.config, "min_lr", 0.0)
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=num_epochs, eta_min=min_lr
            )
        elif scheduler_type == "step":
            # Step decay:
            # reduce lr by gamma every step_size epochs
            step_size = max(1, num_epochs // 3)  # Default: 3 steps
            gamma = 0.1
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif scheduler_type == "exponential":
            # Exponential decay:
            # lr = lr * gamma^epoch
            gamma = 0.95
            return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        else:
            raise ValueError(
                f"Unknown lr_scheduler: {scheduler_type}. "
                "Supported: None, 'cosine', 'step', 'exponential'"
            )

    def _get_optimizer_params(self):
        """Get parameters for optimizer. Override to add extra params in subclasses."""
        return self.model.parameters()

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer with model parameters."""
        opt_name = self.config.optimizer.lower()
        wd = self.config.weight_decay
        params = self._get_optimizer_params()

        if opt_name == "adamw":
            return optim.AdamW(params, lr=self.config.lr, weight_decay=wd)
        elif opt_name == "adam":
            return optim.Adam(params, lr=self.config.lr, weight_decay=wd)
        elif opt_name == "sgd":
            return optim.SGD(params, lr=self.config.lr, momentum=0.9, weight_decay=wd)
        # ----------------------------------------

        raise ValueError(f"Unknown optimizer: {opt_name}")

    def compute_total_loss(self, ce_loss, z_source, z_target, logits_target=None):
        """abstract method for subclasses to implement."""
        raise NotImplementedError

    def get_epoch_metrics(self, epoch: int, is_warmup: bool) -> Dict[str, float]:
        """abstract method for adding custom metrics at the end of each epoch."""
        return {}

    def on_epoch_start(self, epoch: int):
        """hook called at the start of each training epoch."""
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """hook called at the end of each training epoch."""
        pass

    def after_backward(self):
        """hook called after backward() but before optimizer.step()."""
        pass

    def _train_step(self, batch, target_loader_exists: bool) -> Dict[str, float]:
        """performs a single training step (forward, loss, backward) for one batch."""
        if target_loader_exists:
            (s_imgs, s_labels), (t_imgs, t_labels) = batch
            s_imgs, s_labels = s_imgs.to(self.device), s_labels.to(self.device)
            t_imgs, t_labels = t_imgs.to(self.device), t_labels.to(self.device)
            imgs = torch.cat([s_imgs, t_imgs])
        else:
            (s_imgs, s_labels) = batch
            s_imgs, s_labels = s_imgs.to(self.device), s_labels.to(self.device)
            imgs = s_imgs
            t_labels = None  # only for acc calculation

        # --- Forward Pass ---
        logits, z = self.model(imgs)
        mid = s_imgs.size(0)
        logits_s, z_s = logits[:mid], z[:mid]
        logits_t, z_t = (
            (logits[mid:], z[mid:]) if target_loader_exists else (None, None)
        )

        # --- Loss Computation ---
        ce_loss = self.criterion(logits_s, s_labels)
        total_loss, loss_dict = self.compute_total_loss(ce_loss, z_s, z_t, logits_t)

        assert isinstance(
            loss_dict, dict
        ), f"compute_total_loss must return (total_loss, loss_dict), got {type(loss_dict)}"

        step_losses = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                step_losses[key] = value.item() if value.numel() == 1 else 0.0
            else:
                step_losses[key] = float(value)

        # --- Backward Pass ---
        self.optimizer.zero_grad()
        total_loss.backward()
        if self.config.max_norm > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_norm)
        self.after_backward()  # Hook for post-backward operations (e.g., clamping trainable weights)
        self.optimizer.step()

        # --- Metrics for this step ---
        _, pred_s = logits_s.max(1)
        correct_s = pred_s.eq(s_labels).sum().item()
        total_s = s_labels.size(0)

        correct_t = 0
        total_t = 0
        if logits_t is not None and t_labels is not None:
            _, pred_t = logits_t.max(1)
            correct_t = pred_t.eq(t_labels).sum().item()
            total_t = t_labels.size(0)

        return {
            "total_loss": total_loss.item(),
            "ce_loss": ce_loss.item(),
            "correct_s": correct_s,
            "total_s": total_s,
            "correct_t": correct_t,
            "total_t": total_t,
            **step_losses,
        }

    def train_epoch(
        self, source_loader, target_loader=None, epoch=0
    ) -> Dict[str, float]:
        self.model.train()
        self.on_epoch_start(epoch)

        running_loss = running_ce = 0.0
        correct_s = correct_t = total_s = total_t = 0
        running_losses = {}  # all losses are saved in a dict

        loader = zip(source_loader, target_loader) if target_loader else source_loader
        total_steps = (
            min(len(source_loader), len(target_loader))
            if target_loader
            else len(source_loader)
        )
        pbar = tqdm(
            loader,
            total=total_steps,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            leave=False,
        )

        for batch in pbar:
            step_metrics = self._train_step(batch, target_loader is not None)

            running_loss += step_metrics["total_loss"]
            running_ce += step_metrics["ce_loss"]
            correct_s += step_metrics["correct_s"]
            total_s += step_metrics["total_s"]
            correct_t += step_metrics["correct_t"]
            total_t += step_metrics["total_t"]
            # accumulate all losses from loss_dict
            for key, value in step_metrics.items():
                if key not in [
                    "total_loss",
                    "ce_loss",
                    "correct_s",
                    "total_s",
                    "correct_t",
                    "total_t",
                ]:
                    if key not in running_losses:
                        running_losses[key] = 0.0
                    running_losses[key] += value

            # update progress bar ---
            pbar.set_postfix(
                {
                    "loss": f"{running_loss/(pbar.n+1):.4f}",
                    "src_acc": f"{100*correct_s/max(1,total_s):.2f}%",
                    "tgt_acc": (
                        f"{100*correct_t/max(1,total_t):.2f}%" if target_loader else "-"
                    ),
                }
            )

        avg_loss = running_loss / total_steps
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.best_model = copy.deepcopy(self.model.state_dict())

        ce_loss = running_ce / total_steps
        source_acc = 100 * correct_s / max(1, total_s)
        target_acc = 100 * correct_t / max(1, total_t)

        # average all accumulated losses
        avg_losses = {key: value / total_steps for key, value in running_losses.items()}

        metrics = {
            "train_loss": avg_loss,
            "ce_loss": ce_loss,
            "source_acc": source_acc,
            "target_acc": target_acc,
            **avg_losses,
        }

        self.on_epoch_end(epoch, metrics)
        return metrics

    def train(
        self,
        source_loader,
        target_loader=None,
        eval_interval: int = 0,
        diag_max_batches: int = 10,
    ):
        """
        Args:
            eval_interval: Diagnostics schedule. >0: every N epochs, 0: only at end (default)
        """
        # Build training components (criterion + optimizer + scheduler)
        if self.criterion is None:
            class_weights = self._compute_class_weights(source_loader)
            self.criterion = self._build_criterion(class_weights)
        if self.optimizer is None:
            self.optimizer = self._build_optimizer()
        if self.scheduler is None and self.config.lr_scheduler is not None:
            self.scheduler = self._build_scheduler()

        # Initialize class scales for models with learnable scaling
        if hasattr(self.model, "class_scales"):
            class_scales = self._compute_class_scales(source_loader)
            if class_scales is not None:
                with torch.no_grad():
                    self.model.class_scales.copy_(class_scales)
                logger.info(f"Initialized class scales: {class_scales.cpu().numpy()}")
            else:
                logger.warning(
                    "Could not compute class scales, using defaults (all 1.0)"
                )

        if self.config.early_stopping_patience is not None:
            if self.early_stopping is None:
                metric = self.config.early_stopping_metric
                # f1/accuracy are higher-is-better; losses are lower-is-better
                if metric in ("f1", "accuracy"):
                    mode = "max"
                elif metric in ("train_loss", "ce_loss", "da_loss"):
                    mode = "min"
                else:
                    raise ValueError(
                        f"Unknown early stopping metric: {metric}. "
                        "Use one of ['f1','accuracy','train_loss','ce_loss','da_loss']."
                    )

                self.early_stopping = EarlyStopping(
                    patience=self.config.early_stopping_patience,
                    metric_name=metric,
                    mode=mode,
                )
                logger.info(
                    f"Enabled early stopping criterion={metric}, "
                    f"mode={mode}, patience={self.config.early_stopping_patience}\n"
                )

        # Resume-aware epoch loop
        start_epoch = len(self.history.get("train_loss", []))
        for epoch in range(start_epoch, self.config.num_epochs):

            # --- Warmup Logic ---
            is_warmup = epoch < self.config.warmup_epochs
            # assign None as target_loader during warmup phase
            _target_loader = None if is_warmup else target_loader

            metrics = self.train_epoch(source_loader, _target_loader, epoch)

            # Get custom epoch metrics from trainer
            custom_metrics = self.get_epoch_metrics(epoch, is_warmup)
            metrics.update(custom_metrics)

            if target_loader is not None:
                target_f1 = eval_f1_score(self.model, target_loader, self.device)
                metrics["target_f1"] = target_f1
            else:
                metrics["target_f1"] = None

            # current epoch number (for padding new metrics)
            current_epoch_num = len(self.history.get("train_loss", []))

            # all keys that should be in history (union of existing and current metrics)
            all_keys = set(self.history.keys()) | set(metrics.keys())

            # For each key, append the appropriate value
            for k in all_keys:
                if k not in self.history:
                    # New metric: pad with None for all previous epochs
                    self.history[k] = [None] * current_epoch_num

                # Append current value (or None if not in metrics)
                value = metrics.get(k, None)
                self.history[k].append(value)

            # Step learning rate scheduler (skip during warmup)
            if self.scheduler is not None and not is_warmup:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                logger.debug(f"LR scheduler stepped. Current LR: {current_lr:.6f}")

            # Build log message with all metrics
            logger.info(
                "┌──────────────────────────────────────────────────────────────┐"
            )
            log_prefix = " Warmup" if is_warmup else " Epoch"

            # Line 1: Losses
            def fmt(v):
                return f"{v:.2e}" if v is not None and abs(v) < 0.0001 and v != 0.0 else f"{v:.4f}"
            loss_parts = [
                f"{log_prefix} {epoch+1}:",
                f"CE={fmt(metrics['ce_loss'])}",
                f"DA={fmt(metrics.get('da_loss', 0.0))}",
                f"Total={fmt(metrics['train_loss'])}",
            ]
            if "align_loss" in metrics:
                loss_parts.append(f"Align={fmt(metrics['align_loss'])}")
            if "entropy_loss" in metrics:
                loss_parts.append(f"Ent={fmt(metrics['entropy_loss'])}")
            logger.info("  ".join(loss_parts))

            acc_line = f"          SAcc={metrics['source_acc']:.2f}%, TAcc={metrics['target_acc']:.2f}%"
            if metrics.get('target_f1') is not None:
                acc_line += f", TF1={metrics['target_f1']:.4f}"
            logger.info(acc_line)

            # Line 3: Extra metrics (eta, lambda_grl, sigma)
            extra = []
            if "eta_1" in metrics and metrics["eta_1"] is not None:
                extra.append(f"η₁={fmt(metrics['eta_1'])}")
            if "eta_2" in metrics and metrics["eta_2"] is not None:
                extra.append(f"η₂={fmt(metrics['eta_2'])}")
            if "lambda_grl" in metrics:
                extra.append(f"λ_grl={fmt(metrics['lambda_grl'])}")
            if "sigma" in metrics:
                extra.append(f"σ={fmt(metrics['sigma'])}")

            if extra:
                logger.info("          " + ", ".join(extra))
            if self.early_stopping is not None and not is_warmup:
                metric = self.config.early_stopping_metric
                # If using target-based metrics, require target_loader
                if metric == "f1":
                    assert (
                        target_loader is not None
                    ), "early_stopping_metric 'f1' requires a target_loader"
                    metric_value = eval_f1_score(self.model, target_loader, self.device)
                    logger.info(f" Target F1: {fmt(metric_value)}")
                elif metric == "accuracy":
                    assert (
                        target_loader is not None
                    ), "early_stopping_metric 'accuracy' requires a target_loader"
                    metric_value = eval_accuracy(self.model, target_loader, self.device)
                    logger.info(f" Target Accuracy: {metric_value:.2f}%")
                elif metric == "train_loss":
                    metric_value = metrics["train_loss"]
                    logger.info(f" Train loss: {fmt(metric_value)}")
                elif metric == "ce_loss":
                    metric_value = metrics["ce_loss"]
                    logger.info(f" CE loss: {fmt(metric_value)}")
                elif metric == "da_loss":
                    metric_value = metrics["da_loss"]
                    logger.info(f" DA loss: {fmt(metric_value)}")
                else:
                    raise ValueError(f"Unknown early stopping metric: {metric}")

                old_best_score = self.early_stopping.best_score
                should_stop = self.early_stopping(metric_value)
                if (
                    old_best_score is None
                    or self.early_stopping.best_score != old_best_score
                ):
                    self.best_metric_model = copy.deepcopy(self.model.state_dict())
                    logger.info(
                        f" Saved best model based on {metric} "
                        f"(value: {fmt(metric_value)})"
                    )

                if should_stop:
                    logger.info(
                        "└──────────────────────────────────────────────────────────────┘"
                    )
                    break
            logger.info(
                "└──────────────────────────────────────────────────────────────┘"
            )

            # periodic detailed evaluation (optional, expensive)
            if (
                eval_interval > 0
                and diag_max_batches is not None
                and (epoch + 1) % eval_interval == 0
            ):
                diag = compute_epoch_metrics(
                    self.model,
                    eval_loader_source=source_loader,  # or separate eval loader
                    eval_loader_target=target_loader,
                    device=self.device,
                    max_batches=diag_max_batches,
                )
                for k, v in diag.items():
                    self.diag_history[k].append(v)
                logger.info(
                    "┌────────────────────────────────────────────────────────────────────────────────────┐"
                )
                logger.info(
                    f"  └─ Diagnostic: MMD²={diag['mmd2']:.4f}, "
                    f"Sinkhorn={diag['sinkhorn_div']:.4f}, "
                )
                logger.info(
                    f"     Domain AUC={diag['domain_auc']:.3f}, "
                    f"Domain Acc={diag['domain_acc']:.3f}"
                )
                logger.info(
                    f"     Target Macro F1={diag['target_macro_f1']:.2f}%, "
                    f"Target Acc={diag['target_acc']:.2f}%"
                )
                logger.info(
                    f"     Recall Elliptical={diag['recall_elliptical']:.2f}%, "
                    f"Recall Irregular={diag['recall_irregular']:.2f}%, "
                    f"Recall Spiral={diag['recall_spiral']:.2f}%"
                )
                logger.info(
                    "└────────────────────────────────────────────────────────────────────────────────────┘"
                )

        # end-of-training diagnostics (default when eval_interval == 0)
        if (
            diag_max_batches is not None
            and eval_interval == 0
            and target_loader is not None
        ):
            diag = compute_epoch_metrics(
                self.model,
                eval_loader_source=source_loader,
                eval_loader_target=target_loader,
                device=self.device,
                max_batches=diag_max_batches,
            )
            for k, v in diag.items():
                self.diag_history[k].append(v)
            logger.info(
                "┌────────────────────────────────────────────────────────────────────────────────────┐"
            )
            logger.info(
                f"  └─ Final Diagnostic: MMD²={diag['mmd2']:.4f}, "
                f"Sinkhorn={diag['sinkhorn_div']:.4f}, "
            )
            logger.info(
                f"     Domain AUC={diag['domain_auc']:.3f}, "
                f"Domain Acc={diag['domain_acc']:.3f}"
            )
            logger.info(
                f"     Target Macro F1={diag['target_macro_f1']:.2f}%, "
                f"Target Acc={diag['target_acc']:.2f}%"
            )
            logger.info(
                f"     Recall Elliptical={diag['recall_elliptical']:.2f}%, "
                f"Recall Irregular={diag['recall_irregular']:.2f}%, "
                f"Recall Spiral={diag['recall_spiral']:.2f}%"
            )
            logger.info(
                "└────────────────────────────────────────────────────────────────────────────────────┘"
            )

        if self.best_model:
            try:
                current_state = self.model.state_dict()
                best_state = self.best_model
                filtered_best = {
                    k: v
                    for k, v in best_state.items()
                    if k in current_state
                    and getattr(current_state[k], "shape", None)
                    == getattr(v, "shape", None)
                }
                current_state.update(filtered_best)
                self.model.load_state_dict(current_state, strict=True)
            except Exception as e:
                logger.warning(f"Failed to load best_model state_dict strictly: {e}")

        df_hist = pd.DataFrame(self.history)
        df_diag = pd.DataFrame(self.diag_history)

        if not df_hist.empty:
            df_hist.insert(0, "epoch", range(1, len(df_hist) + 1))
            if self.config.warmup_epochs > 0:
                warmup_epoch_numbers = range(1, self.config.warmup_epochs + 1)
                df_hist["epoch_warmup"] = df_hist["epoch"].isin(warmup_epoch_numbers)
            else:
                df_hist["epoch_warmup"] = False

        if not df_diag.empty:
            num_epochs = self.config.num_epochs
            if eval_interval > 0:
                epochs = list(
                    range(
                        eval_interval, eval_interval * len(df_diag) + 1, eval_interval
                    )
                )
                if epochs[-1] < num_epochs:
                    epochs[-1] = num_epochs
            else:
                epochs = [num_epochs]
            df_diag.insert(0, "epoch", epochs)
            if self.config.warmup_epochs > 0:
                warmup_epoch_numbers = range(1, self.config.warmup_epochs + 1)
                df_diag["epoch_warmup"] = df_diag["epoch"].isin(warmup_epoch_numbers)
            else:
                df_diag["epoch_warmup"] = False
            df_diag = df_diag.drop(columns=["confusion_matrix"])

        histories = {
            "history": self.history,
            "diag_history": self.diag_history,
            "history_df": df_hist,
            "diag_history_df": df_diag,
        }

        return histories

    def evaluate(self, loader):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                logits, _ = self.model(imgs)
                _, preds = logits.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        acc = 100.0 * correct / max(1, total)
        return acc

    def save_checkpoint(self, path: str, full_config: Optional[dict] = None):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint.
            full_config: The complete YAML config dict (from run_train.py)
                         to save for easy evaluation.
        """
        import os
        from pathlib import Path

        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "config": self.config,
            "full_config": full_config,
            "best_loss": self.best_loss,
            "history": self.history,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

        if self.best_metric_model is not None:
            best_model_path = str(
                Path(path).parent
                / f"{Path(path).stem}_best_{self.config.early_stopping_metric}.pt"
            )
            best_checkpoint = {
                "model_state_dict": self.best_metric_model,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "config": self.config,
                "full_config": full_config,
                "best_loss": self.best_loss,
                "history": self.history,
            }
            torch.save(best_checkpoint, best_model_path)
            logger.info(f"Best model checkpoint saved to {best_model_path}")

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.
        NOTE: This is for resuming training only.
        """
        # Use weights_only=False to allow loading custom config classes
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        # Build optimizer (subclasses will add their extra parameter groups)
        self.optimizer = self._build_optimizer()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Build and load scheduler if it exists
        if self.config.lr_scheduler is not None:
            self.scheduler = self._build_scheduler()
            if (
                "scheduler_state_dict" in checkpoint
                and checkpoint["scheduler_state_dict"] is not None
            ):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                logger.info(f"Scheduler state loaded from checkpoint")

        self.best_loss = checkpoint["best_loss"]
        self.history = checkpoint["history"]
        self.config = checkpoint["config"]

        logger.info(f"Checkpoint loaded from {path}")
