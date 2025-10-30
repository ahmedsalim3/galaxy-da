from __future__ import annotations

import copy
from typing import Dict, Optional

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
from nebula.modeling.focal_loss import FocalLoss
from nebula.modeling.utils import EarlyStopping

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
        self.best_loss = float("inf")
        self.best_model = None
        self.early_stopping = None
        self.history = {
            "train_loss": [],
            "ce_loss": [],
            "da_loss": [],
            "source_acc": [],
            "target_acc": [],
            # Optional (only for some trainers)
            "eta_1": [],
            "eta_2": [],
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
        if class_weights is not None:
            logger.info(f"Adding class weights to {crit_name} loss function")
        if crit_name == "cross_entropy":
            return nn.CrossEntropyLoss(weight=class_weights)
        elif crit_name == "focal":
            if (
                isinstance(self.config.focal_alpha, str)
                and self.config.focal_alpha == "class_weights"
            ):
                return FocalLoss(
                    gamma=self.config.focal_gamma,
                    alpha=class_weights,
                    reduction=self.config.focal_reduction,
                )
            else:
                alpha_param = (
                    torch.tensor(self.config.focal_alpha, device=self.device)
                    if isinstance(self.config.focal_alpha, (list, float))
                    else None
                )
                return FocalLoss(
                    gamma=self.config.focal_gamma,
                    weight=class_weights,
                    alpha=alpha_param,
                    reduction=self.config.focal_reduction,
                )
        else:
            raise ValueError(f"Unknown criterion: {crit_name}")

    def _compute_class_weights(self, source_loader) -> torch.Tensor:
        if not self.config.use_class_weights:
            return None
        logger.info("Computing class weights from source training data...")

        all_labels = []
        for _, labels in source_loader:
            all_labels.append(labels)
        all_labels = torch.cat(all_labels, dim=0)

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

        logger.info("Class distribution and weights:")
        for c in range(num_classes):
            logger.info(
                f"  Class {c}: {int(class_counts[c])} samples → weight {class_weights[c]:.4f}"
            )

        return class_weights

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

    def compute_total_loss(self, ce_loss, z_source, z_target):
        """abstract method for subclasses to implement."""
        raise NotImplementedError

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
        total_loss, da_loss = self.compute_total_loss(ce_loss, z_s, z_t)

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
            "da_loss": da_loss.item() if da_loss is not None else 0.0,
            "correct_s": correct_s,
            "total_s": total_s,
            "correct_t": correct_t,
            "total_t": total_t,
        }

    def train_epoch(
        self, source_loader, target_loader=None, epoch=0
    ) -> Dict[str, float]:
        self.model.train()
        self.on_epoch_start(epoch)

        running_loss = running_ce = running_da = 0.0
        correct_s = correct_t = total_s = total_t = 0

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
            running_da += step_metrics["da_loss"]
            correct_s += step_metrics["correct_s"]
            total_s += step_metrics["total_s"]
            correct_t += step_metrics["correct_t"]
            total_t += step_metrics["total_t"]

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
        da_loss = running_da / total_steps
        source_acc = 100 * correct_s / max(1, total_s)
        target_acc = 100 * correct_t / max(1, total_t)

        metrics = {
            "train_loss": avg_loss,
            "ce_loss": ce_loss,
            "da_loss": da_loss,
            "source_acc": source_acc,
            "target_acc": target_acc,
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
        # Build training components (criterion + optimizer)
        if self.criterion is None:
            class_weights = self._compute_class_weights(source_loader)
            self.criterion = self._build_criterion(class_weights)
        if self.optimizer is None:
            self.optimizer = self._build_optimizer()

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
            metrics = self.train_epoch(source_loader, target_loader, epoch)
            for k, v in metrics.items():
                self.history[k].append(v)
            logger.info(
                "┌──────────────────────────────────────────────────────────────┐"
            )
            logger.info(
                f" Epoch {epoch+1}: CE={metrics['ce_loss']:.4f}, DA={metrics['da_loss']:.4f}, "
                f"SAcc={metrics['source_acc']:.2f}%, TAcc={metrics['target_acc']:.2f}%"
            )
            if hasattr(self, "trainable_weights"):
                try:
                    eta_1_val = float(self.trainable_weights.eta_1.item())
                    eta_2_val = float(self.trainable_weights.eta_2.item())
                    self.history["eta_1"].append(eta_1_val)
                    self.history["eta_2"].append(eta_2_val)
                    logger.info(
                        f"           eta_1={eta_1_val:.4f}, eta_2={eta_2_val:.4f}"
                    )
                except Exception:
                    self.history["eta_1"].append(None)
                    self.history["eta_2"].append(None)
            else:
                self.history["eta_1"].append(None)
                self.history["eta_2"].append(None)
            if hasattr(self, "current_blur"):
                try:
                    logger.info(f"           sigma={float(self.current_blur):.4f}")
                except Exception:
                    pass

            if self.early_stopping is not None:
                metric = self.config.early_stopping_metric
                # If using target-based metrics, require target_loader
                if metric == "f1":
                    assert (
                        target_loader is not None
                    ), "early_stopping_metric 'f1' requires a target_loader"
                    metric_value = eval_f1_score(self.model, target_loader, self.device)
                    logger.info(f" Target F1: {metric_value:.4f}")
                elif metric == "accuracy":
                    assert (
                        target_loader is not None
                    ), "early_stopping_metric 'accuracy' requires a target_loader"
                    metric_value = eval_accuracy(self.model, target_loader, self.device)
                    logger.info(f" Target Accuracy: {metric_value:.2f}%")
                elif metric == "train_loss":
                    metric_value = metrics["train_loss"]
                    logger.info(f" Train loss: {metric_value:.4f}")
                elif metric == "ce_loss":
                    metric_value = metrics["ce_loss"]
                    logger.info(f" CE loss: {metric_value:.4f}")
                elif metric == "da_loss":
                    metric_value = metrics["da_loss"]
                    logger.info(f" DA loss: {metric_value:.4f}")
                else:
                    raise ValueError(f"Unknown early stopping metric: {metric}")

                if self.early_stopping(metric_value):
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
            self.model.load_state_dict(self.best_model)

        df_hist = pd.DataFrame(self.history)
        df_diag = pd.DataFrame(self.diag_history)

        if not df_hist.empty:
            df_hist.insert(0, "epoch", range(1, len(df_hist) + 1))
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

        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "full_config": full_config,
            "best_loss": self.best_loss,
            "history": self.history,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

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
        self.best_loss = checkpoint["best_loss"]
        self.history = checkpoint["history"]
        self.config = checkpoint["config"]

        logger.info(f"Checkpoint loaded from {path}")
