from __future__ import annotations

import torch
import torch.nn as nn

from nebula.modeling.base import BaseTrainer
from nebula.modeling.configs import (DAAdversarialConfig, DAFixedLambdaConfig,
                                     DATrainableWeightsConfig)
from nebula.modeling.domain_losses import (DomainAdaptationLoss,
                                           TrainableLossWeights)
from nebula.modeling.domin_classifier import DomainClassifier
from nebula.modeling.ota_loss import OTAlignmentLoss
from nebula.modeling.schedulers import (get_lambda_grl_schedule,
                                        get_sigma_schedule)

# -----------------------------------------------------------


class NoDATrainer(BaseTrainer):
    def compute_total_loss(self, ce_loss, *_, **__):
        return ce_loss, {"da_loss": torch.tensor(0.0, device=self.device)}


class DAFixedLambdaTrainer(BaseTrainer):
    def __init__(self, model, config: DAFixedLambdaConfig, device):
        super().__init__(model, config, device)
        self.da_loss_fn = DomainAdaptationLoss(
            method=config.method,
            p=config.sinkhorn_p,
            blur=config.sinkhorn_blur,
        )

        self.lambda_ot = float(getattr(config, "lambda_ot", 0.0))
        self.ot_da_loss = None
        if self.lambda_ot > 0.0:
            ot_lambda_ot = float(getattr(config, "ot_lambda_ot", 0.25))
            ot_lambda_match = float(getattr(config, "ot_lambda_match", 0.5))
            ot_lambda_topk = float(getattr(config, "ot_lambda_topk", 0.05))
            ot_epsilon = float(getattr(config, "ot_epsilon", 0.05))
            ot_n_iter = int(getattr(config, "ot_n_iter", 50))
            match_epsilon = float(getattr(config, "match_epsilon", 0.05))
            match_n_iter = int(getattr(config, "match_n_iter", 20))
            topk = int(getattr(config, "ot_topk", 5))

            self.ot_da_loss = OTAlignmentLoss(
                lambda_ot=ot_lambda_ot,
                lambda_match=ot_lambda_match,
                lambda_topk=ot_lambda_topk,
                topk=topk,
                ot_epsilon=ot_epsilon,
                ot_n_iter=ot_n_iter,
                match_epsilon=match_epsilon,
                match_n_iter=match_n_iter,
            )

    def compute_total_loss(self, ce_loss, z_s, z_t, logits_target=None):
        if z_t is None:
            return ce_loss, {"da_loss": torch.tensor(0.0, device=self.device)}

        z_s_norm = torch.nn.functional.normalize(z_s, p=2, dim=1)
        z_t_norm = torch.nn.functional.normalize(z_t, p=2, dim=1)

        # Geomloss-based DA loss (Sinkhorn/Energy/MMD)
        da_loss = self.da_loss_fn(z_s_norm, z_t_norm)

        # OT-based alignment loss
        align_loss = torch.tensor(0.0, device=self.device)
        if self.ot_da_loss is not None:
            align_loss = self.ot_da_loss(z_s_norm, z_t_norm)

        entropy_loss = torch.tensor(0.0, device=self.device)
        lambda_entropy = getattr(self.config, "lambda_entropy", 0.0)
        if logits_target is not None and lambda_entropy > 0:
            probs_target = torch.nn.functional.softmax(logits_target, dim=1)
            # Compute entropy: -sum(p * log(p + eps)) for numerical stability
            eps = 1e-8
            entropy = -(probs_target * torch.log(probs_target + eps)).sum(dim=1)
            # AVG entropy across batch
            entropy_loss = entropy.mean()

        # Combined loss: classification + geomloss DA + entropy minimization + OT alignment
        total_loss = (
            ce_loss
            + self.config.lambda_da * da_loss
            + lambda_entropy * entropy_loss
            + self.lambda_ot * align_loss
        )

        # Return all losses for logging
        loss_dict = {"da_loss": da_loss}
        if lambda_entropy > 0:
            loss_dict["entropy_loss"] = entropy_loss
        if self.ot_da_loss is not None:
            loss_dict["align_loss"] = align_loss

        return total_loss, loss_dict


class DATrainableWeightsTrainer(DAFixedLambdaTrainer):
    def __init__(self, model, config: DATrainableWeightsConfig, device):
        super().__init__(model, config, device)
        self.trainable_weights = TrainableLossWeights(
            eta_1_init=config.eta_1_init,
            eta_2_init=config.eta_2_init,
        ).to(device)

    def _build_optimizer(self):
        """Build optimizer and add trainable weights parameters."""
        optimizer = super()._build_optimizer()
        optimizer.add_param_group({"params": self.trainable_weights.parameters()})
        return optimizer

    def compute_total_loss(self, ce_loss, z_s, z_t, logits_target=None):
        if z_t is None:
            return ce_loss, {"da_loss": torch.tensor(0.0, device=self.device)}

        # Normalize features
        z_s_norm = torch.nn.functional.normalize(z_s, p=2, dim=1)
        z_t_norm = torch.nn.functional.normalize(z_t, p=2, dim=1)

        da_loss = self.da_loss_fn(z_s_norm, z_t_norm)

        # Optional OT-based alignment loss
        align_loss = torch.tensor(0.0, device=self.device)
        if self.ot_da_loss is not None:
            align_loss = self.ot_da_loss(z_s_norm, z_t_norm)

        # Entropy minimization on target predictions
        entropy_loss = torch.tensor(0.0, device=self.device)
        lambda_entropy = getattr(self.config, "lambda_entropy", 0.0)
        if logits_target is not None and lambda_entropy > 0:
            # Compute softmax probabilities
            probs_target = torch.nn.functional.softmax(logits_target, dim=1)
            # Compute entropy: -sum(p * log(p + eps)) for numerical stability
            eps = 1e-8
            entropy = -(probs_target * torch.log(probs_target + eps)).sum(dim=1)
            # Average entropy across batch
            entropy_loss = entropy.mean()

        # Trainable weights combine CE and DA loss
        # Entropy and alignment losses are added separately (not through trainable weights)
        base_total = self.trainable_weights(ce_loss, da_loss)
        total_loss = (
            base_total + lambda_entropy * entropy_loss + self.lambda_ot * align_loss
        )

        # Return all losses for logging
        loss_dict = {"da_loss": da_loss}
        if lambda_entropy > 0:
            loss_dict["entropy_loss"] = entropy_loss
        if self.ot_da_loss is not None:
            loss_dict["align_loss"] = align_loss

        return total_loss, loss_dict

    def get_epoch_metrics(self, epoch: int, is_warmup: bool):
        """Extract trainable weight values (eta_1, eta_2) for logging."""
        metrics = {}
        if hasattr(self, "trainable_weights") and not is_warmup:
            try:
                metrics["eta_1"] = float(self.trainable_weights.eta_1.item())
                metrics["eta_2"] = float(self.trainable_weights.eta_2.item())
            except Exception:
                metrics["eta_1"] = None
                metrics["eta_2"] = None
        return metrics

    def after_backward(self):
        """
        Hook called after backward() but before optimizer.step().
        """
        # Clamp eta values AFTER backward but BEFORE step
        self.trainable_weights.clamp_weights()


class DATrainableWeightsSigmaTrainer(DATrainableWeightsTrainer):
    """
    Trainer that anneals the Sinkhorn blur parameter (sigma)
    using the on_epoch_start hook.
    """

    def on_epoch_start(self, epoch: int):
        """Update Sinkhorn blur parameter based on the epoch schedule."""
        # We pass self.config (which is DATrainableWeightsSigmaConfig)
        # to the scheduler function.
        new_blur = get_sigma_schedule(epoch, self.config)
        min_blur = getattr(self.config, "sigma_min_blur", 1e-2)
        self.da_loss_fn.update_blur(new_blur, min_blur=min_blur)
        self.current_blur = float(new_blur)

        super().on_epoch_start(epoch)

    def get_epoch_metrics(self, epoch: int, is_warmup: bool):
        """Record current sigma (blur) into history for plotting later."""
        metrics = super().get_epoch_metrics(epoch, is_warmup)
        if hasattr(self, "current_blur") and not is_warmup:
            try:
                metrics["sigma"] = float(self.current_blur)
            except Exception:
                pass
        return metrics


class DAAdversarialTrainer(BaseTrainer):
    """
    Adversarial Domain Adaptation Trainer (DANN).

    Uses gradient reversal and a domain classifier to learn domain-invariant features.
    The model learns to classify the source domain correctly while the feature extractor
    is trained to fool the domain classifier (via gradient reversal).

    Loss: L_total = L_CE + L_domain

    where L_domain uses BCEWithLogitsLoss to distinguish source (1) from target (0).

    ...
    """

    def __init__(self, model, config: DAAdversarialConfig, device):
        super().__init__(model, config, device)

        # Domain classifier for adversarial training
        self.domain_classifier = DomainClassifier(
            latent_dim=config.latent_dim,
            hidden_dim=config.domain_hidden_dim,
            use_projection=config.use_projection,
            projection_dim=config.domain_projection_dim,
        ).to(device)

        # Domain classification loss (binary cross-entropy)
        self.domain_criterion = nn.BCEWithLogitsLoss()

        # Initialize current lambda_grl
        # (will be updated in on_epoch_start if scheduling is used)
        self.current_lambda_grl = config.lambda_grl

        # Optional OT-based alignment loss
        self.lambda_ot = float(getattr(config, "lambda_ot", 0.0))
        self.ot_da_loss = None
        if self.lambda_ot > 0.0:
            ot_lambda_ot = float(getattr(config, "ot_lambda_ot", 0.25))
            ot_lambda_match = float(getattr(config, "ot_lambda_match", 0.5))
            ot_lambda_topk = float(getattr(config, "ot_lambda_topk", 0.05))
            ot_epsilon = float(getattr(config, "ot_epsilon", 0.05))
            ot_n_iter = int(getattr(config, "ot_n_iter", 50))
            match_epsilon = float(getattr(config, "match_epsilon", 0.05))
            match_n_iter = int(getattr(config, "match_n_iter", 20))
            topk = int(getattr(config, "ot_topk", 5))

            self.ot_da_loss = OTAlignmentLoss(
                lambda_ot=ot_lambda_ot,
                lambda_match=ot_lambda_match,
                lambda_topk=ot_lambda_topk,
                topk=topk,
                ot_epsilon=ot_epsilon,
                ot_n_iter=ot_n_iter,
                match_epsilon=match_epsilon,
                match_n_iter=match_n_iter,
            )

    def _build_optimizer(self):
        """Build optimizer and add domain classifier parameters."""
        optimizer = super()._build_optimizer()
        optimizer.add_param_group({"params": self.domain_classifier.parameters()})
        return optimizer

    def compute_total_loss(self, ce_loss, z_s, z_t, logits_target=None):
        """
        Compute total loss including domain adversarial loss, optional OT-based
        alignment loss, and entropy minimization.

        Args:
            ce_loss: Classification loss on source domain.
            z_s: Source latent representations.
            z_t: Target latent representations.
            logits_target: Target logits (for entropy minimization).

        Returns:
            tuple: (total_loss, loss_dict) where loss_dict contains all component losses
        """
        if z_t is None:
            return ce_loss, {"da_loss": torch.tensor(0.0, device=self.device)}

        # Normalize latent features
        z_s = torch.nn.functional.normalize(z_s, p=2, dim=1)
        z_t = torch.nn.functional.normalize(z_t, p=2, dim=1)

        # Create domain labels: source=1, target=0
        batch_size_s = z_s.size(0)
        batch_size_t = z_t.size(0)
        domain_labels = torch.cat(
            [
                torch.ones(batch_size_s, 1, device=self.device),
                torch.zeros(batch_size_t, 1, device=self.device),
            ],
            dim=0,
        )

        # Concatenate source and target latents
        z_combined = torch.cat([z_s, z_t], dim=0)

        # Domain classification with gradient reversal 
        # (use scheduled lambda if available)
        lambda_grl = getattr(self, "current_lambda_grl", self.config.lambda_grl)
        domain_logits = self.domain_classifier(z_combined, lambda_grl)
        domain_loss = self.domain_criterion(domain_logits, domain_labels)

        # Optional OT-based alignment loss
        align_loss = torch.tensor(0.0, device=self.device)
        if self.ot_da_loss is not None:
            align_loss = self.ot_da_loss(z_s, z_t)

        entropy_loss = torch.tensor(0.0, device=self.device)
        if logits_target is not None and self.config.lambda_entropy > 0:
            probs_target = torch.nn.functional.softmax(logits_target, dim=1)
            eps = 1e-8
            entropy = -(probs_target * torch.log(probs_target + eps)).sum(dim=1)
            entropy_loss = entropy.mean()

        # Total loss: classification + domain adversarial + entropy minimization + OT alignment
        total_loss = (
            ce_loss
            + domain_loss
            + self.config.lambda_entropy * entropy_loss
            + self.lambda_ot * align_loss
        )

        # Return all losses for logging
        loss_dict = {"da_loss": domain_loss}
        if self.config.lambda_entropy > 0:
            loss_dict["entropy_loss"] = entropy_loss
        if self.ot_da_loss is not None:
            loss_dict["align_loss"] = align_loss

        return total_loss, loss_dict

    def on_epoch_start(self, epoch: int):
        """Update lambda_grl based on schedule if configured."""
        if (
            hasattr(self.config, "lambda_grl_schedule")
            and self.config.lambda_grl_schedule is not None
        ):
            # Pass epoch (which includes warmup)
            # get_lambda_grl_schedule handles warmup adjustment
            self.current_lambda_grl = get_lambda_grl_schedule(epoch, self.config)
        else:
            self.current_lambda_grl = self.config.lambda_grl

        super().on_epoch_start(epoch)

    def get_epoch_metrics(self, epoch: int, is_warmup: bool):
        """Extract lambda_grl."""
        metrics = {}
        if hasattr(self, "current_lambda_grl") and not is_warmup:
            try:
                metrics["lambda_grl"] = float(self.current_lambda_grl)
            except Exception:
                pass
        return metrics
