from __future__ import annotations

import torch
import torch.nn as nn

from nebula.modeling.base import BaseTrainer
from nebula.modeling.configs import (DAAdversarialConfig, DAFixedLambdaConfig,
                                     DATrainableWeightsConfig)
from nebula.modeling.domain_losses import (DomainAdaptationLoss,
                                           TrainableLossWeights,
                                           get_sigma_schedule)
from nebula.modeling.domin_classifier import DomainClassifier

# -----------------------------------------------------------


class NoDATrainer(BaseTrainer):
    def compute_total_loss(self, ce_loss, *_):
        return ce_loss, torch.tensor(0.0, device=self.device)


class DAFixedLambdaTrainer(BaseTrainer):
    def __init__(self, model, config: DAFixedLambdaConfig, device):
        super().__init__(model, config, device)
        self.da_loss_fn = DomainAdaptationLoss(
            method=config.method,
            p=config.sinkhorn_p,
            blur=config.sinkhorn_blur,
        )

    def compute_total_loss(self, ce_loss, z_s, z_t):
        if z_t is None:
            return ce_loss, torch.tensor(0.0, device=self.device)
        da_loss = self.da_loss_fn(z_s, z_t)
        total_loss = ce_loss + self.config.lambda_da * da_loss
        return total_loss, da_loss


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

    def compute_total_loss(self, ce_loss, z_s, z_t):
        da_loss = (
            self.da_loss_fn(z_s, z_t)
            if z_t is not None
            else torch.tensor(0.0, device=self.device)
        )
        total_loss = self.trainable_weights(ce_loss, da_loss)
        return total_loss, da_loss

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

    def on_epoch_end(self, epoch: int, metrics):
        """Record current sigma (blur) into history for plotting later."""
        try:
            if hasattr(self, "current_blur"):
                if "sigma" not in self.history:
                    self.history["sigma"] = []
                self.history["sigma"].append(float(self.current_blur))
        except Exception:
            pass
        return super().on_epoch_end(epoch, metrics)


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

    def _build_optimizer(self):
        """Build optimizer and add domain classifier parameters."""
        optimizer = super()._build_optimizer()
        optimizer.add_param_group({"params": self.domain_classifier.parameters()})
        return optimizer

    def compute_total_loss(self, ce_loss, z_s, z_t):
        """
        Compute total loss including domain adversarial loss.

        Args:
            ce_loss: Classification loss on source domain
            z_s: Source latent representations
            z_t: Target latent representations

        Returns:
            total_loss, domain_loss
        """
        if z_t is None:
            return ce_loss, torch.tensor(0.0, device=self.device)

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
        domain_logits = self.domain_classifier(z_combined, self.config.lambda_grl)
        # -------------------------------------
        domain_loss = self.domain_criterion(domain_logits, domain_labels)

        # Total loss: classification + domain adversarial
        total_loss = ce_loss + domain_loss

        return total_loss, domain_loss
