import math

import torch
import torch.nn as nn
from geomloss import SamplesLoss

from nebula.commons import Logger

logger = Logger()


class DomainAdaptationLoss:
    def __init__(self, method: str = "sinkhorn", **kwargs):
        self.method = method

        if method == "sinkhorn":
            # GeomLoss with Sinkhorn distance, p=2, blur ~ σ; tune as needed
            p = kwargs.get("p", 2)
            blur = kwargs.get("blur", 10)
            self.loss_fn = SamplesLoss("sinkhorn", p=p, blur=blur)
        elif method == "energy":
            # Energy distance (equivalent to Gaussian MMD with fixed bandwidth)
            p = kwargs.get("p", 2)
            self.loss_fn = SamplesLoss("energy", p=p)
        elif method == "mmd":
            blur = kwargs.get("blur", 10)
            self.loss_fn = SamplesLoss("gaussian", blur=blur)
        elif method == "none":
            self.loss_fn = None
        else:
            raise ValueError(f"Unknown DA method: {method}")

    def __call__(self, z_source: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
        if self.loss_fn is None:
            return torch.tensor(0.0, device=z_source.device)

        # Guard against NaN/Inf in latent features before computing Sinkhorn
        # This can happen with numerical instability or very small blur values
        if not torch.isfinite(z_source).all() or not torch.isfinite(z_target).all():
            # Return zero loss and warn
            logger.warning(
                "NaN/Inf detected in latent features before computing Sinkhorn. Returning zero loss."
            )
            return torch.tensor(0.0, device=z_source.device, requires_grad=True)

        return self.loss_fn(z_source, z_target)

    def update_blur(self, blur: float, min_blur: float = 1e-2):
        """
        Update the blur parameter for Sinkhorn/MMD loss.

        Args:
            blur: New blur value (will be clamped to >= min_blur)
            min_blur: Minimum blur to prevent numerical instability
        """
        if self.method in ["sinkhorn", "mmd"]:
            clamped_blur = max(blur, min_blur)

            p = 2 if self.method == "sinkhorn" else None
            if self.method == "sinkhorn":
                self.loss_fn = SamplesLoss("sinkhorn", p=p, blur=clamped_blur)
            else:
                self.loss_fn = SamplesLoss("gaussian", blur=clamped_blur)


class TrainableLossWeights(nn.Module):
    """Trainable loss weighting coefficients for multi-task learning.

    Trainable loss weighting coefficients for multi-task learning using homoscedastic uncertainty.
    This module implements the approach described in Kendall et al., 2018, "Multi-Task Learning Using Uncertainty to Weigh Losses".
    @ https://arxiv.org/pdf/1705.07115

    Instead of manually setting the relative weights of multiple loss functions, this method treats the weights as trainable parameters
    derived from the **log variance** of each task, which corresponds to the task-dependent (homoscedastic) uncertainty.

    The combined loss for two tasks (e.g., classification and domain adaptation) is computed as:

        Loss = (1 / (2 * eta_1^2)) * L_task1 + (1 / (2 * eta_2^2)) * L_task2 + log(eta_1 * eta_2)

    where `eta_1` and `eta_2` are the exponential of the learned log-variance parameters for each task.
        `L_task1` is the CE loss, `L_task2` is the DA loss

    Attributes:
        log_eta_1 (nn.Parameter): Log-variance parameter for task 1.
        log_eta_2 (nn.Parameter): Log-variance parameter for task 2.

    Methods:
        forward(ce_loss, da_loss): Computes the weighted loss combining two task losses.
        clamp_weights(...): Optional method to restrict learned weights to reasonable ranges.
    """

    def __init__(self, eta_1_init: float = 0.1, eta_2_init: float = 1.0):
        """
        Initialize trainable weights.

        Args:
            eta_1_init: Initial value for classification loss weight.
            eta_2_init: Initial value for DA loss weight.
        """
        super().__init__()
        self.log_eta_1 = nn.Parameter(torch.tensor(eta_1_init).log())
        self.log_eta_2 = nn.Parameter(torch.tensor(eta_2_init).log())

    @property
    def eta_1(self):
        """Get current eta_1 value."""
        return self.log_eta_1.exp()

    @property
    def eta_2(self):
        """Get current eta_2 value."""
        return self.log_eta_2.exp()

    def forward(self, ce_loss: torch.Tensor, da_loss: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted loss with trainable coefficients.

        Loss = (1/(2*eta_1^2)) * L_CE + (1/(2*eta_2^2)) * L_DA + log(eta_1 * eta_2)

        Args:
            ce_loss: Classification (cross-entropy) loss.
            da_loss: Domain adaptation loss.

        Returns:
            Combined weighted loss.
        """
        eta_1 = self.eta_1
        eta_2 = self.eta_2

        weighted_ce = ce_loss / (2 * eta_1**2)
        weighted_da = da_loss / (2 * eta_2**2)
        regularization = torch.log(eta_1 * eta_2)

        return weighted_ce + weighted_da + regularization

    def clamp_weights(self, eta_1_min: float = 1e-3, eta_2_min_factor: float = 0.25):
        """
        Clamp weight parameters to reasonable ranges.

        Args:
            eta_1_min: Minimum value for eta_1.
            eta_2_min_factor: eta_2_min = eta_2_min_factor * eta_1.
        """
        with torch.no_grad():
            eta_1_val = self.eta_1
            eta_2_val = self.eta_2

            # Clamp eta_1
            eta_1_val.clamp_(min=eta_1_min)

            # Clamp eta_2 relative to eta_1
            eta_2_min = eta_2_min_factor * eta_1_val
            eta_2_val.clamp_(min=eta_2_min)

            # Update log parameters
            self.log_eta_1.copy_(eta_1_val.log())
            self.log_eta_2.copy_(eta_2_val.log())


def get_sigma_schedule(epoch: int, config) -> float:
    """
    Compute sigma (blur) value for current epoch using various scheduling strategies.

    Sigma scheduling controls the blur parameter in Sinkhorn distance:
    - As sigma -> 0: closer to OT_0 (more accurate, more expensive, potentially unstable)
    - As sigma -> infinity: closer to MMD (cheaper, less accurate)

    Args:
        epoch: Current epoch number (0-indexed).
        config: Dataclass config object (DATrainableWeightsSigmaConfig) with schedule parameters.

    Returns:
        Blur value for the current epoch (always >= sigma_min_blur).

    Supported schedules:
        - exponential: sigma = initial * (decay_rate ** epoch)
        - linear: sigma = initial - (initial - final) * (epoch / num_epochs)
        - cosine: sigma = final + 0.5 * (initial - final) * (1 + cos(pi * epoch / num_epochs))
        - step: sigma = initial * (gamma ** (epoch // step_size))
        - polynomial: sigma = (initial - final) * (1 - epoch/num_epochs)^power + final
        - constant: sigma = initial (no decay)

    Note: All schedules enforce sigma >= sigma_min_blur to prevent numerical underflow.
    """
    # Get schedule parameters from dataclass config
    schedule_type = getattr(config, "sigma_schedule_type", "exponential")
    initial_blur = getattr(config, "sigma_initial_blur")
    final_blur = getattr(config, "sigma_final_blur")
    num_epochs = getattr(config, "num_epochs")
    min_blur = getattr(config, "sigma_min_blur", 0)

    if schedule_type == "exponential":
        decay_rate = getattr(config, "sigma_decay_rate")
        sigma = initial_blur * (decay_rate**epoch)

    elif schedule_type == "linear":
        progress = min(epoch / max(num_epochs - 1, 1), 1.0)
        sigma = initial_blur - (initial_blur - final_blur) * progress

    elif schedule_type == "cosine":
        progress = min(epoch / max(num_epochs - 1, 1), 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        sigma = final_blur + (initial_blur - final_blur) * cosine_decay

    elif schedule_type == "step":
        step_size = getattr(config, "sigma_step_size")
        step_gamma = getattr(config, "sigma_step_gamma")
        sigma = initial_blur * (step_gamma ** (epoch // step_size))

    elif schedule_type == "polynomial":
        power = getattr(config, "sigma_poly_power")
        progress = min(epoch / max(num_epochs - 1, 1), 1.0)
        sigma = (initial_blur - final_blur) * ((1 - progress) ** power) + final_blur

    elif schedule_type == "constant":
        sigma = initial_blur

    else:
        raise ValueError(
            f"Unknown sigma schedule type: {schedule_type}. "
            f"Supported: exponential, linear, cosine, step, polynomial, constant"
        )

    # Enforce minimum blur to prevent numerical underflow
    # This is IMPORTANT for long training runs with aggressive decay
    return max(sigma, min_blur)
