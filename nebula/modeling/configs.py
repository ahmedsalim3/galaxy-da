from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class BaseTrainerConfig:
    num_epochs: int = 6
    warmup_epochs: int = 0
    lr: float = 1e-4
    optimizer: str = "adamw"
    weight_decay: float = 1e-2
    max_norm: float = 10.0
    criterion: str = "cross_entropy"  # "cross_entropy" | "focal"

    use_class_weights: bool = False
    class_weight_method: str = (
        "effective"  # "balanced" | "effective" (only if use_class_weights=True)
    )
    class_weight_beta: float = (
        0.9999  # Beta for effective number (only if method="effective")
    )

    # If using focal loss
    focal_gamma: float = 2.0  # Focusing parameter
    # Accepts: None | float | list[float] | "class_weights"
    focal_alpha: Optional[Union[float, List[float], str]] = None  # Alpha for focal loss
    focal_reduction: str = "mean"  # "mean" | "sum" | "none"

    # Early stopping
    early_stopping_patience: Optional[int] = (
        None  # Number of epochs to wait before stopping
    )
    early_stopping_metric: str = "f1"  # "f1" or "accuracy" (evaluated on target data)


@dataclass
class NoDAConfig(BaseTrainerConfig):
    pass


@dataclass
class DAFixedLambdaConfig(BaseTrainerConfig):
    lambda_da: float = 0.1
    method: str = "sinkhorn"
    sinkhorn_blur: float = 10.0
    sinkhorn_p: int = 2


@dataclass
class DATrainableWeightsConfig(DAFixedLambdaConfig):
    eta_1_init: float = 0.1
    eta_2_init: float = 1.0


@dataclass
class DATrainableWeightsSigmaConfig(DATrainableWeightsConfig):
    """
    Config for trainable weights with sigma scheduling.

    Sigma scheduling controls the blur parameter in Sinkhorn distance:
    - As sigma -> 0: closer to OT_0 (more accurate, more expensive, potentially unstable)
    - As sigma -> infinity: closer to MMD (cheaper, less accurate)

    Default (exponential): sigma = 10 * 0.6^epoch (from tutorial)
    This anneals from high blur (MMD-like) early in training to lower blur (OT-like) later.

    IMPORTANT: For long training (>50 epochs), use milder decay (0.9-0.95) or set sigma_min_blur
    to prevent numerical underflow. Tutorial used only 6 epochs where 0.6^6 is safe.
    """

    # Schedule type
    sigma_schedule_type: str = (
        "exponential"  # "exponential" | "linear" | "cosine" | "step" | "polynomial" | "constant"
    )

    # Common parameters
    sigma_initial_blur: float = 10.0  # Initial sigma value
    sigma_final_blur: float = 1.0  # Final sigma value (for linear, cosine, polynomial)
    sigma_min_blur: float = 1e-2  # Minimum blur to prevent underflow (enforced floor)

    # Exponential decay: sigma = initial_blur * (decay_rate ** epoch)
    sigma_decay_rate: float = 0.6  # Tutorial uses 0.6 (or 0.4 in some examples)

    # Step decay: sigma = initial_blur * (step_gamma ** (epoch // step_size))
    sigma_step_size: int = 2  # Epochs per step
    sigma_step_gamma: float = 0.5  # Decay factor per step

    # Polynomial decay: sigma = (initial - final) * (1 - epoch/num_epochs)^power + final
    sigma_poly_power: float = (
        2.0  # Polynomial power (1.0 = linear, 2.0 = quadratic, etc.)
    )

    # Cosine annealing: smooth cosine curve from initial to final
    # No additional parameters needed beyond initial_blur and final_blur


@dataclass
class DAAdversarialConfig(BaseTrainerConfig):
    """
    Config for adversarial domain adaptation (DANN).

    Uses a domain classifier with gradient reversal to learn domain-invariant features.
    The feature extractor is trained to fool the domain classifier, encouraging
    representations that are discriminative for the task but indistinguishable
    across domains.

    Reference: "Domain-Adversarial Training of Neural Networks" (Ganin et al., 2016)
    https://arxiv.org/abs/1505.07818
    """

    # Gradient reversal strength
    lambda_grl: float = 0.25  # Strength of gradient reversal (tutorial uses 0.25)

    # Domain classifier architecture
    latent_dim: int = 6272  # Dimension of latent space from feature extractor
    domain_hidden_dim: int = 256  # Hidden dimension in domain classifier
    use_projection: bool = False  # Whether to use projection layer
    domain_projection_dim: int = 128  # Projection dimension if used
