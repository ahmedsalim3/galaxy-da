import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer (GRL) for adversarial domain adaptation.

    During forward pass: passes input unchanged
    During backward pass: multiplies gradients by -lambda_grl

    This enables adversarial training where the feature extractor learns
    domain-invariant representations by fooling the domain classifier.

    Reference: "Domain-Adversarial Training of Neural Networks" (Ganin et al., 2016)
    https://arxiv.org/abs/1505.07818
    """

    @staticmethod
    def forward(ctx, x, lambda_grl):
        """Forward pass: store lambda and return input unchanged."""
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: reverse and scale gradient."""
        return -ctx.lambda_grl * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_grl: float = 1.0) -> torch.Tensor:
    """
    Apply gradient reversal to input tensor.

    Args:
        x: Input tensor
        lambda_grl: Scaling factor for gradient reversal (default: 1.0)

    Returns:
        Tensor with gradient reversal applied
    """
    return GradientReversalFunction.apply(x, lambda_grl)


class DomainClassifier(nn.Module):
    """
    Domain classifier for adversarial domain adaptation (DANN).

    Predicts whether a latent representation comes from source (1) or target (0) domain.
    Uses gradient reversal to encourage domain-invariant features.

    Architecture:
        - Optional projection layer (if latent_dim != hidden_dim)
        - Hidden layer with ReLU
        - Binary output (logits)
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        use_projection: bool = False,
        projection_dim: int = 128,
    ):
        """
        Initialize domain classifier.

        Args:
            latent_dim: Dimension of input latent representation
            hidden_dim: Dimension of hidden layer (default: 256)
            use_projection: Whether to use projection layer before classification
            projection_dim: Dimension of projection layer if used
        """
        super().__init__()
        self.use_projection = use_projection

        if use_projection:
            self.project = nn.Sequential(
                nn.Linear(latent_dim, projection_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
            )
            input_dim = projection_dim
        else:
            input_dim = latent_dim

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),  # Binary classification
        )

    def forward(self, z: torch.Tensor, lambda_grl: float = 1.0) -> torch.Tensor:
        """
        Forward pass with gradient reversal.

        Args:
            z: Latent representation [batch_size, latent_dim]
            lambda_grl: Gradient reversal strength (default: 1.0)

        Returns:
            Domain classification logits [batch_size, 1]
        """
        # Apply gradient reversal
        z_reversed = grad_reverse(z, lambda_grl)

        # Optional projection
        if self.use_projection:
            z_reversed = self.project(z_reversed)

        # Domain classification
        return self.classifier(z_reversed)
