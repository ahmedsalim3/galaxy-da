import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Improved focal loss for multi-class classification with alpha weighting.
    Args:
        gamma: focusing parameter (higher = more focus on hard examples)
        weight: class weights tensor for cross-entropy
        alpha: alpha weights for focal loss (can be different from weight)
        reduction: 'mean'|'sum'|'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        alpha: torch.Tensor | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Compute log-probabilities and true-class probabilities (unweighted)
        log_probs = F.log_softmax(logits, dim=-1)
        true_log_probs = log_probs.gather(dim=1, index=targets.view(-1, 1)).squeeze(1)
        pt = true_log_probs.exp()

        # Standard focal loss term (per-sample)
        focal_term = (1.0 - pt).pow(self.gamma)
        loss = -focal_term * true_log_probs

        # Apply class weights (from imbalance handling) AFTER computing pt
        if self.weight is not None:
            loss = loss * self.weight[targets]

        # Apply alpha weighting if provided (supports tensor or scalar)
        if self.alpha is not None:
            if torch.is_tensor(self.alpha):
                loss = loss * self.alpha[targets]
            else:
                # scalar alpha
                loss = loss * float(self.alpha)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
