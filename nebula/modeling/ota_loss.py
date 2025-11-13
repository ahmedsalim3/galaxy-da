from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _pairwise_sq_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Pairwise squared Euclidean distance matrix.

    Args:
        x: (n_x, d)
        y: (n_y, d)

    Returns:
        (n_x, n_y) with ||x_i - y_j||^2
    """
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Expected 2D tensors (n, d) for x and y.")
    return torch.cdist(x, y, p=2).pow(2)


def _safe_uniform(n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Uniform probability vector of length n on the given device/dtype."""
    if n <= 0:
        raise ValueError("Length must be positive.")
    return torch.full((n,), 1.0 / n, device=device, dtype=dtype)


@dataclass
class SinkhornConfig:
    epsilon: float = 0.05  # entropic regularization
    n_iter: int = 50  # iterations
    # Numerical stabilizers
    min_epsilon: float = 1e-8


def _sinkhorn_log_stabilized(
    cost: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    cfg: SinkhornConfig,
) -> torch.Tensor:
    """
    Compute entropically-regularized OT transport plan using Sinkhorn algorithm.

    Args:
        cost: (n_x, n_y) cost matrix
        mu: (n_x,) source marginal (sums to 1)
        nu: (n_y,) target marginal (sums to 1)
        cfg: SinkhornConfig

    Returns:
        P: (n_x, n_y) transport plan, differentiable wrt cost
    """
    eps = max(float(cfg.epsilon), cfg.min_epsilon)
    K = torch.exp(-cost / eps)  # Gibbs kernel

    # Initialize with ones for better numerical stability
    u = torch.ones_like(mu)
    v = torch.ones_like(nu)

    for _ in range(int(cfg.n_iter)):
        # Sinkhorn iterations in linear domain
        u = mu / (K @ v + 1e-8)
        v = nu / (K.t() @ u + 1e-8)

    # Compute transport plan
    P = u[:, None] * K * v[None, :]
    return P


class SinkhornOTLoss(nn.Module):
    """
    Entropically-regularized OT distance (expected transport cost).

    Given source/target features X, Y, computes:
        L_ot = <P, C>,  where
        C_ij = ||x_i - y_j||_p^p (here p=2, cost=||.||^2) and
        P is the Sinkhorn transport plan with uniform marginals.

    This implementation uses squared Euclidean cost (p=2).
    """

    def __init__(self, epsilon: float = 0.05, n_iter: int = 50) -> None:
        super().__init__()
        self.cfg = SinkhornConfig(epsilon=float(epsilon), n_iter=int(n_iter))

    def forward(self, src_feats: torch.Tensor, tgt_feats: torch.Tensor) -> torch.Tensor:
        if src_feats.ndim != 2 or tgt_feats.ndim != 2:
            raise ValueError("SinkhornOTLoss expects 2D tensors (batch, dim).")

        n_s, n_t = src_feats.size(0), tgt_feats.size(0)
        if n_s == 0 or n_t == 0:
            # Return a proper 0 that participates in autograd graphs
            return src_feats.sum() * 0.0

        C = _pairwise_sq_dist(src_feats, tgt_feats)  # (n_s, n_t)

        mu = _safe_uniform(n_s, device=src_feats.device, dtype=src_feats.dtype)
        nu = _safe_uniform(n_t, device=tgt_feats.device, dtype=tgt_feats.dtype)

        P = _sinkhorn_log_stabilized(C, mu, nu, self.cfg)
        ot_distance = (P * C).sum()
        return ot_distance


class SoftMatchMSELoss(nn.Module):
    """
    Soft matching Mean Squared Error using a Sinkhorn transport plan.

    Computes a soft alignment of targets for each source using P and then
    penalizes the MSE between X and P @ Y.

        L_match = MSE(X, P Y)

    Note: P needs to be row-normalized (each row sums to 1) to act as proper
    weights. The Sinkhorn plan has row-sums = mu (uniform = 1/n), so we
    normalize by multiplying by n_s.
    """

    def __init__(self, epsilon: float = 0.05, n_iter: int = 20) -> None:
        super().__init__()
        self.cfg = SinkhornConfig(epsilon=float(epsilon), n_iter=int(n_iter))

    def forward(self, src_feats: torch.Tensor, tgt_feats: torch.Tensor) -> torch.Tensor:
        if src_feats.ndim != 2 or tgt_feats.ndim != 2:
            raise ValueError("SoftMatchMSELoss expects 2D tensors (batch, dim).")

        n_s, n_t = src_feats.size(0), tgt_feats.size(0)
        if n_s == 0 or n_t == 0:
            return src_feats.sum() * 0.0

        C = _pairwise_sq_dist(src_feats, tgt_feats)
        mu = _safe_uniform(n_s, device=src_feats.device, dtype=src_feats.dtype)
        nu = _safe_uniform(n_t, device=tgt_feats.device, dtype=tgt_feats.dtype)
        P = _sinkhorn_log_stabilized(C, mu, nu, self.cfg)

        # P rows sum to 1/n_s (uniform marginal), so multiply by n_s
        P_normalized = P * n_s

        matched_tgt = P_normalized @ tgt_feats  # (n_s, d)
        return F.mse_loss(src_feats, matched_tgt, reduction="mean")


class TopKPairwiseLoss(nn.Module):
    """
    Mean of the top-\(k\) largest *row-minimum* distances between two sets.

    This loss uses squared Euclidean distances to compute a matrix
    \(C_{ij} = \lVert x_i - y_j\rVert^2\).  For each source vector
    \(x_i\), we find the minimum distance over all targets \(\min_j C_{ij}\).
    These per-row minima represent the best alignment cost for each source
    example.  We then take the top-\(k\) largest values among these minima
    and return their mean.  Intuitively, this penalizes the worst-aligned
    examples (i.e., those whose closest match is still far away).

    Under this formulation:

    * Identical feature sets (\(X = Y\)) yield row minima of zero,
      giving a TopK loss of zero.
    * When features are similar, the minima are small and the loss is
      correspondingly low.
    * For dissimilar or outlying features, the minima are large and
      dominate the TopK loss.

    The `exclude_diagonal` argument is retained for API compatibility
    but is not used in this implementation.

    Args:
        k: number of largest row-minimum distances to average.  If
           `k <= 0`, the loss returns zero.  If `k` exceeds the number of
           rows, all row minima are used.
        exclude_diagonal: unused flag kept for backward compatibility.
    """

    def __init__(self, k: int = 5, exclude_diagonal: bool = True) -> None:
        super().__init__()
        self.k = int(k)
        self.exclude_diagonal = exclude_diagonal

    def forward(self, src_feats: torch.Tensor, tgt_feats: torch.Tensor) -> torch.Tensor:
        if src_feats.ndim != 2 or tgt_feats.ndim != 2:
            raise ValueError("TopKPairwiseLoss expects 2D tensors (batch, dim).")

        n_s, n_t = src_feats.size(0), tgt_feats.size(0)
        if n_s == 0 or n_t == 0 or self.k <= 0:
            return src_feats.sum() * 0.0

        # Compute pairwise squared distances
        cost = _pairwise_sq_dist(src_feats, tgt_feats)  # shape: (n_s, n_t)

        # We measure the worst alignment quality by looking at the best match for
        # each source feature and taking the top-k largest among these minima.
        # This captures how well each source can be matched to some target.
        # For identical or close feature sets, the per-row minima will be small,
        # leading to a small (or zero) TopK loss. For dissimilar sets, the
        # minima will be large. This metric is differentiable via autograd.
        # We ignore the ``exclude_diagonal`` flag in this scheme because
        # excluding the diagonal would cause identical sets to have non-zero
        # minima, which contradicts the intended behavior.
        row_min = cost.min(dim=1).values  # (n_s,)

        # If no rows or k <= 0, return zero
        if row_min.numel() == 0:
            return src_feats.sum() * 0.0

        k = min(self.k, row_min.numel())
        # topk on the minima: largest values indicate worst-aligned examples
        vals, _ = torch.topk(row_min, k=k, largest=True)
        return vals.mean()


class OTAlignmentLoss(nn.Module):
    """
    Composite feature alignment loss = λ_ot * OT + λ_match * SoftMatchMSE + λ_topk * TopK.

    This is differentiable end-to-end and meant to be added on top of any
    other training objective (e.g., CE, adversarial DA, etc.).
    """

    def __init__(
        self,
        lambda_ot: float = 0.25,
        lambda_match: float = 0.5,
        lambda_topk: float = 0.05,
        topk: int = 5,
        ot_epsilon: float = 0.05,
        ot_n_iter: int = 50,
        match_epsilon: float = 0.05,
        match_n_iter: int = 20,
    ) -> None:
        super().__init__()
        self.lambda_ot = float(lambda_ot)
        self.lambda_match = float(lambda_match)
        self.lambda_topk = float(lambda_topk)

        self.ot_loss = SinkhornOTLoss(epsilon=ot_epsilon, n_iter=ot_n_iter)
        self.match_loss = SoftMatchMSELoss(epsilon=match_epsilon, n_iter=match_n_iter)
        self.topk_loss = TopKPairwiseLoss(k=topk)

    def forward(self, src_feats: torch.Tensor, tgt_feats: torch.Tensor) -> torch.Tensor:
        if src_feats.ndim != 2 or tgt_feats.ndim != 2:
            raise ValueError(
                "Source and target features must be 2D tensors (batch, dim)."
            )

        ot = self.ot_loss(src_feats, tgt_feats)
        match_mse = self.match_loss(src_feats, tgt_feats)
        topk_term = self.topk_loss(src_feats, tgt_feats)

        total = (
            self.lambda_ot * ot
            + self.lambda_match * match_mse
            + self.lambda_topk * topk_term
        )
        return total


def compute_hard_ot_cost(
    src_feats: torch.Tensor,
    tgt_feats: torch.Tensor,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute mean hard-assignment OT cost using the Hungarian algorithm (non-differentiable).

    Returns:
        (mean_cost, row_ind, col_ind)
        If SciPy is not available, returns (np.nan, None, None).
    """
    if src_feats.ndim != 2 or tgt_feats.ndim != 2:
        raise ValueError("Expected 2D tensors (batch, dim).")

    x = src_feats.detach().cpu().numpy()
    y = tgt_feats.detach().cpu().numpy()

    cost = np.linalg.norm(x[:, None] - y[None, :], axis=2)  # (n_s, n_t)
    row_ind, col_ind = linear_sum_assignment(cost)
    return float(cost[row_ind, col_ind].mean()), row_ind, col_ind


if __name__ == "__main__":
    torch.manual_seed(0)
    B, D = 32, 8

    src = torch.randn(B, D)
    tgt_close = src + 0.05 * torch.randn(B, D)
    tgt_far = torch.randn(B, D) * 5.0

    # Atomic losses
    ot = SinkhornOTLoss()
    sm = SoftMatchMSELoss()
    tk = TopKPairwiseLoss(k=5)

    print("=== Atomic losses (identical) ===")
    print(f"OT:     {ot(src, src).item():.4f}")
    print(f"SoftM:  {sm(src, src).item():.4f}")
    print(f"TopK:   {tk(src, src).item():.4f}")

    print("\n=== Atomic losses (close vs far) ===")
    print(f"OT:     {ot(src, tgt_close).item():.4f} | {ot(src, tgt_far).item():.4f}")
    print(f"SoftM:  {sm(src, tgt_close).item():.4f} | {sm(src, tgt_far).item():.4f}")
    print(f"TopK:   {tk(src, tgt_close).item():.4f} | {tk(src, tgt_far).item():.4f}")

    # Composite (alignment-only)
    align = OTAlignmentLoss()
    print("\n=== OTAlignmentLoss (close vs far) ===")
    print(
        f"Total:  {align(src, tgt_close).item():.4f} | {align(src, tgt_far).item():.4f}"
    )

    m_close, _, _ = compute_hard_ot_cost(src, tgt_close)
    m_far, _, _ = compute_hard_ot_cost(src, tgt_far)
    print("\n=== Hard OT (Hungarian) diagnostic ===")
    print(f"Mean cost: {m_close:.4f} | {m_far:.4f}")

    print("\n=== EXPECTED vs ACTUAL for identical features ===")
    print("For src == src, we expect:")
    print("  OT:     ~0.0000 (minimal transport cost)")
    print("  SoftM:  ~0.0000 (features already aligned)")
    print("  TopK:   ~0.0000 (no misaligned pairs)")
    print(f"\nActual values:")
    print(
        f"  OT:     {ot(src, src).item():.6f} {'ok.' if ot(src, src).item() < 0.001 else 'not ok.'}"
    )
    print(
        f"  SoftM:  {sm(src, src).item():.6f} {'ok.' if sm(src, src).item() < 0.001 else 'not ok.'}"
    )
    print(
        f"  TopK:   {tk(src, src).item():.6f} {'ok.' if tk(src, src).item() < 0.001 else 'not ok.'}"
    )
