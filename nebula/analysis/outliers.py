import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors


def class_conditional_mahalanobis_scores(
    X_ref: np.ndarray, y_ref: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """
    Compute class-conditional Mahalanobis distance scores for samples in X
    relative to Gaussian models fitted on reference features X_ref per class.

    Returns the minimum Mahalanobis distance across classes for each sample.
    Higher values indicate more outlier-like samples.
    """
    classes = np.unique(y_ref)
    means = {}
    inv_covs = {}
    for c in classes:
        Xc = X_ref[y_ref == c]
        if Xc.shape[0] < 2:
            continue
        means[c] = Xc.mean(axis=0)
        cov = LedoitWolf().fit(Xc).precision_  # inverse covariance
        inv_covs[c] = cov

    scores = []
    for x in X:
        dists = []
        for c in classes:
            if c not in means:
                continue
            diff = x - means[c]
            # Mahalanobis squared distance: diff^T Sigma^{-1} diff
            md2 = float(diff.T @ inv_covs[c] @ diff)
            dists.append(md2)
        scores.append(min(dists) if dists else np.nan)
    return np.array(scores)


def knn_distance_scores(X_ref: np.ndarray, X: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Compute kNN distance-based anomaly scores: average distance to k nearest
    neighbors in the reference set. Higher scores indicate more outlier-like samples.
    """
    if X_ref.shape[0] == 0 or X.shape[0] == 0:
        return np.zeros((X.shape[0],), dtype=float)
    k_eff = min(k, max(1, X_ref.shape[0]))
    nbrs = NearestNeighbors(n_neighbors=k_eff, algorithm="auto").fit(X_ref)
    distances, _ = nbrs.kneighbors(X)
    return distances.mean(axis=1)


def threshold_by_quantile(
    scores: np.ndarray, q: float = 0.95
) -> tuple[np.ndarray, float]:
    """
    Return a boolean mask of scores above the quantile threshold and the threshold value.
    """
    if scores.size == 0 or np.all(np.isnan(scores)):
        return np.zeros_like(scores, dtype=bool), np.nan
    valid = np.isfinite(scores)
    thr = np.quantile(scores[valid], q) if valid.any() else np.nan
    mask = scores >= thr if np.isfinite(thr) else np.zeros_like(scores, dtype=bool)
    return mask, thr


def _rff_features(
    X: np.ndarray,
    n_features: int = 256,
    gamma: float | None = None,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random Fourier features for Gaussian kernel with bandwidth gamma.
    Returns (Phi(X), W, b) so queries can reuse W,b.
    """
    if rng is None:
        rng = np.random.RandomState(42)
    D = X.shape[1]
    if gamma is None:
        # Median heuristic on a subsample
        if X.shape[0] > 2000:
            idx = rng.choice(X.shape[0], size=2000, replace=False)
            Xs = X[idx]
        else:
            Xs = X
        # pairwise distances median
        diffs = Xs[:500] - Xs[None, :500]
        d2 = np.sum(diffs * diffs, axis=-1)
        med = np.median(d2)
        med = max(med, 1e-6)
        gamma = 1.0 / (2.0 * med)
    W = rng.normal(scale=np.sqrt(2 * gamma), size=(D, n_features))
    b = rng.uniform(0, 2 * np.pi, size=(n_features,))
    Z = np.cos(X @ W + b) * np.sqrt(2.0 / n_features)
    return Z, W, b


def rff_parzen_scores(
    X_ref: np.ndarray, X: np.ndarray, n_features: int = 256, gamma: float | None = None
) -> np.ndarray:
    """
    Approximate Parzen window KDE with Gaussian kernel using Random Fourier Features.
    p(x) ≈ phi(x)^T mu_phi, where mu_phi = mean(phi(x_i)). Return negative density as outlier score.
    """
    if X_ref.shape[0] == 0 or X.shape[0] == 0:
        return np.zeros((X.shape[0],), dtype=float)
    rng = np.random.RandomState(123)
    Phi_ref, W, b = _rff_features(X_ref, n_features=n_features, gamma=gamma, rng=rng)
    mu_phi = Phi_ref.mean(axis=0)
    # Compute features for queries with same W,b
    Phi_X = np.cos(X @ W + b) * np.sqrt(2.0 / Phi_ref.shape[1])
    dens = Phi_X @ mu_phi
    # Normalize scores to be comparable: higher score -> more outlier-like
    # Convert to negative density and z-score normalize
    scores = -dens
    scores = (scores - scores.mean()) / (scores.std() + 1e-12)
    return scores
