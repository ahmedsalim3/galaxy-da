from typing import Dict, List, Optional, Tuple

import numpy as np
import ot
import torch
from geomloss import SamplesLoss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _median_sigma(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute median heuristic for Gaussian kernel bandwidth."""
    if X.size == 0 or Y.size == 0:
        return 1.0
    rng = np.random.default_rng(42)
    xs = X[rng.choice(len(X), min(512, len(X)), replace=False)]
    ys = Y[rng.choice(len(Y), min(512, len(Y)), replace=False)]
    Dxx = ((xs[:, None] - xs[None, :]) ** 2).sum(-1)
    Dyy = ((ys[:, None] - ys[None, :]) ** 2).sum(-1)
    Dxy = ((xs[:, None] - ys[None, :]) ** 2).sum(-1)
    d = np.concatenate(
        [
            Dxx[np.triu_indices_from(Dxx, 1)],
            Dyy[np.triu_indices_from(Dyy, 1)],
            Dxy.ravel(),
        ]
    )
    d = d[d > 0]
    if d.size == 0:
        return 1.0
    return float(np.sqrt(0.5 * np.median(d)))


def _gauss(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian kernel matrix."""
    D = ((X[:, None] - Y[None, :]) ** 2).sum(-1) / (2.0 * sigma**2 + 1e-12)
    return np.exp(-D)


def mmd2_unbiased_gaussian(
    X: np.ndarray, Y: np.ndarray, sigma: Optional[float] = None
) -> float:
    """Unbiased MMD^2 estimator with Gaussian kernel."""
    nx, ny = len(X), len(Y)
    if nx < 2 or ny < 2:
        return np.nan

    sigma = _median_sigma(X, Y) if sigma is None else sigma
    if sigma == 0.0:
        return 0.0

    Kxx = _gauss(X, X, sigma)
    np.fill_diagonal(Kxx, 0.0)
    Kyy = _gauss(Y, Y, sigma)
    np.fill_diagonal(Kyy, 0.0)
    Kxy = _gauss(X, Y, sigma)

    term_x = Kxx.sum() / (nx * (nx - 1))
    term_y = Kyy.sum() / (ny * (ny - 1))
    term_xy = 2.0 * Kxy.mean()

    return float(max(0.0, term_x + term_y - term_xy))


def sinkhorn_divergence(
    X: np.ndarray, Y: np.ndarray, blur: float = 0.05, p: int = 2
) -> float:
    """Sinkhorn divergence between two point clouds."""
    if X.size == 0 or Y.size == 0:
        return np.nan
    loss = SamplesLoss("sinkhorn", p=p, blur=blur, debias=True)
    x = torch.as_tensor(X, dtype=torch.float32)
    y = torch.as_tensor(Y, dtype=torch.float32)
    return float(loss(x, y).item())


def domain_probe_auc(
    Xs: np.ndarray, Xt: np.ndarray, seed: int = 123, max_n: int = 4000
) -> Tuple[float, float]:
    """Train domain classifier and compute AUC."""
    rng = np.random.default_rng(seed)
    if len(Xs) > max_n:
        Xs = Xs[rng.choice(len(Xs), max_n, replace=False)]
    if len(Xt) > max_n:
        Xt = Xt[rng.choice(len(Xt), max_n, replace=False)]

    X = np.vstack([Xs, Xt])
    y = np.hstack([np.zeros(len(Xs), int), np.ones(len(Xt), int)])
    idx = rng.permutation(len(X))
    ntr = int(0.7 * len(X))
    tr, te = idx[:ntr], idx[ntr:]

    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X[tr], y[tr])
    probs = clf.predict_proba(X[te])[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y[te], probs)
    acc = (preds == y[te]).mean()

    return float(auc), float(acc)


def domain_probe_auc_per_class(
    Xs: np.ndarray,
    ys: np.ndarray,
    Xt: np.ndarray,
    yt_pred: np.ndarray,
    classes: List[int],
) -> Dict[int, float]:
    """Domain probe AUC per class."""
    out = {}
    for c in classes:
        Xsc = Xs[ys == c]
        Xtc = Xt[yt_pred == c]
        if len(Xsc) < 5 or len(Xtc) < 5:
            out[int(c)] = np.nan
        else:
            out[int(c)] = domain_probe_auc(Xsc, Xtc)[0]
    return out


def sinkhorn_plan_class_mass(
    X: np.ndarray,
    Y: np.ndarray,
    y_src: np.ndarray,
    y_tgt_pred: np.ndarray,
    reg: float = 0.05,
    n_classes: int = 3,
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Compute class-to-class mass from optimal transport plan."""

    ns, nt = len(X), len(Y)
    if ns == 0 or nt == 0:
        return None, None

    a = np.ones(ns) / ns
    b = np.ones(nt) / nt
    M = ((X[:, None] - Y[None, :]) ** 2).sum(-1)
    P = ot.sinkhorn(a, b, M, reg=reg)

    mass = np.zeros((n_classes, n_classes), dtype=np.float64)
    for cs in range(n_classes):
        sm = y_src == cs
        if not sm.any():
            continue
        for ct in range(n_classes):
            tm = y_tgt_pred == ct
            if not tm.any():
                continue
            mass[cs, ct] = P[np.ix_(sm, tm)].sum()

    s = mass.sum()
    if s > 0:
        mass /= s
    on_diag = float(np.trace(mass)) if s > 0 else np.nan

    return mass, on_diag
