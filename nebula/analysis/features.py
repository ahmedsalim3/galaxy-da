from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    max_batches: Optional[int] = None,
    feature_normalize: bool = True,
) -> Dict[str, np.ndarray]:
    """Extract features, logits, labels, and predictions."""
    model.eval()
    Z, Y, LOGITS, PREDS = [], [], [], []

    it = iter(loader)
    n_batches = len(loader) if max_batches is None else min(max_batches, len(loader))

    for _ in range(n_batches):
        images, labels = next(it)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits, z = model(images)

        if feature_normalize:
            z = F.normalize(z, dim=1)

        Z.append(z.detach().cpu().numpy())
        LOGITS.append(logits.detach().cpu().numpy())
        Y.append(labels.detach().cpu().numpy())
        PREDS.append(logits.argmax(1).detach().cpu().numpy())

    latent_dim = Z[0].shape[1] if Z else 1
    num_classes = LOGITS[0].shape[1] if LOGITS else 1

    Z = np.concatenate(Z, 0) if Z else np.empty((0, latent_dim))
    Y = np.concatenate(Y, 0) if Y else np.empty((0,), dtype=int)
    LOGITS = np.concatenate(LOGITS, 0) if LOGITS else np.empty((0, num_classes))
    PREDS = np.concatenate(PREDS, 0) if PREDS else np.empty((0,), dtype=int)

    return {"z": Z, "y": Y, "logits": LOGITS, "preds": PREDS}
