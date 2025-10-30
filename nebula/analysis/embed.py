from typing import Dict, Optional

import numpy as np
import umap
from sklearn.manifold import TSNE


def compute_embedding(
    Z: np.ndarray, dom: np.ndarray, lab: np.ndarray, method: str = "tsne"
) -> Optional[Dict[str, np.ndarray]]:
    """Compute 2D embedding using UMAP or t-SNE."""
    if Z.shape[0] < 2:
        return None

    try:
        if method == "umap":
            reducer = umap.UMAP(
                n_neighbors=30,
                min_dist=0.1,
                metric="euclidean",
                random_state=42,
            )
            XY = reducer.fit_transform(Z)
        elif method == "tsne":
            XY = TSNE(
                n_components=2,
                perplexity=30,
                init="pca",
                learning_rate="auto",
                n_iter=1000,
                random_state=42,
            ).fit_transform(Z)
        else:
            raise ValueError(f"Invalid method: {method}")
        return {"xy": XY, "domain": dom, "label": lab}
    except Exception as e:
        print(f"Error computing embedding with {method}: {e}")
        return None
