from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def plot_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = "pca",
    save_path: Optional[str] = None,
    title: str = "Feature Embeddings",
) -> None:
    method = method.lower()
    if method == "tsne":
        reducer = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=30)
        emb = reducer.fit_transform(features)
    else:
        reducer = PCA(n_components=2)
        emb = reducer.fit_transform(features)

    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=labels, cmap="tab10", s=16, alpha=0.8)
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(scatter, fraction=0.03, pad=0.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
