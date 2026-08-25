from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_time_series(
    x: np.ndarray,
    title: str = "Time Series",
    save_path: Optional[str] = None,
    max_channels: int = 8,
) -> None:
    plt.figure(figsize=(10, 4))
    if x.ndim == 1:
        plt.plot(x)
    else:
        channels = min(x.shape[0], max_channels)
        offset = 0.0
        for i in range(channels):
            plt.plot(x[i] + offset, alpha=0.7, label=f"ch{i}")
            offset += x[i].std() * 2.0
        plt.legend(loc="upper right", ncol=2, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_dataset_samples(
    x: np.ndarray,
    y: np.ndarray,
    num_samples: int = 5,
    save_path: Optional[str] = None,
    title: str = "Dataset Samples",
) -> None:
    num_samples = min(num_samples, x.shape[0])
    plt.figure(figsize=(10, 2 * num_samples))
    for i in range(num_samples):
        plt.subplot(num_samples, 1, i + 1)
        sample = x[i]
        if sample.ndim == 1:
            plt.plot(sample)
        else:
            for ch in range(sample.shape[0]):
                plt.plot(sample[ch], alpha=0.7)
        plt.title(f"Sample {i} | label={y[i]}")
        plt.tight_layout()
    plt.suptitle(title, y=1.02)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()


def plot_ood_distributions(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "OOD Score Distributions",
) -> None:
    plt.figure(figsize=(8, 4))

    def _safe_histplot(scores, color, label):
        """Plot histogram, falling back to a single bar when scores are constant."""
        scores = np.asarray(scores, dtype=float)
        scores = scores[np.isfinite(scores)]
        if len(scores) == 0:
            return
        if np.ptp(scores) < 1e-10:  # all-constant: can't bin
            plt.axvline(scores[0], color=color, linestyle="--", label=f"{label} (constant)")
        else:
            sns.histplot(scores, color=color, label=label, kde=True, stat="density", alpha=0.5)

    _safe_histplot(id_scores, "tab:blue", "ID")
    _safe_histplot(ood_scores, "tab:red", "OOD")
    plt.title(title)
    plt.xlabel("OOD score")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        plt.close()
