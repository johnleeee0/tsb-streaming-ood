from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt


def plot_metrics_bar(
    results: Dict[str, Dict[str, float]],
    metric_keys: Dict[str, str],
    save_path: str,
    title: str,
) -> None:
    # Avoid accumulation if multiple plots are generated in a loop.
    plt.close("all")
    if not results:
        return
    methods = list(results.keys())
    x = list(range(len(methods)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(methods) * 0.6), 4))
    offset = -width * (len(metric_keys) - 1) / 2
    for i, (key, label) in enumerate(metric_keys.items()):
        values = [results[m].get(key, float("nan")) for m in methods]
        ax.bar([v + offset + i * width for v in x], values, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
