from typing import Dict

import numpy as np


def basic_stats(split: Dict[str, np.ndarray]) -> Dict[str, float]:
    x = split["x"]
    return {
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
    }
