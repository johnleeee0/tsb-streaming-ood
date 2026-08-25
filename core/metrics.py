from typing import Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_auroc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_scores))


def compute_aupr(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_scores))


def compute_fpr95(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    thresholds = np.sort(y_scores)
    target_tpr = 0.95
    best_fpr = 1.0
    pos = (y_true == 1)
    neg = (y_true == 0)
    for thr in thresholds:
        tp = ((y_scores >= thr) & pos).sum()
        fp = ((y_scores >= thr) & neg).sum()
        fn = ((y_scores < thr) & pos).sum()
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(neg.sum(), 1)
        if tpr >= target_tpr:
            best_fpr = min(best_fpr, fpr)
    return float(best_fpr)


def compute_id_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true == 0
    if mask.sum() == 0:
        return 0.0
    return float((y_pred[mask] == 0).mean())


def split_scores(
    y_true: np.ndarray, y_scores: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    id_scores = y_scores[y_true == 0]
    ood_scores = y_scores[y_true == 1]
    return id_scores, ood_scores
