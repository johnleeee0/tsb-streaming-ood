from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA

from core.registry import register_ood
from core.base_ood import BaseOODDetector


@register_ood("dfm_pca")
class DFMPcaDetector(BaseOODDetector):
    """
    Deep Feature Modeling with PCA for OOD Detection.

    Following the paper: "TS-OOD: Evaluating Time-Series Out-of-Distribution Detection"
    This implementation fits ONE PCA model PER ID class (not a single global PCA).

    OOD score is computed as the minimum reconstruction error across all ID class models.
    """
    def __init__(self, model: Any, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(model, config)
        self.n_components = int(self.config.get("n_components", 32))
        self.pca_models: Dict[Any, PCA] = {}  # One PCA per ID class
        self.class_labels: Optional[np.ndarray] = None

    def fit(self, x_id: Any, y_id: Optional[Any] = None) -> None:
        """
        Fit one PCA model per ID class.

        Args:
            x_id: Training data (ID classes only)
            y_id: Class labels for training data (REQUIRED for per-class modeling)
        """
        if y_id is None:
            raise ValueError(
                "DFM-PCA requires class labels (y_id) for per-class modeling. "
                "Paper: 'features of each ID class extracted from a given deep layer "
                "to a lower dimensional embeddings via PCA'"
            )

        x_tensor = self._to_tensor(x_id)
        with torch.no_grad():
            feats = self._forward_features(x_tensor).detach().cpu().numpy()

        # Ensure y_id is numpy array
        if isinstance(y_id, torch.Tensor):
            y_id = y_id.cpu().numpy()
        elif not isinstance(y_id, np.ndarray):
            y_id = np.array(y_id)

        self.class_labels = np.unique(y_id)

        # Fit one PCA model per ID class
        for class_label in self.class_labels:
            mask = (y_id == class_label)
            feats_class = feats[mask]

            if len(feats_class) < 2:
                raise ValueError(
                    f"Class {class_label} has only {len(feats_class)} samples. "
                    "Need at least 2 samples per class for PCA."
                )

            # Determine number of components for this class
            max_comp = min(
                self.n_components,
                feats_class.shape[1],  # Feature dimension
                feats_class.shape[0] - 1  # Number of samples - 1
            )

            if max_comp < 1:
                max_comp = 1

            pca = PCA(n_components=max_comp, svd_solver="full")
            pca.fit(feats_class)
            self.pca_models[class_label] = pca

    def score(self, x: Any) -> np.ndarray:
        """
        Compute OOD scores as minimum reconstruction error across all ID class PCAs.

        For each test sample:
            1. Project to low-dimensional space using each class's PCA
            2. Reconstruct back to original feature space
            3. Compute reconstruction error
            4. Return minimum error across all classes (distance to nearest ID class)

        Higher score = more OOD (larger reconstruction error)
        """
        if not self.pca_models:
            raise RuntimeError("DFM-PCA detector must be fit on ID data before scoring.")

        x_tensor = self._to_tensor(x)
        with torch.no_grad():
            feats = self._forward_features(x_tensor).detach().cpu().numpy()

        num_samples = feats.shape[0]
        ood_scores = np.zeros(num_samples)

        # For each test sample
        for i in range(num_samples):
            test_feat = feats[i]

            # Compute reconstruction error for each ID class's PCA model
            reconstruction_errors = []
            for class_label in self.class_labels:
                pca = self.pca_models[class_label]

                # Project to low-dimensional space and reconstruct
                low_dim = pca.transform(test_feat.reshape(1, -1))
                reconstructed = pca.inverse_transform(low_dim)

                # Compute reconstruction error (L2 norm)
                error = np.linalg.norm(test_feat - reconstructed.flatten())
                reconstruction_errors.append(error)

            # OOD score = minimum reconstruction error (distance to nearest ID class PCA)
            # ID sample: low error on its true class → low score
            # OOD sample: high error on ALL classes → high score
            ood_scores[i] = min(reconstruction_errors)

        return ood_scores
