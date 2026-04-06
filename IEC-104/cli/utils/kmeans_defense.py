"""KMeans-based probability correction for ensemble defense.

V4: Probability Correction approach.

Instead of adjusting ensemble weights (which doesn't change argmax after
normalization), directly correct each model's probability vector before
ensemble voting using KMeans cluster class distributions as "second opinion".

Key insight: adversarial samples are crafted to be predicted as class A,
but their features still cluster with class B. We detect this mismatch
and blend the model's proba toward the cluster's class distribution,
potentially flipping the predicted class.
"""

import logging
import pickle

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class KMeansDefense:
    """KMeans-based prediction consistency scorer.

    Parameters
    ----------
    n_clusters : int
        Number of KMeans clusters. Using more clusters than classes allows
        capturing sub-class structure (recommended: 2-4x n_classes).
    threshold_pct : float
        Percentile of training distances for distance-based scoring component.
    """

    def __init__(self, n_clusters=36, threshold_pct=50):
        self.n_clusters = n_clusters
        self.threshold_pct = threshold_pct
        self.km = None
        self.threshold = None
        self.cluster_labels = None      # dominant class per cluster
        self.cluster_class_dist = None  # class distribution per cluster

    def fit(self, X_train, y_train):
        """Fit KMeans on clean training data with class labels.

        Parameters
        ----------
        X_train : ndarray, shape (n_samples, n_features)
        y_train : ndarray, shape (n_samples,) integer class labels
        """
        self.n_classes = int(y_train.max()) + 1
        logger.info(
            f"Fitting KMeans: n_clusters={self.n_clusters}, "
            f"n_classes={self.n_classes}, data={X_train.shape}"
        )

        self.km = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
        )
        self.km.fit(X_train)

        # Map each cluster to its class distribution
        assignments = self.km.predict(X_train)
        self.cluster_class_dist = np.zeros(
            (self.n_clusters, self.n_classes), dtype=np.float32)
        for k in range(self.n_clusters):
            mask = assignments == k
            if mask.sum() > 0:
                for c in range(self.n_classes):
                    self.cluster_class_dist[k, c] = (
                        (y_train[mask] == c).sum() / mask.sum()
                    )

        # Dominant class per cluster
        self.cluster_labels = self.cluster_class_dist.argmax(axis=1)

        # Distance threshold for distance component
        dists = self._min_distances(X_train)
        self.threshold = np.percentile(dists, self.threshold_pct)

        # Log cluster stats
        purity = self.cluster_class_dist.max(axis=1).mean()
        logger.info(
            f"KMeans fit: avg cluster purity={purity:.4f}, "
            f"dist p50={np.median(dists):.4f}, "
            f"p{self.threshold_pct}={self.threshold:.4f}"
        )

    def inconsistency_score(self, X, predicted_proba):
        """Score how inconsistent a model's predictions are with KMeans clusters.

        For each sample:
          1. Find nearest cluster k
          2. Get cluster's class distribution: p_cluster[c]
          3. Compare with model's proba: p_model[c]
          4. inconsistency = 1 - sum(p_cluster * p_model) (dot product)
             If model agrees with cluster → dot product high → score low
             If model disagrees → dot product low → score high

        Also adds distance-based component for samples far from any cluster.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
        predicted_proba : ndarray, shape (n_samples, n_classes)
            Model's probability predictions

        Returns
        -------
        scores : ndarray, shape (n_samples,) in [0, 1]
        """
        # Nearest cluster for each sample
        assignments = self.km.predict(X)

        # Cluster class distribution for each sample's assigned cluster
        cluster_dist = self.cluster_class_dist[assignments]  # (n, n_classes)

        # Prediction-cluster agreement (cosine-like via dot product)
        # Both are probability distributions so values in [0, 1]
        agreement = np.sum(cluster_dist * predicted_proba, axis=1)
        agreement = np.clip(agreement, 0.0, 1.0)

        # Inconsistency: low agreement = high inconsistency
        inconsistency = 1.0 - agreement

        # Distance component: sigmoid around threshold
        dists = self._min_distances(X)
        raw = dists / max(self.threshold, 1e-8)
        dist_score = 1.0 / (1.0 + np.exp(-5.0 * (raw - 1.0)))

        # Combined: max of inconsistency and distance
        # (either signal alone should trigger defense)
        scores = np.maximum(inconsistency, dist_score)

        return scores.astype(np.float32)

    def correct_proba(self, X, model_proba, alpha=0.5):
        """Correct model probabilities using cluster class distribution.

        For each sample i:
          1. Find nearest cluster k
          2. Get cluster's class distribution: cluster_dist[k]
          3. Compute inconsistency = 1 - dot(model_proba[i], cluster_dist[k])
          4. corrected[i] = (1 - α*incon) * model_proba[i] + α*incon * cluster_dist[k]

        When model agrees with cluster: incon ≈ 0 → no correction.
        When model disagrees: incon ≈ 1 → proba shifts toward cluster's class.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input features (used for cluster assignment).
        model_proba : ndarray, shape (n_samples, n_classes)
            Model's probability predictions.
        alpha : float
            Correction strength. 0 = no correction, 1 = full correction.

        Returns
        -------
        corrected_proba : ndarray, shape (n_samples, n_classes)
        """
        assignments = self.km.predict(X)
        cluster_dist = self.cluster_class_dist[assignments]  # (n, n_classes)

        # Inconsistency: how much model and cluster disagree
        agreement = np.sum(cluster_dist * model_proba, axis=1, keepdims=True)
        agreement = np.clip(agreement, 0.0, 1.0)
        inconsistency = 1.0 - agreement  # (n, 1)

        # Blend: model_proba ← cluster_dist when inconsistent
        blend = alpha * inconsistency
        corrected = (1.0 - blend) * model_proba + blend * cluster_dist

        # Re-normalize to ensure valid probability distribution
        row_sums = corrected.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-8)
        corrected = corrected / row_sums

        return corrected.astype(np.float32)

    def _min_distances(self, X):
        """Distance from each sample to its nearest cluster center."""
        return self.km.transform(X).min(axis=1)

    def save(self, path):
        """Save fitted model to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "n_clusters": self.n_clusters,
                "threshold_pct": self.threshold_pct,
                "km": self.km,
                "threshold": self.threshold,
                "cluster_labels": self.cluster_labels,
                "cluster_class_dist": self.cluster_class_dist,
                "n_classes": self.n_classes,
            }, f)
        logger.info(f"KMeansDefense saved to {path}")

    @classmethod
    def load(cls, path):
        """Load a fitted KMeansDefense from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            n_clusters=data["n_clusters"],
            threshold_pct=data["threshold_pct"],
        )
        obj.km = data["km"]
        obj.threshold = data["threshold"]
        obj.cluster_labels = data["cluster_labels"]
        obj.cluster_class_dist = data["cluster_class_dist"]
        obj.n_classes = data["n_classes"]
        logger.info(f"KMeansDefense loaded from {path}")
        return obj
