"""Randomized Smoothing defense for adversarial robustness.

Model-agnostic defense that works with any classifier (tree + DL).
At inference time, adds Gaussian noise N times, collects predictions,
and returns majority vote with optional binomial-test abstention.

Noise is scaled per-feature (sigma is relative to each feature's std),
so sigma=0.1 means "10% of feature standard deviation" — consistent
regardless of raw feature magnitudes.

No training required — only tune sigma and n_samples.

Usage:
    from smoothing import SmoothedClassifier

    sc = SmoothedClassifier(model, n_samples=100, sigma=0.05,
                            feature_std=X_train.std(axis=0))
    preds = sc.predict(X_test)
"""

import numpy as np
from scipy.stats import binomtest


class SmoothedClassifier:
    """Wraps any model with .predict() to add randomized smoothing.

    Parameters
    ----------
    model : object
        Any model with a ``predict(X) -> int array`` method.
    n_samples : int
        Number of noisy copies per input for majority vote.
    sigma : float
        Relative noise scale. Actual noise std per feature =
        sigma * feature_std[j]. If feature_std is None, sigma is
        used directly (absolute noise in raw space).
    feature_std : ndarray or None
        Per-feature standard deviation from training data. When provided,
        noise is scaled per-feature so sigma is unit-independent.
    alpha : float or None
        Significance level for binomial test. If the top class is not
        significantly more likely than the runner-up (p > alpha), the
        prediction is ABSTAIN (-1). Set to None to disable abstention.
    n_classes : int or None
        Number of classes. Auto-detected if None.
    """

    ABSTAIN = -1

    def __init__(self, model, n_samples=100, sigma=0.05, feature_std=None,
                 alpha=0.001, n_classes=None):
        self.model = model
        self.n_samples = n_samples
        self.sigma = sigma
        self.alpha = alpha
        self.n_classes = n_classes

        # Per-feature noise scale
        if feature_std is not None:
            # sigma * std per feature; clamp to avoid zero-std features
            self._noise_scale = sigma * np.maximum(feature_std, 1e-8).astype(np.float32)
        else:
            self._noise_scale = None  # uniform sigma across features

    def _predict_noisy_batch(self, X, n_copies):
        """Generate n_copies noisy versions of each row in X, predict all.

        Parameters
        ----------
        X : ndarray, shape (n_data, n_feat)
        n_copies : int

        Returns
        -------
        all_counts : ndarray, shape (n_data, n_classes)
        """
        n_data, n_feat = X.shape
        all_counts = np.zeros((n_data, self.n_classes), dtype=int)
        idx = np.arange(n_data)

        for _ in range(n_copies):
            if self._noise_scale is not None:
                noise = np.random.randn(n_data, n_feat).astype(np.float32) * self._noise_scale
            else:
                noise = (np.random.randn(n_data, n_feat) * self.sigma).astype(np.float32)
            X_noisy = (X + noise).astype(np.float32)
            preds = np.asarray(self.model.predict(X_noisy)).ravel().astype(int)
            # Vectorized vote accumulation
            valid = (preds >= 0) & (preds < self.n_classes)
            np.add.at(all_counts, (idx[valid], preds[valid]), 1)

        return all_counts

    def _decide_batch(self, all_counts):
        """Apply binomial test to each row. Returns predictions array."""
        preds = np.empty(len(all_counts), dtype=int)
        for i, counts in enumerate(all_counts):
            preds[i] = self._decide(counts)
        return preds

    def _decide(self, counts):
        """Apply binomial test to decide prediction or abstain."""
        sorted_idx = np.argsort(counts)[::-1]
        top_class = sorted_idx[0]
        top_count = counts[top_class]
        runner_count = counts[sorted_idx[1]] if len(sorted_idx) > 1 else 0

        if self.alpha is None:
            return top_class

        total = top_count + runner_count
        if total == 0:
            return self.ABSTAIN

        result = binomtest(int(top_count), int(total), p=0.5,
                           alternative="greater")
        if result.pvalue <= self.alpha:
            return top_class
        return self.ABSTAIN

    def predict(self, X):
        """Smoothed prediction with majority vote.

        Parameters
        ----------
        X : ndarray, shape (n_data, n_features)

        Returns
        -------
        preds : ndarray, shape (n_data,)
            Predicted classes. -1 means ABSTAIN.
        """
        X = np.asarray(X, dtype=np.float32)

        if self.n_classes is None:
            quick = np.asarray(self.model.predict(X[:1])).ravel()
            self.n_classes = max(int(quick.max()) + 1, 2)
            if hasattr(self.model, '_model'):
                m = self.model._model
                if hasattr(m, 'n_classes_'):
                    self.n_classes = int(m.n_classes_)
                elif hasattr(m, 'classes_'):
                    self.n_classes = len(m.classes_)

        all_counts = self._predict_noisy_batch(X, self.n_samples)
        return self._decide_batch(all_counts)

    def predict_detail(self, X):
        """Smoothed prediction returning both predictions and vote counts.

        Returns
        -------
        preds : ndarray, shape (n_data,)
        all_counts : ndarray, shape (n_data, n_classes)
        """
        X = np.asarray(X, dtype=np.float32)

        if self.n_classes is None:
            quick = np.asarray(self.model.predict(X[:1])).ravel()
            self.n_classes = max(int(quick.max()) + 1, 2)

        all_counts = self._predict_noisy_batch(X, self.n_samples)
        preds = self._decide_batch(all_counts)
        return preds, all_counts

    def certify(self, X, y_true):
        """Compute certified accuracy: fraction of correct & non-abstained.

        Returns
        -------
        dict with keys: certified_acc, abstain_rate, correct, abstained, total
        """
        preds = self.predict(X)
        non_abstain = preds != self.ABSTAIN
        correct = (preds == y_true) & non_abstain
        abstained = ~non_abstain

        total = len(X)
        return {
            "certified_acc": correct.sum() / total * 100,
            "abstain_rate": abstained.sum() / total * 100,
            "correct": int(correct.sum()),
            "abstained": int(abstained.sum()),
            "total": total,
        }
