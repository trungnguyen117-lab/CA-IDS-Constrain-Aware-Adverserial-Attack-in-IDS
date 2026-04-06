"""DSE-based adversarial query detection simulation.

Simulates the query detection mechanism from Tiki-Taka (Zhang et al., ToN 2022).

Paper's mechanism: BB attacks send hundreds of queries that are small random
perturbations around the SAME original sample. DSE encodes these queries;
they cluster tightly in embedding space -> detected via k-NN distance.

Simulation: For each adversarial sample (derived from x_clean), generate
synthetic queries mimicking BB attack's search process: random perturbations
around x_clean. Encode queries via DSE. If they cluster tightly (k-NN
distance < tau) -> attack would have been detected mid-process.

Normal traffic: 300 requests from a legitimate user = 300 completely different
samples -> diverse embeddings -> no clustering -> no false detection.
"""

import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .evaluation import compute_asr

logger = logging.getLogger(__name__)


class DSEDetector:
    """Simulates DSE-based query detection.

    For each adversarial sample, simulates BB attack query pattern:
    - Generate N queries as random perturbations around x_clean
      (mimicking ZOO/HSJA probing behavior)
    - Encode via DSE -> queries cluster tightly (all near x_clean)
    - If mean k-NN distance < tau -> detected

    For FPR: simulate normal user sending N diverse requests
    (random clean samples from the dataset) -> no clustering -> no detection.
    """

    def __init__(self, dse_model, X_normal, cfg=None):
        if cfg is None:
            cfg = {}

        self.dse = dse_model
        self.k = cfg.get("k_neighbors", 200)
        self.tau = cfg.get("detection_threshold", 0.008)
        self.n_queries = cfg.get("n_queries", 300)
        self.probe_eps = cfg.get("probe_eps", 0.01)  # perturbation scale

        self.X_normal = X_normal
        logger.info(
            f"DSEDetector ready: k={self.k}, tau={self.tau}, "
            f"n_queries={self.n_queries}, probe_eps={self.probe_eps}"
        )

    def _simulate_attack_queries(self, x_clean):
        """Simulate BB attack probing queries around x_clean.

        BB attacks (ZOO, HSJA) send queries like:
            q_i = x_clean + eps * random_direction_i

        All queries scatter around x_clean with small random perturbations.
        Returns: (n_queries, n_features)
        """
        n_features = len(x_clean)
        noise = np.random.randn(self.n_queries, n_features).astype(np.float32)
        # Scale noise relative to feature magnitudes
        scale = self.probe_eps * (np.abs(x_clean) + 1e-8)
        queries = x_clean[None, :] + noise * scale[None, :]
        return queries

    def _simulate_normal_queries(self):
        """Simulate normal user traffic: diverse clean samples.

        A normal user sends completely different requests (browse different
        pages, different protocols, different sessions). We model this by
        sampling random clean samples from the dataset.
        Returns: (n_queries, n_features)
        """
        n = len(self.X_normal)
        idx = np.random.choice(n, min(self.n_queries, n), replace=False)
        return self.X_normal[idx]

    def _compute_cluster_tightness(self, queries):
        """Compute mean k-NN distance within a set of queries.

        Tight cluster (attack) -> small d_k
        Diverse set (normal) -> large d_k
        """
        emb = self.dse.encode(queries)
        k = min(self.k, len(emb) - 1)
        if k < 1:
            return float("inf")
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(emb)
        dist, _ = nn.kneighbors(emb)
        return dist[:, -1].mean()

    def detect_single(self, x_clean):
        """Simulate detection for one attack attempt targeting x_clean.

        Returns (detected: bool, d_k: float).
        """
        queries = self._simulate_attack_queries(x_clean)
        d_k = self._compute_cluster_tightness(queries)
        return d_k < self.tau, d_k

    def detect_batch(self, X_adv, X_clean=None):
        """Determine which adversarial samples would be detected.

        For each adversarial sample, simulates the attack query process
        around its clean original and checks if DSE would detect it.

        Args:
            X_adv: adversarial samples (N, F)
            X_clean: corresponding clean originals (N, F)

        Returns boolean array: True = detected (would be blocked).
        """
        n = len(X_adv)
        if n == 0:
            return np.zeros(0, dtype=bool)

        if X_clean is None:
            X_clean = X_adv  # fallback

        detected = np.zeros(n, dtype=bool)
        dk_values = np.zeros(n)

        for i in range(n):
            detected[i], dk_values[i] = self.detect_single(X_clean[i])

        n_det = detected.sum()
        logger.debug(
            f"Detection: {n_det}/{n}, "
            f"d_k: mean={dk_values.mean():.6f}, std={dk_values.std():.6f}"
        )
        return detected

    def compute_detection_rate(self, X_adv, X_clean=None):
        """Fraction of adversarial samples that would be detected."""
        detected = self.detect_batch(X_adv, X_clean)
        rate = detected.mean() * 100
        logger.info(f"Detection rate: {rate:.2f}% ({detected.sum()}/{len(detected)})")
        return rate

    def compute_fpr(self, X_clean, n_tests=100):
        """False positive rate: simulate normal users sending diverse queries.

        For each test, sample random clean samples as "queries from one IP"
        (diverse traffic, not an attack). Check if any triggers detection.
        """
        false_positives = 0
        for _ in range(n_tests):
            queries = self._simulate_normal_queries()
            d_k = self._compute_cluster_tightness(queries)
            if d_k < self.tau:
                false_positives += 1

        fpr = false_positives / n_tests * 100
        logger.info(f"FPR on clean data: {fpr:.2f}% ({false_positives}/{n_tests})")
        return fpr

    def adjusted_asr(self, y_true, y_clean_pred, y_adv_pred, X_adv, X_clean=None):
        """ASR adjusted for DSE detection.

        Detected samples are "blocked" -- their prediction reverts to
        clean prediction, meaning the attack fails on those samples.
        """
        detected = self.detect_batch(X_adv, X_clean)
        y_effective = y_adv_pred.copy()
        y_effective[detected] = y_clean_pred[detected]
        adj = compute_asr(y_true, y_clean_pred, y_effective)
        orig = compute_asr(y_true, y_clean_pred, y_adv_pred)
        logger.info(
            f"ASR: {orig:.2f}% -> {adj:.2f}% (DSE blocked {detected.sum()} samples)"
        )
        return adj

    def calibrate_threshold(self, X_clean, X_adv, n_samples=50, n_points=50):
        """Sweep tau by comparing attack vs normal query cluster tightness.

        - Attack: queries = random perturbations around x_clean -> tight cluster
        - Normal: queries = random clean samples -> diverse, loose cluster

        Returns list of (tau, detection_rate, fpr) tuples.
        """
        n_samples = min(n_samples, len(X_adv), len(X_clean))

        # Measure attack cluster tightness
        attack_dk = np.zeros(n_samples)
        for i in range(n_samples):
            queries = self._simulate_attack_queries(X_clean[i])
            attack_dk[i] = self._compute_cluster_tightness(queries)

        # Measure normal cluster tightness
        normal_dk = np.zeros(n_samples)
        for i in range(n_samples):
            queries = self._simulate_normal_queries()
            normal_dk[i] = self._compute_cluster_tightness(queries)

        logger.info(
            f"Cluster tightness — "
            f"attack: mean={attack_dk.mean():.6f}, std={attack_dk.std():.6f} | "
            f"normal: mean={normal_dk.mean():.6f}, std={normal_dk.std():.6f}"
        )

        all_dk = np.concatenate([attack_dk, normal_dk])
        tau_min, tau_max = all_dk.min(), np.percentile(all_dk, 75)
        taus = np.linspace(tau_min, tau_max, n_points)

        results = []
        for tau in taus:
            det_rate = (attack_dk < tau).mean() * 100
            fpr = (normal_dk < tau).mean() * 100
            results.append((tau, det_rate, fpr))

        logger.info("Calibration sweep (tau, det%, fpr%):")
        step = max(1, n_points // 10)
        for tau, det, fpr in results[::step]:
            logger.info(f"  tau={tau:.6f}: det={det:.1f}%, fpr={fpr:.1f}%")

        # Suggest best tau (maximize det - fpr)
        best_idx = max(range(len(results)),
                       key=lambda i: results[i][1] - results[i][2])
        best_tau, best_det, best_fpr = results[best_idx]
        logger.info(
            f"Suggested tau={best_tau:.6f}: det={best_det:.1f}%, fpr={best_fpr:.1f}%"
        )

        return results
