"""Autoencoder-based adversarial sample detector.

Detection logic: adversarial samples have higher reconstruction error
than clean samples. Threshold calibrated from clean data distribution.
"""

import logging

import numpy as np

from .evaluation import compute_asr

logger = logging.getLogger(__name__)


class AEDetector:
    """Detect adversarial samples via autoencoder reconstruction error.

    Threshold is set from clean data: tau = percentile(clean_errors, q).
    Samples with recon_error > tau are flagged as adversarial.
    """

    def __init__(self, ae_model, X_clean, cfg=None):
        if cfg is None:
            cfg = {}

        self.ae = ae_model

        # Fixed threshold takes priority over percentile-based
        fixed_tau = cfg.get("threshold_fixed")
        if fixed_tau is not None:
            self.tau = float(fixed_tau)
        else:
            percentile = cfg.get("threshold_percentile", 99.0)
            clean_errors = self.ae.reconstruction_error(X_clean)
            self.tau = np.percentile(clean_errors, percentile)

        clean_errors = self.ae.reconstruction_error(X_clean)
        self.clean_mean = clean_errors.mean()
        self.clean_std = clean_errors.std()

        tau_source = "fixed" if fixed_tau is not None else f"p{cfg.get('threshold_percentile', 99.0)}"
        logger.info(
            f"AEDetector ready: tau={self.tau:.6f} ({tau_source}), "
            f"clean error: mean={self.clean_mean:.6f}, std={self.clean_std:.6f}"
        )

    def detect(self, X):
        """Detect adversarial samples. Returns boolean array (True=adversarial)."""
        errors = self.ae.reconstruction_error(X)
        detected = errors > self.tau
        logger.debug(
            f"Detection: {detected.sum()}/{len(detected)}, "
            f"error: mean={errors.mean():.6f}, std={errors.std():.6f}"
        )
        return detected

    def compute_detection_rate(self, X_adv):
        """Fraction of adversarial samples detected."""
        detected = self.detect(X_adv)
        rate = detected.mean() * 100
        logger.info(f"Detection rate: {rate:.2f}% ({detected.sum()}/{len(detected)})")
        return rate

    def compute_fpr(self, X_clean):
        """False positive rate on clean data."""
        detected = self.detect(X_clean)
        fpr = detected.mean() * 100
        logger.info(f"FPR on clean: {fpr:.2f}% ({detected.sum()}/{len(detected)})")
        return fpr

    def adjusted_asr(self, y_true, y_clean_pred, y_adv_pred, X_adv):
        """ASR after rejecting detected adversarial samples.

        Blocked samples = attack fail (no prediction returned).
        ASR = successful attacks that slip through / total correctly-classified.
        """
        detected = self.detect(X_adv)
        passed = ~detected
        n_rejected = int(detected.sum())
        n_passed = int(passed.sum())

        correct_mask = y_true == y_clean_pred
        n_correct_total = int(correct_mask.sum())

        if n_correct_total == 0:
            adj = 0.0
        else:
            # Only count flips on samples that passed detection AND were correct
            passed_and_correct = passed & correct_mask
            flipped = y_clean_pred[passed_and_correct] != y_adv_pred[passed_and_correct]
            adj = flipped.sum() / n_correct_total * 100

        orig = compute_asr(y_true, y_clean_pred, y_adv_pred)
        logger.info(
            f"ASR: {orig:.2f}% -> {adj:.2f}% "
            f"(rejected {n_rejected}, passed {n_passed})"
        )
        return adj

    def calibrate_threshold(self, X_clean, X_adv, n_points=50):
        """Sweep threshold and report detection rate vs FPR."""
        clean_errors = self.ae.reconstruction_error(X_clean)
        adv_errors = self.ae.reconstruction_error(X_adv)

        logger.info(
            f"Recon error — "
            f"clean: mean={clean_errors.mean():.6f}, std={clean_errors.std():.6f} | "
            f"adv: mean={adv_errors.mean():.6f}, std={adv_errors.std():.6f}"
        )

        tau_min = np.percentile(clean_errors, 90)
        tau_max = np.percentile(adv_errors, 90)
        # Log-scale sweep to cover the wide range between clean and adv errors
        tau_min = max(tau_min, 1e-6)
        taus = np.geomspace(tau_min, tau_max, n_points)

        results = []
        for tau in taus:
            det_rate = (adv_errors > tau).mean() * 100
            fpr = (clean_errors > tau).mean() * 100
            results.append((tau, det_rate, fpr))

        logger.info("Calibration sweep (tau, det%, fpr%):")
        step = max(1, n_points // 10)
        for tau, det, fpr in results[::step]:
            logger.info(f"  tau={tau:.6f}: det={det:.1f}%, fpr={fpr:.1f}%")

        best_idx = max(range(len(results)),
                       key=lambda i: results[i][1] - results[i][2])
        best_tau, best_det, best_fpr = results[best_idx]
        logger.info(
            f"Suggested tau={best_tau:.6f}: det={best_det:.1f}%, fpr={best_fpr:.1f}%"
        )

        return results
