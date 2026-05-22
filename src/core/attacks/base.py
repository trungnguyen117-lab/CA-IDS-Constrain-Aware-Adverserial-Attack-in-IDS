"""Base classes cho attack generator (Simple + Batched) + timeout utility."""

from __future__ import annotations

import logging
import signal
from abc import ABC, abstractmethod
from contextlib import contextmanager

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@contextmanager
def timeout_context(seconds: int):
    """SIGALRM-based timeout (Unix only). seconds <= 0 → no-op."""
    if seconds is None or seconds <= 0:
        yield
        return

    def handle(signum, frame):
        raise TimeoutError(f"Generation timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, handle)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


class AttackGenerator(ABC):
    def __init__(self, classifier, generator_params: dict | None = None):
        self.attack = None
        self.generator_params = generator_params
        self.classifier = classifier

    @abstractmethod
    def generate(self, x: np.ndarray, y: np.ndarray, input_metadata: dict,
                 mutate_indices: list[int] | None = None, **kwargs):
        """Generate adversarial samples → DataFrame."""
        raise NotImplementedError

    def update_generator_params(self, attack_params: dict):
        if self.generator_params is None:
            self.generator_params = attack_params.copy()
            return attack_params
        return_params = attack_params.copy()
        for key in return_params.keys():
            if key in self.generator_params:
                return_params[key] = self.generator_params[key]
        return return_params

    @staticmethod
    def format_output(x_adv, y, input_metadata):
        y_arr = y.reshape(-1) if y.ndim > 1 else y
        feature_names = input_metadata["feature_names"]
        df = pd.DataFrame(x_adv, columns=feature_names)
        df[input_metadata["label_column"]] = y_arr
        return df


class SimpleAttackGenerator(AttackGenerator):
    ATTACK_CLASS = None
    DEFAULT_PARAMS: dict = {}
    MASK_MODE = "mask"      # 'mask' or 'post_apply'
    NAME = "Attack"

    def __init__(self, classifier, generator_params: dict | None = None):
        super().__init__(classifier, generator_params)
        attack_params = self.update_generator_params({**self.DEFAULT_PARAMS})
        self.attack = self.ATTACK_CLASS(**self.attack_init_kwargs(attack_params))

    def attack_init_kwargs(self, attack_params):
        return {"estimator": self.classifier, **attack_params}

    def generate_raw(self, x, mask):
        if self.MASK_MODE == "mask":
            return self.attack.generate(x=x, mask=mask)
        return self.attack.generate(x=x)

    def generate(self, x, y, input_metadata, mutate_indices=None, **kwargs):
        mutate_indices = mutate_indices or []
        mask = np.ones(x.shape)
        if mutate_indices:
            mask[:, mutate_indices] = 0
        logger.info("[+] Generating adversarial samples with %s", self.NAME)
        x_adv = self.generate_raw(x, mask)
        if self.MASK_MODE == "post_apply" and mutate_indices:
            x_adv[:, mutate_indices] = x[:, mutate_indices]
        return self.format_output(x_adv, y, input_metadata)


class BatchedAttackGenerator(AttackGenerator):
    ATTACK_CLASS = None
    DEFAULT_PARAMS: dict = {}
    MASK_MODE = "mask"
    NAME = "Attack"

    def __init__(self, classifier, generator_params: dict | None = None):
        super().__init__(classifier, generator_params)
        attack_params = self.update_generator_params({**self.DEFAULT_PARAMS})
        self.attack = self.ATTACK_CLASS(classifier=self.classifier, **attack_params)

    def generate_batch(self, xb, maskb):
        if self.MASK_MODE == "mask":
            return self.attack.generate(x=xb, mask=maskb)
        return self.attack.generate(x=xb, y=None)

    def generate(self, x, y, input_metadata, mutate_indices=None, **kwargs):
        mutate_indices = mutate_indices or []
        if x is None or x.size == 0:
            cols = input_metadata.get("feature_names", []) + [input_metadata.get("label_column", "label")]
            return pd.DataFrame(columns=cols)
        mask = np.ones(x.shape)
        if mutate_indices:
            mask[:, mutate_indices] = 0
        batch_size = kwargs.get("batch_size", -1)
        max_retries = int(kwargs.get("max_retries", 3))
        placeholder_mode = kwargs.get("placeholder", "original")
        timeout_seconds = int(kwargs.get("timeout", -1))
        verbose_every = int(kwargs.get("verbose", 0))
        n = x.shape[0]
        if batch_size is None or batch_size <= 0 or batch_size >= n:
            logger.info("[+] %s (single batch)", self.NAME)
            x_adv = self.generate_batch(x, mask)
            if self.MASK_MODE == "post_apply" and mutate_indices:
                x_adv[:, mutate_indices] = x[:, mutate_indices]
        else:
            logger.info("[+] %s in batches of %d", self.NAME, batch_size)
            adv_parts, y_parts = [], []
            total = (n + batch_size - 1) // batch_size
            processed = failed = 0
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                xb, yb, maskb = x[start:end], y[start:end], mask[start:end]
                xb_adv = self.try_batch(xb, maskb, start, end, max_retries, timeout_seconds)
                if xb_adv is None:
                    failed += 1
                    if placeholder_mode == "drop":
                        logger.error("Dropping failed batch %d:%d", start, end)
                    else:
                        logger.error("Using original for failed batch %d:%d", start, end)
                        xb_adv = xb
                if xb_adv is not None:
                    if self.MASK_MODE == "post_apply" and mutate_indices:
                        xb_adv[:, mutate_indices] = xb[:, mutate_indices]
                    adv_parts.append(xb_adv)
                    y_parts.append(yb)
                processed += 1
                if verbose_every > 0 and (processed % verbose_every == 0 or processed == total):
                    logger.info("[%s] Progress: %d/%d, failed=%d",
                                self.NAME, processed, total, failed)
            if not adv_parts:
                cols = input_metadata.get("feature_names", []) + [input_metadata.get("label_column", "label")]
                return pd.DataFrame(columns=cols)
            x_adv = np.concatenate(adv_parts, axis=0)
            y = np.concatenate(y_parts, axis=0)
        return self.format_output(x_adv, y, input_metadata)

    def try_batch(self, xb, maskb, start, end, max_retries, timeout_seconds):
        for attempt in range(1, max_retries + 1):
            try:
                if timeout_seconds > 0:
                    with timeout_context(timeout_seconds):
                        return self.generate_batch(xb, maskb)
                return self.generate_batch(xb, maskb)
            except Exception as e:
                logger.warning("Batch %d:%d attempt %d/%d failed: %s",
                               start, end, attempt, max_retries, e)
        return None
