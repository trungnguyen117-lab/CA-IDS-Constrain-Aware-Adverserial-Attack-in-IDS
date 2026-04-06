"""Base classes for adversarial attack generators.

Hierarchy:
  AttackGenerator (ABC)
    ├─ SimpleAttackGenerator  — single-call attacks (FGSM, PGD, CW, DeepFool, Zoo, BIM, MIM)
    └─ BatchedAttackGenerator — batched attacks with retry/timeout (HSJA, JSMA)
"""

import logging
import signal
from abc import ABC, abstractmethod
from contextlib import contextmanager

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Timeout utility ──────────────────────────────────────────────────────────


@contextmanager
def timeout_context(seconds: int):
    """Context manager for SIGALRM-based timeout (Unix only)."""
    if seconds is None or seconds <= 0:
        yield
        return

    def _handle(signum, frame):
        raise TimeoutError(f"Generation timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handle)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


# ── Base ABC ─────────────────────────────────────────────────────────────────


class AttackGenerator(ABC):

    def __init__(self, classifier, generator_params: dict = None):
        self.attack = None
        self.generator_params = generator_params
        self.classifier = classifier

    @abstractmethod
    def generate(self, x: np.ndarray, y: np.ndarray, input_metadata: dict,
                 mutate_indices: list[int] = [], **kwargs):
        """Generate adversarial samples. Returns a pandas DataFrame."""
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
    def _format_output(x_adv, y, input_metadata):
        """Build DataFrame from adversarial features + labels."""
        y_arr = y.reshape(-1) if y.ndim > 1 else y
        feature_names = input_metadata["feature_names"]
        df = pd.DataFrame(x_adv, columns=feature_names)
        df[input_metadata["label_column"]] = y_arr
        return df


# ── Simple (single-call) attacks ─────────────────────────────────────────────


class SimpleAttackGenerator(AttackGenerator):
    """Base for attacks that run in a single call.

    Subclasses must set:
      ATTACK_CLASS: ART attack class
      DEFAULT_PARAMS: dict of default attack parameters
      MASK_MODE: 'mask' (pass mask kwarg) or 'post_apply' (restore after)
      NAME: display name for logging
    """

    ATTACK_CLASS = None
    DEFAULT_PARAMS = {}
    MASK_MODE = "mask"  # 'mask' or 'post_apply'
    NAME = "Attack"

    def __init__(self, classifier, generator_params: dict = None):
        super().__init__(classifier, generator_params)
        attack_params = self.update_generator_params({**self.DEFAULT_PARAMS})
        self.attack = self.ATTACK_CLASS(
            **self._attack_init_kwargs(attack_params)
        )

    def _attack_init_kwargs(self, attack_params):
        """Build kwargs for ART attack constructor. Override if needed."""
        return {"estimator": self.classifier, **attack_params}

    def _generate_raw(self, x, mask):
        """Call attack.generate with appropriate kwargs. Override for custom calls."""
        if self.MASK_MODE == "mask":
            return self.attack.generate(x=x, mask=mask)
        else:
            return self.attack.generate(x=x)

    def generate(self, x, y, input_metadata, mutate_indices=[], **kwargs):
        mask = np.ones(x.shape)
        if mutate_indices:
            mask[:, mutate_indices] = 0

        logger.info(f"[+] Generating adversarial samples with {self.NAME}")
        x_adv = self._generate_raw(x, mask)

        if self.MASK_MODE == "post_apply" and mutate_indices:
            x_adv[:, mutate_indices] = x[:, mutate_indices]

        return self._format_output(x_adv, y, input_metadata)


# ── Batched attacks (retry + timeout) ────────────────────────────────────────


class BatchedAttackGenerator(AttackGenerator):
    """Base for slow attacks that benefit from batching with retry/timeout.

    Subclasses must set:
      ATTACK_CLASS: ART attack class
      DEFAULT_PARAMS: dict of default attack parameters
      MASK_MODE: 'mask' or 'post_apply'
      NAME: display name for logging

    And may override:
      _generate_batch(xb, maskb): single-batch generation call
    """

    ATTACK_CLASS = None
    DEFAULT_PARAMS = {}
    MASK_MODE = "mask"
    NAME = "Attack"

    def __init__(self, classifier, generator_params: dict = None):
        super().__init__(classifier, generator_params)
        attack_params = self.update_generator_params({**self.DEFAULT_PARAMS})
        self.attack = self.ATTACK_CLASS(classifier=self.classifier, **attack_params)

    def _generate_batch(self, xb, maskb):
        """Generate adv for a single batch. Override for custom calls."""
        if self.MASK_MODE == "mask":
            return self.attack.generate(x=xb, mask=maskb)
        else:
            return self.attack.generate(x=xb, y=None)

    def generate(self, x, y, input_metadata, mutate_indices=[], **kwargs):
        if x is None or x.size == 0:
            cols = input_metadata.get("feature_names", []) + [input_metadata.get("label_column", "label")]
            return pd.DataFrame(columns=cols)

        # Mask setup
        mask = np.ones(x.shape)
        if mutate_indices:
            mask[:, mutate_indices] = 0

        # Batching params from kwargs
        batch_size = kwargs.get("batch_size", -1)
        max_retries = int(kwargs.get("max_retries", 3))
        placeholder_mode = kwargs.get("placeholder", "original")
        timeout_seconds = int(kwargs.get("timeout", -1))
        verbose_every = int(kwargs.get("verbose", 0))
        n_samples = x.shape[0]

        if batch_size is None or batch_size <= 0 or batch_size >= n_samples:
            # Single batch
            logger.info(f"[+] Generating adversarial samples with {self.NAME} (single batch)")
            x_adv = self._generate_batch(x, mask)
            if self.MASK_MODE == "post_apply" and mutate_indices:
                x_adv[:, mutate_indices] = x[:, mutate_indices]
        else:
            # Multi-batch with retry/timeout
            logger.info(f"[+] Generating adversarial samples with {self.NAME} in batches of {batch_size}")
            adv_parts, y_parts = [], []
            total_batches = (n_samples + batch_size - 1) // batch_size
            processed = 0
            failed = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                xb, yb, maskb = x[start:end], y[start:end], mask[start:end]

                xb_adv = self._try_batch(xb, maskb, start, end,
                                         max_retries, timeout_seconds)

                if xb_adv is None:
                    failed += 1
                    if placeholder_mode == "drop":
                        logger.error(f"Dropping failed batch {start}:{end}")
                    else:
                        logger.error(f"Using original for failed batch {start}:{end}")
                        xb_adv = xb

                if xb_adv is not None:
                    if self.MASK_MODE == "post_apply" and mutate_indices:
                        xb_adv[:, mutate_indices] = xb[:, mutate_indices]
                    adv_parts.append(xb_adv)
                    y_parts.append(yb)

                processed += 1
                if verbose_every and verbose_every > 0 and (
                    processed % verbose_every == 0 or processed == total_batches
                ):
                    logger.info(f"[{self.NAME}] Progress: {processed}/{total_batches}, failed={failed}")

            if not adv_parts:
                cols = input_metadata.get("feature_names", []) + [input_metadata.get("label_column", "label")]
                return pd.DataFrame(columns=cols)

            x_adv = np.concatenate(adv_parts, axis=0)
            y = np.concatenate(y_parts, axis=0)

        return self._format_output(x_adv, y, input_metadata)

    def _try_batch(self, xb, maskb, start, end, max_retries, timeout_seconds):
        """Try generating a batch with retries and optional timeout."""
        for attempt in range(1, max_retries + 1):
            try:
                if timeout_seconds > 0:
                    with timeout_context(timeout_seconds):
                        return self._generate_batch(xb, maskb)
                else:
                    return self._generate_batch(xb, maskb)
            except Exception as e:
                logger.warning(f"Batch {start}:{end} attempt {attempt}/{max_retries} failed: {e}")
        return None
