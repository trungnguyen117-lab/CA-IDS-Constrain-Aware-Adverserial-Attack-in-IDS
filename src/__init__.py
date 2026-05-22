"""Shared package for adversarial-robustness pipelines (MODBUS-2023, IIOT-2021, IEC-104).

Each dataset workspace contains ``config.yaml`` + ``model_local/`` (DL class).
Run via the ``aider`` CLI — see ``src/cli.py``.
"""

import os
# macOS arm64 libomp coexist: sklearn / xgboost / lightgbm / catboost all link
# libomp.dylib. Multiple instances + multi-threaded training/loading hits
# OpenMP fatal-exit. ``setdefault`` keeps any value the user already set —
# override with e.g. ``OMP_NUM_THREADS=8 aider ...`` if your environment is
# stable. ``KMP_DUPLICATE_LIB_OK`` silences the duplicate-load assertion.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Import LightGBM before torch — both link libomp.dylib and on macOS the
# torch-bundled copy initialises OMP in a state that segfaults LightGBM
# predictions. Forcing LGBM first locks the right libomp.
import lightgbm  # noqa: F401, E402

# Silence noisy library warnings that flood logs during BB attacks (HSJA/Zoo
# call predict_proba thousands of times → sklearn warns once per call).
import warnings  # noqa: E402

warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Trying to unpickle estimator .* from version",
    category=UserWarning,
)

__version__ = "0.1.0"
