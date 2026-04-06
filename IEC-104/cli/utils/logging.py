"""Logging setup with colored output, warning suppression, and auto file logging."""

import logging
import os
import sys
import warnings

_LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "logs")


def _auto_log_path():
    """Build log dir + filename from sys.argv automatically.

    Script name becomes subdirectory, CLI args become filename.
    E.g. `python cli/pipeline/1_generate_adv.py --target xgb --attack hsja --source train`
    -> logs/1_generate_adv/xgb_hsja_train.log
    """
    script = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    log_dir = os.path.join(_LOGS_DIR, script)

    argv = sys.argv[1:]
    parts = []
    skip_keys = {"--log-level", "--device", "-d", "--log_level"}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in skip_keys:
            i += 2  # skip flag + value
            continue
        if arg.startswith("-"):
            i += 1  # skip flag name
            continue
        parts.append(arg)
        i += 1

    filename = "_".join(parts) + ".log" if parts else "run.log"
    return log_dir, filename


def setup_logging(level="INFO"):
    """Configure root logger with colored console + auto file handler."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    for name in ("art", "numba", "matplotlib", "catboost"):
        logging.getLogger(name).setLevel(logging.WARNING)

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%H:%M:%S"

    # Console handler
    try:
        from colorama import Fore, Style, init
        init(autoreset=True)

        class ColorFormatter(logging.Formatter):
            COLORS = {
                logging.DEBUG: Fore.CYAN,
                logging.INFO: Fore.GREEN,
                logging.WARNING: Fore.YELLOW,
                logging.ERROR: Fore.RED,
                logging.CRITICAL: Fore.RED + Style.BRIGHT,
            }

            def format(self, record):
                color = self.COLORS.get(record.levelno, "")
                record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
                return super().format(record)

        console = logging.StreamHandler()
        console.setFormatter(ColorFormatter(fmt, datefmt=datefmt))
    except ImportError:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(console)

    # Auto file handler
    log_dir, filename = _auto_log_path()
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(log_dir, filename), mode="a", encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(fh)

    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name=None):
    return logging.getLogger(name)
