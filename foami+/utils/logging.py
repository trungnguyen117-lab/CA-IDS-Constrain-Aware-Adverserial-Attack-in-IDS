"""Logging utilities — self-contained copy of src/utils/logging.py."""
import logging
import warnings
from typing import Optional

from colorama import Fore, Style, init

init()

warnings.filterwarnings(
    "ignore",
    message=r".*glibc.*older than 2\.28.*",
    category=FutureWarning,
    module=r"xgboost\.core",
)


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG':    Fore.BLUE,
        'INFO':     Fore.GREEN,
        'WARNING':  Fore.YELLOW,
        'ERROR':    Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }

    def format(self, record: logging.LogRecord) -> str:
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{Style.RESET_ALL}"
            )
        return super().format(record)


def setup_logging(log_level: str = 'INFO') -> None:
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.addHandler(handler)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('art').setLevel(logging.ERROR)
    logging.getLogger('numba').setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name)
