import logging
import os
from typing import Optional


_LOGGER: Optional[logging.Logger] = None


def setup_logging(output_dir: str) -> logging.Logger:
    global _LOGGER
    logger = logging.getLogger("ood_ts")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    _LOGGER = logger
    return logger


def get_logger() -> logging.Logger:
    if _LOGGER is None:
        return setup_logging("runs/default")
    return _LOGGER
