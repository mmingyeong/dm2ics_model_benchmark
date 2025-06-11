"""
logger.py

Description:
    Logging utility for unified console and file output across all modules.

Author:
    Mingyeong Yang (양민경), PhD Student, UST-KASI
Email:
    mmingyeong@kasi.re.kr

Created:
    2025-06-10

Usage:
    from shared.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Training started.")

License:
    For academic use only. Contact the author before redistribution.
"""

import logging
import os
from datetime import datetime

def get_logger(name: str, log_dir: str = "logs", level=logging.INFO) -> logging.Logger:
    """
    Creates a logger that outputs to both console and a timestamped log file.

    Parameters
    ----------
    name : str
        Name of the logger (usually `__name__`).
    log_dir : str
        Directory to save log files.
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicated logs if multiple handlers

    # Formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info(f"Logger initialized. Outputting to {log_path}")
    return logger
