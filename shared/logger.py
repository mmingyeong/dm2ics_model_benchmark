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

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import logging
import os

def get_logger(name: str, log_dir: str = None, filename: str = None) -> logging.Logger:
    """
    Create or retrieve a logger instance with optional file logging.

    Parameters
    ----------
    name : str
        Logger name.
    log_dir : str, optional
        Directory where log file will be saved. If None, uses environment variable
        'DM2ICS_LOGDIR', or disables file logging if that is also unset.
    filename : str, optional
        Name of the log file. Defaults to "{name}.log".

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Avoid duplicate handlers

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Determine log directory
    if log_dir is None:
        log_dir = os.environ.get("DM2ICS_LOGDIR", None)
        if log_dir is None:
            # ✅ fallback: safe default directory in user's home
            home_dir = os.path.expanduser("~")
            log_dir = os.path.join(home_dir, "_dm2ics_model_benchmark", "logs")
    else:
        try:
            os.makedirs(log_dir, exist_ok=True)
            if filename is None:
                filename = f"{name}.log"
            fh = logging.FileHandler(os.path.join(log_dir, filename))
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except PermissionError:
            logger.warning(f"⚠️ Cannot write to log_dir: {log_dir} (Permission Denied)")

    return logger
