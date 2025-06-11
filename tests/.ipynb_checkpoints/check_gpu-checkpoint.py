"""
check_gpu_diagnostic.py

Description:
    Checks for GPU availability using PyTorch and prints out device information.
    Also performs extended diagnostics to help troubleshoot why GPU is not detected.

Author: Mingyeong Yang
Email: mmingyeong@kasi.re.kr
Created: 2025-06-10
"""

import torch
import logging
from datetime import datetime
import os
import subprocess
import sys

# 🔧 Set up logger
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = os.path.join(log_dir, f"check_gpu_diagnostic_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("🚀 Starting extended GPU availability check...")

logger.info(f"Torch version: {torch.__version__}")
logger.info(f"Python version: {sys.version}")
logger.info(f"Platform: {sys.platform}")

# 1. Check nvidia-smi
logger.info("🔍 Checking nvidia-smi...")
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        logger.info("nvidia-smi output:\n" + result.stdout)
    else:
        logger.warning("nvidia-smi failed:\n" + result.stderr)
except FileNotFoundError:
    logger.warning("nvidia-smi not found. NVIDIA driver may not be installed or PATH is not set.")
except Exception as e:
    logger.warning(f"nvidia-smi check error: {e}")

# 2. Check CUDA_HOME and environment variables
cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
logger.info(f"CUDA_HOME/CUDA_PATH: {cuda_home}")
logger.info(f"PATH: {os.environ.get('PATH')}")
logger.info(f"LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")

# 3. Check torch.cuda properties
logger.info("🔍 Checking torch.cuda properties...")
try:
    gpu_available = torch.cuda.is_available()
    logger.info(f"torch.cuda.is_available(): {gpu_available}")
    logger.info(f"torch.version.cuda: {torch.version.cuda}")
    logger.info(f"torch.backends.cudnn.version(): {torch.backends.cudnn.version()}")
    logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    logger.info(f"torch.cuda.current_device(): {torch.cuda.current_device() if gpu_available else 'N/A'}")
    logger.info(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0) if gpu_available else 'N/A'}")
except Exception as e:
    logger.error(f"Error querying torch.cuda: {e}")

# 4. Try importing torch with CUDA extension explicitly
try:
    import importlib.util
    spec = importlib.util.find_spec("torch._C")
    if spec is not None:
        logger.info("torch._C module found.")
    else:
        logger.warning("torch._C module NOT found.")
except Exception as e:
    logger.warning(f"torch._C import check error: {e}")

# 5. Print summary
if not torch.cuda.is_available():
    logger.warning("⚠ No GPU detected by PyTorch.")
    logger.warning("Possible reasons:")
    logger.warning("- NVIDIA driver not installed or not loaded")
    logger.warning("- CUDA toolkit not installed or not in PATH/LD_LIBRARY_PATH")
    logger.warning("- PyTorch not installed with CUDA support")
    logger.warning("- Running in a virtual environment without CUDA bindings")
    logger.warning("- Hardware issue or no NVIDIA GPU present")
else:
    logger.info("✅ GPU detected by PyTorch.")

logger.info("✅ Extended GPU check complete.")
logger.info(f"🔍 Log saved to: {log_path}")
