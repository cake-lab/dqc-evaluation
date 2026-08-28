import logging
import os
import sys
from pathlib import Path

logging.getLogger("qiskit").setLevel(os.getenv("QISKIT_LOG_LEVEL", "WARNING").upper())


def setup_logging(run_dir: Path) -> logging.Logger:
    """Configure logging to file and console.

    Configures a logger that writes:
    - All DEBUG and above messages to output.log file
    - INFO and above messages to stdout console

    Args:
        run_dir: Directory where the log file will be created

    Returns:
        Configured logger instance
    """
    log_file = run_dir / "output.log"

    # Create logger
    logger = logging.getLogger("dqc_evaluation")
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # File handler - logs everything (DEBUG and above)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # Console handler - logs INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Formatter with timestamp and level
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Prevent propagation to parent loggers
    logger.propagate = False

    # Also capture warnings from libraries to file and console
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.addHandler(file_handler)
    warnings_logger.addHandler(console_handler)
    warnings_logger.propagate = False

    return logger
