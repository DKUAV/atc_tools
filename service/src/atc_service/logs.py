"""Configure component logs for the API and worker processes."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(component: str, log_dir: Path) -> None:
    """Send service logs to the console and a per-component rotating file.

    Parameters
    ----------
    component
        Process label, such as api or worker.
    log_dir
        Host-backed directory for the operator logs.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("atc_service")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    console = logging.StreamHandler()
    logfile = RotatingFileHandler(
        log_dir / f"{component}.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    for sink in (console, logfile):
        sink.setFormatter(formatter)
        logger.addHandler(sink)
