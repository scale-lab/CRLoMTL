import logging
import os
import sys
from pathlib import Path
"""def setup_logging(config):
    model_name = config.MODEL.TYPE
    dataset = config.DATA.DATASET
    method = config.METHOD
    tag = config.TAG

    if tag is None:
        tag = f"{model_name}_{dataset}_{method}"

    logs_save_dir = os.path.join(config.OUTPUT, "logs")
    os.makedirs(logs_save_dir, exist_ok=True)

    log_dir = Path(logs_save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"{tag}.log"),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger(__name__)

    logger.info("Config:")
    logger.info(config)

    return logger, tag, logs_save_dir"""





class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()
        try:
            os.fsync(self.stream.fileno())
        except OSError:
            pass


def setup_logging(config):
    model_name = config.MODEL.TYPE
    dataset = config.DATA.DATASET
    method = config.METHOD
    tag = config.TAG

    if tag is None:
        tag = f"{model_name}_{dataset}_{method}"

    logs_save_dir = os.path.join(config.OUTPUT, "logs")
    log_dir = Path(logs_save_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{tag}.log"

    logger = logging.getLogger("CRLoMTL")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    )

    file_handler = FlushFileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Logging to: {log_file}")
    logger.info("Config:")
    logger.info(config)

    return logger, tag, logs_save_dir



