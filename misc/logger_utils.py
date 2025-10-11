#!/usr/bin/env python3
"""
logger_utils.py

Simple logging setup with console and rotating file handler.
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(name: str, log_file: str = 'experiment.log', level=logging.INFO, max_bytes=10_000_000, backup_count=5):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    # Rotating file handler
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count)
    fh.setLevel(level)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    return logger


if __name__ == '__main__':
    logger = setup_logger('test_logger', 'logs/test.log')
    logger.info('Logger initialized')
