"""Minimal logging utility — provides get_logger for art_generator compatibility."""
import logging


def get_logger(name=None):
    return logging.getLogger(name)
