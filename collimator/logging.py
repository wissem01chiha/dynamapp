# Copyright (C) 2024 Collimator, Inc.
# SPDX-License-Identifier: AGPL-3.0-only
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation, version 3. This program is distributed in the hope that it
# will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General
# Public License for more details.  You should have received a copy of the GNU
# Affero General Public License along with this program. If not, see
# <https://www.gnu.org/licenses/>.

# https://www.firedrakeproject.org/_modules/firedrake/logging.html
import logging
import time
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING

BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
BLUE = "\033[34m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
LIGHTGREY = "\033[37m"
RESET = "\033[0m"

__all__ = [
    "logger",
    "set_log_level",
    "set_file_handler",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
]

packages = [__package__, "collimator_profiler"]


class ColorFormatter(logging.Formatter):
    @staticmethod
    def _level_color(level):
        if level >= CRITICAL:
            return RED
        if level >= ERROR:
            return RED
        if level >= WARNING:
            return YELLOW
        if level >= INFO:
            return GREEN
        if level >= DEBUG:
            return BLUE
        return CYAN

    def format(self, record):
        extras: dict | None = record.__dict__.get("extras")
        color = self._level_color(record.levelno)

        ftime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        s = f"{ftime}.{(1000*record.created)%1000:.0f} - {BOLD}[{record.name}][{color}{record.levelname}{RESET}]: {record.getMessage()}{RESET}"

        if extras:
            s += " " + " ".join(f"{LIGHTGREY}{k}{RESET}={v}" for k, v in extras.items())

        return s


__fmt = "%(name)s:%(levelname)s %(message)s"
__formatter = logging.Formatter(fmt=__fmt)
# __formatter = ColorFormatter()
__stream_handler = logging.StreamHandler()
__stream_handler.setFormatter(__formatter)


def set_file_handler(file, formatter=None):
    """Set a file handler to all packages."""
    if formatter is None:
        formatter = __formatter
    fh = logging.FileHandler(file, mode="w")
    fh.setFormatter(formatter)
    for package in packages:
        logger_ = logging.getLogger(package)
        logger_.addHandler(fh)


def set_stream_handler(handler=None):
    """Set the stream handler to all packages."""
    for package in packages:
        logger_ = logging.getLogger(package)
        logger_.addHandler(handler if handler else __stream_handler)


def unset_stream_handler():
    """Remove the stream handler from all packages."""
    for package in packages:
        logger_ = logging.getLogger(package)
        logger_.removeHandler(__stream_handler)


def set_log_level(level, pkg: str | None = None):
    """Set the log level for the specified or all packages.

    Args:
        level: The log level to set.
        pkg: If set, apply the log level only to the specified package.
    """
    if pkg is not None:
        logger_ = logging.getLogger(pkg)
        logger_.setLevel(level)
        return

    for package in packages:
        logger_ = logging.getLogger(package)
        logger_.setLevel(level)


def scope_logging(func):
    """Decorator to log function entry and exit."""

    def wrapper(*args, **kwargs):
        logger_ = logging.getLogger(__package__)
        logger_.debug("*** Entering %s ***", func.__qualname__)
        result = func(*args, **kwargs)
        logger_.debug("*** Exiting %s ***", func.__qualname__)
        return result

    return wrapper


def _block_info(block) -> dict:
    """Returns info about a block as a fake Error object"""
    if not block or not hasattr(block, "name_path_str"):
        return {}

    name_path = block.name_path_str
    uuid_path = block.ui_id_path
    if not name_path or not uuid_path:
        return {}

    return {
        "__error__": {
            "kind": "NotAnError",
            "name_path": name_path,
            "uuid_path": uuid_path,
        }
    }


def logdata(*, block=None, **kwargs):
    """Use this in log.info() and other logging functions to include block info:

    log.info("message", **logdata(block=self))
    """
    # "extra" is for python logging
    # "extras" is for our custom "api" for the frontend
    # "__errors__" is well understood by the frontend
    extras = kwargs or {}
    if block is not None:
        extras.update(_block_info(block))

    if len(extras) == 0:
        return {}

    return {"extra": {"extras": extras}}


logger = logging.getLogger(__package__)


def log(level, msg, *args, **kwargs):
    logger.log(level, msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)


def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)


def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)


def critical(msg, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)
