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

import pytest
from collimator.backend import numpy_api, DEFAULT_BACKEND

import logging
from collimator import logging as collimator_logging


@pytest.fixture(autouse=True)
def configure_logging():
    logger = logging.getLogger()
    level = logger.getEffectiveLevel()
    collimator_logging.set_log_level(level)
    yield


# Make sure we end up with the default backend for the other tests,
# since that's what the rest of the test cases will expect.
@pytest.fixture(autouse=True)
def reset_backend():
    yield
    numpy_api.set_backend(DEFAULT_BACKEND)
