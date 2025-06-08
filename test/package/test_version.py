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

import os
import toml

from collimator import __version__


def _get_version() -> str:
    dirname = os.path.dirname(os.path.abspath(__file__))
    pyproject_path = os.path.join(dirname, "..", "..", "pyproject.toml")
    with open(pyproject_path, "r", encoding="utf-8") as file:
        pyproject = toml.load(file)
    return pyproject["project"]["version"]


def test_version():
    """Checks that version number matches between pyproject.toml and version.py.
    Also checks that version number is in the correct format.
    """
    version = __version__.split(".")
    assert len(version) == 3 or len(version) == 4

    # Check major and minor
    assert version[0] == "2"
    assert version[1] == "0"

    # Check micro
    assert version[2].isdigit()
    assert len(version) == 3 or version[3].startswith("alpha")

    # Check version in pyproject.toml
    assert (
        _get_version() == __version__
    ), "Version mismatch between pyproject.toml and version.py"
