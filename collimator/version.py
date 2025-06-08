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

"""collimator version information."""

# Keep in sync with pyproject.toml
# Carefully bump version number based on changes:
# add .alphaN to prepublish alpha releases
# minor update for major new features
# major update shouldn't happen for now
# TODO: better respect semver (re: breaking api changes)
__version__ = "2.0.6"
