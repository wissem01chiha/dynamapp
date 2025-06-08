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

"""Non-linear Model Predictive Control (NMPC) blocks.

A few of these blocks are implemented using CyIPOPT, which may require
additional installation steps:
- Install CyIPOPT on your system (eg. `sudo apt install coinor-libipopt-dev` on
  Ubuntu)
- `pip install 'pycollimator[nmpc]'`
"""

from .direct_shooting_ipopt_nmpc import DirectShootingNMPC
from .direct_transcription_ipopt_nmpc import DirectTranscriptionNMPC
from .hermite_simpson_ipopt_nmpc import HermiteSimpsonNMPC
from .trajectory_optimization import trajopt


__all__ = [
    "DirectShootingNMPC",
    "DirectTranscriptionNMPC",
    "HermiteSimpsonNMPC",
    "trajopt",
]
