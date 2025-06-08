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

import time

import numpy as np


class MetricsWriter:
    """
    MetricsWriter will write metrics to a CSV file, as they come. This is intended
    for live monitoring of optimization progress.
    """

    def __init__(self, path: str):
        self.path = path
        self.file = open(self.path, "w", encoding="utf-8", buffering=1)
        self.last_write_time = 0
        self._header = None

    def __del__(self):
        self.close()

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None

    # TODO? Use https://docs.python.org/3/library/csv.html#csv.DictWriter instead?
    def write_metrics(self, **values):
        if self._header is None:
            self._header = list(values.keys())
            self.file.write(",".join(self._header) + "\n")

        self.last_write_time = time.time()
        values = [np.asarray(v).tolist() for v in values.values()]
        self.file.write(",".join(str(v) for v in values) + "\n")
