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

"""Generates CSV files for testing purposes."""

import pandas as pd
import numpy as np


def create_test_csv(file_path, include_header=True, rows=5, cols=3):
    """Generates a CSV file for testing."""
    data = np.random.rand(rows, cols)
    if include_header:
        header = [f"col{i}" for i in range(cols)]
    else:
        header = None
    df = pd.DataFrame(data, columns=header)
    df.to_csv(file_path, index=False, header=include_header)


if __name__ == "__main__":
    create_test_csv("with_header.csv", include_header=True, rows=10, cols=5)
    create_test_csv("without_header.csv", include_header=False, rows=10, cols=5)
