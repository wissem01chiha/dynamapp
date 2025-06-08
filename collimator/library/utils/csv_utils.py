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

"""
Utilities for reading csv files and extracting columns of data.
"""

from ...backend import numpy_api as cnp
from ...lazy_loader import LazyLoader

pd = LazyLoader("pd", globals(), "pandas")


def read_csv(file_path, header_as_first_row):
    """Reads a CSV file and returns a DataFrame."""
    try:
        header = 0 if header_as_first_row else None
        return pd.read_csv(
            file_path, skipinitialspace=True, dtype=cnp.float64, header=header
        )
    except Exception as e:
        raise IOError(f"Error reading {file_path}: {e}")


def extract_columns(df, cols):
    """
    Extracts columns from the DataFrame based on information in cols.
    Args:
        df: DataFrame
            The DataFrame to extract columns from.
        cols: int, str, list, tuple
            Either one of the following:
                - A string or integer (for a single column)
                - A list/tuple of strings or integers (for multiple columns)
                - A string representing a slice of columns, e.g. '0:3'
    Returns:
        A numpy array of the extracted columns from the DataFrame.
    """

    if isinstance(cols, (list, tuple)) and len(cols) == 1:
        cols = cols[0]

    # Handle a slice of columns when specified as a string (e.g., '0:3').
    if isinstance(cols, str) and ":" in cols:
        # Extract the start and end indices from the slice string.
        # If either side of the slice is empty, replace it with None for full slice behavior.
        start, end = (int(x) if x else None for x in cols.split(":"))
        extracted_data = df.iloc[:, start:end]

    # Handle integer or list/tuple of integers for direct index access.
    elif isinstance(cols, int) or (
        isinstance(cols, (list, tuple)) and all(isinstance(c, int) for c in cols)
    ):
        extracted_data = (
            df.iloc[:, cols] if isinstance(cols, int) else df.iloc[:, list(cols)]
        )

    # Handle a single column name or list/tuple of column names.
    elif isinstance(cols, str) or (
        isinstance(cols, (list, tuple)) and all(isinstance(c, str) for c in cols)
    ):
        extracted_data = df[cols] if isinstance(cols, str) else df[list(cols)]

    else:
        raise ValueError(
            "Invalid type for 'cols'. "
            "Must be a string, integer, list, tuple, or slice string."
        )

    # Convert to numpy array (with flexibility for library choice) and return.
    return cnp.array(extracted_data.to_numpy())
