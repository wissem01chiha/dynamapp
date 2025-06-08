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
import pytest
import pandas as pd
from collimator.backend import numpy_api as cnp

from collimator.library.utils import read_csv, extract_columns

#################### Fixtures ####################


@pytest.fixture
def csv_with_header():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_script_dir, "assets", "with_header.csv")
    return file_path


@pytest.fixture
def csv_without_header():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_script_dir, "assets", "without_header.csv")
    return file_path


#################### CSV reading ####################


def test_read_csv_with_header(csv_with_header):
    df = read_csv(csv_with_header, header_as_first_row=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 5
    assert df.shape == (10, 5)


def test_read_csv_without_header(csv_without_header):
    df = read_csv(csv_without_header, header_as_first_row=False)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (10, 5)
    assert all(isinstance(col, int) for col in df.columns)


#################### String-based column extraction ####################


def test_extract_columns_single_column(csv_with_header):
    df = read_csv(csv_with_header, header_as_first_row=True)
    extracted = extract_columns(df, "col0")
    assert extracted.shape == (10,)
    assert isinstance(extracted, cnp.ndarray)


def test_extract_columns_single_column_as_a_list(csv_with_header):
    df = read_csv(csv_with_header, header_as_first_row=True)
    extracted = extract_columns(df, ["col0"])
    assert extracted.shape == (10,)
    assert isinstance(extracted, cnp.ndarray)


def test_extract_columns_multiple_columns(csv_with_header):
    df = read_csv(csv_with_header, header_as_first_row=True)
    extracted = extract_columns(df, ["col0", "col1"])
    assert extracted.shape == (10, 2)
    assert isinstance(extracted, cnp.ndarray)


def test_extract_columns_slice(csv_with_header):
    df = read_csv(csv_with_header, header_as_first_row=True)
    extracted = extract_columns(df, "0:3")
    assert extracted.shape == (10, 3)
    assert isinstance(extracted, cnp.ndarray)


def test_extract_columns_slice_with_negative_index(csv_with_header):
    df = read_csv(csv_with_header, header_as_first_row=True)
    extracted = extract_columns(df, "1:-1")
    assert extracted.shape == (10, 3)
    assert isinstance(extracted, cnp.ndarray)


def test_extract_columns_full_front_slice_with_negative_index(csv_with_header):
    df = read_csv(csv_with_header, header_as_first_row=True)
    extracted = extract_columns(df, ":-1")
    assert extracted.shape == (10, 4)
    assert isinstance(extracted, cnp.ndarray)


def test_extract_columns_full_end_slice_with_negative_index(csv_with_header):
    df = read_csv(csv_with_header, header_as_first_row=True)
    extracted = extract_columns(df, "1:")
    assert extracted.shape == (10, 4)
    assert isinstance(extracted, cnp.ndarray)


#################### Index-based column extraction ####################


@pytest.mark.parametrize("csv_file", ["csv_with_header", "csv_without_header"])
def test_extract_columns_single_column_by_index(request, csv_file):
    df = read_csv(
        request.getfixturevalue(csv_file),
        header_as_first_row=(csv_file == "csv_with_header"),
    )
    extracted = extract_columns(df, 0)
    assert extracted.shape == (10,)
    assert isinstance(extracted, cnp.ndarray)


@pytest.mark.parametrize("csv_file", ["csv_with_header", "csv_without_header"])
def test_extract_columns_multiple_columns_by_index(request, csv_file):
    df = read_csv(
        request.getfixturevalue(csv_file),
        header_as_first_row=(csv_file == "csv_with_header"),
    )
    extracted = extract_columns(df, [0, 1])
    assert extracted.shape == (10, 2)
    assert isinstance(extracted, cnp.ndarray)


@pytest.mark.parametrize("csv_file", ["csv_with_header", "csv_without_header"])
def test_extract_columns_with_negative_index(request, csv_file):
    df = read_csv(
        request.getfixturevalue(csv_file),
        header_as_first_row=(csv_file == "csv_with_header"),
    )
    extracted = extract_columns(df, -1)
    assert extracted.shape == (10,)
    assert isinstance(extracted, cnp.ndarray)


@pytest.mark.parametrize("csv_file", ["csv_with_header", "csv_without_header"])
def test_extract_columns_multiple_columns_by_negative_index(request, csv_file):
    df = read_csv(
        request.getfixturevalue(csv_file),
        header_as_first_row=(csv_file == "csv_with_header"),
    )
    extracted = extract_columns(df, [0, -1])
    assert extracted.shape == (10, 2)
    assert isinstance(extracted, cnp.ndarray)


def test_extract_columns_invalid_type(csv_with_header):
    df = read_csv(csv_with_header, header_as_first_row=True)
    # Attempting to extract columns with an invalid type should raise a ValueError
    with pytest.raises(ValueError):
        extract_columns(df, 1.5)
