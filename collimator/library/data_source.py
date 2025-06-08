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

"""DataSource block for loading data into a simulation from a CSV file."""

import numpy as np

from .generic import SourceBlock
from ..framework import parameters
from collimator.lazy_loader import LazyLoader
from collimator.backend import numpy_api as cnp

pd = LazyLoader("pd", globals(), "pandas")


__all__ = [
    "DataSource",
]


class DataSource(SourceBlock):
    """Produces outputs from an imported .csv file.

    The block's output(s) must be synchronized with simulation time. This can be
    achieved by two mechanisms:

    1. Each data row in the file is accompanied by a time value. The time value
        for each row is provided as a column in the data file. For this option,
        the values in the time column must be strictly increasing, with no duplicates,
        from the first data row to the last. The block will check that this condition
        is satisfied at compile time. The column with the time values is identified by
        the column index. This option assumes the left most column is index 0, counting
        up to the right. to select this option, set Time samples as column to True, and
        provide the index of the column.

    2. The time value for each data row is defined using a fixed time step between
        each row. For this option, the Sampling parameter defines the time step.
        The block then computes the time values for each data row starting with zero
        for the first row. Note that by definition, this results in a strictly
        increasing set. To select this option, set `time_samples_as_column` to False,
        and provide the `sampling_interval` value.

    When block output(s) are requested at a simulation time that falls between time
    values for adjacent data rows, there are two options for how the block should
    compute the interpolation:

    1. Zero Order Hold: the block returns data from the row with the lower time value.

    2. Linear: the block performs a linear interpolation between the lower and higher
        time value data rows.

    There are several mechanism for selecting which data columns are included in the
    block output(s). All options are applied using the `data_columns` parameter:

    1. Column name: enter a string that matches a column name in the header. For
        this option, `header_as_first_row` must be set to True. For this option, it
        is only possible to select a single column for the output. The block will
        output a scalar.

    2. Column index: enter an integer index for the desired column. This option
        again assumes the left most column is index 0, counting up to the right. This
        option assumes the same column index regardless of of whether
        `time_samples_as_column` is True or False, therefore it is possible to select
        the same column for time and output. With this option, the block will output
        a scalar.

    3. Slice: enter a slice used to identify a set of sequential columns to be used
        as the desired data for output. The slice works like a NumPy slice. For
        example, if the file has 10 columns, `3:8` will results in the block returning
        a vector of length 5, containing, in order, columns 3,4,5,6,7. Note that
        like NumPy, the second integer in the slice is excluded in the set of
        indices. Only positive integers are allowed for the slice (e.g. `2:-1`,
        `-3:-1`, and `3:` are not allowed).

    Presently, there is only one option for extrapolation beyond the end of data in
    the file. The block will have reached the end of data if the simulation time is
    greater than the time value for the last row of data. Once this occurs, the block
    output(s) will be the values in the last row of data.

    Parameters:
        file_name:
            The name of the imported file which contains the data.
        header_as_first_row:
            Check this box if the first row is meant to be a header.
        time_samples_as_column:
            Check this box to select a column form the file to use as the time values.
            Uncheck it to provide time as a fixed time step between rows.
        time_column:
            Only used when `time_samples_as_column` is True. This is the index of
            the column to be used as time.
        sampling_interval: only used when `time_samples_as_column` is False. Provide
            the fixed time step value here.
        data_columns:
            Enter name, index, or slice to select columns from the data file.
        extrapolation: the extrapolation method.  One of "hold" or "zero".
        interpolation: the interpolation method.  One of "zero_order_hold" or "linear".
    """

    @parameters(
        static=[
            "file_name",
            "data_columns",
            "extrapolation",
            "header_as_first_row",
            "interpolation",
            "sampling_interval",
            "time_column",
            "time_samples_as_column",
        ]
    )
    def __init__(
        self,
        file_name: str,
        data_columns: str = "1",  # slice, e.g. 3:4
        extrapolation: str = "hold",
        header_as_first_row: bool = False,
        interpolation: str = "zero_order_hold",
        sampling_interval: float = 1.0,
        time_column: str = "0",  # @am. could be an int
        time_samples_as_column: bool = False,
        **kwargs,
    ):
        # FIXME: legacy compat
        if "data_integration_id" in kwargs:
            del kwargs["data_integration_id"]

        # name = kwargs.get("name", "no name")
        # print(f"DataSource.init(). name={name} fnam={file_name}")

        # Load and preprocess the csv
        if header_as_first_row:
            header = 0
        else:
            header = None
        # print(f"header={header}")
        usecols = self.str2colindices(data_columns)
        # print(f"usecols={usecols}")

        df = pd.read_csv(
            file_name, header=header, skipinitialspace=True, dtype=np.float64
        )

        # print(f"df after read_csv\n{df}")

        if time_samples_as_column:
            index = int(time_column)
            index = df.columns[index]
        else:
            index = self.make_time_col(0.0, sampling_interval, df.shape[0])
        # print(f"index={index}")
        # print(f"df.columns={df.columns}")
        # print(f"df.columns.isin(usecols)={df.columns.isin(usecols)}")

        # for all colmuns that are not requested by user for output,
        # create a filter that can be used to remove these from the dataframe.
        # this ensures we get the correct output dimension, with the correct values.
        # Note, if one of the columns is time samples, we compute this filter before
        # setting the time column as the index of the dataframe because the user
        # may have provided data_columns as indices, and these would be relative to
        # the file including the time column.
        # first check if usecols and df.columns are the same identifiers
        col_filter = df.columns.isin(usecols)
        if not any(col_filter):
            # if usecols is indices, and df.coliumn is names
            # we need to create a filter that first converts usecols
            # to a list of column names
            # but first ensure that usecols range doens't go beyond
            # the number of columns
            # print(f"len(df.columns)={len(df.columns)}")
            if len(usecols) == 1:
                if usecols[0] > len(df.columns) - 1:
                    usecols[0] = len(df.columns) - 1
            else:
                lwr = min(usecols)
                # add 1 here because max(range(1,4))=3, and range(1,3) would be incorrect.
                upr = max(usecols) + 1
                if upr > len(df.columns) + 1:
                    upr = len(df.columns) + 1
                usecols = range(lwr, upr)

            # print(f"usecols={usecols}")
            col_filter = df.columns.isin(df.columns[usecols])

        # print(f"col_filter={col_filter} before removing time col if its in there.")

        # now that we have computed the filter based on orignal col indices of the file
        # we can move the time col to the dataframe index if that is what was requested
        df.set_index(index, inplace=True)
        # print(f"df after set index\n{df}")

        if time_samples_as_column:
            # if the time col was in the file, and it was used as the dataframe index
            # it would have been part of the filter, but we have since removed the column,
            # this means we need to remove that element of the filter.
            # del col_filter[int(time_column)]
            col_filter = np.delete(col_filter, int(time_column))
            # print(f"col_filter={col_filter} after removing time")

        df = df.loc[:, col_filter]

        # print(f"df when done\n{df}")

        times = cnp.array(df.index.to_numpy())
        data = cnp.array(df.to_numpy())
        max_i_zoh = len(times) - 1
        max_i_interp = len(times) - 2
        output_dim = data.shape[1]
        self._scalar_output = output_dim == 1
        # print(f"self._scalar_output ={self._scalar_output }")

        def get_below_row_idx(time, max_i):
            """
            first we clip the value of 'time' so that it falls inside the
            range of 'times'. this ensures we dont get strange extrapolation behavior.
            then, find the index of 'times' row value that is largest but still smaller
            than 'time'. we use this to locate the rows in 'times' that bound 'time'.
            """
            time_clipped = cnp.clip(time, times[0], times[-1])
            i = cnp.argmin(time_clipped >= times) - 1
            return i, time_clipped

        def _func_zoh(time):
            i, time_clipped = get_below_row_idx(time, max_i_zoh)
            return cnp.where(
                extrapolation == "zero" and time > times[-1],
                cnp.zeros(output_dim),
                data[i, :],
            )

        def _func_interp(time):
            """
            the second lambda function does this:
            y = (yp2-yp1)/(xp2-xp1)*(x-xp1) + yp1
            but does so by operating on the arrays
            ap1 and ap2 which provide the yp1's and yp2's.
            the xp1's and xp2's are time values.
            """
            i, time_clipped = get_below_row_idx(time, max_i_interp)
            ap1 = data[i, :]
            ap2 = data[i + 1, :]

            return cnp.where(
                extrapolation == "zero" and time > times[-1],
                cnp.zeros(output_dim),
                (ap2 - ap1) / (times[i + 1] - times[i]) * (time_clipped - times[i])
                + ap1,
            )

        # wrap output function to return scalar when only one column selected.
        def _wrap_func(_func):
            def _wrapped_func(time):
                output = _func(time)
                return output[0]

            return _wrapped_func

        if interpolation == "zero_order_hold":
            _func = _func_zoh
        else:
            _func = _func_interp

        if self._scalar_output:
            _func = _wrap_func(_func)

        super().__init__(_func, **kwargs)

        if data.size == 0:
            raise ValueError(
                f"DataSource {self.name} did not succeed in getting the requested data columns."
            )

    def make_time_col(self, start, step, n):
        """
        this function computes the time vector from sample
        interval when this is requested by user
        """
        return np.array([i * step + start for i in range(n)])

    def str2colindices(self, s):
        """
        this function parses the data_columns arg and returns
        a range, list, as appropriate to filter for the requested
        columns.
        """
        slice_thing = s.split(":")
        list_thing = s.split(",")
        if s.isdigit():
            # user provided a single index as integer
            index = int(s)
            return [index]
        elif len(slice_thing) == 2:
            start = slice_thing[0]
            end = slice_thing[1]
            if start.isdigit() and end.isdigit():
                # user provided a slice, e.g. 2:4
                return range(int(start), int(end))
            else:
                raise ValueError(f"DataSource data_columns={s} is not a proper slice")
        elif len(list_thing) > 1:
            is_list_of_digits = [i.isdigit() for i in list_thing]
            if all(is_list_of_digits):
                # user prvied a list of indices as integers
                return [int(i) for i in list_thing]
            elif any(is_list_of_digits):
                raise ValueError(
                    f"DataSource data_columns={s} is not a list of either ints or col names. mixting not allowed."
                )
            else:
                # user maybe provided a list of column names
                # we cant know if any or none of the provided names
                # match until we load the csv
                return list_thing

        elif len(slice_thing) == 1 and len(list_thing) == 1:
            # might be a single col name, e.g. 'c1'
            # again, we dont know if it matches the column names
            # until we load the csv
            return [s]
        else:
            raise ValueError(f"DataSource data_columns={s} is not comprehensible")
