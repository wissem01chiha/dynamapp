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

import numpy as np
import pandas as pd
from os import path

"""
generate the follow CSV files for testing DataSource block
t1.csv] no header, no time col, choose data column by index=1

t2.csv] no header, has time data in col=0, choose data column by index=2

t3.csv] has header, no time col, choose data column by index=1

t4.csv] has header, has time data in col=0, choose data column by index=2

t5p0.csv] no header, has time data in col=1, choose data column by index=3
t5p0.csv] has header, has time data in col=1, choose data column=3 by name=c2

t6.csv] no header, has time data in col=0, data in col=1 that is same as time

t9.csv] no header, has time data in col=1, data in col=1 that is same as time
"""


def gen_files(workdir, stop_time):
    times = np.arange(0, stop_time)
    ones = np.ones(np.shape(times))
    sw = np.sin(times)
    cw = np.cos(times)
    df_t1 = pd.DataFrame({"c0": ones, "c1": sw, "c2": cw})

    df_t2 = pd.DataFrame({"time": times, "c0": ones, "c1": sw, "c2": cw})
    df_t5 = pd.DataFrame({"c0": ones, "time": times, "c1": sw, "c2": cw})

    times_t6 = np.arange(-2, stop_time)
    df_t6 = pd.DataFrame({"time": times_t6, "data": times_t6})

    df_t9 = pd.DataFrame({" c0": ones, "time": times, "c1": sw, "c2": cw})

    # Save test dataframes to csv files
    filenames = [
        "t1.csv",
        "t2.csv",
        "t3.csv",
        "t4.csv",
        "t5p0.csv",
        "t5p1.csv",
        "t6.csv",
        "t9.csv",
    ]
    filenames = [path.join(workdir, filename) for filename in filenames]
    df_t1.to_csv(filenames[0], header=False, index=False)
    df_t2.to_csv(filenames[1], header=False, index=False)
    df_t1.to_csv(filenames[2], header=True, index=False)
    df_t2.to_csv(filenames[3], header=True, index=False)
    df_t5.to_csv(filenames[4], header=False, index=False)
    df_t5.to_csv(filenames[5], header=True, index=False)
    df_t6.to_csv(filenames[6], header=False, index=False)
    df_t9.to_csv(filenames[7], header=True, index=False)

    return times, sw, times_t6, cw, filenames
