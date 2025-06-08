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

from scipy.integrate import solve_ivp

import pysindy as ps
from pysindy.utils import lorenz_control


def create_pysindy_model_and_save_csv():
    # Generate data and train native PySINDy model
    rtol = 1e-12
    atol = 1e-12

    integrator_keywords = {}
    integrator_keywords["rtol"] = rtol
    integrator_keywords["method"] = "LSODA"
    integrator_keywords["atol"] = atol

    t_end_train = 2
    dt = 0.002

    # Control input
    def u_fun(t):
        return np.column_stack([np.sin(2 * t), t**2])

    t_train = np.arange(0, t_end_train, dt)
    t_train_span = (t_train[0], t_train[-1])
    x0_train = [-8.0, 8.0, 27.0]
    x_train = solve_ivp(
        lorenz_control,
        t_train_span,
        x0_train,
        t_eval=t_train,
        args=(u_fun,),
        **integrator_keywords,
    ).y.T
    u_train = u_fun(t_train)

    # Instantiate and fit the SINDYc model
    optimizer = ps.STLSQ(
        threshold=0.1, alpha=0.05, max_iter=50, normalize_columns=False
    )
    model = ps.SINDy(optimizer=optimizer)
    model.fit(x_train, u=u_train, t=dt)

    # Save data as csv for wildcat
    datadict = {
        "x_0": x_train[:, 0],
        "x_1": x_train[:, 1],
        "x_2": x_train[:, 2],
        "u_0": u_train[:, 0],
        "u_1": u_train[:, 1],
        "t": t_train,
    }

    df = pd.DataFrame(datadict)
    # add unique identifier to filename so that parallel tests
    # don't delete a file being used by another test
    filename = "lorenz_sindy.csv"
    df.to_csv(filename)


if __name__ == "__main__":
    create_pysindy_model_and_save_csv()
