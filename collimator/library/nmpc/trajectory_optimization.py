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

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

from .hermite_simpson_ipopt_nmpc import HermiteSimpsonNMPC

if TYPE_CHECKING:
    from collimator.framework import SystemBase
    from collimator.backend.typing import Array

__all__ = [
    "trajopt",
]


def trajopt(
    plant: SystemBase,
    t0: float,
    x0: Array,
    Q: Array,
    R: Array,
    N: int,
    tf: float,
    xf: Array = None,
    x_ref: Array = None,
    u_ref: Array = None,
    lb_x: Array = None,
    ub_x: Array = None,
    lb_u: Array = None,
    ub_u: Array = None,
    QN: Array = None,
    constrain_xf: bool = False,
    constrain_uf: bool = False,
    x_guess: bool = None,
    u_guess: bool = None,
):
    dt = tf / N
    nx = Q.shape[0]
    nu = R.shape[0]
    if QN is None:
        QN = np.eye(nx)
    if not constrain_xf:
        QN *= 0.0
    mpc = HermiteSimpsonNMPC(
        plant=plant,
        Q=Q,
        QN=QN,
        R=R,
        N=N,
        dt=dt,
        lb_x=lb_x,
        ub_x=ub_x,
        lb_u=lb_u,
        ub_u=ub_u,
        include_terminal_x_as_constraint=constrain_xf,
        include_terminal_u_as_constraint=constrain_uf,
        x_optvars_0=x_guess,
        u_optvars_0=u_guess,
    )

    if x_ref is None:
        x_ref = np.zeros((N + 1, nx))
        # Only the final point is constrained in this case
        if xf is not None:
            x_ref[-1, :] = xf
    if u_ref is None:
        u_ref = np.zeros((N + 1, nu))

    if x_guess is None:
        if xf is not None:
            x_guess = x0 + (np.arange(N + 1) / N)[:, None] * (xf - x0)
        else:
            x_guess = np.zeros((N + 1, nx))
    if u_guess is None:
        u_guess = np.zeros((N + 1, nu))

    optvars_sol = mpc.solve_trajectory_optimzation(
        t0, x0, x_ref, u_ref, x_guess, u_guess
    )

    u_opt = optvars_sol[: mpc.nu * (mpc.N + 1)].reshape((mpc.N + 1, mpc.nu))
    x_opt = optvars_sol[mpc.nu * (mpc.N + 1) :].reshape((mpc.N + 1, mpc.nx))

    return x_opt, u_opt
