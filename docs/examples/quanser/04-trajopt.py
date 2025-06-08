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
import sys
from time import sleep

import numpy as np

import jax
import jax.numpy as jnp

import collimator
from collimator import library

from controllers import *  # noqa: F403

HOME = os.path.expanduser("~")
sys.path.append(f"{HOME}/Quanser/libraries/python")


#
# Design the optimal trajectory
#
plant = library.QubeServoModel(full_state_output=True, name="qube")

nx, nu, ny = 4, 1, 2  # State, control, measurement dimensions
t0, th = 0.0, 1.0  # Initial and final times
x0 = np.array([0.0, 0.0, 0.0, 0.0])  # Initial state
xf = np.array([0.0, np.pi, 0.0, 0.0])  # Target state
N = 50  # Discretization points

# Optimize the swing-up trajectory, constraining only the final state
t_opt = np.linspace(t0, th, N + 1)
print("Solving for optimal trajectory...")
x_opt, u_opt = collimator.trajopt(
    plant,
    t0=t0,
    tf=th,
    x0=x0,
    xf=xf,
    Q=1e-2 * np.eye(nx),
    R=np.eye(nu),
    N=N,
    constrain_xf=True,
    lb_u=-15.0,
    ub_u=15.0,
)

interp_fun = jax.vmap(jnp.interp, (None, None, 1))

# Offset by at least 2 seconds to make sure the timing is aligned
t_offset = 2.0


def nominal_trajectory_x(t):
    tp = t - t_offset
    return jnp.where(
        (tp > th), xf, jnp.where(tp < 0.0, 0.0, interp_fun(tp, t_opt, x_opt))
    )


def nominal_trajectory_u(t):
    tp = t - t_offset
    return jnp.where((tp > th) | (tp < 0.0), 0.0, interp_fun(tp, t_opt, u_opt))


def make_ffwd_swingup(dt, t0, tf, noise_amplitude=1e2, R=1e0, name="ffwd"):
    # LQR parameters
    Q = np.eye(nx)  # state penalty matrix
    # Q = np.diag([0.0, 1.0, 0.0, 1.0])
    Qf = Q  # terminal state penalty matrix
    N = np.zeros((nx, nu))  # cross cost matrix between state and control vectors

    # Kalman filter parameters
    G = np.eye(nu)
    QN = noise_amplitude * np.eye(nu)
    RN = np.eye(ny)

    ekf = library.ExtendedKalmanFilter.for_continuous_plant(
        library.QubeServoModel(full_state_output=False),
        dt=dt,
        G_func=lambda t: G,
        Q_func=lambda t: QN,
        R_func=lambda t: RN,
        x_hat_0=nominal_trajectory_x(0),
        P_hat_0=1e-4 * np.eye(nx),
        discretized_noise=True,
        name="ekf",
    )

    # create finite-horizon LQR controller with identical nominal and desired trajectories
    lqr = library.FiniteHorizonLinearQuadraticRegulator(
        t0,
        tf,
        library.QubeServoModel(full_state_output=True, name="qube"),
        Qf,
        func_Q=lambda t: Q,
        func_R=lambda t: R * np.eye(nu),
        func_N=lambda t: N,
        func_x_0=nominal_trajectory_x,
        func_u_0=nominal_trajectory_u,
        name="lqr",
    )

    builder = collimator.DiagramBuilder()
    builder.add(ekf, lqr)
    builder.connect(ekf.output_ports[0], lqr.input_ports[0])

    # Export inputs for (u, y)
    builder.export_input(ekf.input_ports[0], "u")
    builder.export_input(ekf.input_ports[1], "y")

    # Export outputs for (u, x_hat)
    builder.export_output(lqr.output_ports[0], "u")
    builder.export_output(ekf.output_ports[0], "x_hat")

    return builder.build(name=name)


def make_diagram(plant, dt, t0, tf, noise_amplitude=1e2, R=1e0):
    swingup_controller = make_ffwd_swingup(dt, t0, tf, noise_amplitude, R)
    balance_controller = make_pid(dt, y_eq=np.array([0.0, np.pi]))  # noqa: F405

    controller = make_switched_controller(  # noqa: F405
        swingup_controller, balance_controller, threshold=0.25
    )

    builder = collimator.DiagramBuilder()
    builder.add(plant, controller)
    builder.connect(plant.output_ports[0], controller.input_ports[0])
    builder.connect(controller.output_ports[0], plant.input_ports[0])

    return builder.build()


#
# Test the feedforward controller
#
dt = 0.001  # 1 kHz control loop
t0 = 0.0
th = t_opt[-1] + t_offset  # Feedforward time horizon

plant_hw = library.QuanserHAL(dt=dt, version=3, hardware=True)
system = make_diagram(plant_hw, dt, t0, th, R=1e1)
context = system.create_context()


# Wait for one second to make sure the communication channel is open
sleep(1.0)

tf = 6.0
results = collimator.simulate(system, context, (0.0, tf))
plant_hw.terminate()
