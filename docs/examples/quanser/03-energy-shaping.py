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

import sys
import os
from time import sleep

import numpy as np
import control

import collimator
from collimator import library

from controllers import *  # noqa: F403

HOME = os.path.expanduser("~")
sys.path.append(f"{HOME}/Quanser/libraries/python")


def make_diagram(
    plant,
    dt,
    rotor_gains,
    pendulum_gains,
    pulse_amplitude=1.0,
    kE=0.01,
    name="root",
):
    y_eq = np.array([0.0, np.pi])

    builder = collimator.DiagramBuilder()

    swingup_controller = make_energy_shaping(dt, kE=kE)  # noqa: F405
    pid_controller = make_pid(dt, rotor_gains, pendulum_gains)  # noqa: F405

    controller = make_switched_controller(swingup_controller, pid_controller, y_eq)  # noqa: F405

    # Add initial impulse to start the swing-up controller
    pulse = make_pulse(amplitude=pulse_amplitude, start_time=2.0, width=0.1)  # noqa: F405
    adder = library.Adder(2, name="adder")

    builder.add(controller, plant, pulse, adder)
    builder.connect(pulse.output_ports[0], adder.input_ports[0])
    builder.connect(controller.output_ports[0], adder.input_ports[1])
    builder.connect(adder.output_ports[0], plant.input_ports[0])
    builder.connect(plant.output_ports[0], controller.input_ports[0])

    return builder.build(name=name)


# Design the balancing controller using LQR
def design_gains(dt):
    x_eq = np.array([0.0, np.pi, 0.0, 0.0])
    u_eq = np.array([0.0])
    plant = library.QubeServoModel(full_state_output=False, name="qube")
    plant.input_ports[0].fix_value(u_eq)
    context = plant.create_context()

    base_context = context.with_continuous_state(x_eq)

    # Linearize and discretize in time
    lin_sys = library.linearize(plant, base_context).sample(dt)

    # Design a state feedback controller
    n = 4
    m = 1
    Q = np.eye(n)  # State cost weighting matrix
    R = 1e1 * np.eye(m)  # Input cost weighting matrix
    K, _, _ = control.dlqr(lin_sys.A, lin_sys.B, Q, R)

    rotor_gains = (-K[0, 0], 0.0, -K[0, 2])  # PD control on theta and theta_dot
    pendulum_gains = (-K[0, 1], 0.0, -K[0, 3])  # PD control on alpha and alpha_dot

    return rotor_gains, pendulum_gains


# Sampling rate (1 kHz)
dt = 1.0 / 1000

# PID gains for the rotor and pendulum based on LQR analysis
rotor_gains, pendulum_gains = design_gains(dt)

plant_hw = library.QuanserHAL(dt=dt, hardware=True, version=3, name="Qube")
system = make_diagram(
    plant_hw,
    dt,
    rotor_gains,
    pendulum_gains,
    kE=0.03,
    pulse_amplitude=1.0,
)
context = system.create_context()

print("Sleeping for 1 second to allow the interface to start...")
sleep(1)

tf = 10.0
results = collimator.simulate(
    system,
    context,
    (0.0, tf),
)

# Make sure to disconnect from the interface
plant_hw.terminate()
