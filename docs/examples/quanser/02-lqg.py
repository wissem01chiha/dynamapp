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

import collimator
from collimator import library

from controllers import *  # noqa: F403

HOME = os.path.expanduser("~")
sys.path.append(f"{HOME}/Quanser/libraries/python")

# Linearize the system and discretize in time
x_eq = np.array([0.0, 0.0, 0.0, 0.0])
y_eq = x_eq[:2]
u_eq = np.array([0.0])

plant = library.QubeServoModel(full_state_output=False, name="qube")
u0 = np.array([0.0])
plant.input_ports[0].fix_value(u0)
context = plant.create_context()

base_context = context.with_continuous_state(x_eq)

# Linearize and discretize in time
lin_sys = library.linearize(plant, base_context)

dt = 0.01  # 100 Hz sampling rate
dt_sys = lin_sys.sample(dt)

n = 4  # State dimension
m = 1  # Input dimension
p = 2  # Output dimension

# Covariance matrices for LQR
Q = np.eye(n)
R = 1e-2 * np.eye(m)

# Covariance matrices for Kalman filter
noise_scale = 1e2
QN = noise_scale * np.eye(m)
RN = np.eye(p)


def make_lqg_diagram(plant, controller, pulse_amplitude=1.0, name="root"):
    # Add initial impulse
    pulse = make_pulse(amplitude=pulse_amplitude, start_time=2.0, width=0.1)  # noqa: F405
    adder = library.Adder(2, name="adder")

    builder = collimator.DiagramBuilder()
    builder.add(plant, controller, pulse, adder)

    # Connect the plant to the LQG controller
    builder.connect(plant.output_ports[0], controller.input_ports[0])

    builder.connect(controller.output_ports[0], adder.input_ports[0])
    builder.connect(pulse.output_ports[0], adder.input_ports[1])
    builder.connect(adder.output_ports[0], plant.input_ports[0])

    return builder.build(name=name)


vhil_plant = library.QuanserHAL(dt=dt, hardware=True, version=3, name="Qube")
balance_controller = make_lqg(dt_sys, dt=dt, Q=Q, R=R, QN=QN, RN=RN, x0=None)  # noqa: F405
system = make_lqg_diagram(vhil_plant, balance_controller, pulse_amplitude=0.0)
system.pprint()
context = system.create_context()

tf = 10.0

print("Sleeping for 1 second to allow the interface to start...")
sleep(1)

results = collimator.simulate(
    system,
    context,
    (0.0, tf),
)

vhil_plant.terminate()
