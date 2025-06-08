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
import jax.numpy as jnp

import collimator
from collimator import library


def make_pulse(amplitude, start_time, width, name="Pulse"):
    builder = collimator.DiagramBuilder()

    # Construct the input signal as the difference of two
    # unit step functions spaced `width` seconds apart
    step1 = library.Step(
        start_value=0.0,
        end_value=amplitude,
        step_time=start_time,
        name="step1",
    )
    step2 = library.Step(
        start_value=0.0,
        end_value=amplitude,
        step_time=start_time + width,
        name="step2",
    )
    adder = library.Adder(2, operators="+-", name="adder")

    # Connect the adder block to the input port of the plant
    builder.add(step1, step2, adder)
    builder.connect(step1.output_ports[0], adder.input_ports[0])
    builder.connect(step2.output_ports[0], adder.input_ports[1])

    builder.export_output(adder.output_ports[0], "u")
    return builder.build(name=name)


class WrapAngle(library.FeedthroughBlock):
    """Center the angle to be within [-pi, pi], with zero at the upright."""

    def __init__(self, name="WrapAngle"):
        super().__init__(self._func, name=name)

    def _func(self, x):
        theta, alpha = x
        return jnp.array(
            [
                theta,
                jnp.mod(alpha, 2 * np.pi) - np.pi,
            ]
        )


def make_energy_shaping(dt, kE=0.01, name="EnergyShaping"):
    # Heuristic Energy-shaping controller
    # V = -kE * (1 - cos(alpha)) * alpha_dot
    #
    # Input port: alpha
    # Output port: voltage

    builder = collimator.DiagramBuilder()

    demux = builder.add(library.Demultiplexer(2, name="y"))

    cos_alpha = builder.add(library.Trigonometric("cos", name="cos_alpha"))
    builder.connect(demux.output_ports[1], cos_alpha.input_ports[0])

    offset = builder.add(library.Constant(1.0, name="offset"))
    energy = builder.add(library.Adder(2, operators="-+", name="energy"))
    builder.connect(cos_alpha.output_ports[0], energy.input_ports[0])
    builder.connect(offset.output_ports[0], energy.input_ports[1])

    # DerivativeDiscrete block
    alpha_dot = builder.add(
        library.DerivativeDiscrete(
            dt=dt,
            filter_type="bilinear",
            filter_coefficient=100.0,
            name="alpha_dot",
        )
    )
    builder.connect(demux.output_ports[1], alpha_dot.input_ports[0])

    product = builder.add(library.Product(2, name="product"))
    builder.connect(energy.output_ports[0], product.input_ports[0])
    builder.connect(alpha_dot.output_ports[0], product.input_ports[1])

    gain = builder.add(library.Gain(kE, name="gain"))
    builder.connect(product.output_ports[0], gain.input_ports[0])

    # Dummy block for the input signal (not used for this type of control)
    u_in = builder.add(library.IOPort(name="u"))

    builder.export_input(u_in.input_ports[0], "u")
    builder.export_input(demux.input_ports[0], "y")

    builder.export_output(gain.output_ports[0], "voltage")

    return builder.build(name=name)


def make_pid(
    dt,
    rotor_gains=(0.3102, 0.0, 0.6527),
    pendulum_gains=(18.3637, 0.0, 1.416),
    y_eq=None,
    filter_coefficient=100.0,
    name="DualPID",
):
    # Dual PID controllers
    # Input ports: theta, alpha
    # Output port: voltage

    if y_eq is None:
        y_eq = np.array([0.0, 0.0])

    builder = collimator.DiagramBuilder()

    demux = builder.add(library.Demultiplexer(2, name="y"))

    y_offset = builder.add(library.Offset(-y_eq, name="y_offset"))  # pylint: disable=E1130
    builder.connect(y_offset.output_ports[0], demux.input_ports[0])

    # PID control on pendulum angle
    kp, ki, kd = pendulum_gains
    pid_alpha = builder.add(
        library.PIDDiscrete(
            kp=kp,
            ki=ki,
            kd=kd,
            dt=dt,
            name="pid_alpha",
            filter_type="bilinear",
            filter_coefficient=filter_coefficient,
        )
    )

    builder.connect(demux.output_ports[1], pid_alpha.input_ports[0])

    # PID control on rotary position
    kp, ki, kd = rotor_gains
    pid_theta = builder.add(
        library.PIDDiscrete(
            kp=kp,
            ki=ki,
            kd=kd,
            dt=dt,
            name="pid_theta",
            filter_type="bilinear",
            filter_coefficient=filter_coefficient,
        )
    )

    builder.connect(demux.output_ports[0], pid_theta.input_ports[0])

    voltage = builder.add(library.Adder(2, operators="++", name="voltage"))
    builder.connect(pid_alpha.output_ports[0], voltage.input_ports[0])
    builder.connect(pid_theta.output_ports[0], voltage.input_ports[1])

    # Dummy block for the input signal (not used for this type of control)
    u_in = builder.add(library.IOPort(name="u"))

    builder.export_input(u_in.input_ports[0], "u")
    builder.export_input(y_offset.input_ports[0], "y")

    builder.export_output(voltage.output_ports[0], "voltage")

    return builder.build(name=name)


def make_switched_controller(
    swingup_controller,
    balance_controller,
    y_eq=None,
    threshold=0.35,
    control_limit=15.0,
    name="controller",
):
    # Full swing-up controller. Energy-shaping control is used when the pendulum
    # is more than 10 degrees from the upright position. PID control is used when
    # the pendulum is within 10 degrees of the upright position.
    #
    # Input ports: theta, alpha
    # Output port: voltage

    if y_eq is None:
        y_eq = np.zeros(2)

    builder = collimator.DiagramBuilder()
    builder.add(swingup_controller, balance_controller)

    y_in = builder.add(library.IOPort(name="y"))

    # Use the wrapped value (mod 2pi) to determine the switching point,
    # but use the raw value for the controllers (since they need smooth
    # estimates for derivatives)
    y_wrapped = builder.add(WrapAngle(name="y_centered"))
    y_offset = builder.add(library.Offset(-y_eq, name="y_offset"))  # pylint: disable=E1130
    builder.connect(y_in.output_ports[0], y_wrapped.input_ports[0])
    builder.connect(y_in.output_ports[0], y_offset.input_ports[0])

    builder.connect(y_offset.output_ports[0], swingup_controller.input_ports[1])
    builder.connect(y_offset.output_ports[0], balance_controller.input_ports[1])

    threshold = builder.add(
        library.Constant(threshold, name="threshold")
    )  # About 10 degrees
    abs_alpha = builder.add(library.Abs(name="abs_alpha"))
    demux = builder.add(library.Demultiplexer(2, name="demux"))  # [rotor, pendulum]
    builder.connect(y_wrapped.output_ports[0], demux.input_ports[0])
    builder.connect(demux.output_ports[1], abs_alpha.input_ports[0])

    near_upright = builder.add(library.Comparator(operator="<", name="near_upright"))
    builder.connect(abs_alpha.output_ports[0], near_upright.input_ports[0])
    builder.connect(threshold.output_ports[0], near_upright.input_ports[1])

    switch = builder.add(library.IfThenElse(name="switch"))
    builder.connect(near_upright.output_ports[0], switch.input_ports[0])
    builder.connect(balance_controller.output_ports[0], switch.input_ports[1])
    builder.connect(swingup_controller.output_ports[0], switch.input_ports[2])

    voltage = builder.add(
        library.Saturate(
            upper_limit=control_limit, lower_limit=-control_limit, name="voltage"
        )
    )
    builder.connect(switch.output_ports[0], voltage.input_ports[0])

    # Some of the controllers will also need the control signal for state estimation
    builder.connect(voltage.output_ports[0], swingup_controller.input_ports[0])
    builder.connect(voltage.output_ports[0], balance_controller.input_ports[0])

    builder.export_input(y_in.input_ports[0], "y")
    builder.export_output(voltage.output_ports[0], "voltage")

    return builder.build(name=name)


def make_lqg(dt_sys, dt, Q, R, QN, RN, name="LQG", x0=None):
    nx = 4
    if x0 is None:
        x0 = np.zeros(nx)
    kf = library.InfiniteHorizonKalmanFilter(
        dt=dt,
        A=dt_sys.A,
        B=dt_sys.B,
        C=dt_sys.C,
        D=dt_sys.D,
        G=0 * dt_sys.B,
        Q=QN,
        R=RN,
        x_hat_0=x0,
        name="kf",
    )
    lqr = library.DiscreteTimeLinearQuadraticRegulator(
        dt=dt, A=dt_sys.A, B=dt_sys.B, Q=Q, R=R, name="lqr"
    )

    # Construct the closed-loop block diagram
    builder = collimator.DiagramBuilder()
    builder.add(kf, lqr)

    # One input port: y
    builder.export_input(kf.input_ports[1], name="y")

    builder.connect(kf.output_ports[0], lqr.input_ports[0])
    builder.connect(lqr.output_ports[0], kf.input_ports[0])

    # Two output ports: u and x_hat
    builder.export_output(lqr.output_ports[0], name="u")
    builder.export_output(kf.output_ports[0], name="x_hat")

    return builder.build(name=name)


# Controller subsystem
def make_mlp_controller(
    nn_config,
    dt,
    sigma,
    delay=0.0,
    filter_coefficient=100.0,
    invert_input=False,
    name="controller",
):
    builder = collimator.DiagramBuilder()

    net = builder.add(library.MLP(name="net", **nn_config))

    # Estimate time derivatives
    theta_dot = builder.add(
        library.DerivativeDiscrete(
            dt=dt,
            filter_type="bilinear",
            filter_coefficient=filter_coefficient,
            name="theta_dot",
        )
    )

    alpha_dot = builder.add(
        library.DerivativeDiscrete(
            dt=dt,
            filter_type="bilinear",
            filter_coefficient=filter_coefficient,
            name="alpha_dot",
        )
    )

    full_state = builder.add(library.Multiplexer(4, name="full_state"))

    # FIXME: The neural network was trained on (y_r - y) instead
    # of (y - y_r), so we might need to invert the input here in order
    # to use the common interface of controllers.py
    inv_gain = -1 if invert_input else 1
    y_in = builder.add(library.Gain(inv_gain, name="y_in"))

    measurements = builder.add(library.Demultiplexer(2, name="measurements"))
    builder.connect(y_in.output_ports[0], measurements.input_ports[0])

    builder.connect(measurements.output_ports[0], full_state.input_ports[0])
    builder.connect(measurements.output_ports[1], full_state.input_ports[1])
    builder.connect(measurements.output_ports[0], theta_dot.input_ports[0])
    builder.connect(theta_dot.output_ports[0], full_state.input_ports[2])
    builder.connect(measurements.output_ports[1], alpha_dot.input_ports[0])
    builder.connect(alpha_dot.output_ports[0], full_state.input_ports[3])

    # Reference signal error
    builder.connect(full_state.output_ports[0], net.input_ports[0])

    # Add noise to the control signal (helpful for training)
    noise = builder.add(
        library.WhiteNoise(dt, noise_power=sigma**2, name="noise", shape=(1,))
    )

    # Add a zero-order hold to avoid continuous re-inference
    adder = builder.add(library.Adder(2, name="noisy_ctrl"))
    builder.connect(net.output_ports[0], adder.input_ports[0])
    builder.connect(noise.output_ports[0], adder.input_ports[1])

    # Add a zero-order hold to avoid continuous re-inference
    zoh = builder.add(library.ZeroOrderHold(dt, name="zoh"))
    builder.connect(adder.output_ports[0], zoh.input_ports[0])

    # Make the output a scalar
    flatten = builder.add(library.Slice("0"))
    builder.connect(zoh.output_ports[0], flatten.input_ports[0])

    # Hold the control signal to zero for the delay period (allows comms
    # to be ready).  Implemented as a unit step function multiplied by the
    # control signal.
    delay_switch = builder.add(
        library.Step(
            start_value=0.0, end_value=1.0, step_time=delay, name="delay_switch"
        )
    )
    u_out = builder.add(library.Product(2, name="u_out"))
    builder.connect(delay_switch.output_ports[0], u_out.input_ports[0])
    builder.connect(flatten.output_ports[0], u_out.input_ports[1])

    # Two inputs: u and y
    # The control signal is not used for anything here, but is needed
    # for compatibility with other controllers that use, e.g. EKF estimators
    u_in = builder.add(library.IOPort(name="u_in"))
    builder.export_input(u_in.input_ports[0])
    builder.export_input(y_in.input_ports[0])

    # Two outputs: u and x_hat
    builder.export_output(u_out.output_ports[0])
    builder.export_output(full_state.output_ports[0])

    return builder.build(name=name)
