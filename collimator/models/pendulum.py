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
import warnings

import jax.numpy as jnp
from ..framework import LeafSystem, DiagramBuilder

from ..library import (
    Demultiplexer,
    Multiplexer,
    Gain,
    Integrator,
    FeedthroughBlock,
    Adder,
)


class Pendulum(LeafSystem):
    """
    Model of a pendulum's dynamics with damping and (optional) external torque.

    Equations of motion:
    𝜃̇ = ω
    ω̇ = -g/L * sin(𝜃) - b/(mL²) * ω + 1/(mL²) * τ


    where:
    - 𝜃 (theta) is the angular displacement,
    - ω (omega) is the angular velocity,
    - g is the acceleration due to gravity,
    - L is the length of the pendulum,
    - b is the damping coefficient,
    - m is the mass of the pendulum,
    - τ (tau) is the external torque.

    The output is the angular displacement 𝜃 or (optionally) the full state [𝜃, ω].
    """

    def __init__(
        self,
        *args,
        x0=[1.0, 0.0],
        m=1.0,
        g=9.81,
        L=1.0,
        b=0.0,
        input_port=False,
        full_state_output=False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.declare_dynamic_parameter("m", m)
        self.declare_dynamic_parameter("g", g)
        self.declare_dynamic_parameter("L", L)
        self.declare_dynamic_parameter("b", b)
        self.declare_continuous_state(default_value=jnp.array(x0), ode=self.ode)

        if input_port:
            self.declare_input_port(name="u")

        if full_state_output:
            self.declare_continuous_state_output(name="x")
        else:

            def _obs_callback(time, state, *inputs, **parameters):
                return state.continuous_state[0]

            self.declare_output_port(
                _obs_callback,
                name="y",
                requires_inputs=False,
            )

    def ode(self, time, state, *inputs, **parameters):
        theta, omega = state.continuous_state
        m = parameters["m"]
        g = parameters["g"]
        L = parameters["L"]
        b = parameters["b"]
        dot_theta = omega

        mLsq = m * L * L
        dot_omega = -(g / L) * jnp.sin(theta) - b * omega / mLsq

        if inputs:
            (tau,) = inputs
            dot_omega += jnp.reshape(tau / mLsq, ())

        return jnp.array([dot_theta, dot_omega])


def PendulumDiagram(
    x0=[1.0, 0.0],
    m=1.0,
    g=9.81,
    L=1.0,
    b=0.0,
    input_port=False,
    full_state_output=False,
    name="pendulum",
):
    """
    Model of a pendulum's dynamics with damping and (optional) external torque.

    Equations of motion:
    𝜃̇ = ω
    ω̇ = -g/L * sin(𝜃) - b/(mL²) * ω + 1/(mL²) * τ


    where:
    - 𝜃 (theta) is the angular displacement,
    - ω (omega) is the angular velocity,
    - g is the acceleration due to gravity,
    - L is the length of the pendulum,
    - b is the damping coefficient,
    - m is the mass of the pendulum,
    - τ (tau) is the external torque.

    The output is the angular displacement 𝜃 or (optionally) the full state [𝜃, ω].
    """
    builder = DiagramBuilder()

    Integrator_0 = builder.add(Integrator(x0, name="Integrator_0"))
    Demux_0 = builder.add(Demultiplexer(2, name="Demux_0"))
    builder.connect(Integrator_0.output_ports[0], Demux_0.input_ports[0])

    Sine_0 = builder.add(FeedthroughBlock(lambda x: jnp.sin(x), name="Sine_0"))
    builder.connect(Demux_0.output_ports[0], Sine_0.input_ports[0])

    Gain_0 = builder.add(Gain(-g / L, name="Gain_0"))
    builder.connect(Sine_0.output_ports[0], Gain_0.input_ports[0])

    # Damping
    Gain_1 = builder.add(Gain(-b / m / L / L, name="Gain_1"))
    builder.connect(Demux_0.output_ports[1], Gain_1.input_ports[0])

    Adder_0 = builder.add(Adder(2, name="Adder_0"))
    builder.connect(Gain_0.output_ports[0], Adder_0.input_ports[0])
    builder.connect(Gain_1.output_ports[0], Adder_0.input_ports[1])

    # Add an optional torque input
    if input_port:
        Gain_2 = builder.add(Gain(1.0 / m / L / L, name="Gain_2"))

        Adder_1 = builder.add(Adder(2, name="Adder_1"))
        builder.connect(Adder_0.output_ports[0], Adder_1.input_ports[0])
        builder.connect(Gain_2.output_ports[0], Adder_1.input_ports[1])

        Mux_0 = builder.add(Multiplexer(2, name="Mux_0"))
        builder.connect(Demux_0.output_ports[1], Mux_0.input_ports[0])
        builder.connect(Adder_1.output_ports[0], Mux_0.input_ports[1])
        builder.connect(Mux_0.output_ports[0], Integrator_0.input_ports[0])

        # Diagram-level inport
        builder.export_input(Gain_2.input_ports[0])
    else:
        Mux_0 = builder.add(Multiplexer(2, name="Mux_0"))
        builder.connect(Demux_0.output_ports[1], Mux_0.input_ports[0])
        builder.connect(Adder_0.output_ports[0], Mux_0.input_ports[1])
        builder.connect(Mux_0.output_ports[0], Integrator_0.input_ports[0])

    if full_state_output:
        builder.export_output(Integrator_0.output_ports[0])
    else:
        # Partial observation
        builder.export_output(Demux_0.output_ports[0])

    return builder.build(name=name)


def animate_pendulum(theta_vec, u_vec, t_vec, length=1.0, interval=10):
    if os.getenv("CI") == "true":
        warnings.warn("Skipping animation in CI environment")
        return

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from IPython.display import HTML

    # Convert theta to Cartesian coordinates
    x = length * jnp.sin(theta_vec)
    y = -length * jnp.cos(theta_vec)

    # Convert theta to Cartesian coordinates
    x = length * jnp.sin(theta_vec)
    y = -length * jnp.cos(theta_vec)

    # Create the plot with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # First subplot for the pendulum
    (line,) = ax1.plot([], [], "o-", lw=2)
    ax1.set_xlim(-length - 0.5, length + 0.5)
    ax1.set_ylim(-length - 0.5, length + 0.5)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_title("Pendulum Animation")

    # Second subplot for the control input
    (line_u,) = ax2.plot([], [], lw=2)
    ax2.set_xlim(0, t_vec[-1])
    ax2.set_ylim(min(u_vec) - 0.5, max(u_vec) + 0.5)
    ax2.set_title("Control Input")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Control Input")

    # Initialize the animation
    def init():
        line.set_data([], [])
        line_u.set_data([], [])
        return line, line_u

    # Update function for the animation
    def update(frame):
        # Update the pendulum
        line.set_data([0, x[frame]], [0, y[frame]])

        # Update the control input plot
        line_u.set_data(t_vec[: frame + 1], u_vec[: frame + 1])

        return line, line_u

    # Create the animation
    ani = FuncAnimation(
        fig, update, frames=len(theta_vec), init_func=init, blit=True, interval=interval
    )

    # Check if the code is running in a Jupyter notebook
    try:
        cfg = get_ipython().config  # noqa: F841
        in_notebook = True
    except NameError:
        in_notebook = False

    if in_notebook:
        plt.close(fig)
        html_content = HTML(ani.to_html5_video())
        # display(html_content)
        return html_content

    try:
        writer = FFMpegWriter(fps=60, metadata={"artist": "Me"}, bitrate=1800)
        savefilename = "planar_quadrotor.mp4"
        ani.save(savefilename, writer=writer)
        print(f"Animation saved as {savefilename}")
    except RuntimeError as e:
        print(f"Error saving animation: {e}")
        plt.close()
