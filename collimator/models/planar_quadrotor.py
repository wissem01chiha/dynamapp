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

from ..framework import LeafSystem


class PlanarQuadrotor(LeafSystem):
    def __init__(self, *args, m=1.0, I_B=1.0, r=0.5, g=9.81, **kwargs):
        super().__init__(*args, **kwargs)
        self.declare_dynamic_parameter("m", m)
        self.declare_dynamic_parameter("I_B", I_B)
        self.declare_dynamic_parameter("r", r)
        self.declare_dynamic_parameter("g", g)

        self.declare_input_port(name="u")

        self.declare_continuous_state(shape=(6,), ode=self.ode)
        self.declare_continuous_state_output()

    def ode(self, time, state, *inputs, **parameters):
        x, y, θ, dx, dy, dθ = state.continuous_state
        (u,) = inputs

        m = parameters["m"]
        I_B = parameters["I_B"]
        r = parameters["r"]
        g = parameters["g"]

        ddx = -(1 / m) * (u[0] + u[1]) * jnp.sin(θ)
        ddy = (1 / m) * (u[0] + u[1]) * jnp.cos(θ) - g
        ddθ = (1 / I_B) * r * (u[0] - u[1])

        return jnp.array([dx, dy, dθ, ddx, ddy, ddθ])

    def animate(
        self,
        xf,
        xlim=None,
        ylim=None,
        figsize=(4, 4),
        interval=50,
        stride=1,
        notebook=True,
    ):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from IPython import display

        xf = xf[:, ::stride]

        xc, yc, θc = xf[0], xf[1], xf[2]

        fig, ax = plt.subplots(figsize=figsize)

        hx, hy = 0.4, 0.1
        r2d = 180 / np.pi
        body = plt.Rectangle(
            (0, 0),
            hx,
            hy,
            angle=0,
            rotation_point="center",
            fc="xkcd:light grey",
            ec="xkcd:light grey",
        )

        ax.add_patch(body)

        if xlim is None:
            xlim = [-1.5, 1.5]
        if ylim is None:
            ylim = [-2, 2]

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.grid()

        def _animate(i):
            # First set xy to be the center, then rotate, then move xy to bottom left
            body.set_xy((xc[i], yc[i]))
            body.angle = θc[i] * r2d
            body.set_xy((xc[i] - hx / 2, yc[i] - hy / 2))

            return (body,)

        anim = FuncAnimation(fig, _animate, frames=xf.shape[1], interval=interval)

        if not notebook:
            return anim

        video = anim.to_html5_video()
        html = display.HTML(video)
        display.display(html)
        plt.close()  # avoid plotting a spare static plot


def animate_planar_quadrotor(
    xf, x_eq=jnp.zeros(6), xlim=None, ylim=None, figsize=(4, 4), interval=50, stride=1
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from IPython.display import HTML

    xf = xf[::stride, :]

    xc, yc, theta_c = xf[:, 0], xf[:, 1], xf[:, 2]

    fig, ax = plt.subplots(figsize=figsize)

    # Quadcopter body dimensions
    hx, hy = 0.6, 0.05
    r2d = 180 / jnp.pi

    # Body of the quadcopter
    body = plt.Rectangle(
        (0, 0),
        hx,
        hy,
        angle=0,
        rotation_point="center",
        fc="xkcd:medium blue",
        ec="xkcd:medium blue",
    )
    ax.add_patch(body)

    # Rotor dimensions
    rotor_width, rotor_height = 0.1, 0.02

    # Rotors of the quadcopter
    rotor1 = plt.Rectangle(
        (-hx / 4, hy / 2),
        rotor_width,
        rotor_height,
        angle=0,
        rotation_point="center",
        fc="xkcd:red",
        ec="xkcd:red",
    )
    rotor2 = plt.Rectangle(
        (hx / 4 - rotor_width, hy / 2),
        rotor_width,
        rotor_height,
        angle=0,
        rotation_point="center",
        fc="xkcd:red",
        ec="xkcd:red",
    )
    ax.add_patch(rotor1)
    ax.add_patch(rotor2)

    # Set plot limits
    if xlim is None:
        xlim = [-1.5, 1.5]
    if ylim is None:
        ylim = [-2, 2]

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid()

    def _animate(i):
        # Update the body position and rotation
        body_center_x = xc[i]
        body_center_y = yc[i]
        body.set_xy((body_center_x - hx / 2, body_center_y - hy / 2))
        body.angle = theta_c[i] * r2d

        # Calculate the new positions for the rotors
        cos_angle = jnp.cos(theta_c[i])
        sin_angle = jnp.sin(theta_c[i])

        # Offset from the center of the body to the center of the rotors
        rotor_offset_x = hx / 3
        rotor_offset_y = hy / 2 + rotor_height

        # Calculate the new positions of the rotor centers
        rotor1_center_x = (
            body_center_x + rotor_offset_x * cos_angle - rotor_offset_y * sin_angle
        )
        rotor1_center_y = (
            body_center_y + rotor_offset_x * sin_angle + rotor_offset_y * cos_angle
        )
        rotor2_center_x = (
            body_center_x - rotor_offset_x * cos_angle - rotor_offset_y * sin_angle
        )
        rotor2_center_y = (
            body_center_y - rotor_offset_x * sin_angle + rotor_offset_y * cos_angle
        )

        # Set the positions of the rotors
        rotor1.set_xy(
            (rotor1_center_x - rotor_width / 2, rotor1_center_y - rotor_height / 2)
        )
        rotor2.set_xy(
            (rotor2_center_x - rotor_width / 2, rotor2_center_y - rotor_height / 2)
        )

        # Set the rotation angle for the rotors
        rotor1.angle = body.angle
        rotor2.angle = body.angle

        return (body, rotor1, rotor2)

    ani = FuncAnimation(fig, _animate, frames=xf.shape[0], interval=interval)

    # Check if the code is running in a Jupyter notebook
    try:
        cfg = get_ipython().config  # noqa: F841
        in_notebook = True
    except NameError:
        in_notebook = False

    if in_notebook:
        plt.close(fig)
        html_content = HTML(ani.to_html5_video())
        return html_content

    try:
        writer = FFMpegWriter(fps=60, metadata={"artist": "Me"}, bitrate=1800)
        savefilename = "pendulum.mp4"
        ani.save(savefilename, writer=writer)
        print(f"Animation saved as {savefilename}")
    except RuntimeError as e:
        print(f"Error saving animation: {e}")
        plt.close()
