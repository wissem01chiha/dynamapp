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

import jax.numpy as jnp

from collimator.framework import LeafSystem


class Acrobot(LeafSystem):
    def __init__(
        self,
        x0=jnp.zeros(4),
        m1=2.4367,
        m2=0.6178,
        l1=0.2563,
        lc1=1.6738,
        lc2=1.5651,
        I1=-4.7443,
        I2=-1.0068,
        g=9.81,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.declare_dynamic_parameter("m1", m1)
        self.declare_dynamic_parameter("m2", m2)
        self.declare_dynamic_parameter("l1", l1)
        self.declare_dynamic_parameter("lc1", lc1)
        self.declare_dynamic_parameter("lc2", lc2)
        self.declare_dynamic_parameter("I1", I1)
        self.declare_dynamic_parameter("I2", I2)
        self.declare_dynamic_parameter("g", g)

        self.declare_input_port(name="tau")

        self.declare_continuous_state(shape=(4,), ode=self.ode, default_value=x0)

        self.declare_continuous_state_output()

    def _eval_output(self, time, state, *inputs, **params):
        x = state.continuous_state
        return x

    def ode(self, time, state, *inputs, **parameters):
        q1, q2, dot_q1, dot_q2 = state.continuous_state

        (tau,) = inputs

        m1 = parameters["m1"]
        m2 = parameters["m2"]
        l1 = parameters["l1"]
        lc1 = parameters["lc1"]
        lc2 = parameters["lc2"]
        I1 = parameters["I1"]
        I2 = parameters["I2"]
        g = parameters["g"]

        # Inertia matrix M(q)
        M = jnp.array(
            [
                [
                    I1 + I2 + m2 * l1**2 + 2 * m2 * l1 * lc2 * jnp.cos(q2),
                    I2 + m2 * l1 * lc2 * jnp.cos(q2),
                ],
                [I2 + m2 * l1 * lc2 * jnp.cos(q2), I2],
            ]
        )

        # Coriolis and centrifugal forces matrix C(q, dot_q)
        C = jnp.array(
            [
                [
                    -2 * m2 * l1 * lc2 * jnp.sin(q2) * dot_q2,
                    -m2 * l1 * lc2 * jnp.sin(q2) * dot_q2,
                ],
                [m2 * l1 * lc2 * jnp.sin(q2) * dot_q1, 0.0],
            ]
        )

        # Gravitational torque vector tau_g(q)
        tau_g = jnp.array(
            [
                [
                    -m1 * g * lc1 * jnp.sin(q1)
                    - m2 * g * (l1 * jnp.sin(q1) + lc2 * jnp.sin(q1 + q2))
                ],
                [-m2 * g * lc2 * jnp.sin(q1 + q2)],
            ]
        )

        # Input matrix B
        B = jnp.array([[0], [1]])

        # Right-hand side of the equation
        dot_q = jnp.array([[dot_q1], [dot_q2]])
        rhs = tau_g + jnp.matmul(B, jnp.array([tau])) - jnp.matmul(C, dot_q)

        # Solving for ddot_q
        ddot_q = jnp.dot(jnp.linalg.inv(M), rhs)

        ddot_q1 = ddot_q[0, 0]
        ddot_q2 = ddot_q[1, 0]

        return jnp.array([dot_q1, dot_q2, ddot_q1, ddot_q2])


def animate_acrobot(
    theta1_array, theta2_array, l1=1, l2=2.0, title="Acrobot", interval=50
):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    from IPython.display import HTML

    fig, ax = plt.subplots()
    pad = 0.2
    ax.set_xlim(-l1 - l2 - pad, l1 + l2 + pad)
    ax.set_ylim(-l1 - l2 - pad, l1 + l2 + pad)
    ax.set_title(title)

    (line,) = ax.plot([], [], "o-", lw=2)

    def init():
        line.set_data([], [])
        return (line,)

    def _animate(frame):
        t1 = theta1_array[frame]
        t2 = theta2_array[frame]

        x1, y1 = l1 * jnp.sin(t1), -l1 * jnp.cos(t1)
        x2, y2 = x1 + l2 * jnp.sin(t1 + t2), y1 - l2 * jnp.cos(t1 + t2)

        line.set_data([0, x1, x2], [0, y1, y2])
        return (line,)

    ani = FuncAnimation(
        fig,
        _animate,
        frames=len(theta1_array),
        init_func=init,
        blit=True,
        interval=interval,
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
        return html_content

    try:
        writer = FFMpegWriter(fps=60, metadata={"artist": "Me"}, bitrate=1800)
        savefilename = "acrobot.mp4"
        ani.save(savefilename, writer=writer)
        print(f"Animation saved as {savefilename}")
    except RuntimeError as e:
        print(f"Error saving animation: {e}")
        plt.close()
