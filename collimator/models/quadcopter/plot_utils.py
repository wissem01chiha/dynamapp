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

"""
Plotting utilities for quadcopter
"""

import os
import warnings

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML

from .transformations import euler_to_rotation_matrix


def plot_sol(sol):
    nvars = 6
    state_names = [r"$x$", r"$y$", r"$z$", r"$\phi$", r"$\theta$", r"$\psi$"]
    state_names = state_names + [r"$\dot{" + name[1:-1] + "}$" for name in state_names]
    fig_state, axs_state = plt.subplots(6, 2, figsize=(11, 7))
    for row_index, row in enumerate(axs_state):
        ax1, ax2 = row
        ax1.plot(
            sol.time, sol.outputs["state"][:, row_index], label=state_names[row_index]
        )
        ax2.plot(
            sol.time,
            sol.outputs["state"][:, row_index + nvars],
            label=state_names[row_index + nvars],
        )
        ax1.legend(loc="best")
        ax2.legend(loc="best")
    fig_state.tight_layout()

    fig_control, axs_control = plt.subplots(2, 2, figsize=(11, 3))
    for row_index, ax in enumerate(axs_control.flatten()):
        ax.plot(
            sol.time,
            sol.outputs["control"][:, row_index],
            "-r",
            label=r"$u_" + str(row_index + 1) + r"$",
            alpha=0.5,
        )
        ax.legend(loc="best")
    fig_control.tight_layout()
    return fig_state, fig_control


def animate_quadcopter(
    sol, tar, xlim=[-2.5, 2.5], ylim=[-2.5, 2.5], zlim=[-1.5, 1.5], interval=100
):
    if os.getenv("CI") == "true":
        warnings.warn("Skipping animation in CI environment")
        return

    nskip = 4
    positions = sol[::nskip, :3]
    orientations = sol[::nskip, 3:]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim3d(xlim)
    ax.set_ylim3d(ylim)
    ax.set_zlim3d(zlim)

    ax.plot(tar[:, 0], tar[:, 1], tar[:, 2], "-b")

    # Initialize line objects for each arm with different colors
    line_colors = ["r", "g", "b", "m"]
    lines = [ax.plot([], [], [], color=color, linewidth=2)[0] for color in line_colors]

    body_length = 0.2
    body = jnp.array(
        [
            [body_length, 0, 0],
            [-body_length, 0, 0],
            [0, body_length, 0],
            [0, -body_length, 0],
        ]
    )

    fig.tight_layout()

    def init():
        return lines

    def animate(i):
        R = euler_to_rotation_matrix(*orientations[i])
        body_rotated = jnp.dot(body, R.T) + positions[i]

        # Update the data for each line (arm)
        for j in range(4):
            if j % 2 == 0:  # Horizontal arms
                lines[j].set_data(
                    [positions[i, 0], body_rotated[j, 0]],
                    [positions[i, 1], body_rotated[j, 1]],
                )
                lines[j].set_3d_properties([positions[i, 2], body_rotated[j, 2]])
            else:  # Vertical arms
                lines[j].set_data(
                    [body_rotated[j, 0], positions[i, 0]],
                    [body_rotated[j, 1], positions[i, 1]],
                )
                lines[j].set_3d_properties([body_rotated[j, 2], positions[i, 2]])

        return lines

    ani = FuncAnimation(
        fig,
        animate,
        frames=len(positions),
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
        # display(html_content)
        return html_content

    try:
        writer = FFMpegWriter(fps=60, metadata={"artist": "Me"}, bitrate=1800)
        savefilename = "quadcopter.mp4"
        ani.save(savefilename, writer=writer)
        print(f"Animation saved as {savefilename}")
    except RuntimeError as e:
        print(f"Error saving animation: {e}")
        plt.close()
