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

from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from ..framework import LeafSystem


class RimlessWheel(LeafSystem):
    """Highly simplified model of 'walking' on a ramp.

    https://underactuated.csail.mit.edu/simple_legs.html#section2
    """

    def __init__(self, *args, g=9.81, L=1.0, N_spokes=7, gamma=np.pi / 12, **kwargs):
        super().__init__(*args, **kwargs)
        self.declare_dynamic_parameter(
            "alpha", np.pi / N_spokes
        )  # Half-angle between spokes
        self.declare_dynamic_parameter("g", g)  # Gravity
        self.declare_dynamic_parameter("L", L)  # Leg length
        self.declare_dynamic_parameter("gamma", gamma)  # Ramp angle

        # For plotting/animation
        self.N_spokes = N_spokes

        # theta, theta_dot states
        self.declare_continuous_state(shape=(2,), ode=self.ode)
        self.declare_continuous_state_output()

        # "toe" position (for visualization)
        self.declare_discrete_state(shape=(), default_value=0.0)
        self.declare_output_port(self.toe_output)

        self.declare_zero_crossing(
            guard=self.downhill_guard,
            reset_map=self.downhill_reset,
            name="downhill_collision",
            direction="positive_then_non_positive",
        )

        self.declare_zero_crossing(
            guard=self.uphill_guard,
            reset_map=self.uphill_reset,
            name="uphill_collision",
            direction="positive_then_non_positive",
        )

    def toe_output(self, time, state, **parameters):
        return state.discrete_state

    def ode(self, time, state, **parameters):
        θ, dθ = state.continuous_state
        g, L = parameters["g"], parameters["L"]
        return jnp.array([dθ, g / L * jnp.sin(θ)])

    def downhill_guard(self, time, state, **parameters):
        θ, dθ = state.continuous_state
        gamma, alpha = parameters["gamma"], parameters["alpha"]
        return (gamma + alpha) - θ

    def downhill_reset(self, time, state, **parameters):
        θm, dθm = state.continuous_state  # "Minus" states
        L, gamma, alpha = parameters["L"], parameters["gamma"], parameters["alpha"]

        # "Plus" states
        θp = gamma - alpha
        dθp = dθm * jnp.cos(2 * alpha)

        xc = jnp.array([θp, dθp])
        xd = state.discrete_state  # toe position
        xd += 2 * L * jnp.sin(alpha)

        state = state.with_continuous_state(xc)
        state = state.with_discrete_state(xd)
        return state

    def uphill_guard(self, time, state, **parameters):
        θ, dθ = state.continuous_state
        gamma, alpha = parameters["gamma"], parameters["alpha"]
        return (gamma - alpha) - θ

    def uphill_reset(self, time, state, **parameters):
        θm, dθm = state.continuous_state  # "Minus" states
        L, gamma, alpha = parameters["L"], parameters["gamma"], parameters["alpha"]

        # "Plus" states
        θp = gamma + alpha
        dθp = dθm * jnp.cos(2 * alpha)

        xc = jnp.array([θp, dθp])
        xd = state.discrete_state  # toe position
        xd -= 2 * L * jnp.sin(alpha)  # Update toe position

        state = state.with_continuous_state(xc)
        state = state.with_discrete_state(xd)
        return state

    def cm(self, xc, toe, p, **kwargs):
        gamma, L = p["gamma"], p["L"]
        θ, _ = xc[:, 0], xc[:, 1]
        x = toe * jnp.cos(gamma) + L * jnp.sin(θ)
        y = -toe * jnp.sin(gamma) + L * jnp.cos(θ)
        return x, y

    def plot_trajectory(self, xc, toe, p, **kwargs):
        import matplotlib.pyplot as plt

        plt.figure(figsize=kwargs.get("figsize", (7, 2)))
        x, y = self.cm(xc, toe, p)

        plt.plot(x, y, lw=2)
        plt.grid()
        plt.show()

    def animate(self, t, xc, toe, p, **kwargs):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        from IPython import display

        gamma, alpha, L = p["gamma"], p["alpha"], p["L"]

        if kwargs.get("interpolation", True):
            dt = kwargs.get("dt", 0.01)
            t_interp = np.arange(t[0], t[-1], dt)
            xc = jax.vmap(partial(jnp.interp, t_interp, t))(xc.T).T
            toe = np.interp(t_interp, t, toe)

        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (7, 3)))

        xlim = kwargs.get("xlim", [-1.0, 10.0])
        ylim = kwargs.get("ylim", [-1.0, 1.0])

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")

        θ = xc[:, 0]
        xcm, ycm = self.cm(xc, toe, p)
        xt, yt = toe * np.cos(gamma), -toe * np.sin(gamma)

        def rotate(x, y, θ):
            return x * np.cos(θ) - y * np.sin(θ), x * np.sin(θ) + y * np.cos(θ)

        # Plot ramp based on starting toe location
        x0, y0 = toe[0] * np.cos(gamma), -toe[0] * np.sin(gamma)

        # Compute equation of ramp in point-slope form
        m = np.tan(-gamma)
        b = y0 - m * x0
        xr = np.linspace(0.9 * xlim[0], 1.1 * xlim[1], 100)
        yr = m * xr + b

        ramp_color = kwargs.get("ramp_color", None)
        (l_ramp,) = ax.plot(xr, yr, "-", c=ramp_color)

        # l_m, = ax.plot([], 'ko', markersize=10)
        wheel_color = kwargs.get("wheel_color", None)
        (l_cm,) = ax.plot([], "o", c=wheel_color, markersize=10)
        wheel_color = l_cm.get_color()
        (l_toe,) = ax.plot([], "-", c=wheel_color)

        l_leg = []
        for _ in range(self.N_spokes):
            l_leg.append(ax.plot([], "-", c=wheel_color)[0])

        def _animate(i):
            l_cm.set_data([xcm[i]], [ycm[i]])
            l_toe.set_data([xcm[i], xt[i]], [ycm[i], yt[i]])
            for j in range(self.N_spokes):
                xl, yl = rotate(0, 1, (j * 2 - 1) * alpha - θ[i])
                l_leg[j].set_data([xcm[i], xcm[i] + L * xl], [ycm[i], ycm[i] + L * yl])

            return l_cm, l_toe, *l_leg

        anim = FuncAnimation(
            fig,
            _animate,
            frames=xc.shape[0],
            interval=kwargs.get("interval", 20),
            blit=True,
        )

        if not kwargs.get("notebook", True):
            return anim

        video = anim.to_html5_video()
        html = display.HTML(video)
        display.display(html)
        plt.close()  # avoid plotting a spare static plot
